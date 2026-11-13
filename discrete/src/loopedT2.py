# -*- coding: utf-8 -*-
"""LOOPED-T2: layer1 (6 heads, frozen) + ONE shared 6-head table applied twice (true weight-tied loop).
Forward:  z1 = argmax(L1 factors);  z2 = argmax(L1 * B[z1] factors);  q = norm(L1 * B[z1] * B[z2]);
          P(y|ctx) = sum_z q(z) O[y,z]
Train: EM. B gets statistics from BOTH rounds (NUM[6+i] + NUM[12+i]). Layer-1 and O handled as usual."""
import os, sys, torch, time, math, json, numpy as np
os.environ['TORCH_CUDA_ARCH_LIST']='8.9'
torch.set_grad_enabled(False)
from torch.utils.cpp_extension import load_inline
from gpu_guard import lease
cuda=open('estep_kernels.cu',encoding='utf-8').read()
cpp=("void estep2_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);\n"
     "void argmax_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);\n"
     "void estep_out_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);")
KER=load_inline(name='dna_estep4k',cpp_sources=cpp,cuda_sources=cuda,functions=['estep2_run','argmax_run','estep_out_run'],
                verbose=False,extra_cuda_cflags=['-O3','-Xcompiler','/Zc:preprocessor','--use_fast_math'])
DEV='cuda'; V=50257; K=1024; L=1024; CH=20000
def log(*a): print('[%s]'%time.strftime('%H:%M:%S'),*a,flush=True)
f=np.load('dna_cache/freq.npy'); s=np.flatnonzero(f>0); NS=len(s)
TM=torch.tensor(np.full(V,NS,dtype=np.int64),device=DEV); TM[s]=torch.arange(NS,device=DEV)
C={k:torch.tensor(np.load('dna_cache/codes_%d.npy'%k).astype(np.int64),device=DEV) for k in (64,256,1024)}
ZID=torch.arange(K+1,device=DEV)
S6=torch.load('dna_cache/dense4_h6k1024_not1.pt'); A1=[t.to(DEV).contiguous() for t in S6['A']]
S2=torch.load('dna_cache/dense9_L2a.pt'); B=[t.to(DEV).contiguous() for t in S2['B']]; OT=S2['OT'].to(DEV).contiguous()
L1MAP=[TM,C[1024],C[256],C[64],C[1024],C[64]]; L1OFF=[0,1,2,3,2,4]; BOFF=[0,1,2,4,6,8]
ROWS=[A1[i].shape[0] for i in range(6)]+[K+1]*12
ROFF=torch.tensor(np.r_[0,np.cumsum(ROWS)[:-1]].astype(np.int32),device=DEV)
NTR=int(sys.argv[1]) if len(sys.argv)>1 else 40_000_000
ROUNDS=int(sys.argv[2]) if len(sys.argv)>2 else 6
ETA=0.30; ALPHA=0.5; BETA=1e-4
CK='dna_cache/loopedT2.pt'
def sidx(mp,rows,d,arr):
    n=arr.shape[0]; out=torch.full((n,),rows-1,dtype=torch.int64,device=DEV)
    if d==0: out[:]=mp[arr.long()]
    elif d<n: out[d:]=mp[arr[:n-d].long()]
    return out
def build_all_idx(x,z1,z2,n,b0):
    cols=[]
    for i in range(6): cols.append(sidx(L1MAP[i],ROWS[i],L1OFF[i],x[b0:b0+n]))
    for i in range(6): cols.append(sidx(ZID,ROWS[6+i],BOFF[i],z1[b0:b0+n]))
    for i in range(6): cols.append(sidx(ZID,ROWS[12+i],BOFF[i],z2[b0:b0+n]))
    return torch.stack(cols,1).to(torch.int16).contiguous()
def argmax_stream(x,z1,z2,nheads,tabs):
    """nheads: 6 (L1) or 12 (L1+B). returns int16 codes for the whole stream."""
    n=x.shape[0]; out=torch.zeros(n,dtype=torch.int16,device=DEV)
    roff=ROFF[:nheads].contiguous(); AT=torch.cat(tabs[:nheads],0).contiguous()
    for b0 in range(0,n,CH):
        e=min(n,b0+CH); m=e-b0
        cols=[]
        for i in range(6): cols.append(sidx(L1MAP[i],ROWS[i],L1OFF[i],x[b0:e]))
        if nheads>6:
            for i in range(6): cols.append(sidx(ZID,ROWS[6+i],BOFF[i],z1[b0:e]))
        IDX=torch.stack(cols,1).to(torch.int16).contiguous()
        KER.argmax_run(AT,roff,IDX,out[b0:e])
    return out
def dev_eval(A1,B,OT):
    va=np.load('D:/Model/data/gpt2/tokens_470m_valid.npy',mmap_mode='r'); st=np.load('dna_cache/starts_dev.npy')
    Bk=torch.tensor(np.stack([np.asarray(va[q:q+L]).astype(np.int64) for q in st]),device=DEV)
    x=Bk[:,:L-1].reshape(-1); n=x.numel()
    tabs=A1+B+B; rr=ROFF; AT=torch.cat(tabs,0).contiguous()
    xf=Bk
    def sb(mp,rows,d):
        f_=torch.full_like(Bk,rows-1)
        if d==0: f_[:]=mp[Bk]
        else: f_[:,d:]=mp[Bk[:,:L-d]]
        return f_[:,:L-1].reshape(-1)
    idx=[]
    for i in range(6): idx.append(sb(L1MAP[i],ROWS[i],L1OFF[i]))
    zbuf=torch.zeros(n,2*K,device=DEV); z1=torch.zeros(n,dtype=torch.int16,device=DEV)
    IDX=torch.stack(idx,1).to(torch.int16).contiguous()
    KER.argmax_run(torch.cat(A1,0).contiguous(),ROFF[:6].contiguous(),IDX,z1)
    # z2
    idx2=list(idx)
    for i in range(6):
        f_=torch.zeros(n,dtype=torch.int64,device=DEV)
        z1b=z1.view(Bk.shape[0],L-1)
        if BOFF[i]==0: f_[:]=z1
        else:
            g=torch.full((Bk.shape[0],L-1),K,dtype=torch.int64,device=DEV)
            g[:,BOFF[i]:]=z1b[:,:L-1-BOFF[i]]
            f_=g.reshape(-1)
        idx2.append(f_)
    IDX2=torch.stack(idx2,1).to(torch.int16).contiguous()
    z2=torch.zeros(n,dtype=torch.int16,device=DEV)
    KER.argmax_run(AT[:sum(ROWS[:12])].contiguous(),ROFF[:12].contiguous(),IDX2,z2)
    idx3=list(idx2)
    for i in range(6):
        f_=torch.zeros(n,dtype=torch.int64,device=DEV); z2b=z2.view(Bk.shape[0],L-1)
        if BOFF[i]==0: f_[:]=z2
        else:
            g=torch.full((Bk.shape[0],L-1),K,dtype=torch.int64,device=DEV)
            g[:,BOFF[i]:]=z2b[:,:L-1-BOFF[i]]
            f_=g.reshape(-1)
        idx3.append(f_)
    IDX3=torch.stack(idx3,1).to(torch.int16).contiguous()
    y=Bk[:,1:].reshape(-1); PQ=torch.empty(n,2*K,device=DEV); LLS=torch.zeros(1,dtype=torch.float64,device=DEV)
    KER.estep_out_run(AT,ROFF,IDX3,y,OT,PQ,LLS)
    q=PQ[:,K:]; p=(q*OT[y]).sum(1)
    return math.exp(-float(torch.log(p).double().sum())/n),float(abs((q[0]@OT.T).sum()-1))
def main():
    with lease('loopedT2',min_free_mb=3000,timeout=900):
        t0=time.time()
        st=torch.load(CK) if os.path.exists(CK) else dict(A1=A1,B=B,OT=OT,hist=[],rnd=0)
        A1s,Bs,OTs,hist,rnd=st['A1'],st['B'],st['OT'],st['hist'],st['rnd']
        tr=np.load('D:/Model/data/gpt2/tokens_470m.npy',mmap_mode='r'); N=min(NTR,len(tr)-1)
        x=torch.tensor(np.asarray(tr[:N+1]).astype(np.int32),device=DEV)
        z1=argmax_stream(x,None,None,6,A1s); log('z1 done %.0fs'%(time.time()-t0))
        for r in range(rnd,ROUNDS):
            z2=argmax_stream(x,z1,None,12,A1s+Bs); log('round %d z2 done %.0fs'%(r,time.time()-t0))
            tabs=A1s+Bs+Bs; AT=torch.cat(tabs,0).contiguous(); TOT=AT.shape[0]
            NUM=torch.zeros(TOT,2*K,device=DEV); AO=torch.zeros(V,K,device=DEV); LLS=torch.zeros(1,dtype=torch.float64,device=DEV)
            PQ=torch.empty(CH,2*K,device=DEV); mx=max(L1OFF+BOFF)
            for b0 in range(mx,N,CH):
                n=min(CH,N-b0)
                IDX=build_all_idx(x,z1,z2,n,b0)
                y=x[b0+1:b0+1+n].long()
                KER.estep_out_run(AT,ROFF,IDX,y,OTs,PQ[:n],LLS)
                for h in range(6,18):
                    lo=int(ROFF[h]); NUM[lo:lo+ROWS[h]].index_add_(0,IDX[:,h].long(),PQ[:n])
                AO.index_add_(0,y,PQ[:n,:K])
            OTs=(AO+BETA)/(AO.sum(0,keepdim=True)+BETA*V)
            for i in range(6):
                lo6=int(ROFF[6+i]); lo12=int(ROFF[12+i]); r6=ROWS[6+i]
                num=NUM[lo6:lo6+r6,:K]+NUM[lo12:lo12+r6,:K]; den=NUM[lo6:lo6+r6,K:]+NUM[lo12:lo12+r6,K:]
                t=Bs[i]*torch.pow((num+ALPHA)/(den+ALPHA),ETA); Bs[i]=(t/t.mean(1,keepdim=True)).clamp_(min=1e-12,max=1e4)
            dp,ne=dev_eval(A1s,Bs,OTs)
            hist.append(dict(round=r,dev_ppl=dp,norm_err=ne,sec=round(time.time()-t0,1))); log(hist[-1])
            torch.save(dict(A1=A1s,B=Bs,OT=OTs,hist=hist,rnd=r+1),CK)
            if r+1<ROUNDS:
                z2=argmax_stream(x,z1,None,12,A1s+Bs)
            del NUM,AO,PQ; torch.cuda.empty_cache()
        log('DONE %.1f min'%((time.time()-t0)/60))
if __name__=='__main__': main()
