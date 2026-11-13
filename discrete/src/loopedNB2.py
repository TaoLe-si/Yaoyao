# -*- coding: utf-8 -*-
"""NB loop heads (full K-codes) vs the current 6. Generic-NB version."""
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
BOFF=json.loads(sys.argv[3]); NB=len(BOFF); TAG=sys.argv[4]; NTR=int(sys.argv[1]); ROUNDS=int(sys.argv[2])
CK='dna_cache/nb_%s.pt'%TAG
def log(*a): print('[%s]'%time.strftime('%H:%M:%S'),*a,flush=True)
f=np.load('dna_cache/freq.npy'); s=np.flatnonzero(f>0); NS=len(s)
TM=torch.tensor(np.full(V,NS,dtype=np.int64),device=DEV); TM[s]=torch.arange(NS,device=DEV)
C={k:torch.tensor(np.load('dna_cache/codes_%d.npy'%k).astype(np.int64),device=DEV) for k in (64,256,1024)}
S6=torch.load('dna_cache/dense4_h6k1024_not1.pt'); A1=[t.to(DEV).contiguous() for t in S6['A']]
S2=torch.load('dna_cache/loopedT2.pt'); B0=[t.to(DEV).contiguous() for t in S2['B']]; OT0=S2['OT']
L1MAP=[TM,C[1024],C[256],C[64],C[1024],C[64]]; L1OFF=[0,1,2,3,2,4]; N1=6
ROWS=[A1[i].shape[0] for i in range(N1)]+[K+1]*(2*NB)
ROFF=torch.tensor(np.r_[0,np.cumsum(ROWS)[:-1]].astype(np.int32),device=DEV)
ZID=torch.arange(K+1,device=DEV)
def sh1(mp,rows,d,A):
    m=A.shape[0]; out=torch.full((m,),rows-1,dtype=torch.int64,device=DEV)
    if d==0: out[:]=mp[A.long()]
    elif d<m: out[d:]=mp[A[:m-d].long()]
    return out
def shz1(rows,d,A):
    m=A.shape[0]; out=torch.full((m,),rows-1,dtype=torch.int64,device=DEV)
    if d==0: out[:]=A.long()
    elif d<m: out[d:]=A[:m-d].long()
    return out
def sh2(mp,rows,d,A):
    f_=torch.full_like(A,rows-1)
    if d==0: f_[:]=mp[A]
    else: f_[:,d:]=mp[A[:,:L-d]]
    return f_[:,:L-1].reshape(-1)
def shz2(rows,d,A):
    f_=torch.full_like(A,rows-1)
    if d==0: f_[:]=A
    else: f_[:,d:]=A[:,:L-d]
    return f_[:,:L-1].reshape(-1)
def main():
    with lease('nb'+TAG,min_free_mb=2600,timeout=1500):
        t0=time.time()
        Bt=[b.clone() for b in B0]+[torch.ones(K+1,K,device=DEV)]*max(0,NB-6)
        Bt=Bt[:NB]; OTs=OT0.clone(); hist=[]; rnd=0
        if os.path.exists(CK):
            st=torch.load(CK); Bt=st['B']; OTs=st['OT']; hist=st['hist']; rnd=st['rnd']
        tr=np.load('D:/Model/data/gpt2/tokens_470m.npy',mmap_mode='r'); N=min(NTR,len(tr)-1)
        x=torch.tensor(np.asarray(tr[:N+1]).astype(np.int32),device=DEV)
        z1=torch.zeros(N+1,dtype=torch.int16,device=DEV); AT1=torch.cat(A1,0).contiguous()
        for b0 in range(0,N+1,CH):
            e=min(N+1,b0+CH)
            KER.argmax_run(AT1,ROFF[:N1].contiguous(),torch.stack([sh1(L1MAP[i],ROWS[i],L1OFF[i],x[b0:e]) for i in range(N1)],1).to(torch.int16).contiguous(),z1[b0:e])
        log('z1 %.0fs'%(time.time()-t0))
        for r in range(rnd,ROUNDS):
            AT2=torch.cat(A1+Bt,0).contiguous(); z2=torch.zeros(N+1,dtype=torch.int16,device=DEV)
            for b0 in range(0,N+1,CH):
                e=min(N+1,b0+CH)
                cols=[sh1(L1MAP[i],ROWS[i],L1OFF[i],x[b0:e]) for i in range(N1)]+[shz1(ROWS[N1+i],BOFF[i],z1[b0:e]) for i in range(NB)]
                KER.argmax_run(AT2,ROFF[:N1+NB].contiguous(),torch.stack(cols,1).to(torch.int16).contiguous(),z2[b0:e])
            log('round %d z2 %.0fs'%(r,time.time()-t0))
            AT=torch.cat(A1+Bt+Bt,0).contiguous()
            NUM=torch.zeros(AT.shape[0],2*K,device=DEV); AO=torch.zeros(V,K,device=DEV); LLS=torch.zeros(1,dtype=torch.float64,device=DEV)
            PQ=torch.empty(CH,2*K,device=DEV); mx=max(L1OFF+BOFF)
            for b0 in range(mx,N,CH):
                n=min(CH,N-b0)
                cols=[sh1(L1MAP[i],ROWS[i],L1OFF[i],x[b0:b0+n]) for i in range(N1)]
                cols+=[shz1(ROWS[N1+i],BOFF[i],z1[b0:b0+n]) for i in range(NB)]
                cols+=[shz1(ROWS[N1+NB+i],BOFF[i],z2[b0:b0+n]) for i in range(NB)]
                IDX=torch.stack(cols,1).to(torch.int16).contiguous()
                y=x[b0+1:b0+1+n].long()
                KER.estep_out_run(AT,ROFF,IDX,y,OTs,PQ[:n],LLS)
                for h in range(N1,N1+2*NB):
                    lo=int(ROFF[h]); NUM[lo:lo+ROWS[h]].index_add_(0,IDX[:,h].long(),PQ[:n])
                AO.index_add_(0,y,PQ[:n,:K])
            OTs=(AO+1e-4)/(AO.sum(0,keepdim=True)+1e-4*V)
            for i in range(NB):
                lo1=int(ROFF[N1+i]); lo2=int(ROFF[N1+NB+i]); rr=ROWS[N1+i]
                num=NUM[lo1:lo1+rr,:K]+NUM[lo2:lo2+rr,:K]; den=NUM[lo1:lo1+rr,K:]+NUM[lo2:lo2+rr,K:]
                t=Bt[i]*torch.pow((num+0.5)/(den+0.5),0.30)
                if i<6: Bt[i]=(t/t.mean(1,keepdim=True)).clamp_(min=1e-12,max=1e4)
                else:   Bt[i]=(t/t.mean(1,keepdim=True)).clamp_(min=1e-12,max=1e4)
            va=np.load('D:/Model/data/gpt2/tokens_470m_valid.npy',mmap_mode='r'); stt=np.load('dna_cache/starts_dev.npy')
            Bk=torch.tensor(np.stack([np.asarray(va[q:q+L]).astype(np.int64) for q in stt]),device=DEV); nb=Bk.shape[0]
            AT=torch.cat(A1+Bt+Bt,0).contiguous()
            IDX1=torch.stack([sh2(L1MAP[i],ROWS[i],L1OFF[i],Bk) for i in range(N1)],1).to(torch.int16).contiguous()
            zz1=torch.zeros(nb*(L-1),dtype=torch.int16,device=DEV)
            KER.argmax_run(torch.cat(A1,0).contiguous(),ROFF[:N1].contiguous(),IDX1,zz1)
            z1b=torch.full((nb,L),K,dtype=torch.int64,device=DEV); z1b[:,:L-1]=zz1.view(nb,L-1).long()
            IDX2=torch.stack([sh2(L1MAP[i],ROWS[i],L1OFF[i],Bk) for i in range(N1)]+[shz2(ROWS[N1+i],BOFF[i],z1b) for i in range(NB)],1).to(torch.int16).contiguous()
            zz2=torch.zeros(nb*(L-1),dtype=torch.int16,device=DEV)
            KER.argmax_run(torch.cat(A1+Bt,0).contiguous(),ROFF[:N1+NB].contiguous(),IDX2,zz2)
            z2b=torch.full((nb,L),K,dtype=torch.int64,device=DEV); z2b[:,:L-1]=zz2.view(nb,L-1).long()
            cols=[sh2(L1MAP[i],ROWS[i],L1OFF[i],Bk) for i in range(N1)]
            cols+=[shz2(ROWS[N1+i],BOFF[i],z1b) for i in range(NB)]
            cols+=[shz2(ROWS[N1+NB+i],BOFF[i],z2b) for i in range(NB)]
            IDX3=torch.stack(cols,1).to(torch.int16).contiguous()
            yv=Bk[:,1:].reshape(-1); nn=yv.numel()
            PQd=torch.empty(nn,2*K,device=DEV); LLd=torch.zeros(1,dtype=torch.float64,device=DEV)
            KER.estep_out_run(AT,ROFF,IDX3,yv,OTs,PQd,LLd)
            p=(PQd[:,K:]*OTs[yv]).sum(1)
            dp=float(math.pow(math.e,-float(torch.log(p.clamp(min=1e-300)).mean())))
            hist.append(dict(round=r,dev_ppl=dp,sec=round(time.time()-t0,1))); log(hist[-1])
            torch.save(dict(B=Bt,OT=OTs,hist=hist,rnd=r+1),CK)
            del NUM,AO,PQ,PQd; torch.cuda.empty_cache()
        npar=sum(int(t.numel()) for t in A1)+sum(int(t.numel()) for t in Bt)+int(OTs.numel())
        log('NB=%d params %d (%.1f%% GPT-2) | loop tables %d'%(NB,npar,100*npar/124439808,sum(int(t.numel()) for t in Bt)))
if __name__=='__main__': main()
