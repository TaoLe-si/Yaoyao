# -*- coding: utf-8 -*-
"""FINAL v2: dense + 10 induction heads + 5 windowed global heads. All heads zero-parameter."""
import os, torch, numpy as np, math, json
os.environ['TORCH_CUDA_ARCH_LIST']='8.9'
torch.set_grad_enabled(False)
from torch.utils.cpp_extension import load_inline
cuda=open('estep_kernels.cu',encoding='utf-8').read()
cpp=("void estep2_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);\n"
     "void argmax_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);\n"
     "void estep_out_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);")
KER=load_inline(name='dna_estep4k',cpp_sources=cpp,cuda_sources=cuda,functions=['estep2_run','argmax_run','estep_out_run'],
                verbose=False,extra_cuda_cflags=['-O3','-Xcompiler','/Zc:preprocessor','--use_fast_math'])
DEV='cuda'; V=50257; K=1024; L=1024
S=torch.load('dna_cache/loopedT2.pt'); A1=S['A1']; Bt=S['B']; OT=S['OT']
f=np.load('dna_cache/freq.npy'); s=np.flatnonzero(f>0); NS=len(s)
TM=torch.tensor(np.full(V,NS,dtype=np.int64),device=DEV); TM[s]=torch.arange(NS,device=DEV)
C={k:torch.tensor(np.load('dna_cache/codes_%d.npy'%k).astype(np.int64),device=DEV) for k in (64,256,1024)}
L1MAP=[TM,C[1024],C[256],C[64],C[1024],C[64]]; L1OFF=[0,1,2,3,2,4]; BOFF=[0,1,2,4,6,8]
ROWS=[A1[i].shape[0] for i in range(6)]+[K+1]*12
ROFF=torch.tensor(np.r_[0,np.cumsum(ROWS)[:-1]].astype(np.int32),device=DEV)
IND=[('s1_w1024','x',1024,4,[8,4,2,1]),('s2_w1024','x2',1024,4,[8,4,2,1]),
 ('s1_w256_k8','x',256,8,[8,4,2,1,1,1,1,1]),('s1_c256_w256','c256',256,4,[8,4,2,1]),
 ('s1_z2_w256','xz2',256,4,[8,4,2,1]),('s2_w512','x2',512,4,[8,4,2,1]),
 ('s1_w192','x',192,4,[8,4,2,1]),('s3_w512','x3',512,4,[8,4,2,1]),
 ('s1_c1024_w256','c1024',256,4,[8,4,2,1]),('s1_w256_k1','x',256,1,[8])]
GL=[32,64,128,256,1024]
def sh2(mp,rows,d,A):
    f_=torch.full_like(A,rows-1)
    if d==0: f_[:]=mp[A]
    else: f_[:,d:]=mp[A[:,:L-d]]
    return f_[:,:L-1].reshape(-1)
def shz(rows,d,A):
    f_=torch.full_like(A,rows-1)
    if d==0: f_[:]=A
    else: f_[:,d:]=A[:,:L-d]
    return f_[:,:L-1].reshape(-1)
def shifted(A,d):
    g=torch.full_like(A,-1-d)
    if d==0: g[:]=A
    else: g[:,d:]=A[:,:L-d]
    return g
def mkcode(k,Bk,z2b):
    return {'x':lambda:Bk,'x2':lambda:Bk*V+shifted(Bk,1),'x3':lambda:(Bk*V+shifted(Bk,1))*V+shifted(Bk,2),
            'c256':lambda:C[256][Bk],'c1024':lambda:C[1024][Bk],'xz2':lambda:Bk*(K+1)+z2b}[k]()
def induction(code,Bk,win,TOPK,WT):
    t=torch.arange(L,device=DEV); P=[]; F=[]
    for b in range(code.shape[0]):
        c=code[b]; mm=(c[:,None]==c[None,:])&(t[None,:]<t[:,None])&(t[None,:]>=t[:,None]-win)
        sc=torch.where(mm,t[None,:].float(),torch.full((L,L),-1.0,device=DEV)); tv,ti=torch.topk(sc,TOPK,dim=1)
        Su=Bk[b][(ti+1).clamp(max=L-1)][:L-1]; M=(tv>=0)[:L-1]
        w=M.float()*torch.tensor(WT,device=DEV,dtype=torch.float32)[None,:]; tot=w.sum(1)
        P.append(torch.where(tot>0,((Su==Bk[b,1:,None]).float()*w).sum(1)/tot.clamp(min=1),torch.zeros_like(tot))); F.append(tot>0)
    return torch.cat(P).cpu().numpy(),torch.cat(F).cpu().numpy()
def glob_win(Bk,W):
    out=[]; tt=torch.arange(L-1,device=DEV)
    a=(tt+1-W).clamp(min=0); pos=(a-1).clamp(min=0); valid=(a>0); den=(tt+1-a).float().clamp(min=1)
    for b in range(Bk.shape[0]):
        xb=Bk[b]; oh=torch.zeros(L,V,device=DEV); oh.scatter_(1,xb[:,None],1.0); Cm=oh.cumsum(0)
        hi=Cm[:L-1]; prev=torch.zeros(L-1,V,device=DEV); prev[valid]=Cm[pos[valid]]
        P=(hi-prev).clamp(min=0)/den[:,None]
        out.append(P.gather(1,Bk[b,1:,None])[:,0]); del oh,Cm,hi,prev,P
    return torch.cat(out).cpu().numpy()
def build(starts):
    va=np.load('D:/Model/data/gpt2/tokens_470m_valid.npy',mmap_mode='r')
    Bk=torch.tensor(np.stack([np.asarray(va[q:q+L]).astype(np.int64) for q in starts]),device=DEV); nb=Bk.shape[0]
    idx1=[sh2(L1MAP[i],ROWS[i],L1OFF[i],Bk) for i in range(6)]
    z1=torch.zeros(nb*(L-1),dtype=torch.int16,device=DEV)
    KER.argmax_run(torch.cat(A1,0).contiguous(),ROFF[:6].contiguous(),torch.stack(idx1,1).to(torch.int16).contiguous(),z1)
    z1b=torch.full((nb,L),K,dtype=torch.int64,device=DEV); z1b[:,:L-1]=z1.view(nb,L-1).long()
    idx2=idx1+[shz(ROWS[6+i],BOFF[i],z1b) for i in range(6)]
    z2=torch.zeros(nb*(L-1),dtype=torch.int16,device=DEV)
    KER.argmax_run(torch.cat(A1+Bt,0).contiguous(),ROFF[:12].contiguous(),torch.stack(idx2,1).to(torch.int16).contiguous(),z2)
    z2b=torch.full((nb,L),K,dtype=torch.int64,device=DEV); z2b[:,:L-1]=z2.view(nb,L-1).long()
    idx3=idx2+[shz(ROWS[12+i],BOFF[i],z2b) for i in range(6)]
    y=Bk[:,1:].reshape(-1); n=y.numel()
    PQ=torch.empty(n,2*K,device=DEV); LLS=torch.zeros(1,dtype=torch.float64,device=DEV)
    KER.estep_out_run(torch.cat(A1+Bt+Bt,0).contiguous(),ROFF,torch.stack(idx3,1).to(torch.int16).contiguous(),y,OT,PQ,LLS)
    pd=(PQ[:,K:]*OT[y]).sum(1).cpu().numpy()
    parts=[pd]; fired=[]
    for (nm,k,win,tk,wt) in IND:
        p,ff=induction(mkcode(k,Bk,z2b),Bk,win,tk,wt); parts.append(p); fired.append(ff)
    for W in GL: parts.append(glob_win(Bk,W))
    return np.stack(parts,1), np.stack(fired,0).astype(np.int64)
def em(c,it=150):
    w=np.full(c.shape[1],1.0/c.shape[1])
    for _ in range(it):
        r=c*w[None,:]; r/=np.maximum(r.sum(1,keepdims=True),1e-300); w=r.mean(0)
    return w
def ppl(x): return float(math.pow(math.e,-float(np.log(np.clip(x,1e-300,1)).mean())))
print('=== dev ===')
dp,dc=build(np.load('dna_cache/starts_dev.npy')); dn=dc.sum(0)
print('  稠密单独              %.3f'%ppl(dp[:,0]))
W={}
for b in range(int(dn.max())+1):
    m=dn==b
    if m.sum()==0: continue
    W[b]=em(dp[m])
px=np.zeros(len(dp))
for b,w in W.items(): m=dn==b; px[m]=(dp[m]*w[None,:]).sum(1)
print('  +10 归纳 +5 全局      %.3f   (bucket dense-w: %s)'%(ppl(px),' '.join('%.3f'%W[b][0] for b in sorted(W))))
print()
print('=== held-out test (权重固定) ===')
tp,tc=build(np.load('dna_cache/starts_test.npy')); tn=tc.sum(0)
print('  稠密单独              %.3f'%ppl(tp[:,0]))
pt=np.zeros(len(tp))
for b,w in W.items():
    m=tn==b
    if m.sum()==0: continue
    pt[m]=(tp[m]*w[None,:]).sum(1)
miss=pt<=0
if miss.any(): pt[miss]=(tp[miss]*W[max(W)][None,:]).sum(1); print('  兜底 %d 个位置'%int(miss.sum()))
print('  +10 归纳 +5 全局      %.3f'%ppl(pt))
npar=sum(int(t.numel()) for t in A1)+sum(int(t.numel()) for t in Bt)+int(OT.numel())
Od=int(torch.unique(OT,dim=0).shape[0])*1024
ded=npar-int(OT.numel())+Od
json.dump(dict(dev=ppl(px),test=ppl(pt),params=npar,params_dedup=ded,unique_O_rows=int(torch.unique(OT,dim=0).shape[0]),
  ind=[dict(name=n,code=k,window=w,topk=t) for n,k,w,t,_ in IND], glob=GL),open('dna_cache/final_v2.json','w'),indent=1)
print()
print('参数 %d (%.1f%%)  |  O去重后 %d (%.1f%%)  |  different O rows %d'%(npar,100*npar/124439808,ded,100*ded/124439808,int(torch.unique(OT,dim=0).shape[0])))
