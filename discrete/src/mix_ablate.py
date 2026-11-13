# -*- coding: utf-8 -*-
"""Quantify each mixture component: dense only / +induction raw / +induction z2 / +global."""
import os, torch, numpy as np, json, math
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
ZID=torch.arange(K+1,device=DEV)
L1MAP=[TM,C[1024],C[256],C[64],C[1024],C[64]]; L1OFF=[0,1,2,3,2,4]; BOFF=[0,1,2,4,6,8]
ROWS=[A1[i].shape[0] for i in range(6)]+[K+1]*12
ROFF=torch.tensor(np.r_[0,np.cumsum(ROWS)[:-1]].astype(np.int32),device=DEV)
AT=torch.cat(A1+Bt+Bt,0).contiguous(); AT12=torch.cat(A1+Bt,0).contiguous(); AT6=torch.cat(A1,0).contiguous()
va=np.load('D:/Model/data/gpt2/tokens_470m_valid.npy',mmap_mode='r'); st=np.load('dna_cache/starts_dev.npy')
Bk=torch.tensor(np.stack([np.asarray(va[q:q+L]).astype(np.int64) for q in st]),device=DEV); nb,L=Bk.shape
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
idx1=[sh2(L1MAP[i],ROWS[i],L1OFF[i],Bk) for i in range(6)]
IDX1=torch.stack(idx1,1).to(torch.int16).contiguous()
z1=torch.zeros(nb*(L-1),dtype=torch.int16,device=DEV); KER.argmax_run(AT6,ROFF[:6].contiguous(),IDX1,z1)
z1b=torch.full((nb,L),K,dtype=torch.int64,device=DEV); z1b[:,:L-1]=z1.view(nb,L-1).long()
idx2=idx1+[shz(ROWS[6+i],BOFF[i],z1b) for i in range(6)]
IDX2=torch.stack(idx2,1).to(torch.int16).contiguous()
z2=torch.zeros(nb*(L-1),dtype=torch.int16,device=DEV); KER.argmax_run(AT12,ROFF[:12].contiguous(),IDX2,z2)
z2b=torch.full((nb,L),K,dtype=torch.int64,device=DEV); z2b[:,:L-1]=z2.view(nb,L-1).long()
idx3=idx2+[shz(ROWS[12+i],BOFF[i],z2b) for i in range(6)]
IDX3=torch.stack(idx3,1).to(torch.int16).contiguous()
y=Bk[:,1:].reshape(-1); n=y.numel()
PQ=torch.empty(n,2*K,device=DEV); LLS=torch.zeros(1,dtype=torch.float64,device=DEV)
KER.estep_out_run(AT,ROFF,IDX3,y,OT,PQ,LLS)
q=PQ[:,K:]; p_dense=(q*OT[y]).sum(1).cpu().numpy()
def induction(code):
    t=torch.arange(L,device=DEV); P=[]; F=[]
    for b in range(code.shape[0]):
        c=code[b]; m=(c[:,None]==c[None,:])&(t[None,:]<t[:,None])
        sc=torch.where(m,t[None,:].float(),torch.full((L,L),-1.0,device=DEV)); tv,ti=torch.topk(sc,4,dim=1)
        Su=Bk[b][(ti+1).clamp(max=L-1)][:L-1]; M=(tv>=0)[:L-1]
        w=M.float()*torch.tensor([8,4,2,1],device=DEV,dtype=torch.float32)[None,:]; tot=w.sum(1)
        P.append(torch.where(tot>0,((Su==Bk[b,1:,None]).float()*w).sum(1)/tot.clamp(min=1),torch.zeros_like(tot))); F.append(tot>0)
    return torch.cat(P).cpu().numpy(),torch.cat(F).cpu().numpy()
p_raw,f_raw=induction(Bk); p_z2,f_z2=induction(Bk*(K+1)+z2b)
gp=[]
for b in range(nb):
    x=Bk[b]; eq=(x[1:,None]==x[None,:]); tri=torch.arange(L,device=DEV)[None,:]<=torch.arange(L-1,device=DEV)[:,None]
    gp.append((eq&tri).float().sum(1)/(torch.arange(L-1,device=DEV).float()+1))
p_glob=torch.cat(gp).cpu().numpy()
def em(c,it=120):
    w=np.full(c.shape[1],1.0/c.shape[1])
    for _ in range(it):
        r=c*w[None,:]; r/=np.maximum(r.sum(1,keepdims=True),1e-300); w=r.mean(0)
    return w
def fit(names,arrs):
    comps=np.stack(arrs,1); bk=f_raw.astype(np.int64)*0
    for i,nm in enumerate(names):
        if nm=='raw': bk=bk+1*f_raw.astype(np.int64)
        if nm=='z2':  bk=bk+2*f_z2.astype(np.int64)
    px=np.zeros(len(p_dense)); W={}
    for b in range(4):
        m=bk==b
        if m.sum()==0: continue
        W[b]=em(comps[m]); px[m]=(comps[m]*W[b][None,:]).sum(1)
    return px,W
def pp(p): return float(math.pow(math.e,-float(np.mean(np.log(np.clip(p,1e-300,1))))))
print('%-42s %8s'%('混合组件','dev PPL'))
print('%-42s %8.3f'%('稠密单独',pp(p_dense)))
for names,arrs,lab in [
   (['raw'],[p_dense,p_raw],'稠密 + 归纳(原始 token)'),
   (['raw','z2'],[p_dense,p_raw,p_z2],'稠密 + 归纳(原始) + 归纳(token,z2)'),
   (['raw','z2','glob'],[p_dense,p_raw,p_z2,p_glob],'稠密 + 归纳×2 + 全局（全开）'),
   (['glob'],[p_dense,p_glob],'稠密 + 仅全局头'),
   (['raw','glob'],[p_dense,p_raw,p_glob],'稠密 + 归纳(原始) + 全局'),
]:
    px,_=fit(names,arrs); print('%-42s %8.3f'%(lab,pp(px)))
