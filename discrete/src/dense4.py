# -*- coding: utf-8 -*-
"""DENSE-4: dense discrete-latent attention model, multiplicative head composition, GIS-damped EM (monotone GEM).
q(z|ctx) proportional to prod_h A_h[f_h(ctx), z]   heads = RoPE positional reads at offsets d_h; V = token id or learned class code
P(y|ctx) = sum_z q(z|ctx) O[y,z]                  exact normalisation over K and over V; no softmax, no exp
E-step: post(z) = q(z) O[y,z] / P(y|ctx).  M-step: O exact; A_h by one GIS step A_h *= ((NUM+a)/(DEN+a))^(1/H).
All parameters are fixed-size dense matrices. No gradients, no context-keyed tables."""
import numpy as np, torch, math, json, time, sys, os
from gpu_guard import lease
from loop_lib import emb, assign, kmeans_w
torch.set_grad_enabled(False)
DEV='cuda'; V=50257; L=1024
cfg=json.load(open(sys.argv[1],encoding='utf-8-sig')); TAG=cfg['tag']; NTR=cfg['NTR']; ITERS=cfg['iters']; WARM=cfg.get('warm'); KZ=cfg['K']; HEADS=[tuple(h) for h in cfg['heads']]
ALPHA=cfg.get('alpha',0.5); ETA=cfg.get('eta',None); BETA=cfg.get('beta',1e-4); CH=cfg.get('chunk',20000); CK='dna_cache/dense4_%s.pt'%TAG
def log(*a): print('[%s]'%time.strftime('%H:%M:%S'),*a,flush=True)
freq=np.load('dna_cache/freq.npy'); seen=np.flatnonzero(freq>0); NS=len(seen)
tokmap=np.full(V,NS,dtype=np.int64); tokmap[seen]=np.arange(NS)
def head_map(kind):
    m=tokmap if kind=='tok' else np.load('dna_cache/codes_%s.npy'%kind[1:]).astype(np.int64)
    return torch.tensor(m,device=DEV),int(m.max())+2          # last row = PAD (no source position)
MAPS=[head_map(k) for k,_ in HEADS]; OFFS=[d for _,d in HEADS]; H=len(HEADS); ROWS=[r for _,r in MAPS]
def nparams(): return sum(r*KZ for r in ROWS)+(NS+1)*KZ
def qpost(A,idx):
    """normalised product of per-head factors. Rescale every 3 heads: each factor is kept in [1e-12, 1e4] by
    clamp_tables(), so a 3-factor product stays within [1e-36, 1e12] (fp32-safe) between rescales."""
    q=A[0][idx[0]].clone()
    for h in range(1,H):
        q*=A[h][idx[h]]
        if h%3==2: q/=q.amax(1,keepdim=True)
    return q/q.sum(1,keepdim=True)
def clamp_tables(A):
    for h in range(len(A)): A[h].clamp_(min=1e-12,max=1e4)
def dev_feats():
    va=np.load('D:/Model/data/gpt2/tokens_470m_valid.npy',mmap_mode='r'); st=np.load('dna_cache/starts_dev.npy')
    B=torch.tensor(np.stack([np.asarray(va[s:s+L]).astype(np.int64) for s in st]),device=DEV); idx=[]
    for h in range(H):
        d=OFFS[h]; m,rows=MAPS[h]
        if d==0: f=m[B]
        else: f=torch.full_like(B,rows-1); f[:,d:]=m[B[:,:L-d]]
        idx.append(f[:,:L-1].reshape(-1))
    return idx,B[:,1:].reshape(-1)
def dev_eval(A,OT,DI,DY):
    lp=torch.zeros((),dtype=torch.float64,device=DEV)
    for s in range(0,len(DY),20000):
        q=qpost(A,[i[s:s+20000] for i in DI]); lp+=torch.log((q*OT[DY[s:s+20000]]).sum(1)).double().sum()
    lp=float(lp)
    q=qpost(A,[i[7:8] for i in DI]); return math.exp(-lp/len(DY)),float(abs((q@OT.T).sum()-1))
def main():
    t0=time.time(); torch.cuda.set_per_process_memory_fraction(0.85)
    tr=np.load('D:/Model/data/gpt2/tokens_470m.npy',mmap_mode='r'); mx=max(OFFS)
    x=torch.tensor(np.asarray(tr[:NTR+1]).astype(np.int32),device=DEV); N=len(x)-1-mx; DI,DY=dev_feats()
    if os.path.exists(CK):
        S=torch.load(CK); A=S['A']; OT=S['OT']; hist=S['hist']; log('resumed',len(hist))
    elif WARM:
        S=torch.load('dna_cache/dense4_%s.pt'%WARM); OT=S['OT']; old={tuple(h):a for h,a in zip(S['cfg']['heads'],S['A'])}
        A=[old[h].clone() if h in old else torch.ones(r,KZ,device=DEV) for h,r in zip(HEADS,ROWS)]; hist=[]; log('warm start from',WARM,'reused heads',[h for h in HEADS if h in old])
    else:
        assert HEADS[0]==('tok',0)
        E=emb(); Ft=torch.zeros(V,64,device=DEV); nt=torch.zeros(V,device=DEV)
        for s0 in range(0,len(x)-1,2_000_000):
            e0=min(len(x)-1,s0+2_000_000); a=x[s0:e0].long(); Ft.index_add_(0,a,E[x[s0+1:e0+1].long()]); nt.index_add_(0,a,torch.ones(e0-s0,device=DEV))
        sv=nt>0; Ft[sv]/=nt[sv][:,None]; Cc=kmeans_w(Ft[sv],nt[sv],KZ,20,seed=7); cls=torch.zeros(V,dtype=torch.int64,device=DEV); cls[sv]=assign(Ft[sv],Cc)
        AO=torch.zeros(V,KZ,device=DEV)
        for s0 in range(0,len(x)-1,2_000_000):
            e0=min(len(x)-1,s0+2_000_000); AO.index_put_((x[s0+1:e0+1].long(),cls[x[s0:e0].long()]),torch.ones(e0-s0,device=DEV),accumulate=True)
        OT=(AO+BETA)/(AO.sum(0,keepdim=True)+BETA*V); del AO
        A=[torch.ones(r,KZ,device=DEV) for r in ROWS]; A[0]*=0.05; A[0][MAPS[0][0],cls]=1.0
        hist=[]
    if cfg.get('total_iters'): ITERS_RUN=max(0,cfg['total_iters']-len(hist))
    else: ITERS_RUN=ITERS
    for it in range(ITERS_RUN):
        NUM=[torch.zeros(r,KZ,device=DEV) for r in ROWS]; DEN=[torch.zeros(r,KZ,device=DEV) for r in ROWS]; AO=torch.zeros(V,KZ,device=DEV); ll=torch.zeros((),dtype=torch.float64,device=DEV)
        for s in range(0,N,CH):
            n=min(CH,N-s); b0=mx+s
            idx=[MAPS[h][0][x[b0-OFFS[h]:b0-OFFS[h]+n].long()] for h in range(H)]; yy=x[b0+1:b0+1+n].long()
            q=qpost(A,idx); pz=q*OT[yy]; p=pz.sum(1); ll+=torch.log(p).double().sum(); post=pz/p[:,None]
            for h in range(H): NUM[h].index_add_(0,idx[h],post); DEN[h].index_add_(0,idx[h],q)
            AO.index_add_(0,yy,post); del q,pz,post
        OT=(AO+BETA)/(AO.sum(0,keepdim=True)+BETA*V); del AO
        for h in range(H):
            A[h]=A[h]*torch.pow((NUM[h]+ALPHA)/(DEN[h]+ALPHA),(ETA if ETA else 1.0/H)); A[h]=A[h]/A[h].mean(1,keepdim=True)
        clamp_tables(A)
        del NUM,DEN; dp,ne=dev_eval(A,OT,DI,DY)
        if hist and dp>hist[-1]['dev_ppl']*1.002 and cfg.get('guard',1): log('dev rose: revert to previous checkpoint and stop'); S=torch.load(CK); A=S['A']; OT=S['OT']; break
        ll=float(ll); hist.append(dict(iter=len(hist),train_ppl_before_update=math.exp(-ll/N),dev_ppl=dp,norm_err=ne,sec=round(time.time()-t0,1))); log(hist[-1])
        torch.save(dict(A=A,OT=OT,hist=hist,cfg=cfg),CK)
    log('params',nparams(),'DONE')
with lease('dense4_'+TAG,min_free_mb=3000,timeout=600): main()
