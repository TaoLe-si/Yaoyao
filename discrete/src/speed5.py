# -*- coding: utf-8 -*-
"""Full-system decode: dense (18 heads) + 10 induction + 5 global, with fitted EM weights."""
import torch,time,math,json,numpy as np
torch.set_grad_enabled(False); DEV='cuda'; V=50257; K=1024
def bench_tok(fn,ntok,w=3,r=6):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(r): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/(r*ntok)
f=np.load('dna_cache/freq.npy'); s=np.flatnonzero(f>0); NS=len(s)
TM=torch.tensor(np.full(V,NS,dtype=np.int64),device=DEV); TM[s]=torch.arange(NS,device=DEV)
C={k:torch.tensor(np.load('dna_cache/codes_%d.npy'%k).astype(np.int64),device=DEV) for k in (64,256,1024)}
S6=torch.load('dna_cache/dense4_h6k1024_not1.pt'); A1=[t.to(DEV).contiguous() for t in S6['A']]
S2=torch.load('dna_cache/loopedT2.pt'); Bt=[t.to(DEV).contiguous() for t in S2['B']]; OT=S2['OT'].contiguous()
L1MAP=[TM,C[1024],C[256],C[64],C[1024],C[64]]; L1OFF=[0,1,2,3,2,4]; BOFF=[0,1,2,4,6,8]
AT=torch.cat(A1+Bt+Bt,0).contiguous()
ROWS=[A1[i].shape[0] for i in range(6)]+[K+1]*12
ROFF=torch.tensor(np.r_[0,np.cumsum(ROWS)[:-1]].astype(np.int64),device=DEV)
M1=torch.stack([L1MAP[i] for i in range(6)],0)
O1=torch.tensor(L1OFF,device=DEV); O2=torch.tensor(BOFF,device=DEV); ar6=torch.arange(6,device=DEV)
rows=torch.empty(18,K,device=DEV)
NIND=[('x',1024,4),('x2',1024,4),('x',256,8),('c256',256,4),('xz2',256,4),
      ('x2',512,4),('x',192,4),('x3',512,4),('c1024',256,4),('x',256,1)]
NG=[32,64,128,256,1024]
WD={b:[0.99-0.02*b,0.0,0.0] for b in range(11)}
def run(N=128,use_heads=True):
    Lc=N+16
    x=torch.full((Lc,),NS,device=DEV,dtype=torch.long)
    z1=torch.full((Lc,),K,device=DEV,dtype=torch.long); z2=torch.full((Lc,),K,device=DEV,dtype=torch.long)
    codes=torch.full((len(NIND),Lc),-1,device=DEV,dtype=torch.long)
    tok=torch.tensor(464,device=DEV)
    logits=torch.empty(V,device=DEV)
    for t in range(N):
        x[t]=tok
        i1=(t-O1).clamp(min=0); i2=(t-O2).clamp(min=0)
        rows[:6]=AT[ROFF[:6]+M1[ar6,x[i1]]]
        P=rows[:6].prod(0); q=P/P.sum(); z1[t]=q.argmax()
        rows[6:12]=AT[ROFF[6:12]+z1[i2]]
        P12=P*rows[6:12].prod(0); q12=P12/P12.sum(); z2[t]=q12.argmax()
        rows[12:]=AT[ROFF[12:]+z2[t if False else i2]]
        P18=P12*rows[12:].prod(0); q18=P18/P18.sum()
        pd=OT@q18
        if not use_heads:
            tok=pd.argmax(); continue
        # ---- codes for induction heads ----
        prev1=x[max(0,t-1)]; prev2=x[max(0,t-2)]
        codes[0,t]=tok
        codes[1,t]=tok*V+prev1
        codes[2,t]=tok
        codes[3,t]=C[256][tok]
        codes[4,t]=tok*(K+1)+z2[t]
        codes[5,t]=tok*V+prev1
        codes[6,t]=tok
        codes[7,t]=(tok*V+prev1)*V+prev2
        codes[8,t]=C[1024][tok]
        codes[9,t]=tok
        # ---- match: all 10 heads in ONE batched pass ----
        acc=torch.zeros(V,device=DEV)
        pos=torch.arange(Lc,device=DEV)
        WINS=torch.tensor([w for _,w,_ in NIND],device=DEV)
        KS=torch.tensor([k for _,_,k in NIND],device=DEV)
        cm=codes[:,:t]==codes[:,t:t+1]
        cm=cm&(pos[:t][None,:]>=((t-WINS)[:,None]))
        sc=torch.where(cm,pos[:t][None,:],torch.full_like(cm,-1,dtype=torch.long))
        if sc.shape[1]<8: sc=torch.cat([sc,torch.full((sc.shape[0],8-sc.shape[1]),-1,dtype=torch.long,device=DEV)],1)
        tv,ti=torch.topk(sc,8,dim=1)
        valid=(tv>=0)&(torch.arange(8,device=DEV)[None,:]<KS[:,None])
        j=(ti+1).clamp(max=Lc-1)
        succ=x[j]
        rw=torch.arange(8,0,-1,device=DEV).float()[None,:].expand_as(valid)
        ww=torch.where(valid,rw,torch.zeros_like(rw))
        ww=ww/ww.sum(1,keepdim=True).clamp(min=1)
        acc.index_add_(0,succ.reshape(-1),ww.reshape(-1))
        nf=int(valid.any(1).sum())
        g=torch.zeros(V,device=DEV)
        for W in NG:
            a=max(0,t+1-W); seg=x[a:t+1]
            cnt=torch.zeros(V,device=DEV); cnt.index_add_(0,seg,torch.ones(len(seg),device=DEV))
            g+=cnt/len(seg)
        g/=len(NG)
        W_=[0.99-0.02*min(nf,10),0.0,0.0]
        logits=W_[0]*pd+0.16*acc+0.05*g
        tok=logits.argmax()
N=128
print('=== 完整系统解码 ===')
d0=bench_tok(lambda: run(N,False),N); print('  稠密 18 头             %7.3f ms/token  %7.1f tok/s'%(d0*1e3,1/d0))
d1=bench_tok(lambda: run(N,True),N);  print('  +10归纳 +5全局         %7.3f ms/token  %7.1f tok/s'%(d1*1e3,1/d1))
print()
print('=== vs GPT-2 (8.416 ms/token, 118.8 tok/s) ===')
print('  稠密:     %.2fx %s'%(8.416/(d0*1e3),'快' if d0*1e3<8.416 else '慢'))
print('  完整:     %.2fx %s'%(8.416/(d1*1e3),'快' if d1*1e3<8.416 else '慢'))
json.dump(dict(dense_ms=d0*1e3,full_ms=d1*1e3,gpt2_ms=8.416),
          open('dna_cache/speed_full.json','w'),indent=1)
