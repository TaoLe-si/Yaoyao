# -*- coding: utf-8 -*-
"""Corrected decode benchmark: tokens/s, batch=1, greedy, same GPU."""
import torch,time,math,json,numpy as np
torch.set_grad_enabled(False); DEV='cuda'; V=50257; K=1024
def bench_tok(fn,ntok,w=3,r=8):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(r): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/(r*ntok)      # seconds per token
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
def run(N=128,mode='full'):
    x=torch.full((N+16,),NS,device=DEV,dtype=torch.long)
    z1=torch.full((N+16,),K,device=DEV,dtype=torch.long)
    z2=torch.full((N+16,),K,device=DEV,dtype=torch.long)
    tok=torch.tensor(464,device=DEV)
    for t in range(N):
        x[t]=tok
        i1=(t-O1).clamp(min=0); i2=(t-O2).clamp(min=0)
        rows[:6]=AT[ROFF[:6]+M1[ar6,x[i1]]]
        P=rows[:6].prod(0); q=P/P.sum(); z1[t]=q.argmax()
        if mode=='q1': tok=(OT@q).argmax(); continue
        rows[6:12]=AT[ROFF[6:12]+z1[i2]]
        P12=P*rows[6:12].prod(0); q12=P12/P12.sum(); z2[t]=q12.argmax()
        if mode=='q12': tok=(OT@q12).argmax(); continue
        rows[12:]=AT[ROFF[12:]+z2[i2]]
        P18=P12*rows[12:].prod(0); q18=P18/P18.sum()
        tok=(OT@q18).argmax()
N=128
print('=== 本模型解码 (batch=1, greedy) ===')
res={}
for m,lab in [('q1','6 头 (仅 q1)'),('q12','12 头 (q1+循环1轮)'),('full','18 头 (完整 T=2)')]:
    d=bench_tok(lambda: run(N,m),N); res[m]=d
    print('  %-22s %7.3f ms/token   %7.1f tok/s'%(lab,d*1e3,1/d))
print()
print('=== GPT-2 small (124.4M) 增量解码 ===')
from transformers import GPT2LMHeadModel
from transformers import DynamicCache
mp='D:/Model/data/gpt2/task_gpt2_eval_model'
g=GPT2LMHeadModel.from_pretrained(mp,torch_dtype=torch.float32).to(DEV).eval()
def run_gpt(N=128):
    cache=DynamicCache()
    ids=torch.tensor([[464]],device=DEV)
    o=g(ids,past_key_values=cache,use_cache=True)
    tok=o.logits[:,-1,:].argmax(-1,keepdim=True)
    for _ in range(N-1):
        o=g(tok,past_key_values=cache,use_cache=True)
        tok=o.logits[:,-1,:].argmax(-1,keepdim=True)
dg=bench_tok(lambda: run_gpt(N),N)
print('  %-22s %7.3f ms/token   %7.1f tok/s'%('GPT-2 small (KV cache)',dg*1e3,1/dg))
print()
print('=== 对比 ===')
print('  %-24s %10s %10s'%('','ms/token','tok/s'))
print('  %-24s %10.3f %10.1f'%('本模型 (18头)',res['full']*1e3,1/res['full']))
print('  %-24s %10.3f %10.1f'%('GPT-2 small',dg*1e3,1/dg))
print('  加速比: %.2fx  (%.0f tok/s vs %.0f tok/s)'%(dg/res['full'],1/res['full'],1/dg))
print()
print('=== 预填充 (prompt 处理) ===')
ids=torch.randint(0,50000,(1,1024),device=DEV)
t0=time.perf_counter()
for _ in range(5): g(ids,use_cache=False)
torch.cuda.synchronize(); dpre=(time.perf_counter()-t0)/5
print('  GPT-2 1024-token prefill : %7.2f ms  (%.0f tok/s)'%(dpre*1e3,1024/dpre))
json.dump(dict(ours18_ms=res['full']*1e3,ours12_ms=res['q12']*1e3,ours6_ms=res['q1']*1e3,
               gpt2_ms=dg*1e3,speedup=dg/res['full'],gpt2_prefill_ms=dpre*1e3),
          open('dna_cache/speed_final.json','w'),indent=1)
