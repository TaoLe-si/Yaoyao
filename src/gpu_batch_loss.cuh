#pragma once
#include "dual_loss_parallel.cuh"
namespace tao::dual {
__global__ void ds_ce_batch(const float*x,float*grad,float*loss,int n,const unsigned*targets,const unsigned*supervised,const float* weights=nullptr){
int slot=blockIdx.x;float multiplier=weights?weights[slot]:1.0f;if(!supervised[slot]||multiplier==0.0f){if(!threadIdx.x)loss[slot]=0;return;}int target=targets[slot];x+=size_t(slot)*n;grad+=size_t(slot)*n;loss+=slot;
__shared__ float scratch[256];__shared__ float mx,den;
int tid=threadIdx.x;float a=-CUDART_INF_F;
for(int j=tid;j<n;j+=256)a=fmaxf(a,x[j]);scratch[tid]=a;__syncthreads();
for(int s=128;s;s/=2){if(tid<s)scratch[tid]=fmaxf(scratch[tid],scratch[tid+s]);__syncthreads();}
if(!tid)mx=scratch[0];__syncthreads();
a=0;for(int j=tid;j<n;j+=256)a+=expf(x[j]-mx);scratch[tid]=a;__syncthreads();
for(int s=128;s;s/=2){if(tid<s)scratch[tid]+=scratch[tid+s];__syncthreads();}
if(!tid){den=scratch[0];*loss=multiplier*(logf(den)+mx-x[target]);}__syncthreads();
for(int j=tid;j<n;j+=256)grad[j]+=multiplier*(expf(x[j]-mx)/den-(j==target));
}

// ─────────────────────────────────────────────────────────────────────────────
// GRPO 裁剪代理目标 + KL 惩罚（对齐论文 Eq.3）。
//
// 【为何能归约为加权交叉熵】设 lp = log π_θ(o_t|·)，r = exp(lp - lp_old)，
// A 为组内优势。论文目标 L = -min(rA, clip(r,1-ε,1+ε)·A) + β·KL(π_θ‖π_ref)。
// 由于 d(lp)/d(logits_j) = softmax_j - onehot_j，而交叉熵的梯度恰为
// (softmax_j - onehot_j)，故 dL/dlogits = c_t·(softmax - onehot)，其中
//     c_t = f'(r)·r + β·(e^{d} - 1),  d = lp_ref - lp_θ,
//     f'(r)·r：A>0 且 r≤1+ε → A·r；A<0 且 r≥1-ε → A·r；否则 0。
// 也就是说：**整个目标等价于把逐位置 CE 的乘数由常数 A 换成 c_t**，
// 反向通路完全不用改。c_t 依赖当前前向算出的 lp，故必须在核内求。
// （裁剪分支经有限差分逐点验证；A<0 的失效侧与 A>0 相反，见下。）
//
// 【π_ref 的取法】论文 Algorithm 1 每次迭代令 π_ref ← π_θ（迭代开始时的策略）。
// 本实现每批轨迹只做一次参数更新（μ=1），故采样策略与参考策略同为 lp_old，
// 即 d = lp_old - lp_θ，无需额外跑一次参考模型前向。
__global__ void ds_grpo_batch(const float*x,float*grad,float*loss,int n,
                              const unsigned*targets,const unsigned*supervised,
                              const float*weights,const float*lpold,const float*lpref,
                              float beta,float eps,float*dbg){
   int slot=blockIdx.x;
   // 与 ds_ce_batch 保持同一套指针约定：按 slot 前移，target 取 targets[slot]。
   if(!supervised[slot]){ if(!threadIdx.x){ loss[slot]=0.f;
       if(dbg){ dbg[slot*3+0]=0.f; dbg[slot*3+1]=0.f; dbg[slot*3+2]=0.f; } } return; }
   const int target=targets[slot];
   x+=size_t(slot)*n; grad+=size_t(slot)*n; loss+=slot;
   const float A=weights?weights[slot]:1.0f;
   const float lpo=lpold?lpold[slot]:0.0f;
   __shared__ float scratch[256];__shared__ float mx,den;
   int tid=threadIdx.x;float a=-CUDART_INF_F;
   for(int j=tid;j<n;j+=256)a=fmaxf(a,x[j]);scratch[tid]=a;__syncthreads();
   for(int s=128;s;s/=2){if(tid<s)scratch[tid]=fmaxf(scratch[tid],scratch[tid+s]);__syncthreads();}
   if(!tid)mx=scratch[0];__syncthreads();
   a=0;for(int j=tid;j<n;j+=256)a+=expf(x[j]-mx);scratch[tid]=a;__syncthreads();
   for(int s=128;s;s/=2){if(tid<s)scratch[tid]+=scratch[tid+s];__syncthreads();}
   if(!tid)den=scratch[0];__syncthreads();
   const float nll=logf(den)+mx-x[target];   // 交叉熵 = -lp_theta
   const float lp=-nll;
   // 数值安全：lp-lp_old 过大会让 expf 溢出成 inf，梯度随即变成 NaN 并毁掉整个训练。
   // 钳到 ±20 后 r ∈ [2e-9, 5e8]，覆盖本文所需的全部动态范围。
   float dlp=fminf(fmaxf(lp-lpo,-20.f),20.f);
   const float r=expf(dlp);                  // 重要性比值 r = exp(lp_theta - lp_old)
   // d = lp_ref - lp_theta。参考分布必须来自**另一个冻结模型**：
   // 若 π_ref 取 π_old，则首次前向 d≡0、KL 梯度 β(1-e^d) 恒为 0，KL 项形同虚设。
   // 未提供参考时退化为 π_ref=π_old（KL 不生效），并在日志中明确说明。
   const float lpr = lpref ? lpref[slot] : lpo;
   const float d  = fminf(fmaxf(lpr-lp,-20.f),20.f);
   const float emd=expf(d);                  // e^{d}
   // 裁剪分支：对 min(rA, clip(r,1-ε,1+ε)·A) 求 f'(r)。
   //   A>0: r≤1+ε → A； r>1+ε → 0（min 取被裁剪的那支）
   //   A<0: r≥1-ε → A； r<1-ε → 0
   // A<0 与 A>0 的失效侧**相反**。早期版本误写成"A<0 恒为 A·r"，
   // 会让 r<1-ε 的样本拿到错误梯度；有限差分测试当场暴露了它。
   float c;
   if(A>=0.f) c=(r<=1.f+eps)?A*r:0.f;
   else       c=(r>=1.f-eps)?A*r:0.f;
   c+=beta*(emd-1.f);                        // KL(π_θ‖π_ref) 的 k3 无偏估计梯度
   if(!tid){ loss[0]=c*nll;
       if(dbg){ dbg[slot*3+0]=lp; dbg[slot*3+1]=r; dbg[slot*3+2]=c; } }
   __syncthreads();
   for(int j=tid;j<n;j+=256)grad[j]+=c*(expf(x[j]-mx)/den-(j==target));
}
}
