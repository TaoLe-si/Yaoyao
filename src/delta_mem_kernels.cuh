#pragma once
// H2R 增量规则矩阵记忆的 CUDA 前向/反向核（dual-state-4-noffn-delta-mem-input-sqrt-d）。
//
// 前向（每 slot、每层）：
//   k^ = k/(||k||+eps)
//   a  = M k^                       (dv)
//   u  = beta (v - a)               (dv)
//   M' = M + u k^T                  (dv x dk)
//   o  = M' q                       (dv)
//
// 反向：记 do = dL/do、dMout = dL/dM'（M' 同时还要当下一个时间步的 M，所以它自己有梯度缓冲）。
// 把 M' = M + u k^T 与 a = M k^ 全式展开，可以解析消去，只需两个 per-slot 中间向量 p、du：
//   c    = q . k^                       (标量)
//   p_i  = sum_j dMout[i,j] k^[j]       (dv)
//   du_i = do_i c + p_i                 (dv)   <- dL/du
//   e    = do . (v - a)                 (标量)
//   dbeta  = sum_i du_i (v_i - a_i)     (标量)
//   dv_i  += beta du_i
//   dM[i,j]  += do_i q_j + dMout[i,j] - beta du_i k^[j]
//   dk^[j]   += beta q_j e + sum_i dMout[i,j] u_i - beta sum_i du_i M[i,j]
//   dq[j]    += sum_i M[i,j] do_i + k^[j] beta e
//
// 单步时 dMout=0，全部退化成只依赖 c、e 两个标量的极简形式（已在有限差分下逐项验证）。
// 跨步复用时多出的三项都是 dMout 的投影，不需要额外保存 M'——dMout 就是 M' 自己的梯度缓冲，
// 也不需要 a 的梯度缓冲（前向的 a 只用于算 e）。
#include "dual_state_cuda_resident.cuh"

namespace tao::dual {

// ---------- 前向 ----------
__global__ void dm_normalize_fwd(const float*k,float*khat,float*norm,unsigned dk){
    unsigned s=blockIdx.x; if(threadIdx.x)return;
    const float* r=k+size_t(s)*dk;
    float ss=0; for(unsigned j=0;j<dk;++j)ss+=r[j]*r[j];
    const float nr=sqrtf(ss); norm[s]=nr;
    const float inv=1.0f/(nr+1e-6f);
    float* o=khat+size_t(s)*dk;
    for(unsigned j=0;j<dk;++j)o[j]=r[j]*inv;
}
__global__ void dm_forward(const float*M,const float*khat,const float*q,const float*v,const float*beta,
                           float*Mout,float*o,float*a,unsigned dv,unsigned dk){
    unsigned s=blockIdx.x,i=threadIdx.x; if(i>=dv)return;
    const float* src=M+(size_t(s)*dv+i)*dk;
    const float* kh=khat+size_t(s)*dk;
    const float* qq=q+size_t(s)*dk;
    float acc=0; for(unsigned j=0;j<dk;++j)acc+=src[j]*kh[j];
    const float b=beta[s];
    const float g=b*(v[size_t(s)*dv+i]-acc);
    float* dst=Mout+(size_t(s)*dv+i)*dk;
    for(unsigned j=0;j<dk;++j)dst[j]=src[j]+g*kh[j];
    float rd=0; for(unsigned j=0;j<dk;++j)rd+=dst[j]*qq[j];
    o[size_t(s)*dv+i]=rd;
    a[size_t(s)*dv+i]=acc;
}
__global__ void dm_sigmoid_fwd(const float*x,float*y,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n)y[i]=ds_sigmoid(x[i]);
}

// ---------- 反向 ----------
// p[i] = sum_j dMout[i,j] k^[j]
__global__ void dm_proj_bwd(const float*dMout,const float*khat,float*p,unsigned dv,unsigned dk){
    unsigned s=blockIdx.x,i=threadIdx.x; if(i>=dv)return;
    const float* r=dMout+(size_t(s)*dv+i)*dk; const float* kh=khat+size_t(s)*dk;
    float z=0; for(unsigned j=0;j<dk;++j)z+=r[j]*kh[j];
    p[size_t(s)*dv+i]=z;
}
// 每 slot 单线程：c、e、du、dbeta 四个量
__global__ void dm_scalars_bwd(const float*q,const float*khat,const float*do_,const float*v,const float*a,
                               const float*p,const float*beta,float*c,float*e,float*du,float*dbeta,
                               unsigned dv,unsigned dk){
    unsigned s=blockIdx.x; if(threadIdx.x)return;
    const float* qq=q+size_t(s)*dk; const float* kh=khat+size_t(s)*dk;
    float cc=0; for(unsigned j=0;j<dk;++j)cc+=qq[j]*kh[j];
    const float* dd=do_+size_t(s)*dv; const float* vv=v+size_t(s)*dv;
    const float* aa=a+size_t(s)*dv; const float* pp=p+size_t(s)*dv;
    float* uu=du+size_t(s)*dv;
    float ee=0,db=0;
    for(unsigned i=0;i<dv;++i){
        const float delta=vv[i]-aa[i];
        const float u=dd[i]*cc+pp[i];
        uu[i]=u;
        ee+=dd[i]*delta;
        db+=u*delta;
    }
    c[s]=cc; e[s]=ee; dbeta[s]=db;
}
// dv[i] += beta du_i 且 dM[i,j] += do_i q_j + dMout[i,j] - beta du_i k^[j]
__global__ void dm_dM_dv_bwd(const float*dMout,const float*khat,const float*q,const float*do_,
                             const float*du,const float*beta,float*dM,float*dv_,unsigned dv,unsigned dk){
    unsigned s=blockIdx.x,i=threadIdx.x; if(i>=dv)return;
    const float b=beta[s],u=du[size_t(s)*dv+i],d=do_[size_t(s)*dv+i];
    dv_[size_t(s)*dv+i]+=b*u;
    const float* kh=khat+size_t(s)*dk; const float* qq=q+size_t(s)*dk;
    const float* mo=dMout+(size_t(s)*dv+i)*dk;
    float* row=dM+(size_t(s)*dv+i)*dk;
    const float s1=d,s2=-b*u;
    for(unsigned j=0;j<dk;++j)row[j]+=s1*qq[j]+s2*kh[j]+mo[j];
}
// dk^[j] += beta q_j e + sum_i dMout[i,j] u_i - beta sum_i du_i M[i,j]
// dq[j]  += sum_i M[i,j] do_i + k^[j] beta e
__global__ void dm_dkhat_dq_bwd(const float*M,const float*khat,const float*q,const float*do_,
                                const float*v,const float*a,const float*du,const float*e,
                                const float*beta,const float*dMout,
                                float*dkhat,float*dq,unsigned dv,unsigned dk){
    unsigned s=blockIdx.x,j=threadIdx.x; if(j>=dk)return;
    const float b=beta[s];
    float h=0,w=0,z=0;
    for(unsigned i=0;i<dv;++i){
        const size_t o=(size_t(s)*dv+i)*dk+j, si=size_t(s)*dv+i;
        const float d=do_[si];
        h+=M[o]*d;
        w+=dMout[o]*(b*(v[si]-a[si]));
        z+=du[si]*M[o];
    }
    const float be=b*e[s];
    dkhat[size_t(s)*dk+j]+=be*q[size_t(s)*dk+j]+w-b*z;
    dq[size_t(s)*dk+j]+=h+khat[size_t(s)*dk+j]*be;
}
// k^ = k/(||k||+eps) 的精确反向（eps 视为常量，但 ||k|| 的依赖保留）
__global__ void dm_normalize_bwd(const float*khat,const float*dkhat,const float*norm,float*dkout,unsigned dk){
    __shared__ float dot;
    unsigned s=blockIdx.x,j=threadIdx.x;
    if(j==0){float z=0; for(unsigned t=0;t<dk;++t)z+=dkhat[size_t(s)*dk+t]*khat[size_t(s)*dk+t]; dot=z;}
    __syncthreads();
    if(j<dk){ const float nr=norm[s]; const float n=nr+1e-6f;
        dkout[size_t(s)*dk+j]+=dkhat[size_t(s)*dk+j]/n - khat[size_t(s)*dk+j]*dot/(nr>1e-12f?nr:1e-12f); }
}
__global__ void dm_sigmoid_bwd(const float*y,const float*dy,float*dx,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){const float s=y[i]; dx[i]+=dy[i]*s*(1.0f-s);}
}
}
