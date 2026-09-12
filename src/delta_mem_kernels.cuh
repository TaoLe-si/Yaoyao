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
//
// 【并行度】本文件曾有三个内核每 slot 只开 1 个线程（<<<slots,1>>>）或用
// <<<slots,dk>>> 让每个线程串行扫 dv 行（跨步访问，完全不合并）。d=3200 时
// 全局只有 8~3200 个线程在跑满一个 36 SM 的 GPU，这是训练吞吐的头号瓶颈。
// 现全部改为块内归约 / 二维分块，且访存按 threadIdx.x 连续（合并）。
#include "dual_state_cuda_resident.cuh"

namespace tao::dual {

// d>1024 后不能再把 dv 直接当 blockDim（CUDA 每块上限 1024 线程）。
// 改为二维 grid：x = slot，y = 行分块；每块 DM_THREADS 个线程覆盖同样多的 M 行。
// 行与行之间完全独立（M 按行存储、每行只读自己的 khat/q/dMout），故加宽不改变语义。
constexpr unsigned DM_THREADS = 256;
// 块内归约用的线程数（红调用）；必须是 2 的幂且 <= 256。
constexpr unsigned DM_RED = 256;
// dm_dkhat_dq_bwd 的列分块宽度与行 stride 并行度。
constexpr unsigned DM_COLS = 128;
constexpr unsigned DM_ROWS = 8;
inline dim3 dm_grid(unsigned slots,unsigned dv){return dim3(slots,(dv+DM_THREADS-1)/DM_THREADS);}
inline dim3 dm_col_grid(unsigned slots,unsigned dk){return dim3(slots,(dk+DM_COLS-1)/DM_COLS);}

// 块内对 red[0..DM_RED-1] 做树形归约，结果留在 red[0]。
__device__ inline void dm_reduce(float*red){
    __syncthreads();
    for(unsigned m=DM_RED>>1;m;m>>=1){if(threadIdx.x<m)red[threadIdx.x]+=red[threadIdx.x+m];__syncthreads();}
}

// ---------- 前向 ----------
// 旧版 <<<slots,1>>>：每 slot 1 个线程串行跑 dk 次，d=3200 时全局仅 8 个线程。
// 现每 slot 一个 DM_RED 线程块做树形归约。
__global__ void dm_normalize_fwd(const float*k,float*khat,float*norm,unsigned dk){
    const unsigned s=blockIdx.x;
    const float* r=k+size_t(s)*dk;
    float* o=khat+size_t(s)*dk;
    __shared__ float red[DM_RED];
    float ss=0;
    for(unsigned j=threadIdx.x;j<dk;j+=blockDim.x)ss+=r[j]*r[j];
    red[threadIdx.x]=ss;
    dm_reduce(red);
    const float nr=sqrtf(red[0]);
    if(!threadIdx.x)norm[s]=nr;
    const float inv=1.0f/(nr+1e-6f);
    for(unsigned j=threadIdx.x;j<dk;j+=blockDim.x)o[j]=r[j]*inv;
}
__global__ void dm_forward(const float*M,const float*khat,const float*q,const float*v,const float*beta,
                           float*Mout,float*o,float*a,unsigned dv,unsigned dk){
    unsigned s=blockIdx.x,i=blockIdx.y*blockDim.x+threadIdx.x; if(i>=dv)return;
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
    unsigned s=blockIdx.x,i=blockIdx.y*blockDim.x+threadIdx.x; if(i>=dv)return;
    const float* r=dMout+(size_t(s)*dv+i)*dk; const float* kh=khat+size_t(s)*dk;
    float z=0; for(unsigned j=0;j<dk;++j)z+=r[j]*kh[j];
    p[size_t(s)*dv+i]=z;
}
// c = q.k^ ； du = do*c + p ； e = do.(v-a) ； dbeta = du.(v-a)
// 旧版每 slot 只有 1 个线程串行扫 dk+dv+dv 次；现每 slot 一个 DM_RED 线程块。
__global__ void dm_scalars_bwd(const float*q,const float*khat,const float*do_,const float*v,const float*a,
                               const float*p,const float*beta,float*c,float*e,float*du,float*dbeta,
                               unsigned dv,unsigned dk){
    const unsigned s=blockIdx.x;
    __shared__ float red[DM_RED];
    // c = sum_j q_j khat_j
    float cc=0;
    for(unsigned j=threadIdx.x;j<dk;j+=blockDim.x)cc+=q[size_t(s)*dk+j]*khat[size_t(s)*dk+j];
    red[threadIdx.x]=cc;
    dm_reduce(red);
    const float cval=red[0];
    if(!threadIdx.x)c[s]=cval;
    __syncthreads();
    // du_i = do_i*c + p_i ，并顺带累加 e、dbeta
    float ee=0,db=0;
    for(unsigned i=threadIdx.x;i<dv;i+=blockDim.x){
        const size_t si=size_t(s)*dv+i;
        const float delta=v[si]-a[si];
        const float u=do_[si]*cval+p[si];
        du[si]=u;
        ee+=do_[si]*delta;
        db+=u*delta;
    }
    // __syncthreads() 已由 dm_reduce 开头的 __syncthreads 保证 du 的写入对全块可见。
    red[threadIdx.x]=ee;
    dm_reduce(red);
    if(!threadIdx.x)e[s]=red[0];
    red[threadIdx.x]=db;
    dm_reduce(red);
    if(!threadIdx.x)dbeta[s]=red[0];
}
// dv[i] += beta du_i 且 dM[i,j] += do_i q_j + dMout[i,j] - beta du_i k^[j]
__global__ void dm_dM_dv_bwd(const float*dMout,const float*khat,const float*q,const float*do_,
                             const float*du,const float*beta,float*dM,float*dv_,unsigned dv,unsigned dk){
    unsigned s=blockIdx.x,i=blockIdx.y*blockDim.x+threadIdx.x; if(i>=dv)return;
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
//
// 旧版 <<<slots,dk>>>：每线程负责一列 j，串行扫 i=0..dv，访问 M[i][j] 的步长是 dk
// —— 相邻 lane 取同一列，访存完全不合并。现改为 (列 128) x (行 stride 8) 的二维块：
// threadIdx.x 取连续列（合并），threadIdx.y 分摊行，最后在块内对 threadIdx.y 归约。
__global__ void dm_dkhat_dq_bwd(const float*M,const float*khat,const float*q,const float*do_,
                                const float*v,const float*a,const float*du,const float*e,
                                const float*beta,const float*dMout,
                                float*dkhat,float*dq,unsigned dv,unsigned dk){
    const unsigned s=blockIdx.x;
    const unsigned j=blockIdx.y*DM_COLS+threadIdx.x;
    const bool active=(j<dk);
    const float b=beta[s];
    float h=0,w=0,z=0;
    if(active){
        for(unsigned i=threadIdx.y;i<dv;i+=DM_ROWS){
            const size_t o=(size_t(s)*dv+i)*dk+j, si=size_t(s)*dv+i;
            const float d=do_[si];
            h+=M[o]*d;
            w+=dMout[o]*(b*(v[si]-a[si]));
            z+=du[si]*M[o];
        }
    }
    __shared__ float sh[3][DM_ROWS][DM_COLS];
    sh[0][threadIdx.y][threadIdx.x]=h;
    sh[1][threadIdx.y][threadIdx.x]=w;
    sh[2][threadIdx.y][threadIdx.x]=z;
    __syncthreads();
    if(threadIdx.y==0&&active){
        float H=sh[0][0][threadIdx.x],W=sh[1][0][threadIdx.x],Z=sh[2][0][threadIdx.x];
        for(unsigned k=1;k<DM_ROWS;++k){H+=sh[0][k][threadIdx.x];W+=sh[1][k][threadIdx.x];Z+=sh[2][k][threadIdx.x];}
        const float be=b*e[s];
        dkhat[size_t(s)*dk+j]+=be*q[size_t(s)*dk+j]+W-b*Z;
        dq[size_t(s)*dk+j]+=H+khat[size_t(s)*dk+j]*be;
    }
}
// k^ = k/(||k||+eps) 的精确反向。旧版由 thread 0 串行算 dot；现块内归约。
__global__ void dm_normalize_bwd(const float*khat,const float*dkhat,const float*norm,float*dkout,unsigned dk){
    const unsigned s=blockIdx.x;
    __shared__ float red[DM_RED];
    float z=0;
    for(unsigned j=threadIdx.x;j<dk;j+=blockDim.x)z+=dkhat[size_t(s)*dk+j]*khat[size_t(s)*dk+j];
    red[threadIdx.x]=z;
    dm_reduce(red);
    const float dot=red[0];
    const float nr=norm[s];
    const float n=nr+1e-6f;
    const float denom=(nr>1e-12f?nr:1e-12f);
    for(unsigned j=threadIdx.x;j<dk;j+=blockDim.x)
        dkout[size_t(s)*dk+j]+=dkhat[size_t(s)*dk+j]/n - khat[size_t(s)*dk+j]*dot/denom;
}
__global__ void dm_sigmoid_bwd(const float*y,const float*dy,float*dx,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){const float s=y[i]; dx[i]+=dy[i]*s*(1.0f-s);}
}
}
