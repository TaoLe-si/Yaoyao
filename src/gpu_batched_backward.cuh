#pragma once
#include "gpu_linear_dx_tiled.cuh"
namespace tao::dual {
__global__ void ds_batch_dx(const float*w,const float*dy,float*dx,int rows,int cols){__shared__ float sums[8][32];int slot=blockIdx.y,j=blockIdx.x*32+threadIdx.x,part=threadIdx.y;float z=0;if(j<cols)for(int i=part;i<rows;i+=8)z+=w[size_t(i)*cols+j]*dy[size_t(slot)*rows+i];sums[part][threadIdx.x]=z;__syncthreads();if(!part&&j<cols){float a=0;for(int k=0;k<8;++k)a+=sums[k][threadIdx.x];dx[size_t(slot)*cols+j]+=a;}}
// 按 slot 分块的 dx：一个 block 覆盖 TILE 个 slot 与一个列分块。
// 权重 (i,j) 每个 slot 组只读一次（原内核每个 slot 都读一遍）；
// dy[base+s][i] 在一个 warp 内被 32 条 lane 广播共享。
#ifndef TAO_BASELINE_GEMM
template<int TILE,int COLS,int CHUNK>
__global__ void ds_dx_tiled_kernel(const float*w,const float*dy,float*dx,int rows,int cols){
    const int base=blockIdx.y*TILE;
    const int j=blockIdx.x*COLS+threadIdx.x;
    const int s=threadIdx.y;
    if(j>=cols)return;
    const int i0=blockIdx.z*CHUNK, i1=min(rows,i0+CHUNK);
    const float* wp=w+size_t(i0)*cols+j;
    const float* dp=dy+size_t(base+s)*rows+i0;
    float acc=0;
    int i=0,n=i1-i0;
    for(;i+3<n;i+=4){
        acc+=wp[size_t(i)*cols]*dp[i]+wp[size_t(i+1)*cols]*dp[i+1]
            +wp[size_t(i+2)*cols]*dp[i+2]+wp[size_t(i+3)*cols]*dp[i+3];
    }
    for(;i<n;++i)acc+=wp[size_t(i)*cols]*dp[i];
    float* out=dx+size_t(base+s)*cols+j;
    if(gridDim.z==1)*out+=acc; else atomicAdd(out,acc);
}
inline bool launch_dx_tiled(const float*w,const float*dy,float*dx,int rows,int cols,int slots){
    constexpr int TILE=8,COLS=32,CHUNK=128;
    if(slots%TILE||slots<TILE||cols<=0||rows<=0)return false;
    const unsigned z=(unsigned(rows)+CHUNK-1)/CHUNK;
    dim3 grid((cols+COLS-1)/COLS,slots/TILE,z);
    if(size_t(grid.x)*grid.y*grid.z>65535u*65535u)return false;
    ds_dx_tiled_kernel<TILE,COLS,CHUNK><<<grid,dim3(COLS,TILE)>>>(w,dy,dx,rows,cols);
    return true;
}
#endif
__global__ void ds_batch_dw(const float*x,const float*dy,float*dw,int rows,int cols,int slots){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<size_t(rows)*cols){float v=dw[i];for(int s=0;s<slots;++s)v+=dy[size_t(s)*rows+i/cols]*x[size_t(s)*cols+i%cols];dw[i]=v;}}
}
