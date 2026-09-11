#pragma once
// 按 slot 分块的批量 matvec：一个 warp 负责一行，寄存器里同时累加 TILE 个 slot，
// 权重行因此只读一次而不是每个 slot 读一次。
// x 的 slot 组先搬到 shared memory，避免每个 warp 重读。
#include <cuda_runtime.h>
namespace tao::dual {
#ifndef TAO_SLOT_TILE
#define TAO_SLOT_TILE 8
#endif
template<int TILE>
__global__ void ds_bmv_tiled_kernel(const float*w,const float*x,float*y,int rows,int cols){
    extern __shared__ float xs[];           // TILE*cols
    const int base=blockIdx.y*TILE;
    for(int i=threadIdx.x;i<TILE*cols;i+=blockDim.x)xs[i]=x[size_t(base)*cols+i];
    __syncthreads();
    const int warp=threadIdx.x/32,lane=threadIdx.x%32;
    const int row=blockIdx.x*(blockDim.x/32)+warp;
    if(row>=rows)return;
    const float*wr=w+size_t(row)*cols;
    float acc[TILE];
    #pragma unroll
    for(int s=0;s<TILE;++s)acc[s]=0.f;
    for(int j=lane;j<cols;j+=32){
        const float wv=wr[j];
        #pragma unroll
        for(int s=0;s<TILE;++s)acc[s]+=wv*xs[size_t(s)*cols+j];
    }
    #pragma unroll
    for(int s=0;s<TILE;++s){
        float z=acc[s];
        #pragma unroll
        for(int k=16;k;k>>=1)z+=__shfl_down_sync(0xffffffffu,z,k);
        if(!lane)y[size_t(base+s)*rows+row]=z;
    }
}
// 返回 false 表示该形状不适用，调用方回退到原内核。
inline bool launch_bmv_tiled(const float*w,const float*x,float*y,int rows,int cols,int slots){
    if(slots%TAO_SLOT_TILE||slots<TAO_SLOT_TILE)return false;
    if(cols<=0||cols>1024||rows<=0)return false;
    const size_t smem=size_t(TAO_SLOT_TILE)*cols*sizeof(float);
    if(smem>48*1024)return false;
    dim3 grid((rows+7)/8,slots/TAO_SLOT_TILE);
    ds_bmv_tiled_kernel<TAO_SLOT_TILE><<<grid,256,smem>>>(w,x,y,rows,cols);
    return true;
}
}
