#pragma once
#include "gpu_linear_dx_tiled.cuh"
namespace tao::dual {
__global__ void ds_batch_dx(const float*w,const float*dy,float*dx,int rows,int cols){__shared__ float sums[8][32];int slot=blockIdx.y,j=blockIdx.x*32+threadIdx.x,part=threadIdx.y;float z=0;if(j<cols)for(int i=part;i<rows;i+=8)z+=w[size_t(i)*cols+j]*dy[size_t(slot)*rows+i];sums[part][threadIdx.x]=z;__syncthreads();if(!part&&j<cols){float a=0;for(int k=0;k<8;++k)a+=sums[k][threadIdx.x];dx[size_t(slot)*cols+j]+=a;}}
__global__ void ds_batch_dw(const float*x,const float*dy,float*dw,int rows,int cols,int slots){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<size_t(rows)*cols){float v=dw[i];for(int s=0;s<slots;++s)v+=dy[size_t(s)*rows+i/cols]*x[size_t(s)*cols+i%cols];dw[i]=v;}}
}
