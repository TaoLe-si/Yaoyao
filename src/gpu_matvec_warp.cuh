#pragma once
#include "dual_state_cuda_probe.cuh"
namespace tao::dual {
__global__ void ds_matvec_warp(const float*a,const float*x,float*y,int rows,int cols){int lane=threadIdx.x%32,row=blockIdx.x*4+threadIdx.x/32;float z=0;if(row<rows)for(int j=lane;j<cols;j+=32)z+=a[size_t(row)*cols+j]*x[j];for(int k=16;k;k/=2)z+=__shfl_down_sync(0xffffffff,z,k);if(!lane&&row<rows)y[row]=z;}
inline void training_matvec(const float*a,const float*x,float*y,int rows,int cols){
#ifdef TAO_WARP_MATVEC
ds_matvec_warp<<<(rows+3)/4,128>>>(a,x,y,rows,cols);
#else
matvec<<<(rows+127)/128,128>>>(a,x,y,rows,cols);
#endif
}
}
