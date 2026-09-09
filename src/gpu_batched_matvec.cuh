#pragma once
#include "gpu_matvec_warp.cuh"
namespace tao::dual {
// Slot-major input/output. No temporal or cross-slot mixing.
__global__ void ds_batched_matvec(const float*w,const float*x,float*y,int rows,int cols,int slots){int slot=blockIdx.y,row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;float z=0;if(slot<slots&&row<rows)for(int j=lane;j<cols;j+=32)z+=w[size_t(row)*cols+j]*x[size_t(slot)*cols+j];for(int k=16;k;k>>=1)z+=__shfl_down_sync(0xffffffff,z,k);if(slot<slots&&row<rows&&!lane)y[size_t(slot)*rows+row]=z;}
}
