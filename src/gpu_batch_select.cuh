#pragma once
#include "dual_state_autograd.cuh"
namespace tao::dual {
__global__ void batch_select_forward(const float*a,const float*b,const float*mask,float*y,int n,int total){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<total)y[i]=mask[i/n]!=0?b[i]:a[i];}
__global__ void batch_select_backward(const float*dy,const float*mask,float*da,float*db,int n,int total){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<total){if(mask[i/n]!=0)db[i]+=dy[i];else da[i]+=dy[i];}}
}
