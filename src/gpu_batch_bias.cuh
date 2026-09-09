#pragma once
#include "dual_state_autograd.cuh"
namespace tao::dual {
__global__ void batch_bias_forward(const float*x,const float*b,float*y,int n,int slots){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n*slots)y[i]=x[i]+b[i%n];}
__global__ void batch_bias_grad(const float*dy,float*db,int n,int slots){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float z=db[i];for(int s=0;s<slots;++s)z+=dy[s*n+i];db[i]=z;}}
}
