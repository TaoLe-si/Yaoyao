#pragma once
#include "dual_state_autograd.cuh"
namespace tao::dual {
__global__ void batch_embed(const float*w,const float*ids,float*y,int d,int slots){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<d*slots)y[i]=w[size_t(unsigned(ids[i/d]))*d+i%d];}
// Each column has one writer; repeated tokens accumulate in slot order.
__global__ void batch_embed_back(const float*dy,const float*ids,float*dw,int d,int slots){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<d)for(int s=0;s<slots;++s)dw[size_t(unsigned(ids[s]))*d+j]+=dy[size_t(s)*d+j];}
}
