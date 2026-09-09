#pragma once
#include "dual_state_cuda_backward.cuh"
namespace tao::dual {
// active controls time padding; reset only applies to active first positions.
__global__ void batch_state_update(const float*old,const float*u,const float*g,float*y,const unsigned*active,const unsigned*reset,int n,int slots){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n*slots){int s=i/n;if(!active[s]){y[i]=old[i];return;}float v=reset[s]?0:old[i];y[i]=v+ds_sigmoid(g[i])*(tanhf(u[i])-v);}}
__global__ void batch_state_backward(const float*old,const float*u,const float*g,const float*dy,float*dold,float*du,float*dg,const unsigned*active,const unsigned*reset,int n,int slots){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n*slots){int s=i/n;if(!active[s]){dold[i]+=dy[i];return;}float v=reset[s]?0:old[i],a=ds_sigmoid(g[i]),c=tanhf(u[i]);if(!reset[s])dold[i]+=dy[i]*(1-a);du[i]+=dy[i]*a*(1-c*c);dg[i]+=dy[i]*(c-v)*a*(1-a);}}
}
