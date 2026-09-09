#pragma once
#include "dual_state_cuda_resident.cuh"
namespace tao::dual {
// All gradient outputs accumulate. Caller zeroes buffers at graph boundaries.
__global__ void ds_linear_dx(const float*w,const float*dy,float*dx,int rows,int cols){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<cols){float z=0;for(int i=0;i<rows;++i)z+=w[i*cols+j]*dy[i];dx[j]+=z;}}
__global__ void ds_linear_dw(const float*x,const float*dy,float*dw,int rows,int cols){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows*cols)dw[i]+=dy[i/cols]*x[i%cols];}
__global__ void ds_rms_backward(const float*x,const float*gamma,const float*dy,float*dx,float*dg,int n){if(threadIdx.x||blockIdx.x)return;float ss=0,dot=0;for(int i=0;i<n;++i){ss+=x[i]*x[i];dot+=dy[i]*gamma[i]*x[i];}float inv=1/sqrtf(ss/n+1e-5f);for(int i=0;i<n;++i){dx[i]+=dy[i]*gamma[i]*inv-x[i]*dot*inv*inv*inv/n;dg[i]+=dy[i]*x[i]*inv;}}
// old state, pre-tanh candidate and pre-sigmoid gate must be retained.
__global__ void ds_update_backward(const float*old,const float*u,const float*g,const float*dy,float*dold,float*du,float*dg,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float a=ds_sigmoid(g[i]),v=tanhf(u[i]);dold[i]+=dy[i]*(1-a);du[i]+=dy[i]*a*(1-v*v);dg[i]+=dy[i]*(v-old[i])*a*(1-a);}}
__global__ void ds_silu_backward(const float*x,const float*dy,float*dx,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float s=ds_sigmoid(x[i]);dx[i]+=dy[i]*(s+x[i]*s*(1-s));}}
}
