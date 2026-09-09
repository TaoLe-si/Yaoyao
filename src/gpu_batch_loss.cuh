#pragma once
#include "dual_loss_parallel.cuh"
namespace tao::dual {
__global__ void ds_ce_batch(const float*x,float*grad,float*loss,int n,const unsigned*targets,const unsigned*supervised){
int slot=blockIdx.x;if(!supervised[slot]){if(!threadIdx.x)loss[slot]=0;return;}int target=targets[slot];float multiplier=1;x+=size_t(slot)*n;grad+=size_t(slot)*n;loss+=slot;
__shared__ float scratch[256];__shared__ float mx,den;
int tid=threadIdx.x;float a=-CUDART_INF_F;
for(int j=tid;j<n;j+=256)a=fmaxf(a,x[j]);scratch[tid]=a;__syncthreads();
for(int s=128;s;s/=2){if(tid<s)scratch[tid]=fmaxf(scratch[tid],scratch[tid+s]);__syncthreads();}
if(!tid)mx=scratch[0];__syncthreads();
a=0;for(int j=tid;j<n;j+=256)a+=expf(x[j]-mx);scratch[tid]=a;__syncthreads();
for(int s=128;s;s/=2){if(tid<s)scratch[tid]+=scratch[tid+s];__syncthreads();}
if(!tid){den=scratch[0];*loss=multiplier*(logf(den)+mx-x[target]);}__syncthreads();
for(int j=tid;j<n;j+=256)grad[j]+=multiplier*(expf(x[j]-mx)/den-(j==target));
}
}
