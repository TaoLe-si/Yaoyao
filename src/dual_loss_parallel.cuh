#pragma once
#include "dual_state_cuda_loss.cuh"
#include <math_constants.h>
namespace tao::dual {
// One cooperative block per token; gradients retain additive seed semantics.
__global__ void ds_ce_parallel(const float*x,float*grad,float*loss,int n,int target,float multiplier){
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
struct BlockLoss {
Device values;size_t used=0;
explicit BlockLoss(size_t capacity):values(capacity){if(!capacity)throw std::invalid_argument("loss capacity");}
void seed(Node output,unsigned target,bool supervised,float multiplier=1){
if(target>=output->value.n||!std::isfinite(multiplier)||multiplier<0)throw std::invalid_argument("loss arguments");
if(!supervised)return;if(used>=values.n)throw std::runtime_error("loss capacity exceeded");
ds_ce_parallel<<<1,256>>>(output->value.p,output->grad.p,values.p+used++,int(output->value.n),target,multiplier);check(cudaGetLastError());
}
double collect(){Vec host(used);if(used)check(cudaMemcpy(host.data(),values.p,used*sizeof(float),cudaMemcpyDeviceToHost));double sum=0;for(float v:host){if(!std::isfinite(v))throw std::runtime_error("nonfinite loss");sum+=v;}used=0;return sum;}
};
}
