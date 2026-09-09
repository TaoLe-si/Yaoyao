#pragma once
#include "dual_state_cuda_probe.cuh"
namespace tao::dual {
__global__ void ds_health(const float*x,size_t n,double*out,unsigned*bad,bool positive,double divisor){__shared__ double sum[256];unsigned i=blockIdx.x*256+threadIdx.x;double v=0;if(i<n){float z=x[i];if(!isfinite(z)||(positive&&z<=0))atomicExch(bad,1u);else{double a=double(z)/divisor;v=a*a;}}sum[threadIdx.x]=v;__syncthreads();for(unsigned k=128;k;k>>=1){if(threadIdx.x<k)sum[threadIdx.x]+=sum[threadIdx.x+k];__syncthreads();}if(!threadIdx.x)atomicAdd(out,sum[0]);}
struct GpuHealth {double*sum=nullptr;unsigned*bad=nullptr;GpuHealth(){check(cudaMalloc(&sum,8));check(cudaMalloc(&bad,4));reset();}~GpuHealth(){cudaFree(sum);cudaFree(bad);}void reset(){check(cudaMemset(sum,0,8));check(cudaMemset(bad,0,4));}void add(Device&v,bool positive=false,double divisor=1){ds_health<<<(v.n+255)/256,256>>>(v.p,v.n,sum,bad,positive,divisor);check(cudaGetLastError());}double finish(){unsigned b;double s;check(cudaMemcpy(&b,bad,4,cudaMemcpyDeviceToHost));check(cudaMemcpy(&s,sum,8,cudaMemcpyDeviceToHost));if(b||!std::isfinite(s))throw std::runtime_error("GPU nonfinite or invalid scale");return s;}
};
}
