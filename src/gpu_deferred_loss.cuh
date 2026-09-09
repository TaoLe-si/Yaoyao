#pragma once
#include "dual_loss_parallel.cuh"
namespace tao::dual {
__global__ void accumulate_loss(const float*values,size_t n,double*total,unsigned*bad){if(threadIdx.x||blockIdx.x)return;double s=0;for(size_t i=0;i<n;++i){float v=values[i];if(!isfinite(v))atomicExch(bad,1u);else s+=v;}*total+=s;}
struct DeferredLoss {BlockLoss block;double*total=nullptr;unsigned*bad=nullptr;explicit DeferredLoss(size_t n):block(n){check(cudaMalloc(&total,8));check(cudaMalloc(&bad,4));reset();}DeferredLoss(const DeferredLoss&)=delete;~DeferredLoss(){cudaFree(total);cudaFree(bad);}void reset(){check(cudaMemsetAsync(total,0,8));check(cudaMemsetAsync(bad,0,4));}void seed(Node y,unsigned target,bool supervised){block.seed(y,target,supervised);}void flush(){if(block.used){accumulate_loss<<<1,1>>>(block.values.p,block.used,total,bad);check(cudaGetLastError());block.used=0;}}double collect(){flush();double v;unsigned b;check(cudaMemcpy(&v,total,8,cudaMemcpyDeviceToHost));check(cudaMemcpy(&b,bad,4,cudaMemcpyDeviceToHost));if(b||!std::isfinite(v))throw std::runtime_error("nonfinite deferred loss");reset();return v;}};
}
