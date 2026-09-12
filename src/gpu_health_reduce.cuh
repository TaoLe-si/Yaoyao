#pragma once
#include "dual_state_cuda_probe.cuh"
namespace tao::dual {
__global__ void ds_health(const float*x,size_t n,double*out,unsigned*bad,bool positive,double divisor){__shared__ double sum[256];unsigned i=blockIdx.x*256+threadIdx.x;double v=0;if(i<n){float z=x[i];if(!isfinite(z)||(positive&&z<=0))atomicExch(bad,1u);else{double a=double(z)/divisor;v=a*a;}}sum[threadIdx.x]=v;__syncthreads();for(unsigned k=128;k;k>>=1){if(threadIdx.x<k)sum[threadIdx.x]+=sum[threadIdx.x+k];__syncthreads();}if(!threadIdx.x)atomicAdd(out,sum[0]);}
// 【常驻缓冲】旧版每次构造/析构都 cudaMalloc + cudaFree，而这两者都会同步整个设备；
// project() 与 update() 每步各构造一次 GpuHealth，等于每步 4 次设备级同步的堆操作。
// 现在 12 字节的归约缓冲在进程内复用一次分配，构造只做 cudaMemset。
struct GpuHealth {
  static double* sum(){static double*p=[](){double*q=nullptr;check(cudaMalloc(&q,8));return q;}();return p;}
  static unsigned* bad(){static unsigned*p=[](){unsigned*q=nullptr;check(cudaMalloc(&q,4));return q;}();return p;}
  GpuHealth(){reset();}
  void reset(){check(cudaMemset(sum(),0,8));check(cudaMemset(bad(),0,4));}
  void add(Device&v,bool positive=false,double divisor=1){ds_health<<<(v.n+255)/256,256>>>(v.p,v.n,sum(),bad(),positive,divisor);check(cudaGetLastError());}
  double finish(){unsigned b;double s;check(cudaMemcpy(&b,bad(),4,cudaMemcpyDeviceToHost));check(cudaMemcpy(&s,sum(),8,cudaMemcpyDeviceToHost));if(b||!std::isfinite(s))throw std::runtime_error("GPU nonfinite or invalid scale");return s;}
};
}
