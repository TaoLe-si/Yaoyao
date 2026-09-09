#include <cuda_runtime.h>
#include <cstdio>
int main(){int n=0;auto e=cudaGetDeviceCount(&n);if(e!=cudaSuccess){printf("CUDA_ERROR %s\n",cudaGetErrorString(e));return 1;}for(int i=0;i<n;++i){cudaDeviceProp p{};if(cudaGetDeviceProperties(&p,i)!=cudaSuccess)return 2;printf("device=%d name=%s total_MiB=%llu compute=%d.%d\n",i,p.name,(unsigned long long)(p.totalGlobalMem/(1024*1024)),p.major,p.minor);}return n?0:3;}
