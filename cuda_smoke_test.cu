#include <cuda_runtime.h>
#include <cstdio>
__global__ void k(float*x){int i=threadIdx.x;if(i<4)x[i]=x[i]*2+1;}
int main(){float h[4]={1,2,3,4},*d;cudaMalloc(&d,16);cudaMemcpy(d,h,16,cudaMemcpyHostToDevice);k<<<1,4>>>(d);cudaDeviceSynchronize();cudaMemcpy(h,d,16,cudaMemcpyDeviceToHost);cudaFree(d);for(float x:h)printf("CUDA_PASS %g %g %g %g\n",h[0],h[1],h[2],h[3]);}