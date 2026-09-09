#pragma once
#include "gpu_rms_parallel.cuh"
namespace tao::dual {
__global__ void batch_rms_forward(const float*x,const float*g,float*y,int n){x+=size_t(blockIdx.x)*n;y+=size_t(blockIdx.x)*n;__shared__ float s[256];int t=threadIdx.x;float a=0;for(int i=t;i<n;i+=256)a+=x[i]*x[i];s[t]=a;__syncthreads();for(int k=128;k;k/=2){if(t<k)s[t]+=s[t+k];__syncthreads();}float inv=rsqrtf(s[0]/n+1e-5f);for(int i=t;i<n;i+=256)y[i]=x[i]*inv*g[i];}
__global__ void batch_rms_backward(const float*x,const float*g,const float*dy,float*dx,float*dg,int n){x+=size_t(blockIdx.x)*n;dy+=size_t(blockIdx.x)*n;dx+=size_t(blockIdx.x)*n;dg+=size_t(blockIdx.x)*n;__shared__ float s[256],d[256];int t=threadIdx.x;float a=0,b=0;for(int i=t;i<n;i+=256){a+=x[i]*x[i];b+=dy[i]*g[i]*x[i];}s[t]=a;d[t]=b;__syncthreads();for(int k=128;k;k/=2){if(t<k){s[t]+=s[t+k];d[t]+=d[t+k];}__syncthreads();}float inv=1/sqrtf(s[0]/n+1e-5f);for(int i=t;i<n;i+=256){dx[i]+=dy[i]*g[i]*inv-x[i]*d[0]*inv*inv*inv/n;dg[i]=dy[i]*x[i]*inv;}}
}
