#pragma once
#include "dual_state_cuda_loss.cuh"
namespace tao::dual {
// Exact per-row ranking reference, O(cols^2), deterministic ties by column index.
__global__ void ds_project(const float*master,float*effective,float*scales,int rows,int cols){int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=rows)return;const float*w=master+size_t(r)*cols;float best=-1,alpha=1;int bestk=cols+1;for(int j=0;j<cols;++j){int count=0;float sum=0;for(int k=0;k<cols;++k)if(fabsf(w[k])>fabsf(w[j])||(fabsf(w[k])==fabsf(w[j])&&k<=j)){++count;sum+=fabsf(w[k]);}float score=sum*sum/count;if(score>best||(score==best&&count<bestk)){best=score;bestk=count;alpha=sum/count;}}if(best==0)alpha=1;scales[r]=alpha;for(int j=0;j<cols;++j){int rank=0;for(int k=0;k<cols;++k)if(fabsf(w[k])>fabsf(w[j])||(fabsf(w[k])==fabsf(w[j])&&k<j))++rank;effective[size_t(r)*cols+j]=(rank<bestk&&w[j]!=0)?copysignf(alpha,w[j]):0;}}
__global__ void ds_adamw(float*w,const float*g,float*m,float*v,int n,float grad_factor,float lr,float decay,float bc1,float bc2){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<n){float z=g[j]*grad_factor;m[j]=.9f*m[j]+.1f*z;v[j]=.999f*v[j]+.001f*z*z;w[j]-=lr*((m[j]/bc1)/(sqrtf(v[j]/bc2)+1e-8f)+decay*w[j]);}}
}
