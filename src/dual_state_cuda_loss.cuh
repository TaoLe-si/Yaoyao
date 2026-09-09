#pragma once
#include "dual_state_autograd.cuh"
namespace tao::dual {
// Reference reduction: one thread; optimize only after integration correctness.
__global__ void ds_cross_entropy(const float*logits,float*grad,float*loss,int n,int target,float multiplier){if(threadIdx.x||blockIdx.x)return;float mx=logits[0];for(int j=1;j<n;++j)mx=fmaxf(mx,logits[j]);float sum=0;for(int j=0;j<n;++j)sum+=expf(logits[j]-mx);loss[0]=multiplier*(logf(sum)+mx-logits[target]);for(int j=0;j<n;++j)grad[j]+=multiplier*(expf(logits[j]-mx)/sum-(j==target));}
inline float seed_loss(Node output,unsigned target,bool supervised,float multiplier=1.f){if(target>=output->value.n||!std::isfinite(multiplier)||multiplier<0)throw std::invalid_argument("loss arguments");if(!supervised)return 0;Device loss(1);ds_cross_entropy<<<1,1>>>(output->value.p,output->grad.p,loss.p,int(output->value.n),target,multiplier);check(cudaGetLastError());return loss.host()[0];}
}
