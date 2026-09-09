#pragma once
#include "dual_state_cuda_optimizer.cuh"
namespace tao::dual {
__global__ void ds_adamw_device(float*w,const float*g,float*m,float*v,int n,const double*norm2,const unsigned*bad,double supervised,float lr,float decay,float bc1,float bc2){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<n&&!*bad&&isfinite(*norm2)){float grad_factor=float(1.0/supervised/fmax(1.0,sqrt(*norm2)));float z=g[j]*grad_factor;m[j]=.9f*m[j]+.1f*z;v[j]=.999f*v[j]+.001f*z*z;w[j]-=lr*((m[j]/bc1)/(sqrtf(v[j]/bc2)+1e-8f)+decay*w[j]);}}
}
