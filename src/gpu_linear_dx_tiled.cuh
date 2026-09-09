#pragma once
#include "dual_state_cuda_backward.cuh"
namespace tao::dual {
__global__ void ds_dx_tiled(const float*w,const float*dy,float*dx,int rows,int cols){__shared__ float sums[8][32];int lane=threadIdx.x,part=threadIdx.y,j=blockIdx.x*32+lane;float z=0;if(j<cols)for(int i=part;i<rows;i+=8)z+=w[size_t(i)*cols+j]*dy[i];sums[part][lane]=z;__syncthreads();if(part==0&&j<cols){float total=0;for(int k=0;k<8;++k)total+=sums[k][lane];dx[j]+=total;}}
}
