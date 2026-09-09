#pragma once
#include "dual_state_cuda_optimizer.cuh"
namespace tao::dual {
// One block per row; bitonic sort by magnitude descending, index ascending.
template<int Width> __global__ void ds_project_sorted(const float*master,float*out,float*scales,int rows,int cols){int row=blockIdx.x;if(row>=rows)return;__shared__ float mag[Width];__shared__ int idx[Width];__shared__ int chosen;__shared__ float alpha;
for(int i=threadIdx.x;i<Width;i+=blockDim.x){mag[i]=i<cols?fabsf(master[size_t(row)*cols+i]):-1.f;idx[i]=i;}__syncthreads();
for(int k=2;k<=Width;k*=2)for(int j=k/2;j;j/=2){for(int i=threadIdx.x;i<Width;i+=blockDim.x){int other=i^j;if(other>i){bool before=mag[i]>mag[other]||(mag[i]==mag[other]&&idx[i]<idx[other]);bool descending=(i&k)==0;if(before!=descending){float m=mag[i];mag[i]=mag[other];mag[other]=m;int n=idx[i];idx[i]=idx[other];idx[other]=n;}}}__syncthreads();}
if(threadIdx.x==0){float sum=0,best=-1;chosen=0;alpha=1;for(int i=0;i<cols;++i){sum+=mag[i];float score=sum*sum/(i+1);if(score>best){best=score;chosen=i+1;alpha=sum/chosen;}}if(best==0)alpha=1;scales[row]=alpha;}__syncthreads();for(int i=threadIdx.x;i<cols;i+=blockDim.x){int col=idx[i];float v=master[size_t(row)*cols+col];out[size_t(row)*cols+col]=(i<chosen&&v!=0)?copysignf(alpha,v):0;}}
inline void project_sorted(Device&master,Device&out,Device&scales,int rows,int cols){if(cols<=0||cols>1024)throw std::runtime_error("sorted width");if(cols<=128)ds_project_sorted<128><<<rows,128>>>(master.p,out.p,scales.p,rows,cols);else if(cols<=256)ds_project_sorted<256><<<rows,128>>>(master.p,out.p,scales.p,rows,cols);else if(cols<=512)ds_project_sorted<512><<<rows,128>>>(master.p,out.p,scales.p,rows,cols);else ds_project_sorted<1024><<<rows,128>>>(master.p,out.p,scales.p,rows,cols);check(cudaGetLastError());}
}
