#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warpSum(float v){
  #pragma unroll
  for(int o=16;o>0;o>>=1) v+=__shfl_down_sync(0xffffffffu,v,o); return v; }
__device__ __forceinline__ float warpMax(float v){
  #pragma unroll
  for(int o=16;o>0;o>>=1) v=fmaxf(v,__shfl_down_sync(0xffffffffu,v,o)); return v; }
template<int TPB> __device__ __forceinline__ float blockSum(float v,float* sm){
  const int lane=threadIdx.x&31,w=threadIdx.x>>5; constexpr int NW=TPB/32;
  v=warpSum(v); if(lane==0) sm[w]=v; __syncthreads();
  if(w==0){ float r=(threadIdx.x<NW)?sm[threadIdx.x]:0.f; r=warpSum(r); if(threadIdx.x==0) sm[0]=r; }
  __syncthreads(); const float o=sm[0]; __syncthreads(); return o; }
template<int TPB> __device__ __forceinline__ float blockMax(float v,float* sm){
  const int lane=threadIdx.x&31,w=threadIdx.x>>5; constexpr int NW=TPB/32;
  v=warpMax(v); if(lane==0) sm[w]=v; __syncthreads();
  if(w==0){ float r=(threadIdx.x<NW)?sm[threadIdx.x]:0.f; r=warpMax(r); if(threadIdx.x==0) sm[0]=r; }
  __syncthreads(); const float o=sm[0]; __syncthreads(); return o; }

template<int K,int TPB>
__global__ void estep2(const float* __restrict__ AT,const int* __restrict__ ROFF,
    const short* __restrict__ IDX,const long long* __restrict__ Y,
    const float* __restrict__ OT,const int H,const int HSC,const int n,
    float* __restrict__ NUMDEN,float* __restrict__ AO,double* __restrict__ LL){
  constexpr int VPT=K/TPB, V4=VPT/4;
  const int pos=blockIdx.x, tid=threadIdx.x;
  extern __shared__ float smem[];
  float* q=smem; float* red=smem+K;
  float4* q4=reinterpret_cast<float4*>(q);
  #pragma unroll
  for(int j=0;j<V4;j++) q4[tid*V4+j]=make_float4(1.f,1.f,1.f,1.f);
  for(int h=0;h<H;++h){
    const long long row=(long long)ROFF[h]+IDX[(long long)pos*H+h];
    const float4* a4=reinterpret_cast<const float4*>(AT+row*K)+tid*V4;
    #pragma unroll
    for(int j=0;j<V4;j++){ float4 t=a4[j]; float4& qq=q4[tid*V4+j];
      qq.x*=t.x; qq.y*=t.y; qq.z*=t.z; qq.w*=t.w; }
    if((h%3)==2){
      float m=0.f;
      #pragma unroll
      for(int j=0;j<V4;j++){ float4 v=q4[tid*V4+j]; m=fmaxf(m,fmaxf(fmaxf(v.x,v.y),fmaxf(v.z,v.w))); }
      m=blockMax<TPB>(m,red); const float inv=1.f/m;
      #pragma unroll
      for(int j=0;j<V4;j++){ float4& qq=q4[tid*V4+j]; qq.x*=inv; qq.y*=inv; qq.z*=inv; qq.w*=inv; }
    }
  }
  float s=0.f;
  #pragma unroll
  for(int j=0;j<V4;j++){ float4 v=q4[tid*V4+j]; s+=v.x+v.y+v.z+v.w; }
  float invs=1.f/blockSum<TPB>(s,red);
  #pragma unroll
  for(int j=0;j<V4;j++){ float4& qq=q4[tid*V4+j]; qq.x*=invs; qq.y*=invs; qq.z*=invs; qq.w*=invs; }
  const long long yy=Y[pos];
  const float4* o4=reinterpret_cast<const float4*>(OT+yy*K)+tid*V4;
  float4 p4[V4]; float ps=0.f;
  #pragma unroll
  for(int j=0;j<V4;j++){ float4 t=o4[j]; float4 v=q4[tid*V4+j];
    p4[j]=make_float4(v.x*t.x,v.y*t.y,v.z*t.z,v.w*t.w);
    ps+=p4[j].x+p4[j].y+p4[j].z+p4[j].w; }
  float invp=1.f/blockSum<TPB>(ps,red);
  #pragma unroll
  for(int j=0;j<V4;j++){ p4[j].x*=invp; p4[j].y*=invp; p4[j].z*=invp; p4[j].w*=invp; }
  for(int h=H-HSC;h<H;++h){
    const long long row=(long long)ROFF[h]+IDX[(long long)pos*H+h];
    float* nm=NUMDEN+row*(2LL*K); float* dn=nm+K;
    #pragma unroll
    for(int j=0;j<V4;j++){ float4 pv=p4[j]; float4 qv=q4[tid*V4+j]; int o=(tid*V4+j)*4;
      atomicAdd(nm+o,pv.x); atomicAdd(nm+o+1,pv.y); atomicAdd(nm+o+2,pv.z); atomicAdd(nm+o+3,pv.w);
      atomicAdd(dn+o,qv.x); atomicAdd(dn+o+1,qv.y); atomicAdd(dn+o+2,qv.z); atomicAdd(dn+o+3,qv.w); }
  }
  { float* ao=AO+yy*K;
    #pragma unroll
    for(int j=0;j<V4;j++){ float4 pv=p4[j]; int o=(tid*V4+j)*4;
      atomicAdd(ao+o,pv.x); atomicAdd(ao+o+1,pv.y); atomicAdd(ao+o+2,pv.z); atomicAdd(ao+o+3,pv.w); } }
  if(tid==0) atomicAdd(LL,(double)logf(ps));
}

void estep2_run(torch::Tensor AT,torch::Tensor ROFF,torch::Tensor IDX,torch::Tensor Y,torch::Tensor OT,
                torch::Tensor NUMDEN,torch::Tensor AO,torch::Tensor LL,int64_t HSC){
  const int n=IDX.size(0),H=IDX.size(1),K=AT.size(1);
  if(K==1024){ constexpr int TPB=128;
    estep2<1024,TPB><<<n,TPB,K*sizeof(float)+(TPB/32)*sizeof(float)>>>(AT.data_ptr<float>(),ROFF.data_ptr<int>(),
      IDX.data_ptr<short>(),Y.data_ptr<long long>(),OT.data_ptr<float>(),H,(int)HSC,n,
      NUMDEN.data_ptr<float>(),AO.data_ptr<float>(),LL.data_ptr<double>());
  } else if(K==512){ constexpr int TPB=128;
    estep2<512,TPB><<<n,TPB,K*sizeof(float)+(TPB/32)*sizeof(float)>>>(AT.data_ptr<float>(),ROFF.data_ptr<int>(),
      IDX.data_ptr<short>(),Y.data_ptr<long long>(),OT.data_ptr<float>(),H,(int)HSC,n,
      NUMDEN.data_ptr<float>(),AO.data_ptr<float>(),LL.data_ptr<double>());
  } else TORCH_CHECK(false,"K must be 512 or 1024");
}

template<int K,int TPB>
__global__ void argmax_kernel(const float* __restrict__ AT,const int* __restrict__ ROFF,
    const short* __restrict__ IDX,const int H,const int n,short* __restrict__ obuf){
  constexpr int VPT=K/TPB, V4=VPT/4;
  const int pos=blockIdx.x, tid=threadIdx.x;
  extern __shared__ float smem[];
  float* q=smem; float* red=smem+K; float* bv=red+(TPB/32); float* bi=bv+TPB;
  float4* q4=reinterpret_cast<float4*>(q);
  #pragma unroll
  for(int j=0;j<V4;j++) q4[tid*V4+j]=make_float4(1.f,1.f,1.f,1.f);
  for(int h=0;h<H;++h){
    const long long row=(long long)ROFF[h]+IDX[(long long)pos*H+h];
    const float4* a4=reinterpret_cast<const float4*>(AT+row*K)+tid*V4;
    #pragma unroll
    for(int j=0;j<V4;j++){ float4 t=a4[j]; float4& qq=q4[tid*V4+j];
      qq.x*=t.x; qq.y*=t.y; qq.z*=t.z; qq.w*=t.w; }
    if((h%3)==2){
      float m=0.f;
      #pragma unroll
      for(int j=0;j<V4;j++){ float4 v=q4[tid*V4+j]; m=fmaxf(m,fmaxf(fmaxf(v.x,v.y),fmaxf(v.z,v.w))); }
      m=blockMax<TPB>(m,red); const float inv=1.f/m;
      #pragma unroll
      for(int j=0;j<V4;j++){ float4& qq=q4[tid*V4+j]; qq.x*=inv; qq.y*=inv; qq.z*=inv; qq.w*=inv; }
    }
  }
  float best=-1e30f; int bidx=0;
  #pragma unroll
  for(int j=0;j<V4;j++){ float4 v=q4[tid*V4+j];
    if(v.x>best){best=v.x;bidx=(tid*V4+j)*4;}
    if(v.y>best){best=v.y;bidx=(tid*V4+j)*4+1;}
    if(v.z>best){best=v.z;bidx=(tid*V4+j)*4+2;}
    if(v.w>best){best=v.w;bidx=(tid*V4+j)*4+3;} }
  bv[tid]=best; bi[tid]=(float)bidx;
  __syncthreads();
  if(tid==0){
    float b=bv[0]; int k=0;
    for(int t=1;t<TPB;t++) if(bv[t]>b){ b=bv[t]; k=t; }
    obuf[pos]=(short)((int)bi[k]);
  }
}
void argmax_run(torch::Tensor AT,torch::Tensor ROFF,torch::Tensor IDX,torch::Tensor obuf){
  const int n=IDX.size(0),H=IDX.size(1),K=AT.size(1);
  const int smem=K*(int)sizeof(float)+(128/32)*(int)sizeof(float)+2*128*(int)sizeof(float);
  if(K==1024){ constexpr int TPB=128;
    argmax_kernel<1024,TPB><<<n,TPB,smem>>>(AT.data_ptr<float>(),ROFF.data_ptr<int>(),IDX.data_ptr<short>(),H,n,obuf.data_ptr<short>());
  } else if(K==512){ constexpr int TPB=128;
    argmax_kernel<512,TPB><<<n,TPB,smem>>>(AT.data_ptr<float>(),ROFF.data_ptr<int>(),IDX.data_ptr<short>(),H,n,obuf.data_ptr<short>());
  } else TORCH_CHECK(false,"K must be 512 or 1024");
}

template<int K,int TPB>
__global__ void estep_out(const float* __restrict__ AT,const int* __restrict__ ROFF,
    const short* __restrict__ IDX,const long long* __restrict__ Y,
    const float* __restrict__ OT,const int H,const int n,
    float* __restrict__ PQ,double* __restrict__ LL){
  constexpr int VPT=K/TPB, V4=VPT/4;
  const int pos=blockIdx.x, tid=threadIdx.x;
  extern __shared__ float smem[];
  float* q=smem; float* red=smem+K;
  float4* q4=reinterpret_cast<float4*>(q);
  #pragma unroll
  for(int j=0;j<V4;j++) q4[tid*V4+j]=make_float4(1.f,1.f,1.f,1.f);
  for(int h=0;h<H;++h){
    const long long row=(long long)ROFF[h]+IDX[(long long)pos*H+h];
    const float4* a4=reinterpret_cast<const float4*>(AT+row*K)+tid*V4;
    #pragma unroll
    for(int j=0;j<V4;j++){ float4 t=a4[j]; float4& qq=q4[tid*V4+j];
      qq.x*=t.x; qq.y*=t.y; qq.z*=t.z; qq.w*=t.w; }
    if((h%3)==2){
      float m=0.f;
      #pragma unroll
      for(int j=0;j<V4;j++){ float4 v=q4[tid*V4+j]; m=fmaxf(m,fmaxf(fmaxf(v.x,v.y),fmaxf(v.z,v.w))); }
      m=blockMax<TPB>(m,red); const float inv=1.f/m;
      #pragma unroll
      for(int j=0;j<V4;j++){ float4& qq=q4[tid*V4+j]; qq.x*=inv; qq.y*=inv; qq.z*=inv; qq.w*=inv; }
    }
  }
  float s=0.f;
  #pragma unroll
  for(int j=0;j<V4;j++){ float4 v=q4[tid*V4+j]; s+=v.x+v.y+v.z+v.w; }
  float invs=1.f/blockSum<TPB>(s,red);
  #pragma unroll
  for(int j=0;j<V4;j++){ float4& qq=q4[tid*V4+j]; qq.x*=invs; qq.y*=invs; qq.z*=invs; qq.w*=invs; }
  const long long yy=Y[pos];
  const float4* o4=reinterpret_cast<const float4*>(OT+yy*K)+tid*V4;
  float4 p4[V4]; float ps=0.f;
  #pragma unroll
  for(int j=0;j<V4;j++){ float4 t=o4[j]; float4 v=q4[tid*V4+j];
    p4[j]=make_float4(v.x*t.x,v.y*t.y,v.z*t.z,v.w*t.w);
    ps+=p4[j].x+p4[j].y+p4[j].z+p4[j].w; }
  float invp=1.f/blockSum<TPB>(ps,red);
  float* dst=PQ+(long long)pos*(2*K);
  #pragma unroll
  for(int j=0;j<V4;j++){ int o=(tid*V4+j)*4; float4 pv=p4[j]; float4 qv=q4[tid*V4+j];
    pv.x*=invp; pv.y*=invp; pv.z*=invp; pv.w*=invp;
    *reinterpret_cast<float4*>(dst+o)=pv;
    *reinterpret_cast<float4*>(dst+K+o)=qv; }
  if(tid==0) atomicAdd(LL,(double)logf(ps));
}
void estep_out_run(torch::Tensor AT,torch::Tensor ROFF,torch::Tensor IDX,torch::Tensor Y,
                   torch::Tensor OT,torch::Tensor PQ,torch::Tensor LL){
  const int n=IDX.size(0),H=IDX.size(1),K=AT.size(1);
  if(K==1024){ constexpr int TPB=128;
    estep_out<1024,TPB><<<n,TPB,K*sizeof(float)+(TPB/32)*sizeof(float)>>>(AT.data_ptr<float>(),ROFF.data_ptr<int>(),
      IDX.data_ptr<short>(),Y.data_ptr<long long>(),OT.data_ptr<float>(),H,n,PQ.data_ptr<float>(),LL.data_ptr<double>());
  } else if(K==512){ constexpr int TPB=128;
    estep_out<512,TPB><<<n,TPB,K*sizeof(float)+(TPB/32)*sizeof(float)>>>(AT.data_ptr<float>(),ROFF.data_ptr<int>(),
      IDX.data_ptr<short>(),Y.data_ptr<long long>(),OT.data_ptr<float>(),H,n,PQ.data_ptr<float>(),LL.data_ptr<double>());
  } else TORCH_CHECK(false,"K must be 512 or 1024");
}
