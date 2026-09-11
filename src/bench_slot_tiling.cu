#define NOMINMAX
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
static void chk(cudaError_t e){if(e!=cudaSuccess){printf("CUDA ERR %s\n",cudaGetErrorString(e));exit(1);}}
static void launch_bmv(const float*w,const float*x,float*y,int rows,int cols,int slots);
// ---- 现状 ----
__global__ void ds_batched_matvec(const float*w,const float*x,float*y,int rows,int cols,int slots){
    int slot=blockIdx.y,row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;
    float z=0;
    if(slot<slots&&row<rows)for(int j=lane;j<cols;j+=32)z+=w[size_t(row)*cols+j]*x[size_t(slot)*cols+j];
    for(int k=16;k;k>>=1)z+=__shfl_down_sync(0xffffffff,z,k);
    if(slot<slots&&row<rows&&!lane)y[size_t(slot)*rows+row]=z;
}
__global__ void ds_batch_dw(const float*x,const float*dy,float*dw,int rows,int cols,int slots){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<size_t(rows)*cols){float a=0;for(int s=0;s<slots;++s)a+=dy[size_t(s)*rows+i/cols]*x[size_t(s)*cols+i%cols];dw[i]+=a;}
}
// ---- 新：一个 warp 一行，lane 沿 cols，寄存器里同时累加 SLOTS 个 slot，权重只读一次 ----
template<int SLOTS>
__global__ void ds_bmv_slots(const float*w,const float*x,float*y,int rows,int cols){
    int warp=threadIdx.x/32,lane=threadIdx.x%32;
    int row=blockIdx.x*(blockDim.x/32)+warp;
    if(row>=rows)return;
    const float*wr=w+size_t(row)*cols;
    float acc[SLOTS];
    #pragma unroll
    for(int s=0;s<SLOTS;++s)acc[s]=0.f;
    for(int j=lane;j<cols;j+=32){
        const float wv=wr[j];
        #pragma unroll
        for(int s=0;s<SLOTS;++s)acc[s]+=wv*x[size_t(s)*cols+j];
    }
    #pragma unroll
    for(int s=0;s<SLOTS;++s){
        float z=acc[s];
        for(int k=16;k;k>>=1)z+=__shfl_down_sync(0xffffffffu,z,k);
        if(!lane)y[size_t(s)*rows+row]=z;
    }
}
static void launch_bmv(const float*w,const float*x,float*y,int rows,int cols,int slots){
    dim3 g((rows+7)/8);
    switch(slots){
      case 1: ds_bmv_slots<1><<<g,256>>>(w,x,y,rows,cols); break;
      case 2: ds_bmv_slots<2><<<g,256>>>(w,x,y,rows,cols); break;
      case 3: ds_bmv_slots<3><<<g,256>>>(w,x,y,rows,cols); break;
      case 4: ds_bmv_slots<4><<<g,256>>>(w,x,y,rows,cols); break;
      case 6: ds_bmv_slots<6><<<g,256>>>(w,x,y,rows,cols); break;
      case 8: ds_bmv_slots<8><<<g,256>>>(w,x,y,rows,cols); break;
      case 12: ds_bmv_slots<12><<<g,256>>>(w,x,y,rows,cols); break;
      case 16: ds_bmv_slots<16><<<g,256>>>(w,x,y,rows,cols); break;
      case 32: ds_bmv_slots<32><<<g,256>>>(w,x,y,rows,cols); break;
      default: ds_batched_matvec<<<dim3((rows+3)/4,slots),128>>>(w,x,y,rows,cols,slots); break;
    }
}
__global__ void ds_dw_slots(const float*x,const float*dy,float*dw,int rows,int cols,int slots){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(rows)*cols)return;
    const int r=int(i/cols), c=int(i%cols);
    float a=0; for(int s=0;s<slots;++s)a+=dy[size_t(s)*rows+r]*x[size_t(s)*cols+c];
    dw[i]+=a;
}
struct Shape{const char*name;int rows,cols;};
int main(){
    setvbuf(stdout,NULL,_IONBF,0);
    cudaDeviceProp pr; chk(cudaGetDeviceProperties(&pr,0));
    printf("GPU: %s  SMs=%d  mem %.0f MHz x %d bit  L2=%d MB\n",pr.name,pr.multiProcessorCount,pr.memoryClockRate/1000.0,pr.memoryBusWidth,pr.l2CacheSize>>20);
    std::vector<Shape> sh={
        {"s.cand.x",128,512},{"s.cand.s",128,128},{"s.gate.x",128,512},{"s.gate.s",128,128},
        {"m.cand.x",512,512},{"m.cand.s",512,128},{"m.cand.m",512,512},
        {"m.gate.x",512,512},{"m.gate.s",512,128},{"m.gate.m",512,512},
        {"read.s",512,128},{"read.m",512,512},{"head",16384,512}};
    for(int slots : {4,8,16,32}){
        double t_old=0,t_new=0,t_dwo=0,t_dwn=0,b_old=0,b_new=0;
        for(auto&s:sh){
            size_t wn=size_t(s.rows)*s.cols;
            float *w,*x,*y,*dw;
            chk(cudaMalloc(&w,wn*4)); chk(cudaMalloc(&x,size_t(slots)*s.cols*4));
            chk(cudaMalloc(&y,size_t(slots)*s.rows*4)); chk(cudaMalloc(&dw,wn*4));
            chk(cudaMemset(w,1,wn*4)); chk(cudaMemset(x,1,size_t(slots)*s.cols*4)); chk(cudaMemset(dw,0,wn*4));
            int iters = s.rows>=16384?40:200;
            cudaEvent_t a,b; chk(cudaEventCreate(&a)); chk(cudaEventCreate(&b)); float ms;
            chk(cudaEventRecord(a));
            for(int i=0;i<iters;++i) ds_batched_matvec<<<dim3((s.rows+3)/4,slots),128>>>(w,x,y,s.rows,s.cols,slots);
            chk(cudaEventRecord(b)); chk(cudaEventSynchronize(b)); chk(cudaEventElapsedTime(&ms,a,b)); t_old+=ms/iters;
            chk(cudaEventRecord(a));
            for(int i=0;i<iters;++i) launch_bmv(w,x,y,s.rows,s.cols,slots);
            chk(cudaEventRecord(b)); chk(cudaEventSynchronize(b)); chk(cudaEventElapsedTime(&ms,a,b)); t_new+=ms/iters;
            chk(cudaEventRecord(a));
            for(int i=0;i<iters;++i) ds_batch_dw<<<(unsigned)((wn+127)/128),128>>>(x,y,dw,s.rows,s.cols,slots);
            chk(cudaEventRecord(b)); chk(cudaEventSynchronize(b)); chk(cudaEventElapsedTime(&ms,a,b)); t_dwo+=ms/iters;
            chk(cudaEventRecord(a));
            for(int i=0;i<iters;++i) ds_dw_slots<<<(unsigned)((wn+127)/128),128>>>(x,y,dw,s.rows,s.cols,slots);
            chk(cudaEventRecord(b)); chk(cudaEventSynchronize(b)); chk(cudaEventElapsedTime(&ms,a,b)); t_dwn+=ms/iters;
            b_old+=double(wn)*slots*4; b_new+=double(wn)*4;
            cudaFree(w);cudaFree(x);cudaFree(y);cudaFree(dw);
        }
        double F=8.0, fo=t_old*F, fn=t_new*F;
        printf("\n=== slots=%d ===\n",slots);
        printf("  前向 matvec  8层合计: 现状 %7.2f ms/token   新 %7.2f ms/token   加速 %5.2fx\n",fo,fn,fo/fn);
        printf("  权重流量/ token   : 现状 %7.1f MB       新 %7.1f MB\n",b_old*F/1e6,b_new*F/1e6);
        printf("  实测带宽          : 现状 %7.0f GB/s     新 %7.0f GB/s\n",b_old*F/(fo/1e3)/1e9,b_new*F/(fn/1e3)/1e9);
        printf("  反向 dw   8层合计 : 现状 %7.2f ms/token   新 %7.2f ms/token   加速 %5.2fx\n",t_dwo*F,t_dwn*F,t_dwo/t_dwn);
    }
    return 0;
}
