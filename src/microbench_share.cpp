#include <cstdio>
#include <cstdint>
#include <chrono>
#include <vector>
#include <thread>
#include <atomic>
#include <immintrin.h>
static float dot1(const int8_t*p,const float*x,size_t cols,float a){
 __m256 s=_mm256_setzero_ps(),a8=_mm256_set1_ps(a);size_t j=0;
 for(;j+16<=cols;j+=16){auto w0=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(p+j)))),a8);
  auto w1=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(p+j+8)))),a8);
  s=_mm256_add_ps(s,_mm256_mul_ps(w0,_mm256_loadu_ps(x+j)));s=_mm256_add_ps(s,_mm256_mul_ps(w1,_mm256_loadu_ps(x+j+8)));}
 alignas(32)float l[8];_mm256_store_ps(l,s);float z=0;for(float v:l)z+=v;return z;}
// 固定每线程工作量，只变线程数 —— 用于判断 8 线程上限是「每核算力」还是「共享资源」
int main(){
 const size_t cols=512,RPThread=512;   // 512 行/线程 = 0.26 MB，单核可驻 L2
 const size_t maxRows=RPThread*8;
 std::vector<int8_t> q(maxRows*cols);std::vector<float> sc(maxRows),x(cols);
 unsigned sd=5150;auto nx=[&](){sd=sd*1664525u+1013904223u;return sd;};
 for(auto&v:q)v=(int8_t)((nx()>>16)%3)-1;for(auto&v:sc)v=0.001f+0.0001f*((nx()>>16)%7);
 for(auto&v:x)v=(nx()>>8)*(1.0f/16777216.0f)-0.5f;
 printf("固定 512 行/线程（0.26 MB/线程，单核可驻 L2），只变线程数:\n");
 printf("  线程  总行数  总MB    G MAC/s   每核 G   相对1T\n");
 double base=0;
 for(int nt:{1,2,3,4,6,8}){
  size_t rows=RPThread*nt;
  std::atomic<size_t> sk{0};std::vector<std::thread> ts;
  auto t0=std::chrono::steady_clock::now();
  for(int t=0;t<nt;++t)ts.emplace_back([&,t]{size_t lo=RPThread*t,hi=RPThread*(t+1);float s=0;
   for(int k=0;k<40;++k)for(size_t r=lo;r<hi;++r)s+=dot1(q.data()+r*cols,x.data(),cols,sc[r]);
   if(s==123456789.f)sk+=1;});
  for(auto&v:ts)v.join();auto t1=std::chrono::steady_clock::now();
  double g=double(rows)*cols*40/std::chrono::duration<double>(t1-t0).count()/1e9;
  if(nt==1)base=g;
  printf("  %4d %7zu %6.2f %10.2f %8.2f %7.2fx\n",nt,rows,double(rows)*cols/1e6,g,g/nt,g/base);
 }
 printf("\n对照: 8 线程 x 16384 行（8.39 MB，远超 L2）上一轮测得 39.66 G\n");
 return 0;}
