#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <vector>
#include <thread>
#include <atomic>
#include <immintrin.h>
// 现有 int8 内核（基线）
static float dot_i8(const int8_t*p,const float*x,size_t cols,float a){
 __m256 s=_mm256_setzero_ps(),a8=_mm256_set1_ps(a);size_t j=0;
 for(;j+16<=cols;j+=16){auto w0=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(p+j)))),a8);
  auto w1=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(p+j+8)))),a8);
  s=_mm256_add_ps(s,_mm256_mul_ps(w0,_mm256_loadu_ps(x+j)));s=_mm256_add_ps(s,_mm256_mul_ps(w1,_mm256_loadu_ps(x+j+8)));}
 alignas(32)float l[8];_mm256_store_ps(l,s);float z=0;for(float v:l)z+=v;return z;}
// 位平面内核：weight = b0 - b1。2 bit/权重。
static float dot_bp(const uint8_t*p0,const uint8_t*p1,const float*x,size_t cols,float a){
 __m256 s=_mm256_setzero_ps();
 const __m256i SH=_mm256_setr_epi32(31,30,29,28,27,26,25,24); // lane l 的 bit l -> bit31
 size_t j=0;
 // 每 8 个权重共用一个字节：lane l 需要该字节的 bit l
 for(;j+8<=cols;j+=8){
  const uint8_t c0=p0[j>>3], c1=p1[j>>3];
  __m256i m0=_mm256_srai_epi32(_mm256_sllv_epi32(_mm256_set1_epi32(c0),SH),31);
  __m256i m1=_mm256_srai_epi32(_mm256_sllv_epi32(_mm256_set1_epi32(c1),SH),31);
  __m256 xv=_mm256_loadu_ps(x+j);
  s=_mm256_add_ps(s,_mm256_sub_ps(_mm256_and_ps(_mm256_castsi256_ps(m0),xv),_mm256_and_ps(_mm256_castsi256_ps(m1),xv)));
 }
 alignas(32)float l[8];_mm256_store_ps(l,s);float z=0;for(float v:l)z+=v;
 return z*a;
}
int main(){
 const size_t cols=512;
 for(size_t rows : {(size_t)2048,(size_t)16384}){
  std::vector<int8_t> q(rows*cols);std::vector<uint8_t> b0(rows*cols/8+1),b1(rows*cols/8+1);
  std::vector<float> sc(rows),x(cols);
  unsigned sd=8123;auto nx=[&](){sd=sd*1664525u+1013904223u;return sd;};
  for(auto&v:q)v=(int8_t)((nx()>>16)%3)-1;
  for(auto&v:sc)v=0.001f+0.0001f*((nx()>>16)%7);
  for(auto&v:x)v=(nx()>>8)*(1.0f/16777216.0f)-0.5f;
  for(size_t i=0;i<rows*cols;++i){if(q[i]==1)b0[i/8]|=uint8_t(1u<<(i%8));if(q[i]==-1)b1[i/8]|=uint8_t(1u<<(i%8));}
  // 校验
  double mx=0;
  for(size_t r=0;r<rows;r+=211){
   float A=dot_i8(q.data()+r*cols,x.data(),cols,sc[r]);
   float B=dot_bp(b0.data()+r*cols/8,b1.data()+r*cols/8,x.data(),cols,sc[r]);
   mx=std::max(mx,std::fabs(double(A)-double(B))/std::max(1e-30,std::fabs(double(A))));}
  auto run=[&](int nt,int kind,int reps){
   std::atomic<size_t> sk{0};std::vector<std::thread> ts;auto t0=std::chrono::steady_clock::now();
   for(int t=0;t<nt;++t)ts.emplace_back([&,t]{size_t lo=rows*t/nt,hi=rows*(t+1)/nt;float s=0;
    for(int k=0;k<reps;++k)for(size_t r=lo;r<hi;++r)
     s+= kind? dot_bp(b0.data()+r*cols/8,b1.data()+r*cols/8,x.data(),cols,sc[r])
             : dot_i8(q.data()+r*cols,x.data(),cols,sc[r]);
    if(s==123456789.f)sk+=1;});
   for(auto&v:ts)v.join();auto t1=std::chrono::steady_clock::now();
   return double(rows)*cols*reps/std::chrono::duration<double>(t1-t0).count()/1e9;};
  int reps=rows>=16384?3:8;
  printf("=== %zu 行 x512  int8 %.2f MB / 位平面 %.2f MB  最大相对差 %.2e ===\n",
         rows,double(rows)*cols/1e6,double(rows)*cols/4/1e6,mx);
  double a1=run(1,0,reps),b1g=run(1,1,reps),a8=run(8,0,reps),b8=run(8,1,reps);
  printf("  int8     1T %6.2f   8T %6.2f\n",a1,a8);
  printf("  位平面   1T %6.2f   8T %6.2f   位平面/int8 @8T = %.2fx\n",b1g,b8,b8/a8);
 }
 return 0;}