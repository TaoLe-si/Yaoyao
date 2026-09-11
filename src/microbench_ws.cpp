#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
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
static float dot4(const int8_t*p,const float*x,size_t cols,float a){
 __m256 s0=_mm256_setzero_ps(),s1=_mm256_setzero_ps(),s2=_mm256_setzero_ps(),s3=_mm256_setzero_ps(),a8=_mm256_set1_ps(a);size_t j=0;
 for(;j+32<=cols;j+=32){
  auto w0=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(p+j)))),a8);
  auto w1=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(p+j+8)))),a8);
  auto w2=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(p+j+16)))),a8);
  auto w3=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(p+j+24)))),a8);
  s0=_mm256_add_ps(s0,_mm256_mul_ps(w0,_mm256_loadu_ps(x+j)));s1=_mm256_add_ps(s1,_mm256_mul_ps(w1,_mm256_loadu_ps(x+j+8)));
  s2=_mm256_add_ps(s2,_mm256_mul_ps(w2,_mm256_loadu_ps(x+j+16)));s3=_mm256_add_ps(s3,_mm256_mul_ps(w3,_mm256_loadu_ps(x+j+24)));}
 s0=_mm256_add_ps(_mm256_add_ps(s0,s1),_mm256_add_ps(s2,s3));
 alignas(32)float l[8];_mm256_store_ps(l,s0);float z=0;for(float v:l)z+=v;return z;}
int main(){
 const size_t cols=512;
 for(size_t rows : {(size_t)256,(size_t)2048,(size_t)16384}){
  std::vector<int8_t> q(rows*cols);std::vector<float> sc(rows),x(cols);
  unsigned sd=31;auto nx=[&](){sd=sd*1664525u+1013904223u;return sd;};
  for(auto&v:q)v=(int8_t)((nx()>>16)%3)-1;for(auto&v:sc)v=0.001f+0.0001f*((nx()>>16)%7);
  for(auto&v:x)v=(nx()>>8)*(1.0f/16777216.0f)-0.5f;
  printf("--- %zu 行 x512  权重 %.2f MB ---\n",rows,double(rows)*cols/1e6);
  auto run=[&](int nt,float(*f)(const int8_t*,const float*,size_t,float),int reps){
   std::atomic<size_t> sk{0};std::vector<std::thread> ts;auto t0=std::chrono::steady_clock::now();
   for(int t=0;t<nt;++t)ts.emplace_back([&,t]{size_t lo=rows*t/nt,hi=rows*(t+1)/nt;float s=0;
    for(int k=0;k<reps;++k)for(size_t r=lo;r<hi;++r)s+=f(q.data()+r*cols,x.data(),cols,sc[r]);
    if(s==123456789.f)sk+=1;});
   for(auto&v:ts)v.join();auto t1=std::chrono::steady_clock::now();
   return double(rows)*cols*reps/std::chrono::duration<double>(t1-t0).count()/1e9;};
  int reps=rows>=16384?3:(rows>=2048?8:30);
  double a1=run(1,dot1,reps),a8=run(8,dot1,reps),b1=run(1,dot4,reps),b8=run(8,dot4,reps);
  printf("  1acc:  1T %6.2f  8T %6.2f  扩展 %4.2fx\n",a1,a8,a8/a1);
  printf("  4acc:  1T %6.2f  8T %6.2f  扩展 %4.2fx   4acc/1acc@8T %4.2fx\n",b1,b8,b8/b1,b8/a8);
 }
 return 0;}