#pragma once
#include "cpu_ternary_avx2.hpp"
struct PipelineRowsUnroll4:CpuTernaryRows{using CpuTernaryRows::CpuTernaryRows;
float dot(size_t row,const float*x)const{__m256 sum=_mm256_setzero_ps(),alpha=_mm256_set1_ps(scale[row]);size_t j=0;const int8_t*p=q.data()+row*cols;for(;j+32<=cols;j+=32){
auto w0=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p+j+0)))),alpha);
auto w1=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p+j+8)))),alpha);
auto w2=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p+j+16)))),alpha);
auto w3=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p+j+24)))),alpha);
auto product0=_mm256_mul_ps(w0,_mm256_loadu_ps(x+j+0));
auto product1=_mm256_mul_ps(w1,_mm256_loadu_ps(x+j+8));
auto product2=_mm256_mul_ps(w2,_mm256_loadu_ps(x+j+16));
auto product3=_mm256_mul_ps(w3,_mm256_loadu_ps(x+j+24));
sum=_mm256_add_ps(sum,product0);
sum=_mm256_add_ps(sum,product1);
sum=_mm256_add_ps(sum,product2);
sum=_mm256_add_ps(sum,product3);
}
for(;j+16<=cols;j+=16){auto w0=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p+j)))),alpha);auto w1=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p+j+8)))),alpha);auto a=_mm256_mul_ps(w0,_mm256_loadu_ps(x+j));auto b=_mm256_mul_ps(w1,_mm256_loadu_ps(x+j+8));sum=_mm256_add_ps(sum,a);sum=_mm256_add_ps(sum,b);}for(;j+8<=cols;j+=8){__m128i bytes=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p+j));__m256 weights=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes)),alpha);sum=_mm256_add_ps(sum,_mm256_mul_ps(weights,_mm256_loadu_ps(x+j)));}alignas(32)float lane[8];_mm256_store_ps(lane,sum);float z=0;for(float v:lane)z+=v;for(;j<cols;++j)z+=(float(p[j])*scale[row])*x[j];return z;}
};
