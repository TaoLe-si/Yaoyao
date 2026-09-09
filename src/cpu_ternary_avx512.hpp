#pragma once
#include "cpu_ternary_avx2.hpp"
inline bool use_cpu_avx512=true;
struct CpuTernaryRows512:CpuTernaryRows{using CpuTernaryRows::CpuTernaryRows;
float dot(size_t row,const float*x)const{if(!use_cpu_avx512)return CpuTernaryRows::dot(row,x);__m256 sum=_mm256_setzero_ps();__m512 alpha=_mm512_set1_ps(scale[row]);size_t j=0;const int8_t*p=q.data()+row*cols;for(;j+16<=cols;j+=16){__m512 w=_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p+j)))),alpha);__m512 product=_mm512_mul_ps(w,_mm512_loadu_ps(x+j));sum=_mm256_add_ps(sum,_mm512_castps512_ps256(product));sum=_mm256_add_ps(sum,_mm512_extractf32x8_ps(product,1));}if(j+8<=cols){__m256 w=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p+j)))),_mm256_set1_ps(scale[row]));sum=_mm256_add_ps(sum,_mm256_mul_ps(w,_mm256_loadu_ps(x+j)));j+=8;}alignas(32)float lane[8];_mm256_store_ps(lane,sum);float z=0;for(float v:lane)z+=v;for(;j<cols;++j)z+=(float(p[j])*scale[row])*x[j];return z;}
};
