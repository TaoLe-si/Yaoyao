#pragma once
#include "cpu_ternary_avx2.hpp"
struct PrefetchRows:CpuTernaryRows{using CpuTernaryRows::CpuTernaryRows;
float dot(size_t row,const float*x)const{__m256 sum=_mm256_setzero_ps(),alpha=_mm256_set1_ps(scale[row]);size_t j=0;const int8_t*p=q.data()+row*cols;for(;j+8<=cols;j+=8){if((j&63)==0&&j+256<cols)_mm_prefetch(reinterpret_cast<const char*>(p+j+256),_MM_HINT_T0);__m256 w=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p+j)))),alpha);sum=_mm256_add_ps(sum,_mm256_mul_ps(w,_mm256_loadu_ps(x+j)));}alignas(32)float lane[8];_mm256_store_ps(lane,sum);float z=0;for(float v:lane)z+=v;for(;j<cols;++j)z+=(float(p[j])*scale[row])*x[j];return z;}
};
