#pragma once
#include <immintrin.h>
#include <cstddef>
inline float tao_dot_avx2(const float*a,const float*b,size_t n){__m256 s=_mm256_setzero_ps();size_t j=0;
#ifdef TAO_CPU_AVX2_UNROLL
__m256 t=s,u=s,vv=s;for(;j+32<=n;j+=32){s=_mm256_add_ps(s,_mm256_mul_ps(_mm256_loadu_ps(a+j),_mm256_loadu_ps(b+j)));t=_mm256_add_ps(t,_mm256_mul_ps(_mm256_loadu_ps(a+j+8),_mm256_loadu_ps(b+j+8)));u=_mm256_add_ps(u,_mm256_mul_ps(_mm256_loadu_ps(a+j+16),_mm256_loadu_ps(b+j+16)));vv=_mm256_add_ps(vv,_mm256_mul_ps(_mm256_loadu_ps(a+j+24),_mm256_loadu_ps(b+j+24)));}s=_mm256_add_ps(_mm256_add_ps(s,t),_mm256_add_ps(u,vv));
#endif
for(;j+8<=n;j+=8)s=_mm256_add_ps(s,_mm256_mul_ps(_mm256_loadu_ps(a+j),_mm256_loadu_ps(b+j)));alignas(32) float v[8];_mm256_store_ps(v,s);float z=0;for(float x:v)z+=x;for(;j<n;++j)z+=a[j]*b[j];return z;}
