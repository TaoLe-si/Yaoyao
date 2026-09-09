#pragma once
#include <immintrin.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <cstdint>
// Exact ternary symbol extraction: reject arbitrary FP32 matrices rather than requantize.
struct CpuTernaryRows {size_t rows,cols;std::vector<int8_t>q;std::vector<float>scale;
CpuTernaryRows(const std::vector<float>&w,size_t r,size_t c):rows(r),cols(c),q(w.size()),scale(r,1){if(!r||!c||w.size()!=r*c)throw std::runtime_error("ternary shape");for(size_t i=0;i<r;++i){float a=0;for(size_t j=0;j<c;++j){float v=w[i*c+j];if(!std::isfinite(v))throw std::runtime_error("nonfinite weight");if(v!=0){if(a&&a!=std::abs(v))throw std::runtime_error("not exact row ternary");a=std::abs(v);q[i*c+j]=v>0?1:-1;}}scale[i]=a?a:1;}}
float dot(size_t row,const float*x)const{__m256 sum=_mm256_setzero_ps(),alpha=_mm256_set1_ps(scale[row]);size_t j=0;const int8_t*p=q.data()+row*cols;for(;j+8<=cols;j+=8){__m128i bytes=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p+j));__m256 weights=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes)),alpha);sum=_mm256_add_ps(sum,_mm256_mul_ps(weights,_mm256_loadu_ps(x+j)));}alignas(32)float lane[8];_mm256_store_ps(lane,sum);float z=0;for(float v:lane)z+=v;for(;j<cols;++j)z+=(float(p[j])*scale[row])*x[j];return z;}
// Four independent lane accumulators preserve dot()'s per-row addition order.
// Compile with explicit AVX2 and strict FP (no contraction/reassociation).
void matvec_four_rows(const float* x,float* y)const{
size_t row=0;
for(;row+4<=rows;row+=4){
    const int8_t* p0=q.data()+row*cols;
    const int8_t* p1=p0+cols;const int8_t* p2=p1+cols;const int8_t* p3=p2+cols;
    __m256 s0=_mm256_setzero_ps(),s1=s0,s2=s0,s3=s0;
    const __m256 a0=_mm256_set1_ps(scale[row]),a1=_mm256_set1_ps(scale[row+1]);
    const __m256 a2=_mm256_set1_ps(scale[row+2]),a3=_mm256_set1_ps(scale[row+3]);
    size_t j=0;
    for(;j+8<=cols;j+=8){
        const __m256 input=_mm256_loadu_ps(x+j);
        const __m256 w0=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p0+j)))),a0);
        const __m256 w1=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p1+j)))),a1);
        const __m256 w2=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p2+j)))),a2);
        const __m256 w3=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p3+j)))),a3);
        s0=_mm256_add_ps(s0,_mm256_mul_ps(w0,input));
        s1=_mm256_add_ps(s1,_mm256_mul_ps(w1,input));
        s2=_mm256_add_ps(s2,_mm256_mul_ps(w2,input));
        s3=_mm256_add_ps(s3,_mm256_mul_ps(w3,input));
    }
    alignas(32)float lane[4][8];
    _mm256_store_ps(lane[0],s0);_mm256_store_ps(lane[1],s1);
    _mm256_store_ps(lane[2],s2);_mm256_store_ps(lane[3],s3);
    for(size_t r=0;r<4;++r){
        float z=0;for(float v:lane[r])z+=v;
        const int8_t* p=q.data()+(row+r)*cols;
        for(size_t k=j;k<cols;++k)z+=(float(p[k])*scale[row+r])*x[k];
        y[row+r]=z;
    }
}
for(;row<rows;++row)y[row]=dot(row,x);
}
};
