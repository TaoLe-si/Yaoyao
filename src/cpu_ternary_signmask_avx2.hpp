#pragma once
#include "cpu_ternary_avx2.hpp"
#include <limits>

// Byte storage, exact extraction/finite rejection inherited unchanged.
// Strict FP required; original eight lanes/reduction/tails, no FMA.
struct CpuTernarySignMaskRows : CpuTernaryRows {
    using CpuTernaryRows::CpuTernaryRows;
    float dot(size_t row,const float* x) const {
        // Subnormal scales must retain baseline DAZ/FTZ behavior of q*scale.
        // Rare fallback also preserves baseline FP exception behavior there.
        if(scale[row]<std::numeric_limits<float>::min())
            return CpuTernaryRows::dot(row,x);
        const __m256i alpha=_mm256_castps_si256(_mm256_set1_ps(scale[row]));
        const __m256i sign=_mm256_set1_epi32(std::numeric_limits<int32_t>::min());
        const __m256i zero=_mm256_setzero_si256();
        const int8_t* p=q.data()+row*cols;
        __m256 sum=_mm256_setzero_ps();size_t j=0;
        for(;j+8<=cols;j+=8) {
            const __m256i symbols=_mm256_cvtepi8_epi32(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p+j)));
            // q==-1 supplies sign bit; q==0 clears every bit -> positive zero.
            const __m256i signed_scale=_mm256_or_si256(alpha,_mm256_and_si256(symbols,sign));
            const __m256 weights=_mm256_castsi256_ps(_mm256_andnot_si256(
                _mm256_cmpeq_epi32(symbols,zero),signed_scale));
            sum=_mm256_add_ps(sum,_mm256_mul_ps(weights,_mm256_loadu_ps(x+j)));
        }
        alignas(32) float lane[8];_mm256_store_ps(lane,sum);
        float z=0;for(float v:lane)z+=v;
        for(;j<cols;++j)z+=(float(p[j])*scale[row])*x[j];
        return z;
    }
};
