#pragma once
#include "cpu_ternary_2bit_avx2.hpp"
#include <array>

// Single bounded alternative: 4 KiB byte -> four FP32 symbols LUT.
// Shared by all matrices. Packed layout/validation remain unchanged.
// Strict FP required, exactly as for the variable-shift candidate.
struct CpuTernary2BitLutRows : CpuTernary2BitRows {
    using CpuTernary2BitRows::CpuTernary2BitRows;
    struct alignas(16) Four { float v[4]; };
    static constexpr std::array<Four,256> make_table() {
        std::array<Four,256> t{};
        for (unsigned b=0;b<256;++b)
            for (unsigned j=0;j<4;++j)
                t[b].v[j]=float(int((b>>(2*j))&3u)-1);
        return t;
    }
    static const std::array<Four,256> table;
    float dot(size_t row,const float* x) const {
        const uint8_t* p=codes.data()+row*stride;
        const __m256 alpha=_mm256_set1_ps(scale[row]);
        __m256 sum=_mm256_setzero_ps();
        size_t j=0;
        for(;j+8<=cols;j+=8) {
            const __m128 lo=_mm_load_ps(table[p[j/4]].v);
            const __m128 hi=_mm_load_ps(table[p[j/4+1]].v);
            const __m256 symbols=_mm256_insertf128_ps(_mm256_castps128_ps256(lo),hi,1);
            const __m256 weights=_mm256_mul_ps(symbols,alpha);
            sum=_mm256_add_ps(sum,_mm256_mul_ps(weights,_mm256_loadu_ps(x+j)));
        }
        alignas(32) float lane[8];
        _mm256_store_ps(lane,sum);
        float z=0;
        for(float v:lane) z+=v;
        for(;j<cols;++j) z+=(float(symbol(row,j))*scale[row])*x[j];
        return z;
    }
};
inline const std::array<CpuTernary2BitLutRows::Four,256>
    CpuTernary2BitLutRows::table=CpuTernary2BitLutRows::make_table();
