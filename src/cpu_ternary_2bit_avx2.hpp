#pragma once
#include "cpu_ternary_avx2.hpp"
#include <limits>

// Experimental row-major 2-bit encoding: 00=-1, 01=0, 10=+1.
// Each row starts on a byte boundary; low bits hold the earliest column.
// Compile AVX2 with strict FP: /fp:strict or -fno-fast-math -ffp-contract=off.
// No FMA, reassociation, or scale-after-dot: matches CpuTernaryRows::dot.
struct CpuTernary2BitRows {
    size_t rows, cols, stride;
    std::vector<uint8_t> codes;
    std::vector<float> scale;

    CpuTernary2BitRows(const std::vector<float>& w, size_t r, size_t c)
        : rows(r), cols(c), stride(c / 4 + (c % 4 != 0)) {
        if (!r || !c || r > std::numeric_limits<size_t>::max() / c || w.size() != r*c)
            throw std::runtime_error("2bit ternary shape");
        // Reuse exact validation; temporary byte storage is released after packing.
        CpuTernaryRows source(w, r, c);
        scale = source.scale;
        codes.assign(r * stride, uint8_t(0x55)); // zero, including padding
        for (size_t i = 0; i < r; ++i)
            for (size_t j = 0; j < c; ++j) {
                const unsigned shift = unsigned(j % 4) * 2;
                auto& b = codes[i*stride+j/4];
                b = uint8_t((b & ~(3u << shift)) |
                            (unsigned(int(source.q[i*c+j])+1) << shift));
            }
    }

    int symbol(size_t row, size_t col) const {
        return int((codes[row*stride+col/4] >> (2*(col%4))) & 3u)-1;
    }
    size_t weight_bytes() const { return codes.size() + scale.size()*sizeof(float); }

    float dot(size_t row, const float* x) const {
        const uint8_t* p = codes.data() + row*stride;
        const __m256 alpha = _mm256_set1_ps(scale[row]);
        const __m256i shifts = _mm256_setr_epi32(0,2,4,6,8,10,12,14);
        const __m256i mask = _mm256_set1_epi32(3), one = _mm256_set1_epi32(1);
        __m256 sum = _mm256_setzero_ps();
        size_t j = 0;
        for (; j+8 <= cols; j += 8) {
            // Exactly two byte loads: no unaligned or tail over-read.
            const unsigned bits = unsigned(p[j/4]) | (unsigned(p[j/4+1]) << 8);
            const __m256i q = _mm256_sub_epi32(_mm256_and_si256(
                _mm256_srlv_epi32(_mm256_set1_epi32(int(bits)), shifts), mask), one);
            const __m256 weights = _mm256_mul_ps(_mm256_cvtepi32_ps(q), alpha);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(weights, _mm256_loadu_ps(x+j)));
        }
        alignas(32) float lane[8];
        _mm256_store_ps(lane, sum);
        float z = 0;
        for (float v : lane) z += v;
        for (; j < cols; ++j) z += (float(symbol(row,j))*scale[row])*x[j];
        return z;
    }
};
