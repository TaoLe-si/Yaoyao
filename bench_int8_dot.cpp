// bench_int8_dot.cpp
// Optimized AVX2 int8 dot product benchmark.
//
// Compare 4 implementations:
//   1. Scalar int8
//   2. AVX2 cvtepi8_epi32 + FMA (my current impl)
//   3. AVX2 cvtepi8_epi16 + mullo + madd (proper int8 chain)
//   4. AVX2 maddubs (uint8 × int8 path, requires signed adjustment)

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <immintrin.h>
#include <cstring>

// ============ 1. Scalar int8 ============
inline int32_t dot_int8_scalar(const int8_t* a, const int8_t* b, int n) {
    int32_t sum = 0;
    for (int i = 0; i < n; ++i) sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
}

// ============ 2. Current: AVX2 cvtepi8_epi32 + FMA ============
inline float dot_int8_cvtfma(const int8_t* a, const int8_t* b, int n) {
    __m256 sumf = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) {
        __m128i va_i = _mm_loadl_epi64((__m128i*)(a + i));
        __m128i vb_i = _mm_loadl_epi64((__m128i*)(b + i));
        __m256 va_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(va_i));
        __m256 vb_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vb_i));
        sumf = _mm256_fmadd_ps(va_f, vb_f, sumf);
    }
    __m128 lo = _mm256_castps256_ps128(sumf);
    __m128 hi = _mm256_extractf128_ps(sumf, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

// ============ 3. Proper int8 chain: cvtepi8_epi16 + mullo + madd ============
inline int32_t dot_int8_proper(const int8_t* a, const int8_t* b, int n) {
    __m256i sum = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    
    int n_aligned = (n / 32) * 32;
    for (int i = 0; i < n_aligned; i += 32) {
        __m256i va = _mm256_loadu_si256((__m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((__m256i*)(b + i));
        
        // Low 128-bit: 16 int8 each -> 16 int16 each
        __m256i va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
        __m256i vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
        __m256i prod_lo = _mm256_mullo_epi16(va_lo, vb_lo);
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(prod_lo, ones));
        
        // High 128-bit
        __m256i va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
        __m256i vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
        __m256i prod_hi = _mm256_mullo_epi16(va_hi, vb_hi);
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(prod_hi, ones));
    }
    
    // Horizontal sum
    __m128i lo = _mm256_castsi256_si128(sum);
    __m128i hi = _mm256_extracti128_si256(sum, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
    s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
    int32_t result = _mm_cvtsi128_si32(s);
    
    // Tail
    for (int i = n_aligned; i < n; ++i) result += (int32_t)a[i] * (int32_t)b[i];
    return result;
}

// ============ 4. Optimized: process 64 int8 per iter (2x unrolled) ============
inline int32_t dot_int8_unrolled(const int8_t* a, const int8_t* b, int n) {
    __m256i sum = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    
    int n_aligned = (n / 64) * 64;
    for (int i = 0; i < n_aligned; i += 64) {
        __m256i va1 = _mm256_loadu_si256((__m256i*)(a + i));
        __m256i vb1 = _mm256_loadu_si256((__m256i*)(b + i));
        __m256i va2 = _mm256_loadu_si256((__m256i*)(a + i + 32));
        __m256i vb2 = _mm256_loadu_si256((__m256i*)(b + i + 32));
        
        // First 32
        __m256i va1_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va1));
        __m256i vb1_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb1));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_mullo_epi16(va1_lo, vb1_lo), ones));
        __m256i va1_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va1, 1));
        __m256i vb1_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb1, 1));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_mullo_epi16(va1_hi, vb1_hi), ones));
        
        // Second 32
        __m256i va2_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va2));
        __m256i vb2_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb2));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_mullo_epi16(va2_lo, vb2_lo), ones));
        __m256i va2_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va2, 1));
        __m256i vb2_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb2, 1));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_mullo_epi16(va2_hi, vb2_hi), ones));
    }
    
    __m128i lo = _mm256_castsi256_si128(sum);
    __m128i hi = _mm256_extracti128_si256(sum, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
    s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
    int32_t result = _mm_cvtsi128_si32(s);
    
    for (int i = n_aligned; i < n; ++i) result += (int32_t)a[i] * (int32_t)b[i];
    return result;
}

int main() {
    const int D = 4096;
    
    std::cout << "================================================================\n";
    std::cout << "  int8 Dot Product Benchmark | D=" << D << "\n";
    std::cout << "================================================================\n\n";
    
    std::vector<int8_t> a(D), b(D);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> ud(-127, 127);
    for (auto& x : a) x = (int8_t)ud(rng);
    for (auto& x : b) x = (int8_t)ud(rng);
    
    // Correctness check
    int32_t s_scalar = dot_int8_scalar(a.data(), b.data(), D);
    int32_t s_proper = dot_int8_proper(a.data(), b.data(), D);
    int32_t s_unrolled = dot_int8_unrolled(a.data(), b.data(), D);
    float s_cvtfma = dot_int8_cvtfma(a.data(), b.data(), D);
    
    std::cout << "  Correctness:\n";
    std::cout << "    scalar:    " << s_scalar << "\n";
    std::cout << "    cvt+fma:   " << (int)s_cvtfma << " (diff=" << std::abs(s_scalar - (int)s_cvtfma) << ")\n";
    std::cout << "    proper:    " << s_proper << " (diff=" << std::abs(s_scalar - s_proper) << ")\n";
    std::cout << "    unrolled:  " << s_unrolled << " (diff=" << std::abs(s_scalar - s_unrolled) << ")\n\n";
    
    // Warmup
    volatile int32_t sink = 0;
    for (int i = 0; i < 1000; ++i) {
        sink += dot_int8_scalar(a.data(), b.data(), D);
        sink += dot_int8_cvtfma(a.data(), b.data(), D);
        sink += dot_int8_proper(a.data(), b.data(), D);
        sink += dot_int8_unrolled(a.data(), b.data(), D);
    }
    if (sink == 0) std::cout << "";
    
    int trials = 100000;
    std::cout << "  Benchmark (" << trials << " iterations):\n\n";
    
    auto bench = [&](auto fn, const char* name) {
        auto t0 = std::chrono::high_resolution_clock::now();
        volatile int32_t acc = 0;
        for (int i = 0; i < trials; ++i) {
            acc += fn(a.data(), b.data(), D);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / (double)trials;
        if (acc == 0) std::cout << "";
        return ns;
    };
    
    double ns_scalar = bench(dot_int8_scalar, "scalar");
    double ns_cvtfma = bench(dot_int8_cvtfma, "cvtfma");
    double ns_proper = bench(dot_int8_proper, "proper");
    double ns_unrolled = bench(dot_int8_unrolled, "unrolled");
    
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "    Scalar:           " << std::setw(8) << ns_scalar << " ns  (1.0x)\n";
    std::cout << "    AVX2 cvt+fma:     " << std::setw(8) << ns_cvtfma << " ns  (" << ns_scalar / ns_cvtfma << "x)\n";
    std::cout << "    AVX2 int8 proper: " << std::setw(8) << ns_proper << " ns  (" << ns_scalar / ns_proper << "x)\n";
    std::cout << "    AVX2 unrolled:    " << std::setw(8) << ns_unrolled << " ns  (" << ns_scalar / ns_unrolled << "x)\n\n";
    
    // Compute throughput
    std::cout << "  Throughput (D/ns):\n";
    std::cout << "    Scalar:           " << std::setprecision(1) << D / ns_scalar << " elem/ns\n";
    std::cout << "    AVX2 cvt+fma:     " << D / ns_cvtfma << " elem/ns\n";
    std::cout << "    AVX2 int8 proper: " << D / ns_proper << " elem/ns\n";
    std::cout << "    AVX2 unrolled:    " << D / ns_unrolled << " elem/ns\n\n";
    
    // Hash Bucket Q4 projection (1024 buckets)
    std::cout << "  Hash Bucket Q4 (1024 buckets * D=4096):\n";
    std::cout << "    Scalar:           " << std::setprecision(2) << (1024 * ns_scalar / 1000) << " µs\n";
    std::cout << "    AVX2 int8 proper: " << (1024 * ns_proper / 1000) << " µs  (vs old 553 µs)\n";
    std::cout << "    AVX2 unrolled:    " << (1024 * ns_unrolled / 1000) << " µs\n\n";
    
    // Full pipeline estimate
    std::cout << "  Full Pipeline (32 layers + Q4 unrolled):\n";
    double layer_us = 4.25 * 32;
    double q4_us = 1024 * ns_unrolled / 1000;
    std::cout << "    32 layers: " << layer_us << " µs\n";
    std::cout << "    Q4 unrolled: " << q4_us << " µs\n";
    std::cout << "    Total: " << (layer_us + q4_us) << " µs/token = " 
              << std::setprecision(0) << 1e6 / (layer_us + q4_us) << " tokens/s\n";
    std::cout << "    vs user 20 t/s: " << std::setprecision(1) << (1e6 / (layer_us + q4_us)) / 20 << "x\n";

    return 0;
}
