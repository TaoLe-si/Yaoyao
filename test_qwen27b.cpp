// test_qwen27b.cpp
// Test hash bucket pool sized for Qwen-27B-class model.
//
// Pool parameters:
//   H = 16384 buckets (large hash table for ~150k vocab with K=3 hashes)
//   D = 4096 (representative hidden size, near LLaMA 8B/13B)
//
// Float pool:  16384 x 4096 x 4 = 256 MB  (fills test buffer)
// Trit packed: 16384 x 4096 / 4 =  16 MB  (fits in L2)
// Trit unpack: 16384 x 4096    =  64 MB  (fits in L2)
//
// Compile: clang++ -O2 -std=c++17 -march=native -o test_qwen27b.exe test_qwen27b.cpp

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <cstdlib>

#ifdef _MSC_VER
#include <intrin.h>
#include <malloc.h>
#else
#include <immintrin.h>
#include <x86intrin.h>
#endif

void* aligned_alloc_portable(size_t size, size_t align) {
#ifdef _MSC_VER
    return _aligned_malloc(size, align);
#else
    return std::aligned_alloc(align, size);
#endif
}

void aligned_free_portable(void* p) {
#ifdef _MSC_VER
    _aligned_free(p);
#else
    std::free(p);
#endif
}

double seconds_since(std::chrono::high_resolution_clock::time_point t0) {
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e9;
}

// ============================================================================
// KERNELS (same as before)
// ============================================================================

float dot_float_scalar(const float* a, const float* b, int D) {
    float s = 0;
    for (int i = 0; i < D; ++i) s += a[i] * b[i];
    return s;
}

int dot_trit_scalar(const int8_t* a, const int8_t* b, int D) {
    int s = 0;
    for (int i = 0; i < D; ++i) {
        int ai = a[i], bi = b[i];
        if (ai == 0 || bi == 0) continue;
        if (ai == bi) s += 1;
        else s -= 1;
    }
    return s;
}

float dot_float_avx2(const float* a, const float* b, int D) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= D; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 s128 = _mm_add_ps(hi, lo);
    __m128 shuf = _mm_movehdup_ps(s128);
    __m128 sums = _mm_add_ps(s128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float r = _mm_cvtss_f32(sums);
    for (; i < D; ++i) r += a[i] * b[i];
    return r;
}

int dot_trit_avx2(const int8_t* a, const int8_t* b, int D) {
    int s = 0;
    int i = 0;
    const __m256i zero = _mm256_setzero_si256();
    const __m256i all_ones = _mm256_set1_epi8(-1);
    for (; i + 32 <= D; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i nz_a = _mm256_andnot_si256(_mm256_cmpeq_epi8(va, zero), all_ones);
        __m256i nz_b = _mm256_andnot_si256(_mm256_cmpeq_epi8(vb, zero), all_ones);
        __m256i both_nz = _mm256_and_si256(nz_a, nz_b);
        __m256i same = _mm256_cmpeq_epi8(va, vb);
        __m256i same_nz = _mm256_and_si256(same, both_nz);
        int both_count = _mm_popcnt_u32(_mm256_movemask_epi8(both_nz));
        int same_count = _mm_popcnt_u32(_mm256_movemask_epi8(same_nz));
        s += 2 * same_count - both_count;
    }
    for (; i < D; ++i) {
        int ai = a[i], bi = b[i];
        if (ai == 0 || bi == 0) continue;
        if (ai == bi) s += 1;
        else s -= 1;
    }
    return s;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Qwen-27B-class hash bucket pool test\n";
    std::cout << "================================================================\n\n";
    
    // Pool sizing
    const size_t H = 16384;
    const int D = 4096;
    const int K = 3;
    
    std::cout << "  Pool parameters:\n";
    std::cout << "    H (buckets)  = " << H << "\n";
    std::cout << "    D (hidden)   = " << D << "\n";
    std::cout << "    K (hashes)   = " << K << "\n\n";
    
    long float_pool_bytes = (long)H * D * 4;
    long trit_packed_bytes = (long)H * ((D + 3) / 4);
    long trit_unpacked_bytes = (long)H * D;
    
    std::cout << "  Pool sizes (single pool):\n";
    std::cout << "    Float:        " << std::setw(10) << float_pool_bytes << " bytes = " 
              << float_pool_bytes / 1048576 << " MB\n";
    std::cout << "    Trit packed:  " << std::setw(10) << trit_packed_bytes << " bytes = " 
              << trit_packed_bytes / 1024 << " KB\n";
    std::cout << "    Trit unpack:  " << std::setw(10) << trit_unpacked_bytes << " bytes = " 
              << trit_unpacked_bytes / 1048576 << " MB\n";
    std::cout << "    Compression:  " << std::fixed << std::setprecision(1) 
              << (float_pool_bytes / (double)trit_packed_bytes) << "x (packed)\n\n";
    
    std::cout << "  Qwen-27B context (for comparison):\n";
    std::cout << "    Qwen2.5-32B actual: hidden=5120, vocab=152064, layers=64\n";
    std::cout << "    Standard embedding table: 152064 x 5120 = 778M params = " 
              << (778L * 1024 * 1024 * 4) / 1048576 / 1024 << " GB float\n";
    std::cout << "    Our hash pool replaces this with H x D = " << H * D 
              << " buckets = " << (H * D * 4) / 1048576 << " MB float\n\n";
    
    // Allocate pools (use full 256 MB for float, 64 MB for trit unpacked)
    const size_t dataset_bytes = 256ULL * 1024 * 1024;
    
    float* pool_f = (float*)aligned_alloc_portable(dataset_bytes, 64);
    int8_t* pool_t = (int8_t*)aligned_alloc_portable(dataset_bytes / 4, 64);  // 64 MB trit
    
    // Fill with random data
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> trit(-1, 1);
    size_t n_float = dataset_bytes / 4;
    size_t n_int8 = dataset_bytes / 4;
    for (size_t i = 0; i < n_float; ++i) pool_f[i] = (float)((int8_t)trit(rng));
    for (size_t i = 0; i < n_int8; ++i) pool_t[i] = (int8_t)trit(rng);
    
    // Query vectors
    std::vector<float> qf(D);
    std::vector<int8_t> qt(D);
    for (int d = 0; d < D; ++d) {
        int t = trit(rng);
        qt[d] = (int8_t)t;
        qf[d] = (float)t;
    }
    
    // Workload setup
    const int N = 30000;
    std::vector<int> tokens(N);
    std::uniform_int_distribution<int> idist(0, 49999);
    for (auto& t : tokens) t = idist(rng);
    
    // Hash range = H for both (since H < total pool capacity, mod H)
    auto h1 = [](int id) { return (size_t)(id % 16384); };
    auto h2 = [](int id) { return (size_t)((id * 3 + 1) % 16384); };
    auto h3 = [](int id) { return (size_t)((id * 7 + 5) % 16384); };
    
    std::cout << "  Workload parameters:\n";
    std::cout << "    N (tokens)   = " << N << "\n";
    std::cout << "    Random token IDs\n\n";
    
    // Float version
    volatile float sf = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        int id = tokens[i];
        const float* b1 = pool_f + h1(id) * D;
        const float* b2 = pool_f + h2(id) * D;
        const float* b3 = pool_f + h3(id) * D;
        float s = 0;
        for (int d = 0; d < D; ++d) {
            float emb = b1[d] + b2[d] + b3[d];
            s += emb * qf[d];
        }
        sf = sf + s;
    }
    double us_f = seconds_since(t0) * 1e6;
    
    // Trit version
    volatile int st = 0;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        int id = tokens[i];
        const int8_t* b1 = pool_t + h1(id) * D;
        const int8_t* b2 = pool_t + h2(id) * D;
        const int8_t* b3 = pool_t + h3(id) * D;
        int s = 0;
        for (int d = 0; d < D; ++d) {
            int emb = (int)b1[d] + (int)b2[d] + (int)b3[d];
            int q = qt[d];
            if (q == 0) continue;
            int es = (emb > 0) ? 1 : -1;
            if (es == q) s += 1;
            else s -= 1;
        }
        st = st + s;
    }
    double us_t = seconds_since(t1) * 1e6;
    
    // Bytes touched per token
    long f_bytes_per_tok = K * D * 4 + D * 4;   // 3 buckets + query (float)
    long t_bytes_per_tok = K * D + D;            // 3 buckets + query (int8)
    long f_total = (long)N * f_bytes_per_tok;
    long t_total = (long)N * t_bytes_per_tok;
    double gbs_f = (double)f_total / (us_f / 1e6) / 1e9;
    double gbs_t = (double)t_total / (us_t / 1e6) / 1e9;
    
    std::cout << "================================================================\n";
    std::cout << "  RESULTS\n";
    std::cout << "================================================================\n\n";
    
    std::cout << "  Float version:\n";
    std::cout << "    Total time:    " << std::fixed << std::setprecision(0) << us_f << " us = " 
              << us_f / 1000.0 << " ms\n";
    std::cout << "    Throughput:    " << (long long)N * 1e6 / us_f << " tok/s\n";
    std::cout << "    Per token:     " << std::setprecision(1) << us_f * 1000 / N << " ns\n";
    std::cout << "    Bytes/tok:     " << f_bytes_per_tok << " (3 buckets + query, all float)\n";
    std::cout << "    Eff bandwidth: " << std::setprecision(2) << gbs_f << " GB/s\n\n";
    
    std::cout << "  Trit version:\n";
    std::cout << "    Total time:    " << us_t << " us = " << us_t / 1000.0 << " ms\n";
    std::cout << "    Throughput:    " << (long long)N * 1e6 / us_t << " tok/s\n";
    std::cout << "    Per token:     " << std::setprecision(1) << us_t * 1000 / N << " ns\n";
    std::cout << "    Bytes/tok:     " << t_bytes_per_tok << " (3 buckets + query, all int8)\n";
    std::cout << "    Eff bandwidth: " << std::setprecision(2) << gbs_t << " GB/s\n\n";
    
    std::cout << "  COMPARISON:\n";
    std::cout << "    Speedup:       " << std::setprecision(2) << us_f / us_t << "x\n";
    std::cout << "    Bytes ratio:   " << std::setprecision(2) 
              << (double)f_bytes_per_tok / t_bytes_per_tok << "x (float/trit per token)\n";
    std::cout << "    Pool ratio:    " << std::setprecision(2) 
              << (float_pool_bytes / (double)trit_unpacked_bytes) << "x (float/unpacked trit)\n\n";
    
    // Theoretical limit check (based on 18 GB/s RAM)
    double gbs_peak = 18.0;
    double theo_f = gbs_peak * 1e9 / f_bytes_per_tok;  // tok/s
    double theo_t = gbs_peak * 1e9 / t_bytes_per_tok;
    std::cout << "  Theoretical limits (at 18 GB/s RAM peak):\n";
    std::cout << "    Float:         " << std::setprecision(0) << theo_f << " tok/s\n";
    std::cout << "    Trit:          " << theo_t << " tok/s\n";
    std::cout << "    Float utilization: " << std::setprecision(1) 
              << (long long)N * 1e6 / us_f / theo_f * 100 << "%\n";
    std::cout << "    Trit utilization:  " 
              << (long long)N * 1e6 / us_t / theo_t * 100 << "%\n\n";
    
    aligned_free_portable(pool_f);
    aligned_free_portable(pool_t);
    
    std::cout << "================================================================\n";
    std::cout << "  Test complete.\n";
    std::cout << "================================================================\n";
    return 0;
}
