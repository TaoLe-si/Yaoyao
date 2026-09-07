// test_ram.cpp
// Excludes L1/L2/L3 cache effects.
//   - Dataset: 256 MB (>> any L3)
//   - Access pattern: random offsets to defeat hardware prefetcher
//
// Measures:
//   1. Pure RAM bandwidth (sequential read/write/copy, random read)
//   2. Trit vs Float dot product, cache-bypassed
//   3. Real hash-bucket workload, cache-bypassed
//
// Compile: clang++ -O2 -std=c++17 -march=native -o test_ram.exe test_ram.cpp

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

// ============================================================================
// Helpers
// ============================================================================

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
// 1. PURE RAM BANDWIDTH
// ============================================================================

void measure_ram_bandwidth() {
    std::cout << "================================================================\n";
    std::cout << "  1. PURE RAM BANDWIDTH (256 MB buffer, exceeds L3)\n";
    std::cout << "================================================================\n\n";
    
    const size_t size = 256ULL * 1024 * 1024;
    char* src = (char*)aligned_alloc_portable(size, 64);
    char* dst = (char*)aligned_alloc_portable(size, 64);
    
    std::mt19937 rng(42);
    for (size_t i = 0; i < size; i += 8) {
        *(uint64_t*)(src + i) = (uint64_t)rng() | ((uint64_t)rng() << 32);
    }
    
    // Sequential read
    {
        volatile uint64_t sink = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < size; i += 64) {
            sink += *(uint64_t*)(src + i);
        }
        double s = seconds_since(t0);
        std::cout << "  Sequential read (8-byte stride):   "
                  << std::fixed << std::setprecision(2) << (size / s / 1e9) << " GB/s\n";
    }
    
    // Sequential write
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < size; i += 64) {
            *(uint64_t*)(dst + i) = 0;
        }
        double s = seconds_since(t0);
        std::cout << "  Sequential write (8-byte stride):  "
                  << std::fixed << std::setprecision(2) << (size / s / 1e9) << " GB/s\n";
    }
    
    // Sequential copy
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::memcpy(dst, src, size);
        double s = seconds_since(t0);
        std::cout << "  Sequential copy (memcpy):          "
                  << std::fixed << std::setprecision(2) << ((2.0 * size) / s / 1e9) << " GB/s  (R+W)\n";
    }
    
    // Random read (large stride, defeats prefetcher)
    {
        const size_t stride = 4096;
        const size_t n = size / stride;
        std::vector<size_t> idx(n);
        for (size_t i = 0; i < n; ++i) idx[i] = i * stride;
        std::mt19937 rng_r(123);
        std::shuffle(idx.begin(), idx.end(), rng_r);
        
        volatile uint64_t sink = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < n; ++i) {
            sink += *(uint64_t*)(src + idx[i]);
        }
        double s = seconds_since(t0);
        std::cout << "  Random read (4 KB stride):         "
                  << std::fixed << std::setprecision(2) << (size / s / 1e9) << " GB/s  (defeats prefetcher)\n";
    }
    
    aligned_free_portable(src);
    aligned_free_portable(dst);
}

// ============================================================================
// 2. DOT PRODUCT KERNELS (same as before, kept here for self-containment)
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
// 3. DOT PRODUCT, CACHE-BYPASSED
// ============================================================================

void test_cache_bypassed_dot() {
    std::cout << "\n================================================================\n";
    std::cout << "  2. DOT PRODUCT, CACHE-BYPASSED (256 MB dataset, random offsets)\n";
    std::cout << "================================================================\n\n";
    
    const size_t dataset_bytes = 256ULL * 1024 * 1024;
    const int N = 5000;  // fewer iterations since each access goes to RAM
    
    int8_t* a8 = (int8_t*)aligned_alloc_portable(dataset_bytes, 64);
    int8_t* b8 = (int8_t*)aligned_alloc_portable(dataset_bytes, 64);
    float* a32 = (float*)aligned_alloc_portable(dataset_bytes, 64);
    float* b32 = (float*)aligned_alloc_portable(dataset_bytes, 64);
    
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> trit(-1, 1);
    size_t n_int8 = dataset_bytes;
    size_t n_float = dataset_bytes / 4;
    for (size_t i = 0; i < n_int8; ++i) {
        a8[i] = (int8_t)trit(rng);
        b8[i] = (int8_t)trit(rng);
    }
    for (size_t i = 0; i < n_float; ++i) {
        a32[i] = (float)((int8_t)trit(rng));
        b32[i] = (float)((int8_t)trit(rng));
    }
    
    std::cout << "  Dataset: 256 MB | " << n_int8 << " int8 elements, " 
              << n_float << " float elements\n";
    std::cout << "  N = " << N << " dot products per D\n\n";
    
    std::cout << "  D       |  Scalar float  |  Scalar trit  |  AVX2 float  |  AVX2 trit\n";
    std::cout << "  --------+----------------+----------------+--------------+------------\n";
    
    for (int D : {768, 2048, 4096}) {
        size_t max_off_t = n_int8 - D;
        size_t max_off_f = n_float - D;
        
        // Scalar float
        volatile float sf = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < N; ++it) {
            size_t oa = ((size_t)it * 7919) % max_off_f;
            size_t ob = ((size_t)it * 6131) % max_off_f;
            sf = sf + dot_float_scalar(a32 + oa, b32 + ob, D);
        }
        double us_f = seconds_since(t0) * 1e6;
        
        // Scalar trit
        volatile int st = 0;
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < N; ++it) {
            size_t oa = ((size_t)it * 7919) % max_off_t;
            size_t ob = ((size_t)it * 6131) % max_off_t;
            st = st + dot_trit_scalar(a8 + oa, b8 + ob, D);
        }
        double us_t = seconds_since(t1) * 1e6;
        
        // AVX2 float
        volatile float af = 0;
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < N; ++it) {
            size_t oa = ((size_t)it * 7919) % max_off_f;
            size_t ob = ((size_t)it * 6131) % max_off_f;
            af = af + dot_float_avx2(a32 + oa, b32 + oa, D);
        }
        double us_avxf = seconds_since(t2) * 1e6;
        
        // AVX2 trit
        volatile int at = 0;
        auto t3 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < N; ++it) {
            size_t oa = ((size_t)it * 7919) % max_off_t;
            size_t ob = ((size_t)it * 6131) % max_off_t;
            at = at + dot_trit_avx2(a8 + oa, b8 + ob, D);
        }
        double us_avxt = seconds_since(t3) * 1e6;
        
        std::cout << "  " << std::setw(7) << D << " |"
                  << std::setw(14) << std::fixed << std::setprecision(0) << us_f << " us |"
                  << std::setw(14) << us_t << " us |"
                  << std::setw(12) << us_avxf << " us |"
                  << std::setw(10) << us_avxt << " us\n";
    }
    
    // Effective RAM bandwidth analysis for AVX2 versions at D=4096
    std::cout << "\n  Effective RAM bandwidth (AVX2, D=4096):\n";
    long f_bytes_per_dot = 2L * 4096 * 4;  // 2 arrays of 4096 floats
    long t_bytes_per_dot = 2L * 4096;      // 2 arrays of 4096 int8
    // Re-run with timing for bandwidth calc
    int D = 4096;
    size_t max_off_t = n_int8 - D;
    size_t max_off_f = n_float - D;
    
    volatile float af = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < N; ++it) {
        size_t oa = ((size_t)it * 7919) % max_off_f;
        size_t ob = ((size_t)it * 6131) % max_off_f;
        af = af + dot_float_avx2(a32 + oa, b32 + oa, D);
    }
    double s_f = seconds_since(t0);
    double gbs_f = (double)N * f_bytes_per_dot / s_f / 1e9;
    
    volatile int at = 0;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < N; ++it) {
        size_t oa = ((size_t)it * 7919) % max_off_t;
        size_t ob = ((size_t)it * 6131) % max_off_t;
        at = at + dot_trit_avx2(a8 + oa, b8 + ob, D);
    }
    double s_t = seconds_since(t1);
    double gbs_t = (double)N * t_bytes_per_dot / s_t / 1e9;
    
    std::cout << "    AVX2 float: " << std::setprecision(2) << gbs_f << " GB/s effective\n";
    std::cout << "    AVX2 trit:  " << gbs_t << " GB/s effective\n";
    std::cout << "    (these are how hard the workloads hit your RAM bandwidth)\n";
    
    aligned_free_portable(a8);
    aligned_free_portable(b8);
    aligned_free_portable(a32);
    aligned_free_portable(b32);
}

// ============================================================================
// 4. REAL HASH-BUCKET WORKLOAD, CACHE-BYPASSED
// ============================================================================

void test_cache_bypassed_workload() {
    std::cout << "\n================================================================\n";
    std::cout << "  3. REAL HASH-BUCKET WORKLOAD, CACHE-BYPASSED\n";
    std::cout << "     (Pool in 256 MB buffer, random token IDs)\n";
    std::cout << "================================================================\n\n";
    
    const size_t dataset_bytes = 256ULL * 1024 * 1024;
    const int K = 3;
    const int N = 30000;
    
    for (int D : {768, 2048, 4096}) {
        // Choose H so total pool size = 256 MB (>> L3)
        // For float pool: H * D * 4 = 256 MB → H = 256M / (D * 4)
        // For trit pool: H * D * 1 = 256 MB → H = 256M / D
        size_t pool_bytes_f = dataset_bytes;
        size_t pool_bytes_t = dataset_bytes;
        size_t n_float = pool_bytes_f / 4;
        size_t n_int8 = pool_bytes_t;
        size_t H_f = n_float / D;
        size_t H_t = n_int8 / D;
        
        std::cout << "  ----- D = " << D << " -----\n";
        std::cout << "  Float pool: H=" << H_f << " buckets  |  Trit pool: H=" << H_t << " buckets\n";
        std::cout << "  Both pools total 256 MB (>> L3)\n\n";
        
        float* pool_f = (float*)aligned_alloc_portable(pool_bytes_f, 64);
        int8_t* pool_t = (int8_t*)aligned_alloc_portable(pool_bytes_t, 64);
        
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> trit(-1, 1);
        for (size_t i = 0; i < n_float; ++i) pool_f[i] = (float)((int8_t)trit(rng));
        for (size_t i = 0; i < n_int8; ++i) pool_t[i] = (int8_t)trit(rng);
        
        std::vector<float> qf(D);
        std::vector<int8_t> qt(D);
        for (int d = 0; d < D; ++d) {
            int t = trit(rng);
            qt[d] = (int8_t)t;
            qf[d] = (float)t;
        }
        
        // Use H_f for both (since hash mod H should match)
        // For trit, take modulo H_f (smaller) to keep comparison fair
        size_t H = H_f;
        std::vector<int> tokens(N);
        std::uniform_int_distribution<int> idist(0, 49999);
        for (auto& t : tokens) t = idist(rng);
        
        auto h1 = [H](int id) { return (size_t)(id % (int)H); };
        auto h2 = [H](int id) { return (size_t)((id * 3 + 1) % (int)H); };
        auto h3 = [H](int id) { return (size_t)((id * 7 + 5) % (int)H); };
        
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
        
        long f_bytes_per_tok = K * D * 4 + D * 4;
        long t_bytes_per_tok = K * D + D;
        double gbs_f = (double)N * f_bytes_per_tok / (us_f / 1e6) / 1e9;
        double gbs_t = (double)N * t_bytes_per_tok / (us_t / 1e6) / 1e9;
        
        std::cout << "    Float:  " << std::fixed << std::setprecision(0) << us_f << " us  | "
                  << (long long)N * 1e6 / us_f << " tok/s  | "
                  << std::setprecision(2) << gbs_f << " GB/s effective\n";
        std::cout << "    Trit:   " << us_t << " us  | "
                  << (long long)N * 1e6 / us_t << " tok/s  | "
                  << gbs_t << " GB/s effective\n";
        std::cout << "    Speedup: " << std::setprecision(2) << us_f / us_t << "x\n\n";
        
        aligned_free_portable(pool_f);
        aligned_free_portable(pool_t);
    }
}

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Cache-bypassed benchmarks: RAM bandwidth + out-of-cache perf\n";
    std::cout << "================================================================\n";
    
    measure_ram_bandwidth();
    test_cache_bypassed_dot();
    test_cache_bypassed_workload();
    
    std::cout << "\n================================================================\n";
    std::cout << "  All tests complete.\n";
    std::cout << "================================================================\n";
    return 0;
}
