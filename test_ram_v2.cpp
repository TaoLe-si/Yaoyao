// test_ram_v2.cpp
// Cache-bypassed benchmarks with corrected methodology.
//
// Bug fix from v1: random-read used 4 KB stride, which only touched 4 MB of
// cache lines (fits in L3). Now uses 64-byte stride so every cache line is
// touched exactly once.

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
// 1. TRUE STREAMING RAM BANDWIDTH
// ============================================================================

void measure_streaming_bandwidth() {
    std::cout << "================================================================\n";
    std::cout << "  1. STREAMING RAM BANDWIDTH (256 MB, every cache line touched once)\n";
    std::cout << "================================================================\n\n";
    
    const size_t size = 256ULL * 1024 * 1024;
    char* src = (char*)aligned_alloc_portable(size, 64);
    char* dst = (char*)aligned_alloc_portable(size, 64);
    
    std::mt19937 rng(42);
    for (size_t i = 0; i < size; i += 8) {
        *(uint64_t*)(src + i) = (uint64_t)rng() | ((uint64_t)rng() << 32);
    }
    
    // (a) Sequential read, 64-byte stride (every cache line, once)
    {
        volatile uint64_t sink = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        const char* p = src;
        for (size_t i = 0; i < size; i += 64) {
            sink += *(uint64_t*)(p + i);
        }
        double s = seconds_since(t0);
        std::cout << "  (a) Streaming read (64-byte stride):  "
                  << std::fixed << std::setprecision(2) << (size / s / 1e9) << " GB/s\n";
    }
    
    // (b) Sequential write, 64-byte stride
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < size; i += 64) {
            *(uint64_t*)(dst + i) = i;
        }
        double s = seconds_since(t0);
        std::cout << "  (b) Streaming write (64-byte stride): "
                  << std::fixed << std::setprecision(2) << (size / s / 1e9) << " GB/s\n";
    }
    
    // (c) Streaming copy (memcpy)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::memcpy(dst, src, size);
        double s = seconds_since(t0);
        std::cout << "  (c) Streaming copy (memcpy):         "
                  << std::fixed << std::setprecision(2) << ((2.0 * size) / s / 1e9) << " GB/s  (R+W)\n";
    }
    
    // (d) Random read, 64-byte stride (every cache line, random order)
    {
        const size_t stride = 64;
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
        std::cout << "  (d) Random read (64-byte stride):    "
                  << std::fixed << std::setprecision(2) << (size / s / 1e9) << " GB/s  (every cache line, random)\n";
    }
    
    // (e) Streaming sum of 256 MB int8 (like our trit data)
    {
        std::vector<int8_t> buf_int8(size);
        for (size_t i = 0; i < size; ++i) buf_int8[i] = (int8_t)(i % 3 - 1);
        
        volatile long long sink = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < size; ++i) sink += buf_int8[i];
        double s = seconds_since(t0);
        std::cout << "  (e) Streaming sum of 256 MB int8:    "
                  << std::fixed << std::setprecision(2) << (size / s / 1e9) << " GB/s\n";
    }
    
    aligned_free_portable(src);
    aligned_free_portable(dst);
}

// ============================================================================
// 2. STREAMING DOT PRODUCT (sequential offsets, cache line touched once)
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

void test_streaming_dot() {
    std::cout << "\n================================================================\n";
    std::cout << "  2. STREAMING DOT PRODUCT\n";
    std::cout << "     Sequential offsets across 256 MB dataset.\n";
    std::cout << "     Each cache line is touched exactly once (true streaming).\n";
    std::cout << "================================================================\n\n";
    
    const size_t dataset_bytes = 256ULL * 1024 * 1024;
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
    
    std::cout << "  D       |  Scalar float  |  Scalar trit  |  AVX2 float  |  AVX2 trit\n";
    std::cout << "  --------+----------------+----------------+--------------+------------\n";
    
    for (int D : {768, 2048, 4096}) {
        int N = (int)(n_int8 / D) - 1;  // number of dot products that fit
        if (N > 5000) N = 5000;
        
        // Scalar float (sequential)
        volatile float sf = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < N; ++it) {
            size_t oa = (size_t)it * D;
            size_t ob = (size_t)it * D;
            sf = sf + dot_float_scalar(a32 + oa, b32 + ob, D);
        }
        double us_f = seconds_since(t0) * 1e6;
        
        // Scalar trit (sequential)
        volatile int st = 0;
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < N; ++it) {
            size_t oa = (size_t)it * D;
            size_t ob = (size_t)it * D;
            st = st + dot_trit_scalar(a8 + oa, b8 + ob, D);
        }
        double us_t = seconds_since(t1) * 1e6;
        
        // AVX2 float
        volatile float af = 0;
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < N; ++it) {
            size_t oa = (size_t)it * D;
            size_t ob = (size_t)it * D;
            af = af + dot_float_avx2(a32 + oa, b32 + ob, D);
        }
        double us_avxf = seconds_since(t2) * 1e6;
        
        // AVX2 trit
        volatile int at = 0;
        auto t3 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < N; ++it) {
            size_t oa = (size_t)it * D;
            size_t ob = (size_t)it * D;
            at = at + dot_trit_avx2(a8 + oa, b8 + ob, D);
        }
        double us_avxt = seconds_since(t3) * 1e6;
        
        std::cout << "  " << std::setw(7) << D << " |"
                  << std::setw(14) << std::fixed << std::setprecision(0) << us_f << " us |"
                  << std::setw(14) << us_t << " us |"
                  << std::setw(12) << us_avxf << " us |"
                  << std::setw(10) << us_avxt << " us\n";
    }
    
    // Effective bandwidth at D=4096, AVX2
    int D = 4096;
    int N = 5000;
    std::cout << "\n  Effective RAM bandwidth (AVX2 streaming, D=" << D << ", N=" << N << "):\n";
    
    long f_bytes = (long)N * D * 4 * 2;  // N dots × D × 4 bytes × 2 arrays
    long t_bytes = (long)N * D * 2;       // N dots × D × 1 byte × 2 arrays
    
    volatile float af = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < N; ++it) {
        size_t oa = (size_t)it * D;
        af = af + dot_float_avx2(a32 + oa, b32 + oa, D);
    }
    double s_f = seconds_since(t0);
    
    volatile int at = 0;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < N; ++it) {
        size_t oa = (size_t)it * D;
        at = at + dot_trit_avx2(a8 + oa, b8 + oa, D);
    }
    double s_t = seconds_since(t1);
    
    std::cout << "    AVX2 float: " << (f_bytes / s_f / 1e9) << " GB/s  (" << f_bytes / 1048576 << " MB read)\n";
    std::cout << "    AVX2 trit:  " << (t_bytes / s_t / 1e9) << " GB/s  (" << t_bytes / 1048576 << " MB read)\n";
    
    aligned_free_portable(a8);
    aligned_free_portable(b8);
    aligned_free_portable(a32);
    aligned_free_portable(b32);
}

// ============================================================================
// 3. REAL HASH-BUCKET WORKLOAD (256 MB pool, random token IDs)
// ============================================================================

void test_workload() {
    std::cout << "\n================================================================\n";
    std::cout << "  3. REAL HASH-BUCKET WORKLOAD (256 MB pool, random token IDs)\n";
    std::cout << "================================================================\n\n";
    
    const size_t dataset_bytes = 256ULL * 1024 * 1024;
    const int K = 3;
    
    std::cout << "  Caveat: per-token working set = 4 × D × sizeof(type). At D=4096:\n";
    std::cout << "    float: 64 KB per token (fits in L1 64 KB)\n";
    std::cout << "    trit:  16 KB per token (comfortably fits L1)\n";
    std::cout << "  So cache helps WITHIN each token. True streaming shown above.\n\n";
    
    for (int D : {768, 2048, 4096}) {
        size_t n_float = dataset_bytes / 4;
        size_t n_int8 = dataset_bytes;
        size_t H = n_float / D;  // same hash range for both
        
        float* pool_f = (float*)aligned_alloc_portable(dataset_bytes, 64);
        int8_t* pool_t = (int8_t*)aligned_alloc_portable(dataset_bytes, 64);
        
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
        
        const int N = 30000;
        std::vector<int> tokens(N);
        std::uniform_int_distribution<int> idist(0, 49999);
        for (auto& t : tokens) t = idist(rng);
        
        auto h1 = [H](int id) { return (size_t)(id % (int)H); };
        auto h2 = [H](int id) { return (size_t)((id * 3 + 1) % (int)H); };
        auto h3 = [H](int id) { return (size_t)((id * 7 + 5) % (int)H); };
        
        // Float
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
        
        // Trit
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
        
        long f_bytes = (long)N * K * D * 4 + (long)N * D * 4;
        long t_bytes = (long)N * K * D + (long)N * D;
        double gbs_f = (double)f_bytes / (us_f / 1e6) / 1e9;
        double gbs_t = (double)t_bytes / (us_t / 1e6) / 1e9;
        
        std::cout << "  ----- D = " << D << " -----\n";
        std::cout << "    Float: " << std::fixed << std::setprecision(0) << us_f << " us  | "
                  << (long long)N * 1e6 / us_f << " tok/s  | "
                  << std::setprecision(2) << gbs_f << " GB/s effective\n";
        std::cout << "    Trit:  " << us_t << " us  | "
                  << (long long)N * 1e6 / us_t << " tok/s  | "
                  << gbs_t << " GB/s effective\n";
        std::cout << "    Speedup: " << std::setprecision(2) << us_f / us_t << "x\n\n";
        
        aligned_free_portable(pool_f);
        aligned_free_portable(pool_t);
    }
}

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Cache-bypassed benchmarks (v2: corrected random stride)\n";
    std::cout << "================================================================\n";
    
    measure_streaming_bandwidth();
    test_streaming_dot();
    test_workload();
    
    std::cout << "\n================================================================\n";
    std::cout << "  All tests complete.\n";
    std::cout << "================================================================\n";
    return 0;
}
