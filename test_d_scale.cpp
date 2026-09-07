// test_d_scale.cpp
// Scaling test: trit design performance at mainstream LLM dimensions.
// Tests pure dot product + real hash-bucket workload for D = 768, 1024, 2048, 4096.

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <array>
#include <string>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <immintrin.h>
#include <x86intrin.h>
#endif

// ============================================================================
// DOT PRODUCT KERNELS
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

// AVX2 trit dot using popcount (pure addition, no multiplication)
int dot_trit_avx2(const int8_t* a, const int8_t* b, int D) {
    int s = 0;
    int i = 0;
    __m256i zero = _mm256_setzero_si256();
    for (; i + 32 <= D; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        
        __m256i nz_a = _mm256_sub_epi8(zero, _mm256_cmpeq_epi8(va, zero));
        __m256i nz_b = _mm256_sub_epi8(zero, _mm256_cmpeq_epi8(vb, zero));
        __m256i both = _mm256_and_si256(nz_a, nz_b);
        __m256i same = _mm256_sub_epi8(zero, _mm256_cmpeq_epi8(va, vb));
        __m256i same_nz = _mm256_and_si256(same, both);
        
        int both_count = _mm_popcnt_u32(_mm256_movemask_epi8(both));
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
// POOL SIZE TABLE
// ============================================================================

void print_pool_sizes() {
    std::cout << "================================================================\n";
    std::cout << "  BUCKET POOL SIZE TABLE (H = 4096 buckets)\n";
    std::cout << "================================================================\n";
    std::cout << "  Model D    Float pool      Trit (packed)   Compression\n";
    std::cout << "  --------  -------------  -------------   -----------\n";
    const int dims[] = {768, 1024, 1536, 2048, 4096, 8192};
    for (int D : dims) {
        long fb = 4096L * D * 4;
        long tb = 4096L * ((D + 3) / 4);
        std::cout << "  " << std::setw(6) << D << "    "
                  << std::setw(8) << fb << " B  "
                  << std::setw(8) << tb << " B       "
                  << std::fixed << std::setprecision(1) << std::setw(5)
                  << (fb / (double)tb) << "x\n";
    }
    std::cout << "  \n";
    std::cout << "  Note: 32 KB = typical L1, 256 KB-1 MB = typical L2.\n";
    std::cout << "  Float pools >= 1 MB spill out of L1 entirely.\n";
    std::cout << "  Trit pools always fit in L1 (max 64 KB at D=8192).\n";
}

// ============================================================================
// PURE DOT PRODUCT AT EACH D
// ============================================================================

void test_dot_at_D(int D, const std::vector<int>& dims_list) {
    std::cout << "\n  ----- D = " << D << " -----\n";
    
    const int N = 20000;  // iterations per D
    
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> trit(-1, 1);
    
    // Generate test data
    std::vector<std::vector<float>> af(N, std::vector<float>(D));
    std::vector<std::vector<float>> bf(N, std::vector<float>(D));
    std::vector<std::vector<int8_t>> at(N, std::vector<int8_t>(D));
    std::vector<std::vector<int8_t>> bt(N, std::vector<int8_t>(D));
    
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < D; ++d) {
            int ta = trit(rng);
            int tb = trit(rng);
            at[i][d] = (int8_t)ta;
            bt[i][d] = (int8_t)tb;
            af[i][d] = (float)ta;
            bf[i][d] = (float)tb;
        }
    }
    
    // Scalar float
    volatile float sf = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) sf += dot_float_scalar(af[i].data(), bf[i].data(), D);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto us_sf = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    // Scalar trit
    volatile int st = 0;
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) st += dot_trit_scalar(at[i].data(), bt[i].data(), D);
    t1 = std::chrono::high_resolution_clock::now();
    auto us_st = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    // AVX2 float
    volatile float af2 = 0;
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) af2 += dot_float_avx2(af[i].data(), bf[i].data(), D);
    t1 = std::chrono::high_resolution_clock::now();
    auto us_avxf = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    // AVX2 trit
    volatile int at2 = 0;
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) at2 += dot_trit_avx2(at[i].data(), bt[i].data(), D);
    t1 = std::chrono::high_resolution_clock::now();
    auto us_avxt = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    // Print
    std::cout << "  Scalar float:   " << std::setw(6) << us_sf << " us  | "
              << std::setw(8) << (long long)N * 1000000 / us_sf << " dots/s\n";
    std::cout << "  Scalar trit:    " << std::setw(6) << us_st << " us  | "
              << std::setw(8) << (long long)N * 1000000 / us_st << " dots/s  ("
              << std::fixed << std::setprecision(2) << (double)us_sf / us_st << "x vs scalar float)\n";
    std::cout << "  AVX2 float:     " << std::setw(6) << us_avxf << " us  | "
              << std::setw(8) << (long long)N * 1000000 / us_avxf << " dots/s\n";
    std::cout << "  AVX2 trit:      " << std::setw(6) << us_avxt << " us  | "
              << std::setw(8) << (long long)N * 1000000 / us_avxt << " dots/s  ("
              << std::fixed << std::setprecision(2) << (double)us_avxf / us_avxt << "x vs AVX2 float)\n";
    std::cout << "  Per-token (AVX2 trit): " << std::fixed << std::setprecision(3)
              << (double)us_avxt / N << " us = " << (double)us_avxt * 1000 / N << " ns\n";
}

// ============================================================================
// REAL WORKLOAD: hash bucket lookup + sum + dot
// ============================================================================

void test_workload_at_D(int D) {
    std::cout << "\n  ----- REAL WORKLOAD D = " << D << ": hash bucket (K=3) + dot -----\n";
    
    const int H = 4096;
    const int K = 3;
    const int N = 50000;
    
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> trit(-1, 1);
    
    // Float pool: H x D
    std::vector<std::vector<float>> pool_f(H, std::vector<float>(D));
    for (int h = 0; h < H; ++h)
        for (int d = 0; d < D; ++d) pool_f[h][d] = (float)trit(rng);
    
    // Trit pool (unpacked int8)
    std::vector<std::vector<int8_t>> pool_t(H, std::vector<int8_t>(D));
    for (int h = 0; h < H; ++h)
        for (int d = 0; d < D; ++d) pool_t[h][d] = (int8_t)trit(rng);
    
    // Query vectors
    std::vector<float> qf(D);
    std::vector<int8_t> qt(D);
    for (int d = 0; d < D; ++d) {
        int t = trit(rng);
        qt[d] = (int8_t)t;
        qf[d] = (float)t;
    }
    
    // Token IDs
    std::vector<int> tokens(N);
    std::uniform_int_distribution<int> idist(0, 49999);
    for (auto& t : tokens) t = idist(rng);
    
    auto h1 = [](int id) { return id % 4096; };
    auto h2 = [](int id) { return (id * 3 + 1) % 4096; };
    auto h3 = [](int id) { return (id * 7 + 5) % 4096; };
    
    // ----- Float version: gather 3 buckets, sum, dot with query (FP multiply) -----
    volatile float sf = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        int id = tokens[i];
        const float* b1 = pool_f[h1(id)].data();
        const float* b2 = pool_f[h2(id)].data();
        const float* b3 = pool_f[h3(id)].data();
        float s = 0;
        for (int d = 0; d < D; ++d) {
            float emb = b1[d] + b2[d] + b3[d];
            s += emb * qf[d];
        }
        sf += s;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto us_f = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    // ----- Trit version: gather 3 buckets, sum (clamp), sign agreement (pure add) -----
    volatile int st = 0;
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        int id = tokens[i];
        const int8_t* b1 = pool_t[h1(id)].data();
        const int8_t* b2 = pool_t[h2(id)].data();
        const int8_t* b3 = pool_t[h3(id)].data();
        int s = 0;
        for (int d = 0; d < D; ++d) {
            int emb = (int)b1[d] + (int)b2[d] + (int)b3[d];  // {-3..3}
            int q = qt[d];
            if (q == 0) continue;
            int es = (emb > 0) ? 1 : -1;
            if (es == q) s += 1;
            else s -= 1;
        }
        st += s;
    }
    t1 = std::chrono::high_resolution_clock::now();
    auto us_t = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    // Bandwidth per inference
    long f_bytes = (long)K * D * 4 + (long)D * 4;  // 3 buckets + query
    long t_bytes = (long)K * D + (long)D;          // 3 buckets (int8) + query (int8)
    
    std::cout << "  Float version: " << std::setw(6) << us_f << " us  | "
              << std::setw(8) << (long long)N * 1000000 / us_f << " tokens/s  | "
              << "bytes/token: " << f_bytes << "\n";
    std::cout << "  Trit version:  " << std::setw(6) << us_t << " us  | "
              << std::setw(8) << (long long)N * 1000000 / us_t << " tokens/s  | "
              << "bytes/token: " << t_bytes << "\n";
    std::cout << "  Speedup:       " << std::fixed << std::setprecision(2)
              << (double)us_f / us_t << "x\n";
    std::cout << "  Per token:     float " << std::fixed << std::setprecision(3)
              << (double)us_f * 1000 / N << " ns  |  trit "
              << (double)us_t * 1000 / N << " ns\n";
}

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Scaling test: trit {-1,0,+1} design at mainstream LLM dims\n";
    std::cout << "================================================================\n";
    
    print_pool_sizes();
    
    std::cout << "\n================================================================\n";
    std::cout << "  PURE DOT PRODUCT (D-dimensional)\n";
    std::cout << "================================================================\n";
    for (int D : {768, 1024, 2048, 4096}) {
        test_dot_at_D(D, {768, 1024, 2048, 4096});
    }
    
    std::cout << "\n================================================================\n";
    std::cout << "  REAL WORKLOAD: hash bucket lookup (K=3) + dot\n";
    std::cout << "================================================================\n";
    for (int D : {768, 2048, 4096}) {
        test_workload_at_D(D);
    }
    
    std::cout << "\n================================================================\n";
    std::cout << "  All tests complete.\n";
    std::cout << "================================================================\n";
    return 0;
}
