// test_vectors.cpp
// CPU-friendly vector operations test for hash bucket embedding table design
// Compile: g++ -O2 -o test_vectors.exe test_vectors.cpp
// Or MSVC: cl /O2 /EHsc test_vectors.cpp /Fe:test_vectors.exe
//
// Tests cover:
//   [1]-[5]  Storage methods for [x,x,x,x] type data
//   [6]-[9]  Basic operations on 4-element vectors
//   [10]     SIMD intrinsics
//   [11]     Performance benchmark: hash bucket lookup pattern

#include <iostream>
#include <iomanip>
#include <array>
#include <vector>
#include <chrono>
#include <cstring>
#include <random>
#include <numeric>
#include <algorithm>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <immintrin.h>
#endif

// ============================================================================
// STORAGE TESTS
// ============================================================================

void test_c_array() {
    std::cout << "\n[1] C-style array: float v[4]\n";
    float v[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::cout << "  size: " << sizeof(v) << " bytes (compile-time fixed)\n";
    std::cout << "  values: ";
    for (int i = 0; i < 4; ++i) std::cout << v[i] << " ";
    std::cout << "\n";
}

void test_std_array() {
    std::cout << "\n[2] std::array<float, 4>\n";
    std::array<float, 4> v = {1.0f, 2.0f, 3.0f, 4.0f};
    std::cout << "  size: " << sizeof(v) << " bytes\n";
    std::cout << "  values: ";
    for (size_t i = 0; i < v.size(); ++i) std::cout << v[i] << " ";
    std::cout << "\n";
}

void test_struct() {
    std::cout << "\n[3] struct Vec4 { x, y, z, w }\n";
    struct Vec4 { float x, y, z, w; };
    Vec4 v{1.0f, 2.0f, 3.0f, 4.0f};
    std::cout << "  size: " << sizeof(v) << " bytes\n";
    std::cout << "  v.x=" << v.x << " v.y=" << v.y << " v.z=" << v.z << " v.w=" << v.w << "\n";
}

void test_structured_bindings() {
    std::cout << "\n[4] Structured bindings (C++17): auto [a,b,c,d] = v\n";
    std::array<float, 4> v = {1.0f, 2.0f, 3.0f, 4.0f};
    auto [a, b, c, d] = v;
    std::cout << "  a=" << a << " b=" << b << " c=" << c << " d=" << d << "\n";
}

void test_bucket_pool() {
    std::cout << "\n[5] Hash bucket pool simulation: vector<array<float,4>>\n";
    const int H = 8;
    std::vector<std::array<float, 4>> pool(H);
    
    // Initialize
    for (int i = 0; i < H; ++i) {
        for (int d = 0; d < 4; ++d) {
            pool[i][d] = static_cast<float>(i) * 0.1f + static_cast<float>(d);
        }
    }
    
    std::cout << "  size: " << pool.size() << " buckets\n";
    std::cout << "  total bytes: " << pool.size() * sizeof(std::array<float,4>) << "\n";
    std::cout << "  Bucket[0]: "; for (auto x : pool[0]) std::cout << std::fixed << std::setprecision(2) << x << " "; std::cout << "\n";
    std::cout << "  Bucket[7]: "; for (auto x : pool[7]) std::cout << std::fixed << std::setprecision(2) << x << " "; std::cout << "\n";
}

// ============================================================================
// OPERATION TESTS
// ============================================================================

void test_add() {
    std::cout << "\n[6] Element-wise add: c = a + b\n";
    std::array<float, 4> a = {1, 2, 3, 4};
    std::array<float, 4> b = {5, 6, 7, 8};
    std::array<float, 4> c;
    for (size_t i = 0; i < 4; ++i) c[i] = a[i] + b[i];
    std::cout << "  a+b = "; for (auto x : c) std::cout << x << " "; std::cout << "\n";
}

void test_scale() {
    std::cout << "\n[7] Scalar multiply: b = a * 2.5\n";
    std::array<float, 4> a = {1, 2, 3, 4};
    float s = 2.5f;
    std::array<float, 4> b;
    for (size_t i = 0; i < 4; ++i) b[i] = a[i] * s;
    std::cout << "  a*2.5 = "; for (auto x : b) std::cout << x << " "; std::cout << "\n";
}

void test_dot() {
    std::cout << "\n[8] Dot product: a · b\n";
    std::array<float, 4> a = {1, 2, 3, 4};
    std::array<float, 4> b = {5, 6, 7, 8};
    float dot = 0;
    for (size_t i = 0; i < 4; ++i) dot += a[i] * b[i];
    std::cout << "  dot = " << dot << "  (expected 70)\n";
}

void test_sum() {
    std::cout << "\n[9] Sum reduction: Σ a[i]\n";
    std::array<float, 4> a = {1, 2, 3, 4};
    float sum = 0;
    for (auto v : a) sum += v;
    std::cout << "  sum = " << sum << "  (expected 10)\n";
}

// ============================================================================
// SIMD TEST (SSE intrinsics work on every x86-64)
// ============================================================================

void test_simd() {
    std::cout << "\n[10] SIMD intrinsics (__m128 / SSE)\n";
    __m128 a = _mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f);
    __m128 b = _mm_set_ps(8.0f, 7.0f, 6.0f, 5.0f);
    __m128 c = _mm_add_ps(a, b);
    float result[4];
    _mm_storeu_ps(result, c);
    std::cout << "  SIMD a+b = "; for (int i = 0; i < 4; ++i) std::cout << result[i] << " "; std::cout << "\n";
    
    // Dot product with SSE: mul then horizontal add
    __m128 mul = _mm_mul_ps(a, b);
    __m128 shuf = _mm_movehdup_ps(mul);
    __m128 sums = _mm_add_ps(mul, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float dot = _mm_cvtss_f32(sums);
    std::cout << "  SIMD a.b = " << dot << "  (expected 70)\n";
}

// ============================================================================
// PERFORMANCE BENCHMARK: hash bucket lookup (simulates candidate A)
// ============================================================================

void benchmark_bucket_lookup() {
    std::cout << "\n[11] PERFORMANCE: hash bucket lookup pattern\n";
    std::cout << "     (simulates: token -> K buckets -> sum into embedding)\n";
    
    const int H = 4096;   // bucket count
    const int D = 64;     // dim per bucket
    const int K = 3;      // hashes per token
    const int N = 1000000; // 1M lookups
    
    // Allocate pool: H x D float = 4096 * 64 * 4 = 1 MB
    std::vector<float> pool(H * D);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : pool) v = dist(rng);
    
    // Token IDs
    std::vector<int> token_ids(N);
    std::uniform_int_distribution<int> idist(0, 49999);
    for (auto& id : token_ids) id = idist(rng);
    
    // 3 simple hash functions
    auto h1 = [](int id) { return id % 4096; };
    auto h2 = [](int id) { return (id * 3 + 1) % 4096; };
    auto h3 = [](int id) { return (id * 7 + 5) % 4096; };
    
    // --- Naive scalar version ---
    auto start = std::chrono::high_resolution_clock::now();
    volatile float sum_scalar = 0;
    for (int t = 0; t < N; ++t) {
        int id = token_ids[t];
        int b1 = h1(id), b2 = h2(id), b3 = h3(id);
        const float* p1 = pool.data() + b1 * D;
        const float* p2 = pool.data() + b2 * D;
        const float* p3 = pool.data() + b3 * D;
        for (int d = 0; d < D; ++d) {
            sum_scalar += p1[d] + p2[d] + p3[d];
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto us_scalar = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // --- SIMD version (process 4 floats at a time) ---
    start = std::chrono::high_resolution_clock::now();
    volatile float sum_simd = 0;
    for (int t = 0; t < N; ++t) {
        int id = token_ids[t];
        int b1 = h1(id), b2 = h2(id), b3 = h3(id);
        const float* p1 = pool.data() + b1 * D;
        const float* p2 = pool.data() + b2 * D;
        const float* p3 = pool.data() + b3 * D;
        for (int d = 0; d < D; d += 4) {
            __m128 v1 = _mm_loadu_ps(p1 + d);
            __m128 v2 = _mm_loadu_ps(p2 + d);
            __m128 v3 = _mm_loadu_ps(p3 + d);
            __m128 vsum = _mm_add_ps(_mm_add_ps(v1, v2), v3);
            // horizontal sum
            __m128 shuf = _mm_movehdup_ps(vsum);
            __m128 sums = _mm_add_ps(vsum, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            sum_simd += _mm_cvtss_f32(sums);
        }
    }
    end = std::chrono::high_resolution_clock::now();
    auto us_simd = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "  Config: H=" << H << " D=" << D << " K=" << K << " N=" << N << "\n";
    std::cout << "  Pool size: " << (H * D * sizeof(float)) / 1024 << " KB\n";
    std::cout << "  ---- Scalar ----\n";
    std::cout << "  Time: " << us_scalar << " us = " << us_scalar/1000.0 << " ms\n";
    std::cout << "  Per lookup: " << (double)us_scalar / N << " us\n";
    std::cout << "  Throughput: " << (long long)N * 1000000 / us_scalar << " lookups/sec\n";
    std::cout << "  ---- SIMD (SSE) ----\n";
    std::cout << "  Time: " << us_simd << " us = " << us_simd/1000.0 << " ms\n";
    std::cout << "  Per lookup: " << (double)us_simd / N << " us\n";
    std::cout << "  Throughput: " << (long long)N * 1000000 / us_simd << " lookups/sec\n";
    std::cout << "  ---- Speedup ----\n";
    std::cout << "  SIMD vs Scalar: " << std::fixed << std::setprecision(2) 
              << (double)us_scalar / us_simd << "x\n";
    std::cout << "  (sum_scalar=" << sum_scalar << ", sum_simd=" << sum_simd << ")\n";
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "==========================================================\n";
    std::cout << "  C++ Vector Test (for hash bucket embedding table design)\n";
    std::cout << "  Target: storage / read / operations on [x,x,x,x] data\n";
    std::cout << "==========================================================\n";
    
    test_c_array();
    test_std_array();
    test_struct();
    test_structured_bindings();
    test_bucket_pool();
    test_add();
    test_scale();
    test_dot();
    test_sum();
    test_simd();
    benchmark_bucket_lookup();
    
    std::cout << "\n==========================================================\n";
    std::cout << "  All tests complete.\n";
    std::cout << "==========================================================\n";
    return 0;
}
