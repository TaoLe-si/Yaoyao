// bench_topk.cpp
// AVX2 Top-K Optimization Benchmark
//
// Compare 4 implementations:
//   1. std::partial_sort (baseline)
//   2. std::nth_element (find K-th, then partial sort)
//   3. Scalar batched (find max in 8, linear scan top-K)
//   4. AVX2 batched (vectorized max + min + insert)

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <immintrin.h>

// ============ 1. std::partial_sort ============
void topk_partial_sort(const float* scores, int V, int K, int* out_idx) {
    std::vector<int> idx(V);
    for (int i = 0; i < V; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
        [&](int a, int b) { return scores[a] > scores[b]; });
    for (int k = 0; k < K; ++k) out_idx[k] = idx[k];
}

// ============ 2. std::nth_element ============
void topk_nth_element(const float* scores, int V, int K, int* out_idx) {
    std::vector<int> idx(V);
    for (int i = 0; i < V; ++i) idx[i] = i;
    std::nth_element(idx.begin(), idx.begin() + K, idx.end(),
        [&](int a, int b) { return scores[a] > scores[b]; });
    std::partial_sort(idx.begin(), idx.begin() + K, idx.begin() + K + 1,
        [&](int a, int b) { return scores[a] > scores[b]; });
    for (int k = 0; k < K; ++k) out_idx[k] = idx[k];
}

// ============ 3. Scalar batched (find max in 8, linear scan top-K) ============
void topk_scalar(const float* scores, int V, int K, int* out_idx) {
    alignas(32) float top_vals[16];
    int top_idx[16];
    for (int k = 0; k < K; ++k) { top_vals[k] = -1e30f; top_idx[k] = -1; }
    int min_pos = 0;
    
    for (int i = 0; i < V; ++i) {
        float v = scores[i];
        if (v > top_vals[min_pos]) {
            top_vals[min_pos] = v;
            top_idx[min_pos] = i;
            // Find new min
            min_pos = 0;
            for (int k = 1; k < K; ++k) {
                if (top_vals[k] < top_vals[min_pos]) min_pos = k;
            }
        }
    }
    for (int k = 0; k < K; ++k) out_idx[k] = top_idx[k];
}

// ============ 4. AVX2 batched (process 8 floats at a time) ============
void topk_avx2(const float* scores, int V, int K, int* out_idx) {
    alignas(32) float top_vals[16];
    int top_idx[16];
    for (int k = 0; k < K; ++k) { top_vals[k] = -1e30f; top_idx[k] = -1; }
    int min_pos = 0;
    
    int V_aligned = (V / 8) * 8;
    
    for (int i = 0; i < V_aligned; i += 8) {
        __m256 vec = _mm256_loadu_ps(&scores[i]);
        
        // Find max in batch (horizontal reduction)
        __m128 lo = _mm256_castps256_ps128(vec);
        __m128 hi = _mm256_extractf128_ps(vec, 1);
        __m128 m = _mm_max_ps(lo, hi);
        m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 0x55));
        float max_val = _mm_cvtss_f32(m);
        
        // Find max index in batch via mask
        __m256 cmp = _mm256_cmp_ps(vec, _mm256_set1_ps(max_val), _CMP_EQ_OQ);
        int mask = _mm256_movemask_ps(cmp);
        int max_off = __builtin_ctz(mask);
        int max_idx = i + max_off;
        
        // Compare to current min in top-K
        if (max_val > top_vals[min_pos]) {
            top_vals[min_pos] = max_val;
            top_idx[min_pos] = max_idx;
            // Find new min
            min_pos = 0;
            for (int k = 1; k < K; ++k) {
                if (top_vals[k] < top_vals[min_pos]) min_pos = k;
            }
        }
    }
    
    // Handle tail
    for (int i = V_aligned; i < V; ++i) {
        float v = scores[i];
        if (v > top_vals[min_pos]) {
            top_vals[min_pos] = v;
            top_idx[min_pos] = i;
            min_pos = 0;
            for (int k = 1; k < K; ++k) {
                if (top_vals[k] < top_vals[min_pos]) min_pos = k;
            }
        }
    }
    
    for (int k = 0; k < K; ++k) out_idx[k] = top_idx[k];
}

// ============ 5. AVX2 batched + AVX2 min scan ============
// Vectorize the min-finding too: process top-K as 16-element AVX2 register
void topk_avx2_v2(const float* scores, int V, int K, int* out_idx) {
    alignas(32) float top_vals[16];
    int top_idx[16];
    for (int k = 0; k < K; ++k) { top_vals[k] = -1e30f; top_idx[k] = -1; }
    
    int V_aligned = (V / 8) * 8;
    
    for (int i = 0; i < V_aligned; i += 8) {
        __m256 vec = _mm256_loadu_ps(&scores[i]);
        
        __m128 lo = _mm256_castps256_ps128(vec);
        __m128 hi = _mm256_extractf128_ps(vec, 1);
        __m128 m = _mm_max_ps(lo, hi);
        m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 0x55));
        float max_val = _mm_cvtss_f32(m);
        
        __m256 cmp = _mm256_cmp_ps(vec, _mm256_set1_ps(max_val), _CMP_EQ_OQ);
        int mask = _mm256_movemask_ps(cmp);
        int max_off = __builtin_ctz(mask);
        int max_idx = i + max_off;
        
        // Find min in top-K using AVX2 (load 8 floats, horizontal min)
        __m256 tv = _mm256_load_ps(top_vals);
        __m128 tlo = _mm256_castps256_ps128(tv);
        __m128 thi = _mm256_extractf128_ps(tv, 1);
        __m128 mn = _mm_min_ps(tlo, thi);
        mn = _mm_min_ps(mn, _mm_movehl_ps(mn, mn));
        mn = _mm_min_ss(mn, _mm_shuffle_ps(mn, mn, 0x55));
        float min_val = _mm_cvtss_f32(mn);
        
        if (max_val > min_val) {
            // Find which position has min
            __m256 cmp_min = _mm256_cmp_ps(tv, _mm256_set1_ps(min_val), _CMP_EQ_OQ);
            int mask_min = _mm256_movemask_ps(cmp_min);
            int min_pos = __builtin_ctz(mask_min);
            
            top_vals[min_pos] = max_val;
            top_idx[min_pos] = max_idx;
        }
    }
    
    for (int k = 0; k < K; ++k) out_idx[k] = top_idx[k];
}

int main() {
    const int V = 50000;  // realistic vocab size
    const int K = 10;
    const int trials = 200;
    
    std::cout << "================================================================\n";
    std::cout << "  AVX2 Top-K Benchmark | V=" << V << ", K=" << K << "\n";
    std::cout << "================================================================\n\n";
    
    // Generate random scores
    std::vector<float> scores(V);
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0, 1.0f);
    for (auto& s : scores) s = nd(rng);
    
    // Verify correctness
    int out_partial[20], out_avx2[20], out_avx2_v2[20], out_scalar[20], out_nth[20];
    topk_partial_sort(scores.data(), V, K, out_partial);
    topk_avx2(scores.data(), V, K, out_avx2);
    topk_avx2_v2(scores.data(), V, K, out_avx2_v2);
    topk_scalar(scores.data(), V, K, out_scalar);
    topk_nth_element(scores.data(), V, K, out_nth);
    
    bool match_partial_avx2 = true;
    for (int k = 0; k < K; ++k) {
        if (scores[out_partial[k]] != scores[out_avx2[k]]) match_partial_avx2 = false;
    }
    bool match_avx2_v2 = true;
    for (int k = 0; k < K; ++k) {
        if (scores[out_partial[k]] != scores[out_avx2_v2[k]]) match_avx2_v2 = false;
    }
    bool match_scalar = true;
    for (int k = 0; k < K; ++k) {
        if (scores[out_partial[k]] != scores[out_scalar[k]]) match_scalar = false;
    }
    bool match_nth = true;
    for (int k = 0; k < K; ++k) {
        if (scores[out_partial[k]] != scores[out_nth[k]]) match_nth = false;
    }
    
    std::cout << "  Correctness:\n";
    std::cout << "    scalar matches partial_sort: " << (match_scalar ? "YES" : "NO") << "\n";
    std::cout << "    AVX2 matches partial_sort: " << (match_partial_avx2 ? "YES" : "NO") << "\n";
    std::cout << "    AVX2-v2 matches partial_sort: " << (match_avx2_v2 ? "YES" : "NO") << "\n";
    std::cout << "    nth_element matches: " << (match_nth ? "YES" : "NO") << "\n\n";
    
    std::cout << "  Top-10 indices and scores:\n";
    std::cout << "    partial_sort: ";
    for (int k = 0; k < K; ++k) std::cout << out_partial[k] << "(" << std::fixed << std::setprecision(2) << scores[out_partial[k]] << ") ";
    std::cout << "\n    AVX2:         ";
    for (int k = 0; k < K; ++k) std::cout << out_avx2[k] << "(" << scores[out_avx2[k]] << ") ";
    std::cout << "\n\n";
    
    // Warmup
    int out[20];
    for (int i = 0; i < 100; ++i) {
        topk_partial_sort(scores.data(), V, K, out);
        topk_nth_element(scores.data(), V, K, out);
        topk_scalar(scores.data(), V, K, out);
        topk_avx2(scores.data(), V, K, out);
        topk_avx2_v2(scores.data(), V, K, out);
        if (out[0] == -99999) std::cout << "";
    }
    
    // Benchmark
    std::cout << "  Benchmark (" << trials << " iterations):\n\n";
    
    auto bench = [&](auto fn, const char* name) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < trials; ++i) {
            fn(scores.data(), V, K, out);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / (double)trials;
        return us;
    };
    
    double us_partial = bench(topk_partial_sort, "partial_sort");
    double us_nth = bench(topk_nth_element, "nth_element");
    double us_scalar = bench(topk_scalar, "scalar");
    double us_avx2 = bench(topk_avx2, "AVX2");
    double us_avx2_v2 = bench(topk_avx2_v2, "AVX2-v2");
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "    std::partial_sort:   " << std::setw(8) << us_partial << " µs  (baseline)\n";
    std::cout << "    std::nth_element:    " << std::setw(8) << us_nth << " µs  (" 
              << std::setprecision(1) << us_partial / us_nth << "× speedup)\n";
    std::cout << "    scalar batched:      " << std::setw(8) << us_scalar << " µs  (" 
              << std::setprecision(1) << us_partial / us_scalar << "× speedup)\n";
    std::cout << "    AVX2 batched:        " << std::setw(8) << us_avx2 << " µs  (" 
              << std::setprecision(1) << us_partial / us_avx2 << "× speedup)\n";
    std::cout << "    AVX2 + AVX2-min:     " << std::setw(8) << us_avx2_v2 << " µs  (" 
              << std::setprecision(1) << us_partial / us_avx2_v2 << "× speedup)\n\n";
    
    // Throughput
    std::cout << "  Throughput (V/t_us µs):\n";
    std::cout << "    partial_sort:  " << std::setprecision(0) << V / us_partial << " elements/µs\n";
    std::cout << "    nth_element:   " << V / us_nth << " elements/µs\n";
    std::cout << "    scalar:        " << V / us_scalar << " elements/µs\n";
    std::cout << "    AVX2:          " << V / us_avx2 << " elements/µs\n";
    std::cout << "    AVX2-v2:       " << V / us_avx2_v2 << " elements/µs\n";
    
    return 0;
}
