// bench_q2a_avx2.cpp
// AVX2 SIMD version of Q2-A + sum + polynomial head.
//
// Key optimizations:
//   - 8 floats at a time for h update (FMA, round, clamp)
//   - 16 int16s at a time for s update
//   - 8-wide horizontal sum for head
//
// Compile: clang++ -O2 -std=c++17 -march=native -mavx2 -mfma -o bench_q2a_avx2.exe bench_q2a_avx2.cpp

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <immintrin.h>

// ============ SCALAR VERSION ============

inline void q2a_step_scalar(const int8_t* h_old, const int16_t* s_old,
                            const int8_t* x_t, const float* alpha,
                            int8_t* h_new, int16_t* s_new, int D) {
    for (int d = 0; d < D; ++d) {
        float v = alpha[d] * (float)h_old[d] + (1.0f - alpha[d]) * (float)x_t[d];
        int r = (int)std::lroundf(v);
        if (r > 4) r = 4;
        if (r < -4) r = -4;
        h_new[d] = (int8_t)r;

        int s = (int)s_old[d] + (int)x_t[d];
        if (s > 64) s = 64;
        if (s < -64) s = -64;
        s_new[d] = (int16_t)s;
    }
}

inline float poly_head_scalar(const int8_t* h, const int16_t* s,
                              const float* w_h, const float* w_s,
                              const float* w_s2, const float* w_hs, float b, int D) {
    float acc = b;
    for (int d = 0; d < D; ++d) {
        float hd = (float)h[d];
        float sd = (float)s[d];
        acc += w_h[d] * hd + w_s[d] * sd + w_s2[d] * sd * sd + w_hs[d] * hd * sd;
    }
    return acc;
}

// ============ AVX2 VERSION ============

inline void q2a_step_avx2(const int8_t* h_old, const int16_t* s_old,
                          const int8_t* x_t, const float* alpha,
                          int8_t* h_new, int16_t* s_new, int D) {
    const __m256 one_f = _mm256_set1_ps(1.0f);
    const __m256 lo_f = _mm256_set1_ps(-4.0f);
    const __m256 hi_f = _mm256_set1_ps(4.0f);
    const __m256i lo_i = _mm256_set1_epi16(-64);
    const __m256i hi_i = _mm256_set1_epi16(64);

    int d = 0;

    // Process 16 dims per main iter (for s); 8 dims per h sub-iter
    for (; d + 16 <= D; d += 16) {
        // ====== H UPDATE: 2x 8-wide ======
        // First 8 dims
        {
            __m128i h_i8 = _mm_loadl_epi64((const __m128i*)(h_old + d));
            __m256i h_i32 = _mm256_cvtepi8_epi32(h_i8);
            __m256 h_f = _mm256_cvtepi32_ps(h_i32);

            __m128i x_i8 = _mm_loadl_epi64((const __m128i*)(x_t + d));
            __m256i x_i32 = _mm256_cvtepi8_epi32(x_i8);
            __m256 x_f = _mm256_cvtepi32_ps(x_i32);

            __m256 alpha_f = _mm256_loadu_ps(alpha + d);
            __m256 v = _mm256_fmadd_ps(alpha_f, _mm256_sub_ps(h_f, x_f), x_f);
            v = _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            v = _mm256_min_ps(_mm256_max_ps(v, lo_f), hi_f);
            __m256i v_i32 = _mm256_cvtps_epi32(v);
            __m128i v_i16 = _mm256_cvtepi32_epi16(v_i32);
            // store lower 8 bytes (need to pack 8 int16 -> 8 int8)
            __m128i v_packed = _mm_packs_epi16(v_i16, v_i16);  // saturating pack int16->int8
            _mm_storel_epi64((__m128i*)(h_new + d), v_packed);
        }
        // Next 8 dims
        {
            __m128i h_i8 = _mm_loadl_epi64((const __m128i*)(h_old + d + 8));
            __m256i h_i32 = _mm256_cvtepi8_epi32(h_i8);
            __m256 h_f = _mm256_cvtepi32_ps(h_i32);

            __m128i x_i8 = _mm_loadl_epi64((const __m128i*)(x_t + d + 8));
            __m256i x_i32 = _mm256_cvtepi8_epi32(x_i8);
            __m256 x_f = _mm256_cvtepi32_ps(x_i32);

            __m256 alpha_f = _mm256_loadu_ps(alpha + d + 8);
            __m256 v = _mm256_fmadd_ps(alpha_f, _mm256_sub_ps(h_f, x_f), x_f);
            v = _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            v = _mm256_min_ps(_mm256_max_ps(v, lo_f), hi_f);
            __m256i v_i32 = _mm256_cvtps_epi32(v);
            __m128i v_i16 = _mm256_cvtepi32_epi16(v_i32);
            __m128i v_packed = _mm_packs_epi16(v_i16, v_i16);
            _mm_storel_epi64((__m128i*)(h_new + d + 8), v_packed);
        }

        // ====== S UPDATE: 16-wide ======
        __m256i s_v = _mm256_loadu_si256((const __m256i*)(s_old + d));
        __m128i x_i8_16 = _mm_loadu_si128((const __m128i*)(x_t + d));
        __m256i x_i16 = _mm256_cvtepi8_epi16(x_i8_16);
        __m256i sum = _mm256_add_epi16(s_v, x_i16);
        sum = _mm256_min_epi16(_mm256_max_epi16(sum, lo_i), hi_i);
        _mm256_storeu_si256((__m256i*)(s_new + d), sum);
    }

    // Tail (scalar)
    for (; d < D; ++d) {
        float v = alpha[d] * (float)h_old[d] + (1.0f - alpha[d]) * (float)x_t[d];
        int r = (int)std::lroundf(v);
        if (r > 4) r = 4;
        if (r < -4) r = -4;
        h_new[d] = (int8_t)r;

        int s = (int)s_old[d] + (int)x_t[d];
        if (s > 64) s = 64;
        if (s < -64) s = -64;
        s_new[d] = (int16_t)s;
    }
}

inline float poly_head_avx2(const int8_t* h, const int16_t* s,
                            const float* w_h, const float* w_s,
                            const float* w_s2, const float* w_hs, float b, int D) {
    __m256 acc = _mm256_setzero_ps();
    int d = 0;
    for (; d + 8 <= D; d += 8) {
        __m128i h_i8 = _mm_loadl_epi64((const __m128i*)(h + d));
        __m256i h_i32 = _mm256_cvtepi8_epi32(h_i8);
        __m256 h_f = _mm256_cvtepi32_ps(h_i32);

        __m128i s_i16 = _mm_loadu_si128((const __m128i*)(s + d));
        __m256i s_i32 = _mm256_cvtepi16_epi32(s_i16);
        __m256 s_f = _mm256_cvtepi32_ps(s_i32);

        __m256 s2 = _mm256_mul_ps(s_f, s_f);
        __m256 hs = _mm256_mul_ps(h_f, s_f);

        acc = _mm256_fmadd_ps(_mm256_loadu_ps(w_h + d), h_f, acc);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(w_s + d), s_f, acc);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(w_s2 + d), s2, acc);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(w_hs + d), hs, acc);
    }
    // Horizontal sum
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float total = _mm_cvtss_f32(sums) + b;

    for (; d < D; ++d) {
        float hd = (float)h[d];
        float sd = (float)s[d];
        total += w_h[d] * hd + w_s[d] * sd + w_s2[d] * sd * sd + w_hs[d] * hd * sd;
    }
    return total;
}

// ============ BENCHMARK ============

struct BenchResult {
    double us_step, us_head, us_total;
    double speedup_step, speedup_total;
};

BenchResult bench(int D, int seq_len, int iters) {
    std::vector<int8_t> h_old(D, 0), h_new(D);
    std::vector<int16_t> s_old(D, 0), s_new(D);
    std::vector<int8_t> x_t(D);
    std::vector<float> alpha(D, 0.9f);
    std::vector<float> w_h(D), w_s(D), w_s2(D), w_hs(D);
    std::default_random_engine rng(42);
    std::normal_distribution<float> nd(0, 0.1f);
    for (auto& v : w_h) v = nd(rng);
    for (auto& v : w_s) v = nd(rng);
    for (auto& v : w_s2) v = nd(rng);
    for (auto& v : w_hs) v = nd(rng);
    std::uniform_int_distribution<int> ud(-3, 3);
    for (auto& v : x_t) v = ud(rng);
    float b = 0;

    // SCALAR
    // Warmup
    for (int i = 0; i < 100; ++i) {
        q2a_step_scalar(h_old.data(), s_old.data(), x_t.data(), alpha.data(),
                        h_new.data(), s_new.data(), D);
        std::swap(h_old, h_new); std::swap(s_old, s_new);
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    float acc1 = 0;
    for (int i = 0; i < iters; ++i) {
        q2a_step_scalar(h_old.data(), s_old.data(), x_t.data(), alpha.data(),
                        h_new.data(), s_new.data(), D);
        acc1 += poly_head_scalar(h_new.data(), s_new.data(),
                                w_h.data(), w_s.data(), w_s2.data(), w_hs.data(), b, D);
        std::swap(h_old, h_new); std::swap(s_old, s_new);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us_scalar = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    // Reset state
    std::fill(h_old.begin(), h_old.end(), 0);
    std::fill(s_old.begin(), s_old.end(), 0);

    // AVX2
    for (int i = 0; i < 100; ++i) {
        q2a_step_avx2(h_old.data(), s_old.data(), x_t.data(), alpha.data(),
                      h_new.data(), s_new.data(), D);
        std::swap(h_old, h_new); std::swap(s_old, s_new);
    }
    t0 = std::chrono::high_resolution_clock::now();
    float acc2 = 0;
    for (int i = 0; i < iters; ++i) {
        q2a_step_avx2(h_old.data(), s_old.data(), x_t.data(), alpha.data(),
                      h_new.data(), s_new.data(), D);
        acc2 += poly_head_avx2(h_new.data(), s_new.data(),
                                w_h.data(), w_s.data(), w_s2.data(), w_hs.data(), b, D);
        std::swap(h_old, h_new); std::swap(s_old, s_new);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double us_avx2 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    if (acc1 == 0 || acc2 == 0) std::cout << "";  // prevent opt

    double ups_scalar = us_scalar / iters;
    double ups_avx2 = us_avx2 / iters;
    return {ups_scalar, ups_avx2, ups_scalar - ups_avx2, ups_scalar / ups_avx2, (double)iters};
}

int main() {
    std::cout << "================================================================\n";
    std::cout << "  AVX2 SIMD Benchmark: Q2-A + sum + polynomial head\n";
    std::cout << "================================================================\n\n";

    std::cout << "  | D    | Scalar (us) | AVX2 (us) | Speedup | SEQ ms  | tokens/s |\n";
    std::cout << "  |------|-------------|-----------|---------|---------|----------|\n";

    const int D_LIST[] = {1024, 2048, 4096, 8192};
    for (int D : D_LIST) {
        const int seq_len = 1024;
        const int iters = 3000;
        auto r = bench(D, seq_len, iters);
        double seq_ms_scalar = r.us_step * seq_len / 1000.0;
        double seq_ms_avx2 = (r.us_step / r.speedup_step) * seq_len / 1000.0;
        double tokens_per_sec = 1e6 / (r.us_step / r.speedup_step);
        std::cout << "  | " << std::setw(4) << D
                  << " | " << std::setw(11) << std::fixed << std::setprecision(2) << r.us_step
                  << " | " << std::setw(9) << std::fixed << std::setprecision(2) << r.us_step / r.speedup_step
                  << " | " << std::setw(6) << std::fixed << std::setprecision(2) << r.speedup_step << "x"
                  << "  | " << std::setw(7) << std::fixed << std::setprecision(2) << seq_ms_avx2
                  << " | " << std::setw(8) << std::fixed << std::setprecision(0) << tokens_per_sec
                  << " |\n";
    }

    // Detailed per-component
    std::cout << "\n  Detailed per-component (D=4096):\n\n";

    int D = 4096;
    std::vector<int8_t> h_old(D, 0), h_new(D);
    std::vector<int16_t> s_old(D, 0), s_new(D);
    std::vector<int8_t> x_t(D);
    std::vector<float> alpha(D, 0.9f);
    std::vector<float> w_h(D), w_s(D), w_s2(D), w_hs(D);
    std::default_random_engine rng(42);
    std::normal_distribution<float> nd(0, 0.1f);
    for (auto& v : w_h) v = nd(rng);
    for (auto& v : w_s) v = nd(rng);
    for (auto& v : w_s2) v = nd(rng);
    for (auto& v : w_hs) v = nd(rng);
    std::uniform_int_distribution<int> ud(-3, 3);
    for (auto& v : x_t) v = ud(rng);

    int iters = 5000;

    // Scalar step only
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        q2a_step_scalar(h_old.data(), s_old.data(), x_t.data(), alpha.data(),
                        h_new.data(), s_new.data(), D);
        std::swap(h_old, h_new); std::swap(s_old, s_new);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double scalar_step = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / iters;

    // AVX2 step only
    std::fill(h_old.begin(), h_old.end(), 0);
    std::fill(s_old.begin(), s_old.end(), 0);
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        q2a_step_avx2(h_old.data(), s_old.data(), x_t.data(), alpha.data(),
                      h_new.data(), s_new.data(), D);
        std::swap(h_old, h_new); std::swap(s_old, s_new);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double avx2_step = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / iters;

    // Scalar head only
    t0 = std::chrono::high_resolution_clock::now();
    float acc1 = 0;
    for (int i = 0; i < iters; ++i) {
        acc1 += poly_head_scalar(h_new.data(), s_new.data(),
                                  w_h.data(), w_s.data(), w_s2.data(), w_hs.data(), 0, D);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double scalar_head = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / iters;

    // AVX2 head only
    t0 = std::chrono::high_resolution_clock::now();
    float acc2 = 0;
    for (int i = 0; i < iters; ++i) {
        acc2 += poly_head_avx2(h_new.data(), s_new.data(),
                                w_h.data(), w_s.data(), w_s2.data(), w_hs.data(), 0, D);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double avx2_head = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / iters;

    if (acc1 == 0 || acc2 == 0) std::cout << "";

    std::cout << "  | Component  | Scalar (us) | AVX2 (us) | Speedup |\n";
    std::cout << "  |-------------|-------------|-----------|---------|\n";
    std::cout << "  | Step only   | " << std::setw(11) << std::fixed << std::setprecision(2) << scalar_step
              << " | " << std::setw(9) << std::fixed << std::setprecision(2) << avx2_step
              << " | " << std::setw(6) << std::fixed << std::setprecision(2) << scalar_step / avx2_step << "x |\n";
    std::cout << "  | Head only   | " << std::setw(11) << std::fixed << std::setprecision(2) << scalar_head
              << " | " << std::setw(9) << std::fixed << std::setprecision(2) << avx2_head
              << " | " << std::setw(6) << std::fixed << std::setprecision(2) << scalar_head / avx2_head << "x |\n";
    std::cout << "  | Step + Head | " << std::setw(11) << std::fixed << std::setprecision(2) << scalar_step + scalar_head
              << " | " << std::setw(9) << std::fixed << std::setprecision(2) << avx2_step + avx2_head
              << " | " << std::setw(6) << std::fixed << std::setprecision(2) << (scalar_step + scalar_head) / (avx2_step + avx2_head) << "x |\n";

    std::cout << "\n  27B-class estimate (32 layers, D=4096):\n";
    double per_token_layer = avx2_step + avx2_head;
    double total_32layer = 32 * per_token_layer / 1000.0;  // ms
    double tokens_per_sec = 1e6 / (32 * per_token_layer);
    std::cout << "    Single-thread, 32 layers: " << std::fixed << std::setprecision(2) << total_32layer << " ms/token\n";
    std::cout << "                                " << std::fixed << std::setprecision(0) << tokens_per_sec << " tokens/s\n";
    double seq_ms = 32 * per_token_layer * 1024 / 1000.0;
    std::cout << "    1024-token sequence: " << std::fixed << std::setprecision(1) << seq_ms << " ms\n";

    std::cout << "\n================================================================\n";
    return 0;
}
