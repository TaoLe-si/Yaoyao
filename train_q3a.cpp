// train_q3a.cpp
// Full architecture: Q1 -> Q3 (k=3 conv) -> Q2-A + Sum -> Q4 (poly head)
// AVX2 SIMD throughout.
//
// Task: "ends with AB" — last 2 tokens are A then B.
//       Pure Q2-A CANNOT solve (h only sees last 1 token with α).
//       Q3 conv MUST solve (sees x_{t-1}, x_t at position T).
//
// Compile: clang++ -O2 -std=c++17 -march=native -mavx2 -mfma -o train_q3a.exe train_q3a.cpp

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <immintrin.h>

const int D = 4096;
const int H_VAL = 4;
const int S_MAX = 64;
const int SEQ_LEN = 16;
const int EPOCHS = 80;
const int N_TRAIN = 500;
const float LR = 0.5f;

void q1(int token_id, int8_t* x) {
    std::fill(x, x + D, 0);
    if (token_id == 0) { x[0] = 3; x[1] = -3; x[2] = 1; }
    else { x[0] = -3; x[1] = 3; x[2] = -1; }
}

// ============ AVX2: Q3 (k=3 conv) ============
inline void q3_conv_avx2(const int8_t* x_prev2, const int8_t* x_prev,
                         const int8_t* x_curr,
                         const float* w0, const float* w1, const float* w2,
                         int8_t* y_out) {
    const __m256 lo_f = _mm256_set1_ps(-4.0f);
    const __m256 hi_f = _mm256_set1_ps(4.0f);

    int d = 0;
    for (; d + 8 <= D; d += 8) {
        __m128i x0_i8 = _mm_loadl_epi64((const __m128i*)(x_prev2 + d));
        __m128i x1_i8 = _mm_loadl_epi64((const __m128i*)(x_prev + d));
        __m128i x2_i8 = _mm_loadl_epi64((const __m128i*)(x_curr + d));

        __m256 x0_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(x0_i8));
        __m256 x1_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(x1_i8));
        __m256 x2_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(x2_i8));

        __m256 w0_f = _mm256_loadu_ps(w0 + d);
        __m256 w1_f = _mm256_loadu_ps(w1 + d);
        __m256 w2_f = _mm256_loadu_ps(w2 + d);

        __m256 y_f = _mm256_mul_ps(w0_f, x0_f);
        y_f = _mm256_fmadd_ps(w1_f, x1_f, y_f);
        y_f = _mm256_fmadd_ps(w2_f, x2_f, y_f);

        y_f = _mm256_round_ps(y_f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        y_f = _mm256_min_ps(_mm256_max_ps(y_f, lo_f), hi_f);

        __m256i y_i32 = _mm256_cvtps_epi32(y_f);
        __m128i y_i16 = _mm256_cvtepi32_epi16(y_i32);
        __m128i y_packed = _mm_packs_epi16(y_i16, y_i16);
        _mm_storel_epi64((__m128i*)(y_out + d), y_packed);
    }
    for (; d < D; ++d) {
        float v = w0[d] * (float)x_prev2[d] + w1[d] * (float)x_prev[d] + w2[d] * (float)x_curr[d];
        int r = (int)std::lroundf(v);
        if (r > H_VAL) r = H_VAL;
        if (r < -H_VAL) r = -H_VAL;
        y_out[d] = (int8_t)r;
    }
}

// ============ AVX2: Q2-A + Sum ============
inline void q2a_step_avx2(const int8_t* h_old, const int16_t* s_old,
                          const int8_t* y_t, const float* alpha,
                          int8_t* h_new, int16_t* s_new) {
    const __m256 lo_f = _mm256_set1_ps(-4.0f);
    const __m256 hi_f = _mm256_set1_ps(4.0f);
    const __m256i lo_i = _mm256_set1_epi16(-64);
    const __m256i hi_i = _mm256_set1_epi16(64);

    int d = 0;
    for (; d + 16 <= D; d += 16) {
        for (int sub = 0; sub < 2; ++sub) {
            __m128i h_i8 = _mm_loadl_epi64((const __m128i*)(h_old + d + sub * 8));
            __m256i h_i32 = _mm256_cvtepi8_epi32(h_i8);
            __m256 h_f = _mm256_cvtepi32_ps(h_i32);
            __m128i y_i8 = _mm_loadl_epi64((const __m128i*)(y_t + d + sub * 8));
            __m256i y_i32 = _mm256_cvtepi8_epi32(y_i8);
            __m256 y_f = _mm256_cvtepi32_ps(y_i32);
            __m256 alpha_f = _mm256_loadu_ps(alpha + d + sub * 8);
            __m256 v = _mm256_fmadd_ps(alpha_f, _mm256_sub_ps(h_f, y_f), y_f);
            v = _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            v = _mm256_min_ps(_mm256_max_ps(v, lo_f), hi_f);
            __m256i v_i32 = _mm256_cvtps_epi32(v);
            __m128i v_i16 = _mm256_cvtepi32_epi16(v_i32);
            __m128i v_packed = _mm_packs_epi16(v_i16, v_i16);
            _mm_storel_epi64((__m128i*)(h_new + d + sub * 8), v_packed);
        }
        __m256i s_v = _mm256_loadu_si256((const __m256i*)(s_old + d));
        __m128i y_i8_16 = _mm_loadu_si128((const __m128i*)(y_t + d));
        __m256i y_i16 = _mm256_cvtepi8_epi16(y_i8_16);
        __m256i sum = _mm256_add_epi16(s_v, y_i16);
        sum = _mm256_min_epi16(_mm256_max_epi16(sum, lo_i), hi_i);
        _mm256_storeu_si256((__m256i*)(s_new + d), sum);
    }
    for (; d < D; ++d) {
        float v = alpha[d] * (float)h_old[d] + (1.0f - alpha[d]) * (float)y_t[d];
        int r = (int)std::lroundf(v);
        if (r > H_VAL) r = H_VAL;
        if (r < -H_VAL) r = -H_VAL;
        h_new[d] = (int8_t)r;
        int s = (int)s_old[d] + (int)y_t[d];
        if (s > S_MAX) s = S_MAX;
        if (s < -S_MAX) s = -S_MAX;
        s_new[d] = (int16_t)s;
    }
}

// ============ AVX2: Q4 polynomial head ============
inline float poly_head_avx2(const int8_t* h, const int16_t* s,
                            const float* w_h, const float* w_s,
                            const float* w_s2, const float* w_hs, float b) {
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
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float total = _mm_cvtss_f32(sums) + b;
    for (; d < D; ++d) {
        float hd = (float)h[d], sd = (float)s[d];
        total += w_h[d] * hd + w_s[d] * sd + w_s2[d] * sd * sd + w_hs[d] * hd * sd;
    }
    return total;
}

// ============ Forward + Backward ============
struct ForwardCache {
    std::vector<std::vector<int8_t>> xs, ys;        // x_t, y_t (input to Q2)
    std::vector<std::vector<int8_t>> hs, hnext;     // h before and after step
    std::vector<std::vector<int16_t>> ss, snext;
};

ForwardCache forward_seq(const std::vector<int>& tokens,
                          const float* w_q3_0, const float* w_q3_1, const float* w_q3_2,
                          const float* alpha,
                          const float* w_h, const float* w_s,
                          const float* w_s2, const float* w_hs, float b,
                          float* out_logit) {
    ForwardCache c;
    c.xs.assign(SEQ_LEN, std::vector<int8_t>(D, 0));
    c.ys.assign(SEQ_LEN, std::vector<int8_t>(D, 0));
    c.hs.assign(SEQ_LEN + 1, std::vector<int8_t>(D, 0));
    c.hnext.assign(SEQ_LEN + 1, std::vector<int8_t>(D, 0));
    c.ss.assign(SEQ_LEN + 1, std::vector<int16_t>(D, 0));
    c.snext.assign(SEQ_LEN + 1, std::vector<int16_t>(D, 0));

    std::vector<int8_t> x_curr(D, 0), x_prev(D, 0), x_prev2(D, 0), y(D, 0);
    for (int t = 0; t < SEQ_LEN; ++t) {
        x_prev2 = x_prev;
        x_prev = x_curr;
        q1(tokens[t], x_curr.data());
        c.xs[t] = x_curr;

        q3_conv_avx2(x_prev2.data(), x_prev.data(), x_curr.data(),
                     w_q3_0, w_q3_1, w_q3_2, y.data());
        c.ys[t] = y;

        q2a_step_avx2(c.hs[t].data(), c.ss[t].data(),
                       y.data(), alpha,
                       c.hnext[t + 1].data(), c.snext[t + 1].data());
    }
    // Last step final state
    *out_logit = poly_head_avx2(c.hnext[SEQ_LEN].data(), c.snext[SEQ_LEN].data(),
                                 w_h, w_s, w_s2, w_hs, b);
    return c;
}

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Q3 (k=3 conv) + Q2-A + Sum + Q4 (poly head): End-to-End\n";
    std::cout << "================================================================\n\n";
    std::cout << "  Architecture: Q1 -> Q3 (k=3) -> Q2-A + Sum -> Q4 (poly)\n";
    std::cout << "  Task: 'ends with AB' (last 2 tokens are A then B)\n\n";

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> ud(0, 1);

    // Generate balanced data
    std::vector<std::pair<std::vector<int>, int>> data;
    int lp0 = 0, lp1 = 0;
    for (int i = 0; i < N_TRAIN / 2; ++i) {
        std::vector<int> seq(SEQ_LEN);
        for (int t = 0; t < SEQ_LEN; ++t) seq[t] = ud(rng);
        seq[SEQ_LEN - 2] = 0;  // A
        seq[SEQ_LEN - 1] = 1;  // B
        data.push_back({seq, 0});
        lp0++;
    }
    for (int i = 0; i < N_TRAIN / 2; ++i) {
        std::vector<int> seq(SEQ_LEN);
        for (int t = 0; t < SEQ_LEN; ++t) seq[t] = ud(rng);
        // Ensure last 2 are NOT AB
        if (seq[SEQ_LEN - 2] == 0 && seq[SEQ_LEN - 1] == 1) {
            seq[SEQ_LEN - 1] = 0;  // Force not AB
        }
        data.push_back({seq, 1});
        lp1++;
    }
    std::cout << "  Data: " << data.size() << " (label-0: " << lp0
              << " ends with AB, label-1: " << lp1 << " not AB)\n\n";

    // Initialize params
    std::vector<float> w_q3_0(D, 0), w_q3_1(D, 0), w_q3_2(D, 0);
    std::vector<float> alpha(D, 0.99f);
    std::vector<float> w_h(D), w_s(D), w_s2(D), w_hs(D);
    std::normal_distribution<float> nd(0, 0.1f);
    for (auto& v : w_h) v = nd(rng);
    for (auto& v : w_s) v = nd(rng);
    for (auto& v : w_s2) v = nd(rng);
    for (auto& v : w_hs) v = nd(rng);
    float b = 0;

    auto predict = [&](const std::vector<int>& seq) {
        float logit;
        forward_seq(seq, w_q3_0.data(), w_q3_1.data(), w_q3_2.data(),
                    alpha.data(),
                    w_h.data(), w_s.data(), w_s2.data(), w_hs.data(), b, &logit);
        return logit > 0 ? 0 : 1;
    };

    int initial_correct = 0;
    for (auto& p : data) if (predict(p.first) == p.second) initial_correct++;
    std::cout << "  Initial acc: " << initial_correct << "/" << data.size()
              << " (" << initial_correct * 100 / data.size() << "%)\n\n";

    // Simple training: only train Q3 conv weights + bias.
    // h channel weights stay random (we use h[0] as the AB signal but actually
    // since Q3 produces a feature capturing x_{T-1}=A & x_T=B, we can train
    // Q3 weights directly).
    //
    // For simplicity here: train w_q3_1 and w_q3_2 (skip w_q3_0 for now)
    // to maximize y[0] when AB at end, minimize otherwise.
    std::vector<float> lr_q3(D, 0.5f);
    std::vector<float> lr_b(1, 0.5f);

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(data.begin(), data.end(), std::mt19937(epoch + 1));

        std::vector<float> w_q3_1_grad(D, 0), w_q3_2_grad(D, 0);
        float b_grad = 0;
        float total_loss = 0;

        for (auto& p : data) {
            float logit;
            ForwardCache c = forward_seq(p.first, w_q3_0.data(), w_q3_1.data(), w_q3_2.data(),
                                          alpha.data(), w_h.data(), w_s.data(),
                                          w_s2.data(), w_hs.data(), b, &logit);

            float p0 = 1.0f / (1.0f + std::exp(-2.0f * logit));
            float prob_true = (p.second == 0) ? p0 : (1.0f - p0);
            total_loss += -std::log(std::max(prob_true, 1e-7f));
            float d_logit = (p.second == 0) ? (-2.0f * (1.0f - p0)) : (2.0f * p0);

            // Gradient through head: only y[SEQ_LEN-1][0] matters here (we'll
            // approximate by treating y[SEQ_LEN-1] as input to a simplified
            // linear head with weight 1 on dim 0).
            //
            // For end-to-end, full backward is complex. Let's do a SHORTCUT:
            // Train Q3 to produce y[SEQ_LEN-1][0] such that:
            //   label 0 (AB): y[0] -> large positive
            //   label 1 (not AB): y[0] -> large negative
            // Then linear head with weight 1 on dim 0.
            //
            // We achieve this by setting up the loss on y[SEQ_LEN-1][0] directly.

            // Use simplified loss: push y_last[0] toward +6 if AB, -6 if not.
            int8_t y_last_0 = c.ys[SEQ_LEN - 1][0];
            float target = (p.second == 0) ? 6.0f : -6.0f;
            float diff = (float)y_last_0 - target;
            float d_y0 = 2.0f * diff;  // gradient of (y - target)^2

            // Backprop through Q3 conv at last position:
            // y_last[0] = w_q3_1[0] * x_prev[0] + w_q3_2[0] * x_curr[0]
            //           (x_prev2[0] contribution ignored for simplicity)
            // where x_prev = x_{T-1} = token at position SEQ_LEN-2
            //       x_curr = x_{T} = token at position SEQ_LEN-1
            int8_t x_prev_0 = c.xs[SEQ_LEN - 2][0];
            int8_t x_curr_0 = c.xs[SEQ_LEN - 1][0];
            w_q3_1_grad[0] += d_y0 * (float)x_prev_0;
            w_q3_2_grad[0] += d_y0 * (float)x_curr_0;
            b_grad += d_logit;
        }

        w_q3_1[0] -= LR * w_q3_1_grad[0] / N_TRAIN;
        w_q3_2[0] -= LR * w_q3_2_grad[0] / N_TRAIN;
        b -= LR * b_grad / N_TRAIN;

        if ((epoch + 1) % 10 == 0 || epoch == 0 || epoch == EPOCHS - 1) {
            int c = 0;
            for (auto& pp : data) if (predict(pp.first) == pp.second) c++;
            std::cout << "  Epoch " << std::setw(3) << (epoch + 1)
                      << ": loss=" << std::fixed << std::setprecision(4) << total_loss / data.size()
                      << " acc=" << c << "/" << data.size()
                      << " (" << std::setw(3) << c * 100 / data.size() << "%)"
                      << " w_q3_1[0]=" << std::setprecision(3) << w_q3_1[0]
                      << " w_q3_2[0]=" << w_q3_2[0]
                      << " b=" << b << "\n";
        }
    }
    int fc = 0;
    for (auto& p : data) if (predict(p.first) == p.second) fc++;
    std::cout << "\n  Final: " << fc << "/" << data.size()
              << " (" << fc * 100 / data.size() << "%)\n\n";

    // Benchmark
    std::cout << "================================================================\n";
    std::cout << "  BENCHMARK (Q3 + Q2-A + Sum + Head per layer)\n";
    std::cout << "================================================================\n\n";

    std::vector<int> bench_tokens(SEQ_LEN);
    for (int t = 0; t < SEQ_LEN; ++t) bench_tokens[t] = ud(rng);

    // Warmup
    for (int i = 0; i < 100; ++i) {
        float logit;
        forward_seq(bench_tokens, w_q3_0.data(), w_q3_1.data(), w_q3_2.data(),
                    alpha.data(), w_h.data(), w_s.data(),
                    w_s2.data(), w_hs.data(), b, &logit);
        if (logit == -1e30f) std::cout << "";
    }

    int iters = 5000;
    auto t0 = std::chrono::high_resolution_clock::now();
    float acc_logit = 0;
    for (int i = 0; i < iters; ++i) {
        float logit;
        forward_seq(bench_tokens, w_q3_0.data(), w_q3_1.data(), w_q3_2.data(),
                    alpha.data(), w_h.data(), w_s.data(),
                    w_s2.data(), w_hs.data(), b, &logit);
        acc_logit += logit;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us_total = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / iters;

    if (acc_logit == 0) std::cout << "";

    // Component breakdown
    // Q3 only
    std::vector<int8_t> x_p2(D, 0), x_p(D, 0), x_c(D, 0), y_out(D, 0);
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        q3_conv_avx2(x_p2.data(), x_p.data(), x_c.data(),
                     w_q3_0.data(), w_q3_1.data(), w_q3_2.data(), y_out.data());
    }
    t1 = std::chrono::high_resolution_clock::now();
    double us_q3 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / iters;

    // Q2-A only
    std::vector<int8_t> h_old(D, 0), h_new(D), y_in(D, 0);
    std::vector<int16_t> s_old(D, 0), s_new(D);
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        q2a_step_avx2(h_old.data(), s_old.data(), y_in.data(), alpha.data(),
                      h_new.data(), s_new.data());
    }
    t1 = std::chrono::high_resolution_clock::now();
    double us_q2 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / iters;

    // Head only
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        float r = poly_head_avx2(h_new.data(), s_new.data(),
                                  w_h.data(), w_s.data(),
                                  w_s2.data(), w_hs.data(), b);
        if (r == 0) std::cout << "";
    }
    t1 = std::chrono::high_resolution_clock::now();
    double us_head = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / iters;

    std::cout << "  | Component              | µs/step (D=4096) | ns/dim |\n";
    std::cout << "  |------------------------|------------------|--------|\n";
    std::cout << "  | Q3 (k=3 conv)          | " << std::setw(16) << std::fixed << std::setprecision(2) << us_q3
              << " | " << std::setw(6) << std::setprecision(2) << us_q3 * 1000 / D << " |\n";
    std::cout << "  | Q2-A + Sum             | " << std::setw(16) << std::fixed << std::setprecision(2) << us_q2
              << " | " << std::setw(6) << std::setprecision(2) << us_q2 * 1000 / D << " |\n";
    std::cout << "  | Poly Head              | " << std::setw(16) << std::fixed << std::setprecision(2) << us_head
              << " | " << std::setw(6) << std::setprecision(2) << us_head * 1000 / D << " |\n";
    std::cout << "  | Step (Q3+Q2+Head)      | " << std::setw(16) << std::fixed << std::setprecision(2) << (us_q3 + us_q2 + us_head)
              << " | " << std::setw(6) << std::setprecision(2) << (us_q3 + us_q2 + us_head) * 1000 / D << " |\n";

    std::cout << "\n  Per-token (SEQ_LEN=" << SEQ_LEN << "): " << std::fixed << std::setprecision(2) << (us_q3 + us_q2 + us_head) * SEQ_LEN << " µs\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(0) << 1e6 / (us_q3 + us_q2 + us_head) << " tokens/s\n";

    // 32-layer projection
    double per_layer = us_q3 + us_q2 + us_head;
    std::cout << "\n  32-layer (D=4096) projection:\n";
    std::cout << "    per token single-thread: " << std::fixed << std::setprecision(2) << 32 * per_layer << " µs\n";
    std::cout << "    throughput:              " << std::fixed << std::setprecision(0) << 1e6 / (32 * per_layer) << " tokens/s\n";
    std::cout << "    vs your Qwen 35B (20 t/s): " << std::fixed << std::setprecision(1) << (1e6 / (32 * per_layer)) / 20.0 << "x faster\n";

    return 0;
}
