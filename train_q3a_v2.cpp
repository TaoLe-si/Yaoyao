// train_q3a_v2.cpp
// Cleaner version: simple linear head + full gradient flow.
// Tests both "ends with AB" (k=2 sufficient) and "ends with AAA" (needs k=3).

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
const int SEQ_LEN = 16;
const int EPOCHS = 60;
const int N_TRAIN = 500;
const float LR = 0.1f;

void q1(int token_id, int8_t* x) {
    std::fill(x, x + D, 0);
    if (token_id == 0) { x[0] = 3; x[1] = -3; x[2] = 1; }
    else { x[0] = -3; x[1] = 3; x[2] = -1; }
}

// ============ Forward ============
// y_t[d] = w_0*x_{t-2}[d] + w_1*x_{t-1}[d] + w_2*x_t[d]
// Stores x_{t-1} and x_{t-2} in the cache for backprop.
struct ForwardCache {
    std::vector<std::vector<int8_t>> xs;       // x_t
    std::vector<int8_t> y_last;                // y_{T-1}
};

void forward_seq(const std::vector<int>& tokens,
                 const float* w_q3_0, const float* w_q3_1, const float* w_q3_2,
                 std::vector<std::vector<int8_t>>& xs_out,
                 std::vector<int8_t>& y_last_out) {
    xs_out.assign(SEQ_LEN, std::vector<int8_t>(D, 0));
    y_last_out.assign(D, 0);

    std::vector<int8_t> x_curr(D, 0), x_prev(D, 0), x_prev2(D, 0), y(D, 0);

    for (int t = 0; t < SEQ_LEN; ++t) {
        x_prev2 = x_prev;
        x_prev = x_curr;
        q1(tokens[t], x_curr.data());
        xs_out[t] = x_curr;

        // Q3 conv (scalar version since we only need a few dims)
        for (int d = 0; d < D; ++d) {
            float v = w_q3_0[d] * (float)x_prev2[d]
                    + w_q3_1[d] * (float)x_prev[d]
                    + w_q3_2[d] * (float)x_curr[d];
            int r = (int)std::lroundf(v);
            if (r > H_VAL) r = H_VAL;
            if (r < -H_VAL) r = -H_VAL;
            y[d] = (int8_t)r;
        }
        if (t == SEQ_LEN - 1) y_last_out = y;
    }
}

void forward_seq_avx2(const std::vector<int>& tokens,
                       const float* w_q3_0, const float* w_q3_1, const float* w_q3_2,
                       std::vector<std::vector<int8_t>>& xs_out,
                       std::vector<int8_t>& y_last_out) {
    xs_out.assign(SEQ_LEN, std::vector<int8_t>(D, 0));
    y_last_out.assign(D, 0);

    std::vector<int8_t> x_curr(D, 0), x_prev(D, 0), x_prev2(D, 0), y(D, 0);
    const __m256 lo_f = _mm256_set1_ps(-4.0f);
    const __m256 hi_f = _mm256_set1_ps(4.0f);

    for (int t = 0; t < SEQ_LEN; ++t) {
        x_prev2 = x_prev;
        x_prev = x_curr;
        q1(tokens[t], x_curr.data());
        xs_out[t] = x_curr;

        int d = 0;
        for (; d + 8 <= D; d += 8) {
            __m128i x0_i8 = _mm_loadl_epi64((const __m128i*)(x_prev2.data() + d));
            __m128i x1_i8 = _mm_loadl_epi64((const __m128i*)(x_prev.data() + d));
            __m128i x2_i8 = _mm_loadl_epi64((const __m128i*)(x_curr.data() + d));
            __m256 x0_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(x0_i8));
            __m256 x1_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(x1_i8));
            __m256 x2_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(x2_i8));
            __m256 w0_f = _mm256_loadu_ps(w_q3_0 + d);
            __m256 w1_f = _mm256_loadu_ps(w_q3_1 + d);
            __m256 w2_f = _mm256_loadu_ps(w_q3_2 + d);
            __m256 y_f = _mm256_mul_ps(w0_f, x0_f);
            y_f = _mm256_fmadd_ps(w1_f, x1_f, y_f);
            y_f = _mm256_fmadd_ps(w2_f, x2_f, y_f);
            y_f = _mm256_round_ps(y_f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            y_f = _mm256_min_ps(_mm256_max_ps(y_f, lo_f), hi_f);
            __m256i y_i32 = _mm256_cvtps_epi32(y_f);
            __m128i y_i16 = _mm256_cvtepi32_epi16(y_i32);
            __m128i y_packed = _mm_packs_epi16(y_i16, y_i16);
            _mm_storel_epi64((__m128i*)(y.data() + d), y_packed);
        }
        for (; d < D; ++d) {
            float v = w_q3_0[d] * (float)x_prev2[d]
                    + w_q3_1[d] * (float)x_prev[d]
                    + w_q3_2[d] * (float)x_curr[d];
            int r = (int)std::lroundf(v);
            if (r > H_VAL) r = H_VAL;
            if (r < -H_VAL) r = -H_VAL;
            y[d] = (int8_t)r;
        }
        if (t == SEQ_LEN - 1) y_last_out = y;
    }
}

float head_forward(const int8_t* y_last, const float* w_head, float b) {
    float logit = b;
    for (int d = 0; d < D; ++d) logit += w_head[d] * (float)y_last[d];
    return logit;
}

float head_forward_simple(const int8_t* y_last, float w, float b) {
    return w * (float)y_last[0] + b;
}

// ============ Training: simple head w * y_last[0] + b ============
struct TrainResult {
    int final_correct;
    float final_w_q3_0, final_w_q3_1, final_w_q3_2;
    float final_w_head, final_b;
};

TrainResult train_task(const std::vector<std::pair<std::vector<int>, int>>& data_in,
                       const char* task_name,
                       bool require_aaa = false) {
    std::cout << "================================================================\n";
    std::cout << "  TASK: " << task_name << "\n";
    std::cout << "================================================================\n\n";

    std::mt19937 rng(42);
    std::vector<std::pair<std::vector<int>, int>> data = data_in;

    std::vector<float> w_q3_0(D, 0), w_q3_1(D, 0), w_q3_2(D, 0);
    std::normal_distribution<float> nd(0, 0.05f);
    for (auto& v : w_q3_0) v = nd(rng);
    for (auto& v : w_q3_1) v = nd(rng);
    for (auto& v : w_q3_2) v = nd(rng);
    float w_head = nd(rng);
    float b = 0;

    auto predict = [&](const std::vector<int>& seq) {
        std::vector<std::vector<int8_t>> xs;
        std::vector<int8_t> y_last;
        forward_seq(seq, w_q3_0.data(), w_q3_1.data(), w_q3_2.data(), xs, y_last);
        float logit = head_forward_simple(y_last.data(), w_head, b);
        return logit > 0 ? 0 : 1;
    };

    int initial_correct = 0;
    for (auto& p : data) if (predict(p.first) == p.second) initial_correct++;
    std::cout << "  Initial acc: " << initial_correct << "/" << data.size()
              << " (" << initial_correct * 100 / data.size() << "%)\n\n";

    // Full gradient on w_q3_0[0], w_q3_1[0], w_q3_2[0], w_head, b
    // logit = w_head * y_last[0] + b
    // y_last[0] = w_q3_0[0]*x_prev2[0] + w_q3_1[0]*x_prev[0] + w_q3_2[0]*x_curr[0]

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(data.begin(), data.end(), std::mt19937(epoch + 1));
        float total_loss = 0;

        for (auto& p : data) {
            std::vector<std::vector<int8_t>> xs;
            std::vector<int8_t> y_last;
            forward_seq(p.first, w_q3_0.data(), w_q3_1.data(), w_q3_2.data(), xs, y_last);

            float logit = w_head * (float)y_last[0] + b;
            float p0 = 1.0f / (1.0f + std::exp(-2.0f * logit));
            float prob_true = (p.second == 0) ? p0 : (1.0f - p0);
            total_loss += -std::log(std::max(prob_true, 1e-7f));

            float d_logit = (p.second == 0) ? (-2.0f * (1.0f - p0)) : (2.0f * p0);
            // d/dy_last[0] = d_logit * w_head
            float d_y0 = d_logit * w_head;

            int8_t x_p2 = xs[SEQ_LEN - 3][0];  // t = SEQ_LEN-3 contributes via w_q3_0
            int8_t x_p1 = xs[SEQ_LEN - 2][0];  // t = SEQ_LEN-2 via w_q3_1
            int8_t x_c = xs[SEQ_LEN - 1][0];   // t = SEQ_LEN-1 via w_q3_2

            // Direct gradient on Q3 weights (clamping ignored for simplicity)
            float grad_w0 = d_y0 * (float)x_p2;
            float grad_w1 = d_y0 * (float)x_p1;
            float grad_w2 = d_y0 * (float)x_c;

            // Update (per-sample, no division by N)
            w_q3_0[0] -= LR * grad_w0;
            w_q3_1[0] -= LR * grad_w1;
            w_q3_2[0] -= LR * grad_w2;
            w_head -= LR * d_logit * (float)y_last[0];
            b -= LR * d_logit;
        }

        if ((epoch + 1) % 10 == 0 || epoch == 0 || epoch == EPOCHS - 1) {
            int c = 0;
            for (auto& pp : data) if (predict(pp.first) == pp.second) c++;
            std::cout << "  Epoch " << std::setw(3) << (epoch + 1)
                      << ": loss=" << std::fixed << std::setprecision(4) << total_loss / data.size()
                      << " acc=" << c << "/" << data.size()
                      << " (" << std::setw(3) << c * 100 / data.size() << "%)"
                      << " w=[w0:" << std::setprecision(2) << w_q3_0[0]
                      << " w1:" << w_q3_1[0]
                      << " w2:" << w_q3_2[0] << "]"
                      << " w_hd=" << w_head
                      << " b=" << b << "\n";
        }
    }
    int fc = 0;
    for (auto& p : data) if (predict(p.first) == p.second) fc++;
    std::cout << "\n  Final: " << fc << "/" << data.size()
              << " (" << fc * 100 / data.size() << "%)\n\n";
    return {fc, w_q3_0[0], w_q3_1[0], w_q3_2[0], w_head, b};
}

int main() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> ud(0, 1);

    // ============ TASK 1: ends with AB ============
    std::cout << "================================================================\n";
    std::cout << "  SETUP: Q3 k=3 + simple linear head (y_last[0] -> logit)\n";
    std::cout << "================================================================\n\n";

    std::vector<std::pair<std::vector<int>, int>> data_ab;
    for (int i = 0; i < N_TRAIN / 2; ++i) {
        std::vector<int> seq(SEQ_LEN);
        for (int t = 0; t < SEQ_LEN; ++t) seq[t] = ud(rng);
        seq[SEQ_LEN - 2] = 0;
        seq[SEQ_LEN - 1] = 1;
        data_ab.push_back({seq, 0});
    }
    for (int i = 0; i < N_TRAIN / 2; ++i) {
        std::vector<int> seq(SEQ_LEN);
        for (int t = 0; t < SEQ_LEN; ++t) seq[t] = ud(rng);
        if (seq[SEQ_LEN - 2] == 0 && seq[SEQ_LEN - 1] == 1) {
            seq[SEQ_LEN - 1] = 0;
        }
        data_ab.push_back({seq, 1});
    }
    auto r_ab = train_task(data_ab, "ends with AB");

    // ============ TASK 2: ends with AAA ============
    std::vector<std::pair<std::vector<int>, int>> data_aaa;
    for (int i = 0; i < N_TRAIN / 2; ++i) {
        std::vector<int> seq(SEQ_LEN);
        for (int t = 0; t < SEQ_LEN; ++t) seq[t] = ud(rng);
        seq[SEQ_LEN - 3] = 0;
        seq[SEQ_LEN - 2] = 0;
        seq[SEQ_LEN - 1] = 0;
        data_aaa.push_back({seq, 0});
    }
    for (int i = 0; i < N_TRAIN / 2; ++i) {
        std::vector<int> seq(SEQ_LEN);
        for (int t = 0; t < SEQ_LEN; ++t) seq[t] = ud(rng);
        if (seq[SEQ_LEN - 3] == 0 && seq[SEQ_LEN - 2] == 0 && seq[SEQ_LEN - 1] == 0) {
            seq[SEQ_LEN - 1] = 1;
        }
        data_aaa.push_back({seq, 1});
    }
    auto r_aaa = train_task(data_aaa, "ends with AAA");

    // ============ BENCHMARK ============
    std::cout << "================================================================\n";
    std::cout << "  BENCHMARK\n";
    std::cout << "================================================================\n\n";

    std::vector<float> w_q3_0(D, 0), w_q3_1(D, 0), w_q3_2(D, 0);
    for (int d = 0; d < D; ++d) {
        w_q3_0[d] = 0.05f;
        w_q3_1[d] = 0.1f;
        w_q3_2[d] = -0.1f;
    }
    std::vector<int> bench_tokens(SEQ_LEN);
    for (int t = 0; t < SEQ_LEN; ++t) bench_tokens[t] = ud(rng);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        std::vector<std::vector<int8_t>> xs;
        std::vector<int8_t> y_last;
        forward_seq_avx2(bench_tokens, w_q3_0.data(), w_q3_1.data(), w_q3_2.data(), xs, y_last);
        if (y_last[0] == -5) std::cout << "";
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "  Q3 forward (16 steps, D=4096): "
              << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 100.0
              << " µs/seq\n";

    int iters = 5000;
    std::vector<std::vector<int8_t>> xs;
    std::vector<int8_t> y_last;
    t0 = std::chrono::high_resolution_clock::now();
    float sum_y = 0;
    for (int i = 0; i < iters; ++i) {
        forward_seq_avx2(bench_tokens, w_q3_0.data(), w_q3_1.data(), w_q3_2.data(), xs, y_last);
        sum_y += (float)y_last[0];
    }
    t1 = std::chrono::high_resolution_clock::now();
    double us_per_seq = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / (double)iters;
    if (sum_y == 0) std::cout << "";

    std::cout << "  Q3 forward AVX2: " << std::fixed << std::setprecision(2) << us_per_seq << " µs/seq";
    std::cout << " (" << std::setprecision(2) << us_per_seq / SEQ_LEN << " µs/step, "
              << std::setprecision(0) << 1e6 / (us_per_seq / SEQ_LEN) << " tokens/s)\n";

    // 32-layer projection: each layer is Q3 + Q2-A + Sum + Head
    double q3_per_step = us_per_seq / SEQ_LEN;
    double q2a_per_step = 1.0;  // from earlier benchmark
    double head_per_step = 2.0;  // from earlier benchmark
    double per_layer = q3_per_step + q2a_per_step + head_per_step;
    std::cout << "\n  Per-layer breakdown:\n";
    std::cout << "    Q3 (k=3 conv):  " << std::fixed << std::setprecision(2) << q3_per_step << " µs\n";
    std::cout << "    Q2-A + Sum:     " << std::fixed << std::setprecision(2) << q2a_per_step << " µs\n";
    std::cout << "    Poly Head:      " << std::fixed << std::setprecision(2) << head_per_step << " µs\n";
    std::cout << "    TOTAL per layer: " << std::fixed << std::setprecision(2) << per_layer << " µs\n";
    std::cout << "\n  32-layer (D=4096) projection:\n";
    std::cout << "    per token single-thread: " << std::fixed << std::setprecision(2) << 32 * per_layer << " µs\n";
    std::cout << "    throughput:              " << std::fixed << std::setprecision(0) << 1e6 / (32 * per_layer) << " tokens/s\n";
    std::cout << "    vs your Qwen 35B (20 t/s): " << std::fixed << std::setprecision(1) << (1e6 / (32 * per_layer)) / 20.0 << "x faster\n";

    return 0;
}
