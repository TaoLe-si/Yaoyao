// train_q2a_sum.cpp
// Q2-A + SUM channel: bypass the EMA order-sensitivity.
//
// Architecture:
//   h[d] in {-1, 0, +1} * H_VAL    (EMA state, bounded)
//   s[d] in Z (cumulative sum, soft-clamped)
//   alpha[d] in [0, 1]              (learned per-dim)
//
// Forward per step:
//   h_new[d] = clamp(alpha[d]·h_old[d] + (1-alpha[d])·x_t[d], ±H_VAL)
//   s_new[d] = clamp(s_old[d] + x_t[d], ±S_MAX)
//
// Linear head on concat(h, s).
//
// Task: 16-token MAJORITY (order-invariant).

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>

const int D = 4096;
const int H_VAL = 4;
const int S_MAX = 64;     // for 16-token x ∈ [-3,+3], sum ≤ 48
const int SEQ_LEN = 16;
const int EPOCHS = 80;
const int N_TRAIN = 500;
const float LR_W = 0.5f;
const float LR_ALPHA = 0.02f;

void q1_lookup(int token_id, int8_t* out_x) {
    std::fill(out_x, out_x + D, 0);
    if (token_id == 0) { out_x[0] = 3; out_x[1] = -3; out_x[2] = 1; }
    else { out_x[0] = -3; out_x[1] = 3; out_x[2] = -1; }
}

inline void q2a_step(const int8_t* h_old, const int16_t* s_old,
                     const int8_t* x_t, const float* alpha,
                     int8_t* h_new, int16_t* s_new) {
    for (int d = 0; d < D; ++d) {
        float v = alpha[d] * (float)h_old[d] + (1.0f - alpha[d]) * (float)x_t[d];
        int r = (int)std::lroundf(v);
        if (r > H_VAL) r = H_VAL;
        if (r < -H_VAL) r = -H_VAL;
        h_new[d] = (int8_t)r;

        int s = (int)s_old[d] + (int)x_t[d];
        if (s > S_MAX) s = S_MAX;
        if (s < -S_MAX) s = -S_MAX;
        s_new[d] = (int16_t)s;
    }
}

inline float linear_forward(const int8_t* h, const int16_t* s,
                            const float* w_h, const float* w_s, float b) {
    float sum = b;
    for (int d = 0; d < D; ++d) {
        sum += (float)h[d] * w_h[d] + (float)s[d] * w_s[d];
    }
    return sum;
}

inline float sigmoid_2x(float x) {
    return 1.0f / (1.0f + std::exp(-2.0f * x));
}

struct SeqState {
    std::vector<std::vector<int8_t>> hs;
    std::vector<std::vector<int16_t>> ss;
    std::vector<std::vector<int8_t>> xs;
};

SeqState forward_seq(const std::vector<int>& tokens, const float* alpha) {
    SeqState s;
    s.hs.resize(SEQ_LEN + 1);
    s.ss.resize(SEQ_LEN + 1);
    s.xs.resize(SEQ_LEN);
    for (int i = 0; i <= SEQ_LEN; ++i) {
        s.hs[i].assign(D, 0);
        s.ss[i].assign(D, 0);
    }
    for (int i = 0; i < SEQ_LEN; ++i) s.xs[i].assign(D, 0);
    std::vector<int8_t> xt(D);
    for (int t = 0; t < SEQ_LEN; ++t) {
        q1_lookup(tokens[t], xt.data());
        s.xs[t] = xt;
        q2a_step(s.hs[t].data(), s.ss[t].data(), xt.data(), alpha,
                 s.hs[t + 1].data(), s.ss[t + 1].data());
    }
    return s;
}

int predict(const std::vector<int>& tokens, const float* alpha,
            const float* w_h, const float* w_s, float b) {
    SeqState s = forward_seq(tokens, alpha);
    float logit = linear_forward(s.hs.back().data(), s.ss.back().data(),
                                  w_h, w_s, b);
    return logit > 0 ? 0 : 1;
}

float compute_grad(const std::vector<int>& tokens, int label,
                   const float* alpha,
                   const float* w_h, const float* w_s, float b,
                   std::vector<float>& w_h_grad, std::vector<float>& w_s_grad,
                   std::vector<float>& alpha_grad, float& b_grad) {
    SeqState s = forward_seq(tokens, alpha);

    float logit = linear_forward(s.hs.back().data(), s.ss.back().data(),
                                  w_h, w_s, b);
    float p0 = sigmoid_2x(logit);
    float prob_true = (label == 0) ? p0 : (1.0f - p0);
    float loss = -std::log(std::max(prob_true, 1e-7f));

    float d_logit = (label == 0) ? (-2.0f * (1.0f - p0)) : (2.0f * p0);

    // Initial gradients
    std::vector<float> d_h(D), d_s(D);
    for (int d = 0; d < D; ++d) {
        d_h[d] = d_logit * w_h[d];
        d_s[d] = d_logit * w_s[d];
        w_h_grad[d] += d_logit * (float)s.hs.back()[d];
        w_s_grad[d] += d_logit * (float)s.ss.back()[d];
    }
    b_grad += d_logit;

    // BPTT through Q2-A h channel
    for (int t = SEQ_LEN - 1; t >= 0; --t) {
        std::vector<float> d_h_old(D);
        for (int d = 0; d < D; ++d) {
            d_h_old[d] = alpha[d] * d_h[d];
            alpha_grad[d] += ((float)s.hs[t][d] - (float)s.xs[t][d]) * d_h[d];
        }
        d_h = std::move(d_h_old);
    }

    return loss;
}

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Q2-A + SUM channel: solve majority (order-invariant)\n";
    std::cout << "================================================================\n\n";

    std::cout << "  Architecture:\n";
    std::cout << "    h  (EMA,  bounded [-4, +4])  -- near-term\n";
    std::cout << "    s  (sum,  bounded [-64,+64]) -- global aggregate (ORDER-INVARIANT)\n";
    std::cout << "    alpha (learned)\n\n";

    std::cout << "  Linear head: logit = w_h.h + w_s.s + b\n";
    std::cout << "  The 's' channel makes order-invariant tasks solvable.\n\n";

    std::mt19937 rng(42);
    std::vector<std::pair<std::vector<int>, int>> data;
    int cnt_label_0 = 0, cnt_label_1 = 0;
    for (int i = 0; i < N_TRAIN; ++i) {
        std::vector<int> seq(SEQ_LEN);
        int count_a = 0;
        for (int t = 0; t < SEQ_LEN; ++t) {
            seq[t] = (rng() % 2 == 0) ? 0 : 1;
            if (seq[t] == 0) count_a++;
        }
        if (count_a == SEQ_LEN / 2) { i--; continue; }
        int label = (count_a > SEQ_LEN / 2) ? 0 : 1;
        if (label == 0) cnt_label_0++; else cnt_label_1++;
        data.push_back({seq, label});
    }
    std::cout << "  Data: " << data.size() << " sequences (" << cnt_label_0
              << " label-0, " << cnt_label_1 << " label-1)\n";
    std::cout << "  Task: majority A (label 0) vs majority B (label 1)\n\n";

    std::vector<float> alpha(D, 0.99f);   // near-uniform EMA for h channel
    std::vector<float> w_h(D, 0.0f), w_s(D, 0.0f);
    float b = 0.0f;
    std::normal_distribution<float> nd(0.0f, 0.1f);
    for (auto& v : w_h) v = nd(rng);
    for (auto& v : w_s) v = nd(rng);

    int correct = 0;
    for (auto& p : data) {
        if (predict(p.first, alpha.data(), w_h.data(), w_s.data(), b) == p.second)
            correct++;
    }
    std::cout << "  Initial acc: " << correct << "/" << data.size()
              << " (" << correct * 100 / data.size() << "%)\n";
    std::cout << "  (Expect ~50% since alpha=0.99 makes h small, head untrained)\n\n";

    std::vector<float> w_h_grad(D, 0), w_s_grad(D, 0), alpha_grad(D, 0);
    float b_grad = 0;

    std::cout << "  Epoch | Loss    | Acc (sum+ema)\n";
    std::cout << "  ------+---------+--------------\n";

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(data.begin(), data.end(), rng);
        std::fill(w_h_grad.begin(), w_h_grad.end(), 0);
        std::fill(w_s_grad.begin(), w_s_grad.end(), 0);
        std::fill(alpha_grad.begin(), alpha_grad.end(), 0);
        b_grad = 0;
        float total_loss = 0;

        for (auto& p : data) {
            float loss = compute_grad(p.first, p.second, alpha.data(),
                                       w_h.data(), w_s.data(), b,
                                       w_h_grad, w_s_grad, alpha_grad, b_grad);
            total_loss += loss;
        }

        for (int d = 0; d < D; ++d) {
            w_h[d] -= LR_W * w_h_grad[d] / N_TRAIN;
            w_s[d] -= LR_W * w_s_grad[d] / N_TRAIN;
            alpha[d] -= LR_ALPHA * alpha_grad[d] / N_TRAIN;
            if (alpha[d] < 0.05f) alpha[d] = 0.05f;
            if (alpha[d] > 0.99f) alpha[d] = 0.99f;
        }
        b -= LR_W * b_grad / N_TRAIN;

        if ((epoch + 1) % 5 == 0 || epoch == 0 || epoch == EPOCHS - 1) {
            int c = 0;
            for (auto& pp : data) {
                if (predict(pp.first, alpha.data(), w_h.data(), w_s.data(), b) == pp.second) c++;
            }
            std::cout << "  " << std::setw(5) << (epoch + 1) << " | "
                      << std::fixed << std::setprecision(4) << std::setw(7)
                      << total_loss / data.size() << " | "
                      << c << "/" << data.size()
                      << " (" << std::setw(3) << c * 100 / data.size() << "%)\n";
        }
    }

    int final_c = 0;
    for (auto& p : data) {
        if (predict(p.first, alpha.data(), w_h.data(), w_s.data(), b) == p.second)
            final_c++;
    }
    std::cout << "\n  Final acc: " << final_c << "/" << data.size()
              << " (" << final_c * 100 / data.size() << "%)\n\n";

    std::cout << "  Learned parameters (key dims):\n";
    std::cout << "    alpha[0]   = " << std::setprecision(3) << alpha[0] << "\n";
    std::cout << "    w_h[0]     = " << w_h[0] << "  (head weight on h)\n";
    std::cout << "    w_s[0]     = " << w_s[0] << "  (head weight on sum s)\n";
    std::cout << "    w_h[1]     = " << w_h[1] << "\n";
    std::cout << "    w_s[1]     = " << w_s[1] << "\n";
    std::cout << "    b          = " << b << "\n\n";

    // Look at what s[0] values look like
    SeqState s_ex = forward_seq(data[0].first, alpha.data());
    std::cout << "  Example s[0] trajectory for first sequence:\n";
    std::cout << "    Label = " << data[0].second << ", tokens = ";
    for (int t : data[0].first) std::cout << t << " ";
    std::cout << "\n    s[0] = ";
    for (int t = 0; t <= SEQ_LEN; ++t) std::cout << (int)s_ex.ss[t][0] << " ";
    std::cout << "\n    h[0] = ";
    for (int t = 0; t <= SEQ_LEN; ++t) std::cout << (int)s_ex.hs[t][0] << " ";
    std::cout << "\n\n";

    std::cout << "  Timing (inference, 10000 iters):\n";
    std::vector<int8_t> h_old(D, 0), h_new(D), x_t(D);
    std::vector<int16_t> s_old(D, 0), s_new(D);
    q1_lookup(0, x_t.data());
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; ++i) {
        q2a_step(h_old.data(), s_old.data(), x_t.data(), alpha.data(),
                 h_new.data(), s_new.data());
        std::swap(h_old, h_new);
        std::swap(s_old, s_new);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    std::cout << "    " << std::setprecision(2) << us / 10000 << " us / step\n";
    std::cout << "    (vs original Q2-A: ~10.75 us -- sum channel adds "
              << std::setprecision(2) << (us / 10000 - 10.75) << " us)\n\n";

    std::cout << "  Memory (per token):\n";
    std::cout << "    h:  " << D << " bytes (int8)\n";
    std::cout << "    s:  " << D * 2 << " bytes (int16)\n";
    std::cout << "    x_t: " << D << " bytes (int8)\n";
    std::cout << "    Total per-step state: " << (D + D * 2 + D) / 1024 << " KB\n";
    std::cout << "    (Still O(D), not O(t*D) -- no KV cache!)\n";

    std::cout << "\n================================================================\n";
    return 0;
}
