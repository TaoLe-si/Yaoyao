// train_q2a_v2.cpp
// Q2-A training v2: full-batch SGD + tuned hyperparameters
//
// Changes from v1:
//   - Full-batch updates (accumulate gradients over N_TRAIN samples)
//   - Higher LR for both w and alpha
//   - Larger weight init
//   - More data, more epochs
//
// Forward (per step):
//   h_new[d] = round(alpha[d]*h_old[d] + (1-alpha[d])*x_t[d]), clamp [-4, +4]
//
// Task: 16-token sequence, majority-A (label 0) vs majority-B (label 1).
//
// Compile: clang++ -O2 -std=c++17 -march=native -o train_q2a_v2.exe train_q2a_v2.cpp

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>

const int D = 4096;
const int H_VAL = 4;
const int SEQ_LEN = 16;
const int EPOCHS = 200;
const int N_TRAIN = 500;
const float LR_W = 0.5f;
const float LR_ALPHA = 0.3f;
const float W_INIT_STD = 0.1f;

void q1_lookup(int token_id, int8_t* out_x) {
    std::fill(out_x, out_x + D, 0);
    if (token_id == 0) { out_x[0] = 3; out_x[1] = -3; out_x[2] = 1; }
    else { out_x[0] = -3; out_x[1] = 3; out_x[2] = -1; }
}

inline void q2a_step(const int8_t* h_old, const int8_t* x_t, const float* alpha,
                     int8_t* h_new) {
    for (int d = 0; d < D; ++d) {
        float v = alpha[d] * (float)h_old[d] + (1.0f - alpha[d]) * (float)x_t[d];
        int r = (int)std::lroundf(v);
        if (r > H_VAL) r = H_VAL;
        if (r < -H_VAL) r = -H_VAL;
        h_new[d] = (int8_t)r;
    }
}

float linear_forward(const int8_t* h, const float* w, float b) {
    float sum = b;
    for (int d = 0; d < D; ++d) sum += (float)h[d] * w[d];
    return sum;
}

inline float sigmoid_2x(float x) {
    return 1.0f / (1.0f + std::exp(-2.0f * x));
}

struct SeqState {
    std::vector<std::vector<int8_t>> hs;
    std::vector<std::vector<int8_t>> xs;
};

SeqState forward_seq(const std::vector<int>& tokens, const float* alpha) {
    SeqState s;
    s.xs.resize(SEQ_LEN);
    s.hs.resize(SEQ_LEN + 1);
    for (int i = 0; i <= SEQ_LEN; ++i) {
        s.hs[i].assign(D, 0);
    }
    std::vector<int8_t> xt(D);
    for (int t = 0; t < SEQ_LEN; ++t) {
        q1_lookup(tokens[t], xt.data());
        s.xs[t] = xt;
        q2a_step(s.hs[t].data(), xt.data(), alpha, s.hs[t + 1].data());
    }
    return s;
}

int predict(const std::vector<int>& tokens, const float* alpha,
            const float* w, float b) {
    SeqState s = forward_seq(tokens, alpha);
    float logit = linear_forward(s.hs.back().data(), w, b);
    return logit > 0 ? 0 : 1;
}

// Forward + backward, returns loss, accumulates gradients (no updates)
float compute_grad(const std::vector<int>& tokens, int label,
                   const float* alpha, const float* w, float b,
                   std::vector<float>& w_grad, std::vector<float>& alpha_grad,
                   float& b_grad) {
    SeqState s = forward_seq(tokens, alpha);

    float logit = linear_forward(s.hs.back().data(), w, b);
    float p0 = sigmoid_2x(logit);
    float prob_true = (label == 0) ? p0 : (1.0f - p0);
    float loss = -std::log(std::max(prob_true, 1e-7f));

    float d_logit = (label == 0) ? (-2.0f * (1.0f - p0)) : (2.0f * p0);

    std::vector<float> d_h(D);
    for (int d = 0; d < D; ++d) {
        d_h[d] = d_logit * w[d];
        w_grad[d] += d_logit * (float)s.hs.back()[d];
    }
    b_grad += d_logit;

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
    std::cout << "  Q2-A v2: integer state + learned per-dim forget rate\n";
    std::cout << "  Full-batch SGD with tuned hyperparameters\n";
    std::cout << "================================================================\n\n";

    std::cout << "  Task: 16-token sequence classification\n";
    std::cout << "    majority-A -> label 0,  majority-B -> label 1\n";
    std::cout << "  D = " << D << ", h range = [-" << H_VAL << ", +" << H_VAL << "]\n";
    std::cout << "  LR_W = " << LR_W << ", LR_ALPHA = " << LR_ALPHA << "\n";
    std::cout << "  N_TRAIN = " << N_TRAIN << ", EPOCHS = " << EPOCHS << "\n";
    std::cout << "  Working set: ~" << (D * 9 / 1024) << " KB (all in L1)\n\n";

    // Generate data
    std::mt19937 rng(42);
    std::vector<std::pair<std::vector<int>, int>> data;
    for (int i = 0; i < N_TRAIN; ++i) {
        std::vector<int> seq(SEQ_LEN);
        int count_a = 0;
        for (int t = 0; t < SEQ_LEN; ++t) {
            seq[t] = (rng() % 2 == 0) ? 0 : 1;
            if (seq[t] == 0) count_a++;
        }
        int label = (count_a > SEQ_LEN / 2) ? 0 : 1;
        if (count_a == SEQ_LEN / 2) { i--; continue; }
        data.push_back({seq, label});
    }

    // Count class balance
    int n_class0 = 0;
    for (auto& p : data) if (p.second == 0) n_class0++;
    std::cout << "  Class balance: " << n_class0 << " label-0, "
              << (int)data.size() - n_class0 << " label-1\n\n";

    // Init
    std::vector<float> alpha(D, 0.9f);
    std::vector<float> w(D, 0.0f);
    float b = 0.0f;
    std::normal_distribution<float> nd(0.0f, W_INIT_STD);
    for (auto& wi : w) wi = nd(rng);

    std::vector<float> w_grad(D), alpha_grad(D);
    float b_grad;

    // Initial accuracy
    int correct = 0;
    for (auto& p : data) {
        if (predict(p.first, alpha.data(), w.data(), b) == p.second) correct++;
    }
    std::cout << "  Initial acc: " << correct << " / " << data.size()
              << " (" << correct * 100 / data.size() << "%)\n\n";

    std::cout << "  Epoch | Avg Loss | Accuracy |   alpha[0]   alpha[1]   alpha[2]\n";
    std::cout << "  ------+----------+----------+---------------------------------\n";

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(data.begin(), data.end(), rng);
        std::fill(w_grad.begin(), w_grad.end(), 0.0f);
        std::fill(alpha_grad.begin(), alpha_grad.end(), 0.0f);
        b_grad = 0;
        float total_loss = 0;

        // Full-batch: accumulate gradients over all samples
        for (auto& p : data) {
            float loss = compute_grad(p.first, p.second, alpha.data(), w.data(), b,
                                       w_grad, alpha_grad, b_grad);
            total_loss += loss;
        }

        // Apply updates
        for (int d = 0; d < D; ++d) {
            w[d] -= LR_W * w_grad[d] / N_TRAIN;
            alpha[d] -= LR_ALPHA * alpha_grad[d] / N_TRAIN;
            if (alpha[d] < 0.05f) alpha[d] = 0.05f;
            if (alpha[d] > 0.99f) alpha[d] = 0.99f;
        }
        b -= LR_W * b_grad / N_TRAIN;

        if ((epoch + 1) % 5 == 0 || epoch == 0 || epoch == EPOCHS - 1) {
            int c = 0;
            for (auto& pp : data) {
                if (predict(pp.first, alpha.data(), w.data(), b) == pp.second) c++;
            }
            std::cout << "  " << std::setw(5) << (epoch + 1) << " | "
                      << std::fixed << std::setprecision(4) << std::setw(8)
                      << total_loss / data.size() << " | "
                      << c << "/" << data.size()
                      << " (" << std::setw(3) << c * 100 / data.size() << "%) | "
                      << std::setprecision(3)
                      << std::setw(8) << alpha[0] << " "
                      << std::setw(8) << alpha[1] << " "
                      << std::setw(8) << alpha[2] << "\n";
        }
    }

    int final_c = 0;
    for (auto& p : data) {
        if (predict(p.first, alpha.data(), w.data(), b) == p.second) final_c++;
    }
    std::cout << "\n  Final acc: " << final_c << " / " << data.size()
              << " (" << final_c * 100 / data.size() << "%)\n\n";

    std::cout << "  Learned alpha distribution:\n";
    double sum = 0; float mn = 1.0f, mx = 0.0f;
    int cnt_high = 0, cnt_mid = 0, cnt_low = 0;
    for (float a : alpha) {
        sum += a;
        if (a < mn) mn = a;
        if (a > mx) mx = a;
        if      (a > 0.85f) cnt_high++;
        else if (a < 0.5f)  cnt_low++;
        else                cnt_mid++;
    }
    std::cout << "    Mean:        " << std::fixed << std::setprecision(3) << sum / D << "\n";
    std::cout << "    Range:       [" << mn << ", " << mx << "]\n";
    std::cout << "    High (>0.85): " << cnt_high << " / " << D
              << " (" << cnt_high * 100.0 / D << "%)\n";
    std::cout << "    Mid  (0.5-0.85): " << cnt_mid << " / " << D << "\n";
    std::cout << "    Low  (<0.5):  " << cnt_low << " / " << D
              << " (" << cnt_low * 100.0 / D << "%)\n";

    std::cout << "\n  Alpha for the 3 informative dims:\n";
    std::cout << "    alpha[0]   = " << std::setprecision(3) << alpha[0]
              << "  (sees +3 for A, -3 for B)\n";
    std::cout << "    alpha[1]   = " << alpha[1]
              << "  (sees -3 for A, +3 for B)\n";
    std::cout << "    alpha[2]   = " << alpha[2]
              << "  (sees +1 for A, -1 for B)\n";
    std::cout << "    alpha[100] = " << alpha[100] << "  (irrelevant dim)\n";

    // Show learned w for the 3 informative dims
    std::cout << "\n  Learned w for the 3 informative dims:\n";
    std::cout << "    w[0]   = " << std::setprecision(3) << w[0]
              << "  (multiplies h[0])\n";
    std::cout << "    w[1]   = " << w[1] << "  (multiplies h[1])\n";
    std::cout << "    w[2]   = " << w[2] << "  (multiplies h[2])\n";
    std::cout << "    w[100] = " << w[100] << "  (irrelevant dim, should stay small)\n";
    std::cout << "    b      = " << b << "\n";

    std::cout << "\n  Per-step timing (Q2-A forward, 10000 iters):\n";
    std::vector<int8_t> h_old(D, 0), h_new(D), x_t(D);
    q1_lookup(0, x_t.data());

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; ++i) {
        q2a_step(h_old.data(), x_t.data(), alpha.data(), h_new.data());
        std::swap(h_old, h_new);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    std::cout << "    " << std::setprecision(2) << us / 10000 << " us / step\n";
    std::cout << "    ~" << (long long)(1e6 / (us / 10000)) << " steps/sec\n";

    std::cout << "\n================================================================\n";
    return 0;
}
