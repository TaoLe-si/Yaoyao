// train_q2a_v3.cpp
// Q2-A training v3: alpha init at 0.99 (uniform weights), smaller LR_ALPHA.
//
// Theory check first:
//   For alpha = 0.99, weights w_j = (1-alpha)*alpha^(15-j)
//     w_0 = 0.0086, w_15 = 0.01   (ratio 0.86 - near uniform)
//   For alpha = 0.9, weights:
//     w_0 = 0.021, w_15 = 0.10    (ratio 0.21 - skewed)
//
// alpha = 0.99 makes h_t ~ uniform sum of all x_t -> order-invariant aggregation
//
// Compile: clang++ -O2 -std=c++17 -march=native -o train_q2a_v3.exe train_q2a_v3.cpp

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
const int EPOCHS = 150;
const int N_TRAIN = 500;
const float LR_W = 0.5f;
const float LR_ALPHA = 0.02f;  // smaller, to avoid oscillation
const float ALPHA_INIT = 0.99f;  // high - nearly uniform weights

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
    for (int i = 0; i <= SEQ_LEN; ++i) s.hs[i].assign(D, 0);
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
    std::cout << "  Q2-A v3: alpha init = 0.99, smaller LR_ALPHA\n";
    std::cout << "================================================================\n\n";

    std::cout << "  Task: 16-token majority-A vs majority-B\n";
    std::cout << "  Theory: alpha=0.99 makes weights near-uniform (ratio 0.86)\n";
    std::cout << "          -> h_t approximates uniform sum -> order-invariant\n";
    std::cout << "  D = " << D << ", ALPHA_INIT = " << ALPHA_INIT << "\n";
    std::cout << "  LR_W = " << LR_W << ", LR_ALPHA = " << LR_ALPHA << "\n";
    std::cout << "  N_TRAIN = " << N_TRAIN << ", EPOCHS = " << EPOCHS << "\n\n";

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

    std::vector<float> alpha(D, ALPHA_INIT);
    std::vector<float> w(D, 0.0f);
    float b = 0.0f;
    std::normal_distribution<float> nd(0.0f, 0.1f);
    for (auto& wi : w) wi = nd(rng);

    std::vector<float> w_grad(D), alpha_grad(D);
    float b_grad;

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

        for (auto& p : data) {
            float loss = compute_grad(p.first, p.second, alpha.data(), w.data(), b,
                                       w_grad, alpha_grad, b_grad);
            total_loss += loss;
        }

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

    std::cout << "  Learned alpha (3 informative dims + 1 irrelevant):\n";
    std::cout << "    alpha[0]   = " << std::setprecision(3) << alpha[0] << "\n";
    std::cout << "    alpha[1]   = " << alpha[1] << "\n";
    std::cout << "    alpha[2]   = " << alpha[2] << "\n";
    std::cout << "    alpha[100] = " << alpha[100] << "\n";

    std::cout << "\n  Learned w (3 informative dims + 1 irrelevant):\n";
    std::cout << "    w[0]   = " << std::setprecision(3) << w[0] << "\n";
    std::cout << "    w[1]   = " << w[1] << "\n";
    std::cout << "    w[2]   = " << w[2] << "\n";
    std::cout << "    w[100] = " << w[100] << "\n";
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
