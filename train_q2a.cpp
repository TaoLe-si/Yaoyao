// train_q2a.cpp
// Q2-A: integer state, learned per-dim forget rate alpha.

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
const int EPOCHS = 80;
const int N_TRAIN = 200;
float LR = 0.01f;

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

float train_seq(const std::vector<int>& tokens, int label,
                const float* alpha, std::vector<float>& alpha_grad,
                std::vector<float>& w, float& b) {

    SeqState s = forward_seq(tokens, alpha);

    float logit = linear_forward(s.hs.back().data(), w.data(), b);
    float p0 = sigmoid_2x(logit);
    float p1 = 1.0f - p0;

    float prob_true = (label == 0) ? p0 : p1;
    float loss = -std::log(std::max(prob_true, 1e-7f));

    float d_logit = (label == 0) ? (-2.0f * (1.0f - p0)) : (2.0f * p0);

    std::vector<float> d_h(D);
    for (int d = 0; d < D; ++d) {
        d_h[d] = d_logit * w[d];
        w[d] -= LR * d_logit * (float)s.hs.back()[d];
    }
    b -= LR * d_logit;

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
    std::cout << "  Q2-A: integer state + learned per-dim forget rate\n";
    std::cout << "================================================================\n\n";

    std::cout << "  Task: 16-token sequence classification\n";
    std::cout << "         majority-A -> label 0,  majority-B -> label 1\n";
    std::cout << "  D = " << D << ", h range = [-" << H_VAL << ", +" << H_VAL << "]\n";
    std::cout << "  Working set: ~" << (D * 9 / 1024) << " KB (all in L1)\n\n";

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

    std::vector<float> alpha(D, 0.9f);
    std::vector<float> alpha_grad(D, 0.0f);
    std::vector<float> w(D, 0.0f);
    float b = 0.0f;
    std::normal_distribution<float> nd(0.0f, 0.01f);
    for (auto& wi : w) wi = nd(rng);

    int correct = 0;
    for (auto& p : data) {
        if (predict(p.first, alpha.data(), w.data(), b) == p.second) correct++;
    }
    std::cout << "  Initial acc: " << correct << " / " << data.size()
              << " (" << correct * 100 / data.size() << "%)\n\n";

    std::cout << "  Epoch | Avg Loss | Accuracy\n";
    std::cout << "  ------+----------+---------\n";
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(data.begin(), data.end(), rng);
        float total_loss = 0;
        for (auto& p : data) {
            std::fill(alpha_grad.begin(), alpha_grad.end(), 0.0f);
            float loss = train_seq(p.first, p.second, alpha.data(), alpha_grad, w, b);
            total_loss += loss;
            for (int d = 0; d < D; ++d) {
                alpha[d] -= LR * alpha_grad[d];
                if (alpha[d] < 0.05f) alpha[d] = 0.05f;
                if (alpha[d] > 0.99f) alpha[d] = 0.99f;
            }
        }
        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            int c = 0;
            for (auto& pp : data) {
                if (predict(pp.first, alpha.data(), w.data(), b) == pp.second) c++;
            }
            std::cout << "  " << std::setw(5) << (epoch + 1) << " | "
                      << std::fixed << std::setprecision(4) << std::setw(8)
                      << total_loss / data.size() << " | "
                      << c << "/" << data.size()
                      << " (" << c * 100 / data.size() << "%)\n";
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
              << " (" << cnt_high * 100.0 / D << "%)  <- strongly remembers\n";
    std::cout << "    Mid  (0.5-0.85): " << cnt_mid << " / " << D << "\n";
    std::cout << "    Low  (<0.5):  " << cnt_low << " / " << D
              << " (" << cnt_low * 100.0 / D << "%)  <- mostly overwrites\n";

    std::cout << "\n  Alpha for the 3 informative dims:\n";
    std::cout << "    alpha[0]   = " << std::setprecision(3) << alpha[0]
              << "  (sees +3 for A, -3 for B)\n";
    std::cout << "    alpha[1]   = " << alpha[1]
              << "  (sees -3 for A, +3 for B)\n";
    std::cout << "    alpha[2]   = " << alpha[2]
              << "  (sees +1 for A, -1 for B)\n";
    std::cout << "    alpha[100] = " << alpha[100] << "  (irrelevant dim, always 0)\n";

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
    std::cout << "    (working set ~20 KB, all in L1)\n";

    std::cout << "\n================================================================\n";
    return 0;
}
