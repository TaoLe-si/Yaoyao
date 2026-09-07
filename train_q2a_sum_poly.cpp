// train_q2a_sum_poly.cpp
// Q2-A + sum channel + POLYNOMIAL FEATURES in head.
// Features: h, s, s^2, h*s. Adds nonlinearity without MLP.

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>

const int D = 4096;
const int H_VAL = 4;
const int S_MAX = 64;
const int SEQ_LEN = 16;
const int EPOCHS = 100;
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

struct SeqState {
    std::vector<std::vector<int8_t>> hs;
    std::vector<std::vector<int16_t>> ss;
    std::vector<std::vector<int8_t>> xs;
};

SeqState forward_seq(const std::vector<int>& tokens, const float* alpha) {
    SeqState s;
    s.hs.resize(SEQ_LEN + 1); s.ss.resize(SEQ_LEN + 1); s.xs.resize(SEQ_LEN);
    for (int i = 0; i <= SEQ_LEN; ++i) { s.hs[i].assign(D, 0); s.ss[i].assign(D, 0); }
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

// Polynomial features: [h, s, s^2, h*s]
// Compute logit = sum_d (w_h[d]*h[d] + w_s[d]*s[d] + w_s2[d]*s[d]^2 + w_hs[d]*h[d]*s[d]) + b
inline float poly_forward(const int8_t* h, const int16_t* s,
                         const float* w_h, const float* w_s,
                         const float* w_s2, const float* w_hs, float b) {
    float sum = b;
    for (int d = 0; d < D; ++d) {
        float hd = (float)h[d];
        float sd = (float)s[d];
        sum += w_h[d] * hd + w_s[d] * sd + w_s2[d] * sd * sd + w_hs[d] * hd * sd;
    }
    return sum;
}

inline float sigmoid_2x(float x) {
    return 1.0f / (1.0f + std::exp(-2.0f * x));
}

int predict(const std::vector<int>& tokens, const float* alpha,
            const float* w_h, const float* w_s, const float* w_s2, const float* w_hs,
            float b) {
    SeqState s = forward_seq(tokens, alpha);
    float logit = poly_forward(s.hs.back().data(), s.ss.back().data(),
                                w_h, w_s, w_s2, w_hs, b);
    return logit > 0 ? 0 : 1;
}

float compute_grad(const std::vector<int>& tokens, int label,
                   const float* alpha,
                   const float* w_h, const float* w_s, const float* w_s2, const float* w_hs,
                   float b,
                   std::vector<float>& w_h_grad, std::vector<float>& w_s_grad,
                   std::vector<float>& w_s2_grad, std::vector<float>& w_hs_grad,
                   std::vector<float>& alpha_grad, float& b_grad) {
    SeqState s = forward_seq(tokens, alpha);

    float logit = poly_forward(s.hs.back().data(), s.ss.back().data(),
                                w_h, w_s, w_s2, w_hs, b);
    float p0 = sigmoid_2x(logit);
    float prob_true = (label == 0) ? p0 : (1.0f - p0);
    float loss = -std::log(std::max(prob_true, 1e-7f));

    float d_logit = (label == 0) ? (-2.0f * (1.0f - p0)) : (2.0f * p0);

    std::vector<float> d_h(D);
    for (int d = 0; d < D; ++d) {
        float hd = (float)s.hs.back()[d];
        float sd = (float)s.ss.back()[d];
        // logit = w_h*h + w_s*s + w_s2*s^2 + w_hs*h*s + b
        // d_logit/d_h = w_h + w_hs*s
        // d_logit/d_s = w_s + 2*w_s2*s + w_hs*h
        d_h[d] = d_logit * (w_h[d] + w_hs[d] * sd);
        w_h_grad[d] += d_logit * hd;
        w_s_grad[d] += d_logit * sd;
        w_s2_grad[d] += d_logit * sd * sd;
        w_hs_grad[d] += d_logit * hd * sd;
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

struct TrainResult {
    int correct;
};

TrainResult train_and_eval(std::vector<std::pair<std::vector<int>, int>>& data,
                            std::vector<float>& alpha,
                            std::vector<float>& w_h, std::vector<float>& w_s,
                            std::vector<float>& w_s2, std::vector<float>& w_hs,
                            float& b, const char* task_name) {
    std::vector<float> w_h_grad(D, 0), w_s_grad(D, 0);
    std::vector<float> w_s2_grad(D, 0), w_hs_grad(D, 0);
    std::vector<float> alpha_grad(D, 0);
    float b_grad = 0;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(data.begin(), data.end(), std::mt19937(epoch + 1));
        std::fill(w_h_grad.begin(), w_h_grad.end(), 0);
        std::fill(w_s_grad.begin(), w_s_grad.end(), 0);
        std::fill(w_s2_grad.begin(), w_s2_grad.end(), 0);
        std::fill(w_hs_grad.begin(), w_hs_grad.end(), 0);
        std::fill(alpha_grad.begin(), alpha_grad.end(), 0);
        b_grad = 0;
        float total_loss = 0;
        for (auto& p : data) {
            float loss = compute_grad(p.first, p.second, alpha.data(),
                                       w_h.data(), w_s.data(), w_s2.data(), w_hs.data(), b,
                                       w_h_grad, w_s_grad, w_s2_grad, w_hs_grad,
                                       alpha_grad, b_grad);
            total_loss += loss;
        }
        for (int d = 0; d < D; ++d) {
            w_h[d] -= LR_W * w_h_grad[d] / N_TRAIN;
            w_s[d] -= LR_W * w_s_grad[d] / N_TRAIN;
            w_s2[d] -= LR_W * w_s2_grad[d] / N_TRAIN;
            w_hs[d] -= LR_W * w_hs_grad[d] / N_TRAIN;
            alpha[d] -= LR_ALPHA * alpha_grad[d] / N_TRAIN;
            if (alpha[d] < 0.05f) alpha[d] = 0.05f;
            if (alpha[d] > 0.99f) alpha[d] = 0.99f;
        }
        b -= LR_W * b_grad / N_TRAIN;

        if ((epoch + 1) % 20 == 0 || epoch == 0 || epoch == EPOCHS - 1) {
            int c = 0;
            for (auto& pp : data) {
                if (predict(pp.first, alpha.data(), w_h.data(), w_s.data(),
                            w_s2.data(), w_hs.data(), b) == pp.second) c++;
            }
            std::cout << "  " << task_name << " Epoch " << std::setw(3) << (epoch + 1)
                      << ": loss=" << std::fixed << std::setprecision(4) << total_loss / data.size()
                      << " acc=" << c << "/" << data.size()
                      << " (" << std::setw(3) << c * 100 / data.size() << "%)\n";
        }
    }
    int final_c = 0;
    for (auto& p : data) {
        if (predict(p.first, alpha.data(), w_h.data(), w_s.data(),
                    w_s2.data(), w_hs.data(), b) == p.second) final_c++;
    }
    return {final_c};
}

int main() {
    std::mt19937 rng(42);

    // ============ TASK A: Exact count ============
    std::cout << "================================================================\n";
    std::cout << "  TASK A: count_a = 8 (s=0) vs count_a in {4, 12} (s=±24)\n";
    std::cout << "================================================================\n\n";
    std::cout << "  Polynomial features: [h, s, s^2, h*s]\n";
    std::cout << "  s^2 is the KEY feature: 0 for count=8, 576 for count in {4,12}\n\n";

    std::vector<std::pair<std::vector<int>, int>> data_A;
    while ((int)data_A.size() < N_TRAIN) {
        std::vector<int> seq(SEQ_LEN);
        int count_a = 0;
        for (int t = 0; t < SEQ_LEN; ++t) {
            seq[t] = (rng() % 2 == 0) ? 0 : 1;
            if (seq[t] == 0) count_a++;
        }
        int label;
        if (count_a == 8) label = 0;
        else if (count_a == 4 || count_a == 12) label = 1;
        else continue;
        data_A.push_back({seq, label});
    }
    int label_A_0 = 0, label_A_1 = 0;
    for (auto& p : data_A) (p.second == 0) ? label_A_0++ : label_A_1++;
    std::cout << "  Data: " << data_A.size() << " sequences (label-0: "
              << label_A_0 << ", label-1: " << label_A_1 << ")\n\n";

    {
        std::vector<float> alpha(D, 0.99f);
        std::vector<float> w_h(D, 0.0f), w_s(D, 0.0f), w_s2(D, 0.0f), w_hs(D, 0.0f);
        float b = 0.0f;
        std::normal_distribution<float> nd(0.0f, 0.1f);
        for (auto& v : w_h) v = nd(rng);
        for (auto& v : w_s) v = nd(rng);
        for (auto& v : w_s2) v = nd(rng);
        for (auto& v : w_hs) v = nd(rng);
        auto r = train_and_eval(data_A, alpha, w_h, w_s, w_s2, w_hs, b, "TaskA");

        std::cout << "\n  Task A learned (key dims):\n";
        std::cout << "    w_s[0]   = " << std::setprecision(3) << w_s[0] << "\n";
        std::cout << "    w_s2[0]  = " << w_s2[0] << "  (must be NEGATIVE for |s| detection)\n";
        std::cout << "    w_hs[0]  = " << w_hs[0] << "\n";
        std::cout << "    b        = " << b << "\n\n";
    }

    // ============ TASK B: XOR ============
    std::cout << "================================================================\n";
    std::cout << "  TASK B: XOR(majority, last_token)\n";
    std::cout << "================================================================\n\n";
    std::cout << "  Polynomial features: [h, s, s^2, h*s]\n";
    std::cout << "  h*s is the KEY feature: positive when s,h same sign (label 0)\n";
    std::cout << "                          negative when s,h differ (label 1)\n\n";

    std::vector<std::pair<std::vector<int>, int>> data_B;
    for (int i = 0; i < N_TRAIN; ++i) {
        std::vector<int> seq(SEQ_LEN);
        int count_a = 0;
        for (int t = 0; t < SEQ_LEN; ++t) {
            seq[t] = (rng() % 2 == 0) ? 0 : 1;
            if (seq[t] == 0) count_a++;
        }
        if (count_a == SEQ_LEN / 2) { i--; continue; }
        int M = (count_a > SEQ_LEN / 2) ? 0 : 1;
        int L = (seq[SEQ_LEN - 1] == 0) ? 0 : 1;
        int label = M ^ L;
        data_B.push_back({seq, label});
    }
    int label_B_0 = 0, label_B_1 = 0;
    for (auto& p : data_B) (p.second == 0) ? label_B_0++ : label_B_1++;
    std::cout << "  Data: " << data_B.size() << " sequences (label-0: "
              << label_B_0 << ", label-1: " << label_B_1 << ")\n\n";

    {
        std::vector<float> alpha(D, 0.7f);
        std::vector<float> w_h(D, 0.0f), w_s(D, 0.0f), w_s2(D, 0.0f), w_hs(D, 0.0f);
        float b = 0.0f;
        std::normal_distribution<float> nd(0.0f, 0.1f);
        for (auto& v : w_h) v = nd(rng);
        for (auto& v : w_s) v = nd(rng);
        for (auto& v : w_s2) v = nd(rng);
        for (auto& v : w_hs) v = nd(rng);
        auto r = train_and_eval(data_B, alpha, w_h, w_s, w_s2, w_hs, b, "TaskB");

        std::cout << "\n  Task B learned (key dims):\n";
        std::cout << "    alpha[0] = " << std::setprecision(3) << alpha[0] << "\n";
        std::cout << "    w_s[0]   = " << w_s[0] << "\n";
        std::cout << "    w_h[0]   = " << w_h[0] << "\n";
        std::cout << "    w_hs[0]  = " << w_hs[0] << "  (should be POSITIVE for XOR)\n";
        std::cout << "    b        = " << b << "\n\n";
    }

    std::cout << "================================================================\n";
    return 0;
}
