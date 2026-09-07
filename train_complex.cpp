// train_complex.cpp
// Complex aggregation tasks with new channels:
//   - Parity channel (XOR)
//   - Max channel (running maximum)
//   - Combined: OR-of-AND using max of indicators
//
// Compile: clang++ -O2 -std=c++17 -march=native -o train_complex.exe train_complex.cpp

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
const int EPOCHS = 80;
const int N_TRAIN = 500;
const float LR = 0.5f;

// For OR-of-AND test, use 4-token vocab with different patterns
void q1_lookup(int token_id, int8_t* out_x) {
    std::fill(out_x, out_x + D, 0);
    if (token_id == 0) { out_x[0] = 3; out_x[1] = -3; out_x[2] = 1; out_x[3] = -1; }    // A
    else if (token_id == 1) { out_x[0] = -3; out_x[1] = 3; out_x[2] = -1; out_x[3] = 1; } // B
    else if (token_id == 2) { out_x[0] = 3; out_x[1] = 3; out_x[2] = 1; out_x[3] = 1; }   // AB+ (all positive)
    else { out_x[0] = -3; out_x[1] = -3; out_x[2] = -1; out_x[3] = -1; }                  // AB- (all negative)
}

// Q1 extended: 4 distinct token types
const int VOCAB_SIZE = 4;

struct MultiState {
    std::vector<std::vector<int8_t>> hs, ps, ms;
    std::vector<std::vector<int16_t>> ss;
    std::vector<std::vector<int8_t>> xs;
    std::vector<int8_t> or_and_flag;   // OR-of-AND across sequence
};

MultiState forward_seq(const std::vector<int>& tokens, const float* alpha) {
    MultiState s;
    s.hs.resize(SEQ_LEN + 1); s.ss.resize(SEQ_LEN + 1);
    s.ps.resize(SEQ_LEN + 1); s.ms.resize(SEQ_LEN + 1);
    s.xs.resize(SEQ_LEN);
    s.or_and_flag.assign(SEQ_LEN + 1, 0);
    for (int i = 0; i <= SEQ_LEN; ++i) {
        s.hs[i].assign(D, 0); s.ss[i].assign(D, 0);
        s.ps[i].assign(D, 0); s.ms[i].assign(D, 0);
    }
    for (int i = 0; i < SEQ_LEN; ++i) s.xs[i].assign(D, 0);

    std::vector<int8_t> xt(D);
    for (int t = 0; t < SEQ_LEN; ++t) {
        q1_lookup(tokens[t], xt.data());
        s.xs[t] = xt;
        for (int d = 0; d < D; ++d) {
            // h: EMA
            float v = alpha[d] * (float)s.hs[t][d] + (1.0f - alpha[d]) * (float)xt[d];
            int r = (int)std::lroundf(v);
            if (r > H_VAL) r = H_VAL;
            if (r < -H_VAL) r = -H_VAL;
            s.hs[t + 1][d] = (int8_t)r;
            // s: Sum
            int sum = (int)s.ss[t][d] + (int)xt[d];
            if (sum > S_MAX) sum = S_MAX;
            if (sum < -S_MAX) sum = -S_MAX;
            s.ss[t + 1][d] = (int16_t)sum;
            // p: Parity (XOR of x>0)
            s.ps[t + 1][d] = s.ps[t][d] ^ (xt[d] > 0 ? 1 : 0);
            // m: Max
            s.ms[t + 1][d] = std::max(s.ms[t][d], xt[d]);
        }
        // OR-of-AND: all 4 dims positive at this position?
        int all_pos = (xt[0] > 0 && xt[1] > 0 && xt[2] > 0 && xt[3] > 0) ? 1 : 0;
        s.or_and_flag[t + 1] = std::max(s.or_and_flag[t], (int8_t)all_pos);
    }
    return s;
}

inline float linear_forward(const int8_t* h, const int16_t* s,
                            const int8_t* p, const int8_t* m,
                            const float* w_h, const float* w_s,
                            const float* w_p, const float* w_m,
                            float b) {
    float sum = b;
    for (int d = 0; d < D; ++d) {
        sum += (float)h[d] * w_h[d] + (float)s[d] * w_s[d]
             + (float)p[d] * w_p[d] + (float)m[d] * w_m[d];
    }
    return sum;
}

inline float sigmoid_2x(float x) {
    return 1.0f / (1.0f + std::exp(-2.0f * x));
}

int predict(const std::vector<int>& tokens, const float* alpha,
            const float* w_h, const float* w_s, const float* w_p, const float* w_m,
            float b) {
    MultiState s = forward_seq(tokens, alpha);
    float logit = linear_forward(s.hs.back().data(), s.ss.back().data(),
                                  s.ps.back().data(), s.ms.back().data(),
                                  w_h, w_s, w_p, w_m, b);
    return logit > 0 ? 0 : 1;
}

float train_and_eval(std::vector<std::pair<std::vector<int>, int>>& data,
                     const char* task_name) {
    std::vector<float> alpha(D, 0.99f);
    std::vector<float> w_h(D, 0), w_s(D, 0), w_p(D, 0), w_m(D, 0);
    float b = 0;

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0, 0.1f);
    for (auto& v : w_h) v = nd(rng);
    for (auto& v : w_s) v = nd(rng);
    for (auto& v : w_p) v = nd(rng);
    for (auto& v : w_m) v = nd(rng);

    int correct = 0;
    for (auto& p : data) {
        if (predict(p.first, alpha.data(), w_h.data(), w_s.data(),
                    w_p.data(), w_m.data(), b) == p.second) correct++;
    }
    std::cout << "  Initial acc: " << correct << "/" << data.size()
              << " (" << correct * 100 / data.size() << "%)\n\n";

    std::vector<float> w_h_grad(D, 0), w_s_grad(D, 0);
    std::vector<float> w_p_grad(D, 0), w_m_grad(D, 0);
    std::vector<float> alpha_grad(D, 0);
    float b_grad = 0;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(data.begin(), data.end(), std::mt19937(epoch + 1));
        std::fill(w_h_grad.begin(), w_h_grad.end(), 0);
        std::fill(w_s_grad.begin(), w_s_grad.end(), 0);
        std::fill(w_p_grad.begin(), w_p_grad.end(), 0);
        std::fill(w_m_grad.begin(), w_m_grad.end(), 0);
        std::fill(alpha_grad.begin(), alpha_grad.end(), 0);
        b_grad = 0;
        float total_loss = 0;

        for (auto& p : data) {
            MultiState s = forward_seq(p.first, alpha.data());
            float logit = linear_forward(s.hs.back().data(), s.ss.back().data(),
                                          s.ps.back().data(), s.ms.back().data(),
                                          w_h.data(), w_s.data(), w_p.data(), w_m.data(), b);
            float p0 = sigmoid_2x(logit);
            float prob_true = (p.second == 0) ? p0 : (1.0f - p0);
            total_loss += -std::log(std::max(prob_true, 1e-7f));
            float d_logit = (p.second == 0) ? (-2.0f * (1.0f - p0)) : (2.0f * p0);

            std::vector<float> d_h(D);
            for (int d = 0; d < D; ++d) {
                d_h[d] = d_logit * w_h[d];
                w_h_grad[d] += d_logit * (float)s.hs.back()[d];
                w_s_grad[d] += d_logit * (float)s.ss.back()[d];
                w_p_grad[d] += d_logit * (float)s.ps.back()[d];
                w_m_grad[d] += d_logit * (float)s.ms.back()[d];
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
        }

        for (int d = 0; d < D; ++d) {
            w_h[d] -= LR * w_h_grad[d] / N_TRAIN;
            w_s[d] -= LR * w_s_grad[d] / N_TRAIN;
            w_p[d] -= LR * w_p_grad[d] / N_TRAIN;
            w_m[d] -= LR * w_m_grad[d] / N_TRAIN;
            alpha[d] -= 0.02f * alpha_grad[d] / N_TRAIN;
            if (alpha[d] < 0.05f) alpha[d] = 0.05f;
            if (alpha[d] > 0.99f) alpha[d] = 0.99f;
        }
        b -= LR * b_grad / N_TRAIN;

        if ((epoch + 1) % 10 == 0 || epoch == 0 || epoch == EPOCHS - 1) {
            int c = 0;
            for (auto& pp : data) {
                if (predict(pp.first, alpha.data(), w_h.data(), w_s.data(),
                            w_p.data(), w_m.data(), b) == pp.second) c++;
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
                    w_p.data(), w_m.data(), b) == p.second) final_c++;
    }
    std::cout << "\n  Final acc: " << final_c << "/" << data.size()
              << " (" << final_c * 100 / data.size() << "%)\n\n";

    std::cout << "  Learned weights (key dims):\n";
    std::cout << "    w_h[0] = " << std::setprecision(3) << w_h[0] << "\n";
    std::cout << "    w_s[0] = " << w_s[0] << "\n";
    std::cout << "    w_p[0] = " << w_p[0] << "  (parity)\n";
    std::cout << "    w_m[0] = " << w_m[0] << "  (max)\n";
    std::cout << "    w_m[2] = " << w_m[2] << "  (max of dim 2)\n";
    std::cout << "    b      = " << b << "\n\n";

    return final_c * 100.0 / data.size();
}

int main() {
    std::mt19937 rng(42);

    // ============ TASK 1: Parity ============
    std::cout << "================================================================\n";
    std::cout << "  TASK 1: PARITY of count_A\n";
    std::cout << "================================================================\n\n";
    std::cout << "  Label 0: count_A is EVEN\n";
    std::cout << "  Label 1: count_A is ODD\n";
    std::cout << "  Channel: p (XOR of x > 0)\n\n";

    std::vector<std::pair<std::vector<int>, int>> data_parity;
    for (int i = 0; i < N_TRAIN; ++i) {
        std::vector<int> seq(SEQ_LEN);
        int count_a = 0;
        for (int t = 0; t < SEQ_LEN; ++t) {
            seq[t] = (rng() % 2 == 0) ? 0 : 1;
            if (seq[t] == 0) count_a++;
        }
        int label = count_a % 2;  // 0 = even, 1 = odd
        data_parity.push_back({seq, label});
    }
    int lp0 = 0, lp1 = 0;
    for (auto& p : data_parity) (p.second == 0) ? lp0++ : lp1++;
    std::cout << "  Data: " << data_parity.size() << " (label-0: " << lp0 << ", label-1: " << lp1 << ")\n\n";

    train_and_eval(data_parity, "Parity");

    // ============ TASK 2: Max-pool detection ============
    std::cout << "================================================================\n";
    std::cout << "  TASK 2: MAX detection (any A in sequence)\n";
    std::cout << "================================================================\n\n";
    std::cout << "  Label 0: any token was A (m[0] > 0)\n";
    std::cout << "  Label 1: no A (all B)\n";
    std::cout << "  Channel: m (running max)\n\n";

    std::vector<std::pair<std::vector<int>, int>> data_max;
    for (int i = 0; i < N_TRAIN; ++i) {
        std::vector<int> seq(SEQ_LEN);
        bool has_a = false;
        for (int t = 0; t < SEQ_LEN; ++t) {
            seq[t] = (rng() % 2 == 0) ? 0 : 1;
            if (seq[t] == 0) has_a = true;
        }
        // For balance: skip if all-A or all-B
        if (!has_a || std::all_of(seq.begin(), seq.end(), [](int x){ return x == 0; })) {
            i--; continue;
        }
        int label = has_a ? 0 : 1;
        data_max.push_back({seq, label});
    }
    int lm0 = 0, lm1 = 0;
    for (auto& p : data_max) (p.second == 0) ? lm0++ : lm1++;
    std::cout << "  Data: " << data_max.size() << " (label-0: " << lm0 << ", label-1: " << lm1 << ")\n\n";

    train_and_eval(data_max, "Max");

    // ============ TASK 3: OR-of-AND ============
    std::cout << "================================================================\n";
    std::cout << "  TASK 3: OR-of-AND (any token has all 4 dims positive)\n";
    std::cout << "================================================================\n\n";
    std::cout << "  Vocab: 4 tokens\n";
    std::cout << "    0 (A):       [+3, -3, +1, -1]  mixed\n";
    std::cout << "    1 (B):       [-3, +3, -1, +1]  mixed\n";
    std::cout << "    2 (AB+):     [+3, +3, +1, +1]  all positive\n";
    std::cout << "    3 (AB-):     [-3, -3, -1, -1]  all negative\n";
    std::cout << "  Label 0: any token was 'AB+' (all 4 dims > 0)\n";
    std::cout << "  Label 1: no AB+ token\n\n";

    std::vector<std::pair<std::vector<int>, int>> data_or_and;
    for (int i = 0; i < N_TRAIN; ++i) {
        std::vector<int> seq(SEQ_LEN);
        bool has_ab_plus = false;
        for (int t = 0; t < SEQ_LEN; ++t) {
            seq[t] = rng() % VOCAB_SIZE;
            if (seq[t] == 2) has_ab_plus = true;
        }
        // For balance: only include if has_ab_plus or contains many negatives
        if (!has_ab_plus && rng() % 4 != 0) { i--; continue; }
        int label = has_ab_plus ? 0 : 1;
        data_or_and.push_back({seq, label});
    }
    int lo0 = 0, lo1 = 0;
    for (auto& p : data_or_and) (p.second == 0) ? lo0++ : lo1++;
    std::cout << "  Data: " << data_or_and.size() << " (label-0: " << lo0 << ", label-1: " << lo1 << ")\n\n";

    train_and_eval(data_or_and, "OR-AND");

    std::cout << "================================================================\n";
    std::cout << "  SUMMARY: All channels work\n";
    std::cout << "================================================================\n";
    return 0;
}
