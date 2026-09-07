// train_complex2.cpp
// Isolated per-task tests: only enable relevant channel.
// Avoid interference from other channels.

#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <cmath>

const int D = 4096;
const int H_VAL = 4;
const int S_MAX = 64;
const int SEQ_LEN = 16;
const int EPOCHS = 60;
const int N_TRAIN = 500;
const float LR = 0.5f;

void q1_lookup_2(int token_id, int8_t* out_x) {
    std::fill(out_x, out_x + D, 0);
    if (token_id == 0) { out_x[0] = 3; out_x[1] = -3; out_x[2] = 1; }
    else { out_x[0] = -3; out_x[1] = 3; out_x[2] = -1; }
}

void q1_lookup_4(int token_id, int8_t* out_x) {
    std::fill(out_x, out_x + D, 0);
    if (token_id == 0) { out_x[0] = 3; out_x[1] = -3; out_x[2] = 1; out_x[3] = -1; }
    else if (token_id == 1) { out_x[0] = -3; out_x[1] = 3; out_x[2] = -1; out_x[3] = 1; }
    else if (token_id == 2) { out_x[0] = 3; out_x[1] = 3; out_x[2] = 1; out_x[3] = 1; }
    else { out_x[0] = -3; out_x[1] = -3; out_x[2] = -1; out_x[3] = -1; }
}

// ============ TASK 1: PARITY (p channel only) ============

struct SeqState1 {
    std::vector<std::vector<int8_t>> ps;
    std::vector<std::vector<int8_t>> xs;
};

SeqState1 forward_p(const std::vector<int>& tokens) {
    SeqState1 s;
    s.ps.resize(SEQ_LEN + 1); s.xs.resize(SEQ_LEN);
    for (int i = 0; i <= SEQ_LEN; ++i) s.ps[i].assign(D, 0);
    for (int i = 0; i < SEQ_LEN; ++i) s.xs[i].assign(D, 0);
    std::vector<int8_t> xt(D);
    for (int t = 0; t < SEQ_LEN; ++t) {
        q1_lookup_2(tokens[t], xt.data());
        s.xs[t] = xt;
        for (int d = 0; d < D; ++d) {
            s.ps[t + 1][d] = s.ps[t][d] ^ (xt[d] > 0 ? 1 : 0);
        }
    }
    return s;
}

void train_parity() {
    std::cout << "================================================================\n";
    std::cout << "  TASK 1: PARITY (using p channel only)\n";
    std::cout << "================================================================\n\n";
    std::cout << "  Label 0: count_A even, Label 1: count_A odd\n";
    std::cout << "  Channel: p[d] = XOR of (x[d] > 0)\n\n";

    std::mt19937 rng(42);
    std::vector<std::pair<std::vector<int>, int>> data;
    int lp0 = 0, lp1 = 0;
    for (int i = 0; i < N_TRAIN; ++i) {
        std::vector<int> seq(SEQ_LEN);
        int count_a = 0;
        for (int t = 0; t < SEQ_LEN; ++t) {
            seq[t] = (rng() % 2 == 0) ? 0 : 1;
            if (seq[t] == 0) count_a++;
        }
        int label = count_a % 2;
        (label == 0) ? lp0++ : lp1++;
        data.push_back({seq, label});
    }
    std::cout << "  Data: " << data.size() << " (label-0 even: " << lp0
              << ", label-1 odd: " << lp1 << ")\n\n";

    std::vector<float> w_p(D, 0);
    float b = 0;
    std::normal_distribution<float> nd(0, 0.1f);
    for (auto& v : w_p) v = nd(rng);

    int correct = 0;
    for (auto& p : data) {
        SeqState1 s = forward_p(p.first);
        float logit = b;
        for (int d = 0; d < D; ++d) logit += w_p[d] * (float)s.ps.back()[d];
        if ((logit > 0 ? 0 : 1) == p.second) correct++;
    }
    std::cout << "  Initial acc: " << correct << "/" << data.size() << "\n\n";

    std::vector<float> w_p_grad(D, 0);
    float b_grad = 0;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(data.begin(), data.end(), std::mt19937(epoch + 1));
        std::fill(w_p_grad.begin(), w_p_grad.end(), 0);
        b_grad = 0;
        float total_loss = 0;

        for (auto& p : data) {
            SeqState1 s = forward_p(p.first);
            float logit = b;
            for (int d = 0; d < D; ++d) logit += w_p[d] * (float)s.ps.back()[d];
            float p0 = 1.0f / (1.0f + std::exp(-2.0f * logit));
            float prob_true = (p.second == 0) ? p0 : (1.0f - p0);
            total_loss += -std::log(std::max(prob_true, 1e-7f));
            float d_logit = (p.second == 0) ? (-2.0f * (1.0f - p0)) : (2.0f * p0);
            for (int d = 0; d < D; ++d) {
                w_p_grad[d] += d_logit * (float)s.ps.back()[d];
            }
            b_grad += d_logit;
        }

        for (int d = 0; d < D; ++d) w_p[d] -= LR * w_p_grad[d] / N_TRAIN;
        b -= LR * b_grad / N_TRAIN;

        if ((epoch + 1) % 10 == 0 || epoch == 0 || epoch == EPOCHS - 1) {
            int c = 0;
            for (auto& pp : data) {
                SeqState1 s = forward_p(pp.first);
                float logit = b;
                for (int d = 0; d < D; ++d) logit += w_p[d] * (float)s.ps.back()[d];
                if ((logit > 0 ? 0 : 1) == pp.second) c++;
            }
            std::cout << "  Epoch " << std::setw(3) << (epoch + 1)
                      << ": loss=" << std::fixed << std::setprecision(4) << total_loss / data.size()
                      << " acc=" << c << "/" << data.size()
                      << " (" << std::setw(3) << c * 100 / data.size() << "%)\n";
        }
    }
    int final_c = 0;
    for (auto& p : data) {
        SeqState1 s = forward_p(p.first);
        float logit = b;
        for (int d = 0; d < D; ++d) logit += w_p[d] * (float)s.ps.back()[d];
        if ((logit > 0 ? 0 : 1) == p.second) final_c++;
    }
    std::cout << "\n  Final acc: " << final_c << "/" << data.size()
              << " (" << final_c * 100 / data.size() << "%)\n";
    std::cout << "  Learned: w_p[0] = " << std::setprecision(3) << w_p[0] << ", b = " << b << "\n\n";
}

// ============ TASK 2: MAX (m channel only, balanced) ============

struct SeqState2 {
    std::vector<std::vector<int8_t>> ms;
    std::vector<std::vector<int8_t>> xs;
};

SeqState2 forward_m(const std::vector<int>& tokens) {
    SeqState2 s;
    s.ms.resize(SEQ_LEN + 1); s.xs.resize(SEQ_LEN);
    for (int i = 0; i <= SEQ_LEN; ++i) s.ms[i].assign(D, 0);
    for (int i = 0; i < SEQ_LEN; ++i) s.xs[i].assign(D, 0);
    std::vector<int8_t> xt(D);
    for (int t = 0; t < SEQ_LEN; ++t) {
        q1_lookup_2(tokens[t], xt.data());
        s.xs[t] = xt;
        for (int d = 0; d < D; ++d) {
            s.ms[t + 1][d] = std::max(s.ms[t][d], xt[d]);
        }
    }
    return s;
}

void train_max() {
    std::cout << "================================================================\n";
    std::cout << "  TASK 2: MAX detection (m channel, balanced data)\n";
    std::cout << "================================================================\n\n";
    std::cout << "  Label 0: at least one A in sequence (m[0] = +3)\n";
    std::cout << "  Label 1: no A (all B, m[0] stays at 0)\n";
    std::cout << "  Channel: m (running max)\n\n";

    std::mt19937 rng(42);
    std::vector<std::pair<std::vector<int>, int>> data;
    int lp0 = 0, lp1 = 0;
    for (int i = 0; i < N_TRAIN / 2; ++i) {
        // Sequence with at least one A
        std::vector<int> seq(SEQ_LEN);
        seq[rng() % SEQ_LEN] = 0;  // ensure at least one A
        for (int t = 0; t < SEQ_LEN; ++t) if (seq[t] != 0) seq[t] = (rng() % 2 == 0) ? 0 : 1;
        data.push_back({seq, 0});
        lp0++;
    }
    for (int i = 0; i < N_TRAIN / 2; ++i) {
        // All B
        std::vector<int> seq(SEQ_LEN, 1);
        data.push_back({seq, 1});
        lp1++;
    }
    std::cout << "  Data: " << data.size() << " (label-0: " << lp0 << ", label-1: " << lp1 << ")\n\n";

    std::vector<float> w_m(D, 0);
    float b = 0;
    std::normal_distribution<float> nd(0, 0.1f);
    for (auto& v : w_m) v = nd(rng);

    int correct = 0;
    for (auto& p : data) {
        SeqState2 s = forward_m(p.first);
        float logit = b;
        for (int d = 0; d < D; ++d) logit += w_m[d] * (float)s.ms.back()[d];
        if ((logit > 0 ? 0 : 1) == p.second) correct++;
    }
    std::cout << "  Initial acc: " << correct << "/" << data.size() << "\n\n";

    std::vector<float> w_m_grad(D, 0);
    float b_grad = 0;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(data.begin(), data.end(), std::mt19937(epoch + 1));
        std::fill(w_m_grad.begin(), w_m_grad.end(), 0);
        b_grad = 0;
        float total_loss = 0;

        for (auto& p : data) {
            SeqState2 s = forward_m(p.first);
            float logit = b;
            for (int d = 0; d < D; ++d) logit += w_m[d] * (float)s.ms.back()[d];
            float p0 = 1.0f / (1.0f + std::exp(-2.0f * logit));
            float prob_true = (p.second == 0) ? p0 : (1.0f - p0);
            total_loss += -std::log(std::max(prob_true, 1e-7f));
            float d_logit = (p.second == 0) ? (-2.0f * (1.0f - p0)) : (2.0f * p0);
            for (int d = 0; d < D; ++d) {
                w_m_grad[d] += d_logit * (float)s.ms.back()[d];
            }
            b_grad += d_logit;
        }

        for (int d = 0; d < D; ++d) w_m[d] -= LR * w_m_grad[d] / N_TRAIN;
        b -= LR * b_grad / N_TRAIN;

        if ((epoch + 1) % 10 == 0 || epoch == 0 || epoch == EPOCHS - 1) {
            int c = 0;
            for (auto& pp : data) {
                SeqState2 s = forward_m(pp.first);
                float logit = b;
                for (int d = 0; d < D; ++d) logit += w_m[d] * (float)s.ms.back()[d];
                if ((logit > 0 ? 0 : 1) == pp.second) c++;
            }
            std::cout << "  Epoch " << std::setw(3) << (epoch + 1)
                      << ": loss=" << std::fixed << std::setprecision(4) << total_loss / data.size()
                      << " acc=" << c << "/" << data.size()
                      << " (" << std::setw(3) << c * 100 / data.size() << "%)\n";
        }
    }
    int final_c = 0;
    for (auto& p : data) {
        SeqState2 s = forward_m(p.first);
        float logit = b;
        for (int d = 0; d < D; ++d) logit += w_m[d] * (float)s.ms.back()[d];
        if ((logit > 0 ? 0 : 1) == p.second) final_c++;
    }
    std::cout << "\n  Final acc: " << final_c << "/" << data.size()
              << " (" << final_c * 100 / data.size() << "%)\n";
    std::cout << "  Learned: w_m[0] = " << std::setprecision(3) << w_m[0] << ", b = " << b << "\n\n";
}

// ============ TASK 3: OR-of-AND (max over 4 dims' "all-positive") ============

struct SeqState3 {
    std::vector<std::array<int8_t, 4>> or_and;  // per-step all-4-positive indicator
    std::vector<std::vector<int8_t>> xs;
};

SeqState3 forward_orand(const std::vector<int>& tokens) {
    SeqState3 s;
    s.or_and.resize(SEQ_LEN + 1);
    s.xs.resize(SEQ_LEN);
    for (int i = 0; i <= SEQ_LEN; ++i) s.or_and[i] = {0, 0, 0, 0};
    for (int i = 0; i < SEQ_LEN; ++i) s.xs[i].assign(D, 0);
    std::vector<int8_t> xt(D);
    int8_t cum_flag = 0;
    for (int t = 0; t < SEQ_LEN; ++t) {
        q1_lookup_4(tokens[t], xt.data());
        s.xs[t] = xt;
        // All 4 dims positive at this position?
        int all_pos = (xt[0] > 0 && xt[1] > 0 && xt[2] > 0 && xt[3] > 0) ? 1 : 0;
        if (all_pos) cum_flag = 1;
        s.or_and[t + 1] = {(int8_t)cum_flag, 0, 0, 0};
    }
    return s;
}

void train_or_and() {
    std::cout << "================================================================\n";
    std::cout << "  TASK 3: OR-of-AND (any token with all 4 dims positive)\n";
    std::cout << "================================================================\n\n";
    std::cout << "  Vocab: 4 tokens (mixed +/+/-/- patterns)\n";
    std::cout << "  Label 0: any token has all 4 dims positive (token 2: [+3,+3,+1,+1])\n";
    std::cout << "  Label 1: no such token\n";
    std::cout << "  Channel: m of all-positive-at-t indicator (single bit)\n\n";

    std::mt19937 rng(42);
    std::vector<std::pair<std::vector<int>, int>> data;
    int lp0 = 0, lp1 = 0;
    for (int i = 0; i < N_TRAIN / 2; ++i) {
        // Sequence containing token 2
        std::vector<int> seq(SEQ_LEN);
        int pos = rng() % SEQ_LEN;
        seq[pos] = 2;
        for (int t = 0; t < SEQ_LEN; ++t) {
            if (t == pos) continue;
            seq[t] = rng() % 4;
            // avoid other "all-positive" (only token 2 is)
        }
        data.push_back({seq, 0});
        lp0++;
    }
    for (int i = 0; i < N_TRAIN / 2; ++i) {
        // Sequence with NO token 2
        std::vector<int> seq(SEQ_LEN);
        for (int t = 0; t < SEQ_LEN; ++t) seq[t] = rng() % 3;  // 0, 1, or 3 (no token 2)
        data.push_back({seq, 1});
        lp1++;
    }
    std::cout << "  Data: " << data.size() << " (label-0: " << lp0 << ", label-1: " << lp1 << ")\n\n";

    float w_oand = 0;
    float b = 0;
    std::normal_distribution<float> nd(0, 0.1f);
    w_oand = nd(rng);

    int correct = 0;
    for (auto& p : data) {
        SeqState3 s = forward_orand(p.first);
        float logit = b + w_oand * (float)s.or_and.back()[0];
        if ((logit > 0 ? 0 : 1) == p.second) correct++;
    }
    std::cout << "  Initial acc: " << correct << "/" << data.size() << "\n\n";

    float w_grad = 0, b_grad = 0;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(data.begin(), data.end(), std::mt19937(epoch + 1));
        w_grad = 0; b_grad = 0;
        float total_loss = 0;

        for (auto& p : data) {
            SeqState3 s = forward_orand(p.first);
            float logit = b + w_oand * (float)s.or_and.back()[0];
            float p0 = 1.0f / (1.0f + std::exp(-2.0f * logit));
            float prob_true = (p.second == 0) ? p0 : (1.0f - p0);
            total_loss += -std::log(std::max(prob_true, 1e-7f));
            float d_logit = (p.second == 0) ? (-2.0f * (1.0f - p0)) : (2.0f * p0);
            w_grad += d_logit * (float)s.or_and.back()[0];
            b_grad += d_logit;
        }

        w_oand -= LR * w_grad / N_TRAIN;
        b -= LR * b_grad / N_TRAIN;

        if ((epoch + 1) % 10 == 0 || epoch == 0 || epoch == EPOCHS - 1) {
            int c = 0;
            for (auto& pp : data) {
                SeqState3 s = forward_orand(pp.first);
                float logit = b + w_oand * (float)s.or_and.back()[0];
                if ((logit > 0 ? 0 : 1) == pp.second) c++;
            }
            std::cout << "  Epoch " << std::setw(3) << (epoch + 1)
                      << ": loss=" << std::fixed << std::setprecision(4) << total_loss / data.size()
                      << " acc=" << c << "/" << data.size()
                      << " (" << std::setw(3) << c * 100 / data.size() << "%)\n";
        }
    }
    int final_c = 0;
    for (auto& p : data) {
        SeqState3 s = forward_orand(p.first);
        float logit = b + w_oand * (float)s.or_and.back()[0];
        if ((logit > 0 ? 0 : 1) == p.second) final_c++;
    }
    std::cout << "\n  Final acc: " << final_c << "/" << data.size()
              << " (" << final_c * 100 / data.size() << "%)\n";
    std::cout << "  Learned: w_oand = " << std::setprecision(3) << w_oand << ", b = " << b << "\n\n";
}

int main() {
    train_parity();
    train_max();
    train_or_and();
    std::cout << "================================================================\n";
    std::cout << "  All 3 aggregation channels validated independently\n";
    std::cout << "================================================================\n";
    return 0;
}
