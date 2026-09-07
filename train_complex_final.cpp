// train_complex_final.cpp
// Final clean implementation of all 3 complex tasks.
// Task 3 uses OR-of-AND flag (not max channel) because max alone is insufficient.

#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <cmath>

const int D = 4096;
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

// ========== TASK 1: PARITY ==========
std::vector<std::vector<int8_t>> forward_p(const std::vector<int>& tokens) {
    std::vector<std::vector<int8_t>> ps(SEQ_LEN + 1, std::vector<int8_t>(D, 0));
    std::vector<int8_t> xt(D);
    for (int t = 0; t < SEQ_LEN; ++t) {
        q1_lookup_2(tokens[t], xt.data());
        for (int d = 0; d < D; ++d) {
            ps[t + 1][d] = ps[t][d] ^ (xt[d] > 0 ? 1 : 0);
        }
    }
    return ps;
}

void train_parity() {
    std::cout << "================================================================\n";
    std::cout << "  TASK 1: PARITY (p channel, XOR)\n";
    std::cout << "================================================================\n\n";
    std::mt19937 rng(42);
    std::vector<std::pair<std::vector<int>, int>> data;
    for (int i = 0; i < N_TRAIN; ++i) {
        std::vector<int> seq(SEQ_LEN);
        int count_a = 0;
        for (int t = 0; t < SEQ_LEN; ++t) {
            seq[t] = (rng() % 2 == 0) ? 0 : 1;
            if (seq[t] == 0) count_a++;
        }
        data.push_back({seq, count_a % 2});
    }

    std::vector<float> w(D, 0);
    std::normal_distribution<float> nd(0, 0.1f);
    for (auto& v : w) v = nd(rng);
    float b = 0;

    auto predict = [&](const std::vector<int>& seq) {
        auto ps = forward_p(seq);
        float logit = b;
        for (int d = 0; d < D; ++d) logit += w[d] * (float)ps.back()[d];
        return logit > 0 ? 0 : 1;
    };

    int initial_correct = 0;
    for (auto& p : data) if (predict(p.first) == p.second) initial_correct++;
    std::cout << "  Initial acc: " << initial_correct << "/" << data.size() << "\n\n";

    std::vector<float> w_grad(D, 0);
    float b_grad = 0;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(data.begin(), data.end(), std::mt19937(epoch + 1));
        std::fill(w_grad.begin(), w_grad.end(), 0);
        b_grad = 0;
        for (auto& p : data) {
            auto ps = forward_p(p.first);
            float logit = b;
            for (int d = 0; d < D; ++d) logit += w[d] * (float)ps.back()[d];
            float p0 = 1.0f / (1.0f + std::exp(-2.0f * logit));
            float d_logit = (p.second == 0) ? (-2.0f * (1.0f - p0)) : (2.0f * p0);
            for (int d = 0; d < D; ++d) w_grad[d] += d_logit * (float)ps.back()[d];
            b_grad += d_logit;
        }
        for (int d = 0; d < D; ++d) w[d] -= LR * w_grad[d] / N_TRAIN;
        b -= LR * b_grad / N_TRAIN;

        if ((epoch + 1) % 20 == 0 || epoch == 0 || epoch == EPOCHS - 1) {
            int c = 0;
            for (auto& pp : data) if (predict(pp.first) == pp.second) c++;
            std::cout << "  Epoch " << std::setw(3) << (epoch + 1)
                      << ": acc=" << c << "/" << data.size()
                      << " (" << std::setw(3) << c * 100 / data.size() << "%)"
                      << " w[0]=" << std::fixed << std::setprecision(2) << w[0] << "\n";
        }
    }
    int fc = 0;
    for (auto& p : data) if (predict(p.first) == p.second) fc++;
    std::cout << "\n  Final: " << fc << "/" << data.size()
              << " (" << fc * 100 / data.size() << "%)\n\n";
}

// ========== TASK 2: MAX ==========
std::vector<std::vector<int8_t>> forward_m(const std::vector<int>& tokens) {
    std::vector<std::vector<int8_t>> ms(SEQ_LEN + 1, std::vector<int8_t>(D, 0));
    std::vector<int8_t> xt(D);
    for (int t = 0; t < SEQ_LEN; ++t) {
        q1_lookup_2(tokens[t], xt.data());
        for (int d = 0; d < D; ++d) {
            ms[t + 1][d] = (ms[t][d] > xt[d]) ? ms[t][d] : xt[d];
        }
    }
    return ms;
}

void train_max() {
    std::cout << "================================================================\n";
    std::cout << "  TASK 2: MAX detection (m channel)\n";
    std::cout << "================================================================\n\n";
    std::mt19937 rng(42);
    std::vector<std::pair<std::vector<int>, int>> data;
    for (int i = 0; i < N_TRAIN / 2; ++i) {
        std::vector<int> seq(SEQ_LEN);
        seq[rng() % SEQ_LEN] = 0;
        for (int t = 0; t < SEQ_LEN; ++t)
            if (seq[t] != 0) seq[t] = (rng() % 2 == 0) ? 0 : 1;
        data.push_back({seq, 0});
    }
    for (int i = 0; i < N_TRAIN / 2; ++i) {
        std::vector<int> seq(SEQ_LEN, 1);
        data.push_back({seq, 1});
    }

    std::vector<float> w(D, 0);
    std::normal_distribution<float> nd(0, 0.1f);
    for (auto& v : w) v = nd(rng);
    float b = 0;

    auto predict = [&](const std::vector<int>& seq) {
        auto ms = forward_m(seq);
        float logit = b;
        for (int d = 0; d < D; ++d) logit += w[d] * (float)ms.back()[d];
        return logit > 0 ? 0 : 1;
    };

    int initial_correct = 0;
    for (auto& p : data) if (predict(p.first) == p.second) initial_correct++;
    std::cout << "  Initial acc: " << initial_correct << "/" << data.size() << "\n\n";

    std::vector<float> w_grad(D, 0);
    float b_grad = 0;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(data.begin(), data.end(), std::mt19937(epoch + 1));
        std::fill(w_grad.begin(), w_grad.end(), 0);
        b_grad = 0;
        for (auto& p : data) {
            auto ms = forward_m(p.first);
            float logit = b;
            for (int d = 0; d < D; ++d) logit += w[d] * (float)ms.back()[d];
            float p0 = 1.0f / (1.0f + std::exp(-2.0f * logit));
            float d_logit = (p.second == 0) ? (-2.0f * (1.0f - p0)) : (2.0f * p0);
            for (int d = 0; d < D; ++d) w_grad[d] += d_logit * (float)ms.back()[d];
            b_grad += d_logit;
        }
        for (int d = 0; d < D; ++d) w[d] -= LR * w_grad[d] / N_TRAIN;
        b -= LR * b_grad / N_TRAIN;

        if ((epoch + 1) % 20 == 0 || epoch == 0 || epoch == EPOCHS - 1) {
            int c = 0;
            for (auto& pp : data) if (predict(pp.first) == pp.second) c++;
            std::cout << "  Epoch " << std::setw(3) << (epoch + 1)
                      << ": acc=" << c << "/" << data.size()
                      << " (" << std::setw(3) << c * 100 / data.size() << "%)"
                      << " w[0]=" << std::fixed << std::setprecision(2) << w[0] << "\n";
        }
    }
    int fc = 0;
    for (auto& p : data) if (predict(p.first) == p.second) fc++;
    std::cout << "\n  Final: " << fc << "/" << data.size()
              << " (" << fc * 100 / data.size() << "%)\n\n";
}

// ========== TASK 3: OR-of-AND (using flag, NOT max) ==========
std::vector<int8_t> forward_flag(const std::vector<int>& tokens) {
    std::vector<int8_t> flag(SEQ_LEN + 1, 0);
    std::vector<int8_t> xt(D);
    int cum = 0;
    for (int t = 0; t < SEQ_LEN; ++t) {
        q1_lookup_4(tokens[t], xt.data());
        int all_pos = (xt[0] > 0 && xt[1] > 0 && xt[2] > 0 && xt[3] > 0);
        if (all_pos) cum = 1;
        flag[t + 1] = (int8_t)cum;
    }
    return flag;
}

void train_or_and() {
    std::cout << "================================================================\n";
    std::cout << "  TASK 3: OR-of-AND (using flag, FIXED data)\n";
    std::cout << "================================================================\n\n";
    std::cout << "  Vocab: 4 tokens\n";
    std::cout << "    0 (A):   [+3, -3, +1, -1]  mixed\n";
    std::cout << "    1 (B):   [-3, +3, -1, +1]  mixed\n";
    std::cout << "    2 (AB+): [+3, +3, +1, +1]  ALL POSITIVE\n";
    std::cout << "    3 (AB-): [-3, -3, -1, -1]  all negative\n\n";
    std::cout << "  Label 0: any token == AB+ (token 2)\n";
    std::cout << "  Label 1: no AB+ (no token 2)\n\n";
    std::cout << "  State: flag[t+1] = OR of all_pos_at_t (single bit)\n\n";

    std::mt19937 rng(42);
    std::vector<std::pair<std::vector<int>, int>> data;
    for (int i = 0; i < N_TRAIN / 2; ++i) {
        std::vector<int> seq(SEQ_LEN);
        int pos = rng() % SEQ_LEN;
        seq[pos] = 2;
        for (int t = 0; t < SEQ_LEN; ++t) {
            if (t == pos) continue;
            int r = rng() % 4;
            seq[t] = r;  // includes 2 too — that's fine, label 0 just needs AT LEAST ONE token 2
        }
        data.push_back({seq, 0});
    }
    for (int i = 0; i < N_TRAIN / 2; ++i) {
        std::vector<int> seq(SEQ_LEN);
        for (int t = 0; t < SEQ_LEN; ++t) {
            int r = rng() % 3;
            seq[t] = (r == 2) ? 3 : r;  // EXCLUDE token 2
        }
        data.push_back({seq, 1});
    }
    std::cout << "  Data: " << data.size() << " (250 +250, label-1 NEVER has token 2)\n\n";

    // Use D weights for stability
    std::vector<float> w(D, 0);
    std::normal_distribution<float> nd(0, 0.1f);
    for (auto& v : w) v = nd(rng);
    float b = 0;

    auto predict = [&](const std::vector<int>& seq) {
        auto flag = forward_flag(seq);
        float logit = b + w[0] * (float)flag.back();
        return logit > 0 ? 0 : 1;
    };

    int initial_correct = 0;
    for (auto& p : data) if (predict(p.first) == p.second) initial_correct++;
    std::cout << "  Initial acc: " << initial_correct << "/" << data.size() << "\n\n";

    std::vector<float> w_grad(D, 0);
    float b_grad = 0;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(data.begin(), data.end(), std::mt19937(epoch + 1));
        std::fill(w_grad.begin(), w_grad.end(), 0);
        b_grad = 0;
        for (auto& p : data) {
            auto flag = forward_flag(p.first);
            float logit = b + w[0] * (float)flag.back();
            float p0 = 1.0f / (1.0f + std::exp(-2.0f * logit));
            float d_logit = (p.second == 0) ? (-2.0f * (1.0f - p0)) : (2.0f * p0);
            w_grad[0] += d_logit * (float)flag.back();
            b_grad += d_logit;
        }
        w[0] -= LR * w_grad[0] / N_TRAIN;
        b -= LR * b_grad / N_TRAIN;

        if ((epoch + 1) % 10 == 0 || epoch == 0 || epoch == EPOCHS - 1) {
            int c = 0;
            for (auto& pp : data) if (predict(pp.first) == pp.second) c++;
            std::cout << "  Epoch " << std::setw(3) << (epoch + 1)
                      << ": acc=" << c << "/" << data.size()
                      << " (" << std::setw(3) << c * 100 / data.size() << "%)"
                      << " w[0]=" << std::fixed << std::setprecision(2) << w[0]
                      << " b=" << std::setprecision(2) << b << "\n";
        }
    }
    int fc = 0;
    for (auto& p : data) if (predict(p.first) == p.second) fc++;
    std::cout << "\n  Final: " << fc << "/" << data.size()
              << " (" << fc * 100 / data.size() << "%)\n";

    // Test set
    std::cout << "\n  Test set (held-out):\n";
    std::vector<std::pair<std::vector<int>, int>> test_data;
    std::mt19937 test_rng(123);
    int tc = 0;
    for (int i = 0; i < 100; ++i) {
        std::vector<int> seq(SEQ_LEN);
        if (i < 50) {
            int pos = test_rng() % SEQ_LEN;
            seq[pos] = 2;
            for (int t = 0; t < SEQ_LEN; ++t) if (t != pos) seq[t] = test_rng() % 4;
            test_data.push_back({seq, 0});
        } else {
            for (int t = 0; t < SEQ_LEN; ++t) {
                int r = test_rng() % 3;
                seq[t] = (r == 2) ? 3 : r;
            }
            test_data.push_back({seq, 1});
        }
    }
    for (auto& p : test_data) if (predict(p.first) == p.second) tc++;
    std::cout << "    Generalization: " << tc << "/" << test_data.size()
              << " (" << tc * 100 / test_data.size() << "%)\n\n";
}

int main() {
    train_parity();
    train_max();
    train_or_and();
    return 0;
}
