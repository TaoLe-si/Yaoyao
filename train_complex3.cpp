// train_complex3.cpp
// OR-of-AND with D-dim weights, no shared state issues.

#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <cmath>

const int D = 4096;
const int SEQ_LEN = 16;
const int EPOCHS = 40;
const int N_TRAIN = 500;
const float LR = 0.5f;

void q1_lookup_4(int token_id, int8_t* out_x) {
    std::fill(out_x, out_x + D, 0);
    if (token_id == 0) { out_x[0] = 3; out_x[1] = -3; out_x[2] = 1; out_x[3] = -1; }
    else if (token_id == 1) { out_x[0] = -3; out_x[1] = 3; out_x[2] = -1; out_x[3] = 1; }
    else if (token_id == 2) { out_x[0] = 3; out_x[1] = 3; out_x[2] = 1; out_x[3] = 1; }
    else { out_x[0] = -3; out_x[1] = -3; out_x[2] = -1; out_x[3] = -1; }
}

// For each dim d, track if any x_t[d] > 0 ever (in that dim).
// Then OR across the 4 dims = "all 4 dims had at least one positive somewhere".
struct SeqState {
    std::vector<std::vector<int8_t>> ms;  // m[d] = max(x_t[d])
    std::vector<std::vector<int8_t>> xs;
};

SeqState forward(const std::vector<int>& tokens) {
    SeqState s;
    s.ms.resize(SEQ_LEN + 1); s.xs.resize(SEQ_LEN);
    for (int i = 0; i <= SEQ_LEN; ++i) s.ms[i].assign(D, 0);
    for (int i = 0; i < SEQ_LEN; ++i) s.xs[i].assign(D, 0);
    std::vector<int8_t> xt(D);
    for (int t = 0; t < SEQ_LEN; ++t) {
        q1_lookup_4(tokens[t], xt.data());
        s.xs[t] = xt;
        for (int d = 0; d < D; ++d) {
            int8_t x = xt[d];
            int8_t m = s.ms[t][d];
            s.ms[t + 1][d] = (m > x) ? m : x;  // max
        }
    }
    return s;
}

// OR-of-AND: any token has all 4 dims > 0
// m[d] > 0 means dim d had a positive somewhere
// "all 4 dims ever positive" = m[0] > 0 AND m[1] > 0 AND m[2] > 0 AND m[3] > 0
// Linear head on 4 max values can learn this if weights > 0:
//   label 0: all positive -> logit = sum w[d] * m[d] > 0
//   label 1: at least one dim never positive -> at least one m[d] <= 0

int main() {
    std::cout << "================================================================\n";
    std::cout << "  TASK 3 (REDO): OR-of-AND using 4 max channels + linear head\n";
    std::cout << "================================================================\n\n";
    std::cout << "  Vocab: 4 tokens\n";
    std::cout << "    0 (A):       [+3, -3, +1, -1]  mixed\n";
    std::cout << "    1 (B):       [-3, +3, -1, +1]  mixed\n";
    std::cout << "    2 (AB+):     [+3, +3, +1, +1]  all positive\n";
    std::cout << "    3 (AB-):     [-3, -3, -1, -1]  all negative\n";
    std::cout << "  Label 0: any token has all 4 dims > 0 (token 2)\n";
    std::cout << "  Label 1: no such token (no all-positive token)\n\n";
    std::cout << "  Head: w[0]*m[0] + w[1]*m[1] + w[2]*m[2] + w[3]*m[3] + b\n";
    std::cout << "  Where m[d] is max over t of x_t[d]\n\n";

    std::mt19937 rng(42);
    std::vector<std::pair<std::vector<int>, int>> data;
    int lp0 = 0, lp1 = 0;
    for (int i = 0; i < N_TRAIN / 2; ++i) {
        std::vector<int> seq(SEQ_LEN);
        int pos = rng() % SEQ_LEN;
        seq[pos] = 2;
        for (int t = 0; t < SEQ_LEN; ++t) {
            if (t == pos) continue;
            seq[t] = rng() % 4;
        }
        data.push_back({seq, 0});
        lp0++;
    }
    for (int i = 0; i < N_TRAIN / 2; ++i) {
        std::vector<int> seq(SEQ_LEN);
        for (int t = 0; t < SEQ_LEN; ++t) { int r = rng() % 3; seq[t] = (r == 2) ? 3 : r; }
        data.push_back({seq, 1});
        lp1++;
    }
    std::cout << "  Data: " << data.size() << " (label-0: " << lp0
              << ", label-1: " << lp1 << ")\n\n";

    std::vector<float> w(D, 0);
    std::normal_distribution<float> nd(0, 0.1f);
    for (auto& v : w) v = nd(rng);
    float b = 0;

    auto predict = [&](const std::vector<int>& seq) {
        SeqState s = forward(seq);
        float logit = b;
        for (int d = 0; d < 4; ++d) logit += w[d] * (float)s.ms.back()[d];
        return logit > 0 ? 0 : 1;
    };

    int c0 = 0;
    for (auto& p : data) if (predict(p.first) == p.second) c0++;
    std::cout << "  Initial acc: " << c0 << "/" << data.size() << "\n\n";

    std::vector<float> w_grad(D, 0);
    float b_grad = 0;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(data.begin(), data.end(), std::mt19937(epoch + 1));
        std::fill(w_grad.begin(), w_grad.end(), 0);
        b_grad = 0;
        float total_loss = 0;

        for (auto& p : data) {
            SeqState s = forward(p.first);
            float logit = b;
            for (int d = 0; d < 4; ++d) logit += w[d] * (float)s.ms.back()[d];
            float p0 = 1.0f / (1.0f + std::exp(-2.0f * logit));
            float prob_true = (p.second == 0) ? p0 : (1.0f - p0);
            total_loss += -std::log(std::max(prob_true, 1e-7f));
            float d_logit = (p.second == 0) ? (-2.0f * (1.0f - p0)) : (2.0f * p0);
            for (int d = 0; d < 4; ++d) {
                w_grad[d] += d_logit * (float)s.ms.back()[d];
            }
            b_grad += d_logit;
        }

        for (int d = 0; d < 4; ++d) w[d] -= LR * w_grad[d] / N_TRAIN;
        b -= LR * b_grad / N_TRAIN;

        if ((epoch + 1) % 5 == 0 || epoch == 0 || epoch == EPOCHS - 1) {
            int c = 0;
            for (auto& pp : data) if (predict(pp.first) == pp.second) c++;
            std::cout << "  Epoch " << std::setw(3) << (epoch + 1)
                      << ": loss=" << std::fixed << std::setprecision(4) << total_loss / data.size()
                      << " acc=" << c << "/" << data.size()
                      << " (" << std::setw(3) << c * 100 / data.size() << "%)"
                      << " w=[" << std::setprecision(2) << w[0] << "," << w[1] << "," << w[2] << "," << w[3] << "]"
                      << " b=" << b << "\n";
        }
    }

    int final_c = 0;
    for (auto& p : data) if (predict(p.first) == p.second) final_c++;
    std::cout << "\n  Final acc: " << final_c << "/" << data.size()
              << " (" << final_c * 100 / data.size() << "%)\n\n";

    // Verify on separate test set
    std::cout << "  Test set verification:\n";
    std::vector<std::pair<std::vector<int>, int>> test_data;
    int test_correct = 0;
    std::mt19937 test_rng(123);
    for (int i = 0; i < 100; ++i) {
        std::vector<int> seq(SEQ_LEN);
        if (i < 50) {
            // Has token 2
            int pos = test_rng() % SEQ_LEN;
            seq[pos] = 2;
            for (int t = 0; t < SEQ_LEN; ++t) {
                if (t == pos) continue;
                seq[t] = test_rng() % 4;
            }
            test_data.push_back({seq, 0});
        } else {
            // No token 2
            for (int t = 0; t < SEQ_LEN; ++t) seq[t] = test_rng() % 3;
            test_data.push_back({seq, 1});
        }
    }
    for (auto& p : test_data) if (predict(p.first) == p.second) test_correct++;
    std::cout << "    Generalization: " << test_correct << "/" << test_data.size()
              << " (" << test_correct * 100 / test_data.size() << "%)\n\n";

    return 0;
}
