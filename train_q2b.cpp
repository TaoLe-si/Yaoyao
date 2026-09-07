// train_q2b.cpp
// Q2-B v5: Test 3 modes side-by-side.
//   A) W FROZEN at init (g ~ 0 -> running average -> Q2-A-equivalent baseline)
//   B) W trained (the actual Q2-B idea, harder to stabilize)
//   C) Head only (W + g ~ 0, just train linear head)
//
// This isolates where the training difficulty comes from.

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>

const int D = 256;
const int SEQ_LEN = 16;
const int EPOCHS = 60;
const int N_TRAIN = 500;
const float LR = 0.5f;

void q1_lookup(int token_id, int8_t* out_x) {
    std::fill(out_x, out_x + D, 0);
    if (token_id == 0) { out_x[0] = 3; out_x[1] = -3; out_x[2] = 1; }
    else { out_x[0] = -3; out_x[1] = 3; out_x[2] = -1; }
}

struct SeqState {
    std::vector<std::vector<float>> hs, gs;
    std::vector<std::vector<int8_t>> xs;
};

void run_mode(const char* label, bool train_W, std::mt19937& rng) {
    std::cout << "\n================================================================\n";
    std::cout << "  " << label << "\n";
    std::cout << "================================================================\n\n";

    std::vector<std::pair<std::vector<int>, int>> data;
    for (int i = 0; i < N_TRAIN; ++i) {
        std::vector<int> seq(SEQ_LEN);
        for (int t = 0; t < SEQ_LEN; ++t) seq[t] = (rng() % 2 == 0) ? 0 : 1;
        data.push_back({seq, seq[SEQ_LEN - 1]});
    }

    std::vector<float> W(D * D);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    for (auto& w : W) w = nd(rng);

    std::vector<float> w(D, 0.0f);
    float b = 0.0f;
    std::normal_distribution<float> nd2(0.0f, 0.1f);
    for (auto& wi : w) wi = nd2(rng);

    auto forward = [&](const std::vector<int>& tokens, bool record) {
        SeqState s;
        s.hs.resize(SEQ_LEN + 1); s.gs.resize(SEQ_LEN); s.xs.resize(SEQ_LEN);
        for (int i = 0; i <= SEQ_LEN; ++i) s.hs[i].assign(D, 0.0f);
        for (int i = 0; i < SEQ_LEN; ++i) { s.gs[i].assign(D, 0.0f); s.xs[i].assign(D, 0); }
        std::vector<float> h_new(D);
        std::vector<int8_t> x_t(D);
        for (int t = 0; t < SEQ_LEN; ++t) {
            q1_lookup(tokens[t], x_t.data());
            s.xs[t] = x_t;
            for (int d = 0; d < D; ++d) {
                float z = 0;
                for (int j = 0; j < D; ++j) z += W[d * D + j] * (float)x_t[j];
                float g = std::tanh(z);
                if (record) s.gs[t][d] = g;
                h_new[d] = 0.5f * (1.0f + g) * s.hs[t][d] + 0.5f * (1.0f - g) * (float)x_t[d];
            }
            s.hs[t + 1] = h_new;
        }
        return s;
    };

    auto predict = [&](const std::vector<int>& tokens) {
        SeqState s = forward(tokens, false);
        float sum = b;
        for (int d = 0; d < D; ++d) sum += s.hs.back()[d] * w[d];
        return sum > 0 ? 0 : 1;
    };

    int initial = 0;
    for (auto& p : data) if (predict(p.first) == p.second) initial++;
    std::cout << "  Initial acc: " << initial << "/" << data.size() << "\n\n";

    std::vector<float> W_grad(D * D, 0.0f);
    std::vector<float> w_grad(D, 0.0f);

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(data.begin(), data.end(), rng);
        std::fill(W_grad.begin(), W_grad.end(), 0.0f);
        std::fill(w_grad.begin(), w_grad.end(), 0.0f);
        float b_grad = 0;
        float total_loss = 0;

        for (auto& p : data) {
            SeqState s = forward(p.first, true);
            float sum = b;
            for (int d = 0; d < D; ++d) sum += s.hs.back()[d] * w[d];
            float p0 = 1.0f / (1.0f + std::exp(-2.0f * sum));
            float prob_true = (p.second == 0) ? p0 : (1.0f - p0);
            total_loss += -std::log(std::max(prob_true, 1e-7f));
            float d_logit = (p.second == 0) ? (-2.0f * (1.0f - p0)) : (2.0f * p0);

            std::vector<float> d_h(D);
            for (int d = 0; d < D; ++d) {
                d_h[d] = d_logit * w[d];
                w_grad[d] += d_logit * s.hs.back()[d];
            }
            b_grad += d_logit;

            if (train_W) {
                for (int t = SEQ_LEN - 1; t >= 0; --t) {
                    std::vector<float> d_h_old(D, 0.0f);
                    for (int d = 0; d < D; ++d) {
                        float g = s.gs[t][d];
                        d_h_old[d] = 0.5f * (1.0f + g) * d_h[d];
                        float d_g = 0.5f * (s.hs[t][d] - (float)s.xs[t][d]) * d_h[d];
                        float d_z = d_g * (1.0f - g * g);
                        for (int j = 0; j < D; ++j) W_grad[d * D + j] += d_z * (float)s.xs[t][j];
                    }
                    d_h = std::move(d_h_old);
                }
            }
        }

        for (int d = 0; d < D; ++d) w[d] -= LR * w_grad[d] / N_TRAIN;
        b -= LR * b_grad / N_TRAIN;

        if (train_W) {
            for (size_t i = 0; i < W.size(); ++i) {
                float up = LR * W_grad[i] / N_TRAIN;
                if (up > 0.2f) up = 0.2f;
                if (up < -0.2f) up = -0.2f;
                W[i] -= up;
            }
        }

        if ((epoch + 1) % 10 == 0 || epoch == 0 || epoch == EPOCHS - 1) {
            int c = 0;
            for (auto& pp : data) if (predict(pp.first) == pp.second) c++;
            std::cout << "  Epoch " << std::setw(3) << (epoch + 1) << ": loss=" << std::fixed << std::setprecision(4) << total_loss / data.size() << " acc=" << c << "/" << data.size() << "\n";
        }
    }
}

int main() {
    std::cout << "Q2-B training difficulty analysis (3 modes)\n";
    std::cout << "  Task: predict last token\n";
    std::cout << "  All modes: full-batch, LR=0.5, D=256\n\n";

    std::mt19937 rng(42);

    // Mode A: train head only, W frozen (effectively Q2-A α=0.5)
    run_mode("Mode A: W FROZEN (running average baseline)", false, rng);

    // Mode B: train head + W (Q2-B with input-dependent gate)
    run_mode("Mode B: W trained (true Q2-B)", true, rng);

    return 0;
}
