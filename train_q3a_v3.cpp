// train_q3a_v3.cpp
// Fixed: proper mean gradient, plus use s channel for 3-gram patterns.

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <immintrin.h>

const int D = 4096;
const int H_VAL = 4;
const int S_MAX = 64;
const int SEQ_LEN = 16;
const int EPOCHS = 60;
const int N_TRAIN = 500;
const float LR = 0.5f;

void q1(int token_id, int8_t* x) {
    std::fill(x, x + D, 0);
    if (token_id == 0) { x[0] = 3; x[1] = -3; x[2] = 1; }
    else { x[0] = -3; x[1] = 3; x[2] = -1; }
}

void forward_seq(const std::vector<int>& tokens,
                 const float* w_q3_0, const float* w_q3_1, const float* w_q3_2,
                 std::vector<std::vector<int8_t>>& xs_out,
                 std::vector<int8_t>& y_last_out) {
    xs_out.assign(SEQ_LEN, std::vector<int8_t>(D, 0));
    y_last_out.assign(D, 0);
    std::vector<int8_t> x_curr(D, 0), x_prev(D, 0), x_prev2(D, 0), y(D, 0);
    for (int t = 0; t < SEQ_LEN; ++t) {
        x_prev2 = x_prev;
        x_prev = x_curr;
        q1(tokens[t], x_curr.data());
        xs_out[t] = x_curr;
        for (int d = 0; d < D; ++d) {
            float v = w_q3_0[d] * (float)x_prev2[d]
                    + w_q3_1[d] * (float)x_prev[d]
                    + w_q3_2[d] * (float)x_curr[d];
            int r = (int)std::lroundf(v);
            if (r > H_VAL) r = H_VAL;
            if (r < -H_VAL) r = -H_VAL;
            y[d] = (int8_t)r;
        }
        if (t == SEQ_LEN - 1) y_last_out = y;
    }
}

void train_task(const std::vector<std::pair<std::vector<int>, int>>& data_in,
                const char* task_name) {
    std::cout << "================================================================\n";
    std::cout << "  TASK: " << task_name << "\n";
    std::cout << "================================================================\n\n";

    std::mt19937 rng(42);
    std::vector<std::pair<std::vector<int>, int>> data = data_in;

    std::vector<float> w_q3_0(D, 0), w_q3_1(D, 0), w_q3_2(D, 0);
    std::normal_distribution<float> nd(0, 0.05f);
    for (auto& v : w_q3_0) v = nd(rng);
    for (auto& v : w_q3_1) v = nd(rng);
    for (auto& v : w_q3_2) v = nd(rng);
    float w_head = nd(rng);
    float b = 0;

    auto predict = [&](const std::vector<int>& seq) {
        std::vector<std::vector<int8_t>> xs;
        std::vector<int8_t> y_last;
        forward_seq(seq, w_q3_0.data(), w_q3_1.data(), w_q3_2.data(), xs, y_last);
        float logit = w_head * (float)y_last[0] + b;
        return logit > 0 ? 0 : 1;
    };

    int initial_correct = 0;
    for (auto& p : data) if (predict(p.first) == p.second) initial_correct++;
    std::cout << "  Initial acc: " << initial_correct << "/" << data.size()
              << " (" << initial_correct * 100 / data.size() << "%)\n\n";

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(data.begin(), data.end(), std::mt19937(epoch + 1));
        float total_loss = 0;
        float gw0 = 0, gw1 = 0, gw2 = 0, gwh = 0, gb = 0;

        for (auto& p : data) {
            std::vector<std::vector<int8_t>> xs;
            std::vector<int8_t> y_last;
            forward_seq(p.first, w_q3_0.data(), w_q3_1.data(), w_q3_2.data(), xs, y_last);

            float logit = w_head * (float)y_last[0] + b;
            float p0 = 1.0f / (1.0f + std::exp(-2.0f * logit));
            float prob_true = (p.second == 0) ? p0 : (1.0f - p0);
            total_loss += -std::log(std::max(prob_true, 1e-7f));

            float d_logit = (p.second == 0) ? (-2.0f * (1.0f - p0)) : (2.0f * p0);
            float d_y0 = d_logit * w_head;

            int8_t x_p2 = xs[SEQ_LEN - 3][0];
            int8_t x_p1 = xs[SEQ_LEN - 2][0];
            int8_t x_c = xs[SEQ_LEN - 1][0];

            gw0 += d_y0 * (float)x_p2;
            gw1 += d_y0 * (float)x_p1;
            gw2 += d_y0 * (float)x_c;
            gwh += d_logit * (float)y_last[0];
            gb += d_logit;
        }

        // MEAN gradient (divide by N_TRAIN)
        w_q3_0[0] -= LR * gw0 / N_TRAIN;
        w_q3_1[0] -= LR * gw1 / N_TRAIN;
        w_q3_2[0] -= LR * gw2 / N_TRAIN;
        w_head -= LR * gwh / N_TRAIN;
        b -= LR * gb / N_TRAIN;

        if ((epoch + 1) % 10 == 0 || epoch == 0 || epoch == EPOCHS - 1) {
            int c = 0;
            for (auto& pp : data) if (predict(pp.first) == pp.second) c++;
            std::cout << "  Epoch " << std::setw(3) << (epoch + 1)
                      << ": loss=" << std::fixed << std::setprecision(4) << total_loss / data.size()
                      << " acc=" << c << "/" << data.size()
                      << " (" << std::setw(3) << c * 100 / data.size() << "%)"
                      << " w=[w0:" << std::setprecision(2) << w_q3_0[0]
                      << " w1:" << w_q3_1[0]
                      << " w2:" << w_q3_2[0] << "]"
                      << " w_hd=" << std::setprecision(2) << w_head
                      << " b=" << std::setprecision(2) << b << "\n";
        }
    }
    int fc = 0;
    for (auto& p : data) if (predict(p.first) == p.second) fc++;
    std::cout << "\n  Final: " << fc << "/" << data.size()
              << " (" << fc * 100 / data.size() << "%)\n\n";
}

int main() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> ud(0, 1);

    std::cout << "================================================================\n";
    std::cout << "  Q3 k=3 conv + simple head: AB and AAA tests\n";
    std::cout << "================================================================\n\n";

    // TASK 1: ends with AB
    std::vector<std::pair<std::vector<int>, int>> data_ab;
    for (int i = 0; i < N_TRAIN / 2; ++i) {
        std::vector<int> seq(SEQ_LEN);
        for (int t = 0; t < SEQ_LEN; ++t) seq[t] = ud(rng);
        seq[SEQ_LEN - 2] = 0;
        seq[SEQ_LEN - 1] = 1;
        data_ab.push_back({seq, 0});
    }
    for (int i = 0; i < N_TRAIN / 2; ++i) {
        std::vector<int> seq(SEQ_LEN);
        for (int t = 0; t < SEQ_LEN; ++t) seq[t] = ud(rng);
        if (seq[SEQ_LEN - 2] == 0 && seq[SEQ_LEN - 1] == 1) seq[SEQ_LEN - 1] = 0;
        data_ab.push_back({seq, 1});
    }
    train_task(data_ab, "ends with AB");

    // TASK 2: ends with AAA
    std::vector<std::pair<std::vector<int>, int>> data_aaa;
    for (int i = 0; i < N_TRAIN / 2; ++i) {
        std::vector<int> seq(SEQ_LEN);
        for (int t = 0; t < SEQ_LEN; ++t) seq[t] = ud(rng);
        seq[SEQ_LEN - 3] = 0;
        seq[SEQ_LEN - 2] = 0;
        seq[SEQ_LEN - 1] = 0;
        data_aaa.push_back({seq, 0});
    }
    for (int i = 0; i < N_TRAIN / 2; ++i) {
        std::vector<int> seq(SEQ_LEN);
        for (int t = 0; t < SEQ_LEN; ++t) seq[t] = ud(rng);
        if (seq[SEQ_LEN - 3] == 0 && seq[SEQ_LEN - 2] == 0 && seq[SEQ_LEN - 1] == 0)
            seq[SEQ_LEN - 1] = 1;
        data_aaa.push_back({seq, 1});
    }
    train_task(data_aaa, "ends with AAA");

    return 0;
}
