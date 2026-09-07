// train_q4a_min.cpp
// Minimal test: explicit Q1 (one-hot-like) + Q3 + Q2-A + Sum + Q4 linear.
// Verify the full pipeline reaches 100% on majority prediction.

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
const int EPOCHS = 40;
const int N_TRAIN = 500;
const int V = 4;
const float LR = 1.0f;
const float WD = 0.05f;

// ============ Q1: explicit per-token embedding ============
// Token v: dims [2v, 2v+1] active (+3, -3), others 0
void q1(int token_id, int8_t* x) {
    std::fill(x, x + D, 0);
    if (token_id >= 0 && token_id < V) {
        x[token_id * 2] = 3;
        x[token_id * 2 + 1] = -3;
    }
}

void q3_conv(const int8_t* x_p2, const int8_t* x_p, const int8_t* x_c,
             const float* w0, const float* w1, const float* w2, int8_t* y) {
    for (int d = 0; d < D; ++d) {
        float v = w0[d] * (float)x_p2[d] + w1[d] * (float)x_p[d] + w2[d] * (float)x_c[d];
        int r = (int)std::lroundf(v);
        if (r > H_VAL) r = H_VAL; if (r < -H_VAL) r = -H_VAL;
        y[d] = (int8_t)r;
    }
}

void q2a_step(const int8_t* h_old, const int16_t* s_old,
              const int8_t* y, const float* alpha,
              int8_t* h_new, int16_t* s_new) {
    for (int d = 0; d < D; ++d) {
        float v = alpha[d] * (float)h_old[d] + (1.0f - alpha[d]) * (float)y[d];
        int r = (int)std::lroundf(v);
        if (r > H_VAL) r = H_VAL; if (r < -H_VAL) r = -H_VAL;
        h_new[d] = (int8_t)r;
        int s = (int)s_old[d] + (int)y[d];
        if (s > S_MAX) s = S_MAX; if (s < -S_MAX) s = -S_MAX;
        s_new[d] = (int16_t)s;
    }
}

void forward_seq2(const std::vector<int>& tokens,
                  const float* w0, const float* w1, const float* w2, const float* alpha,
                  std::vector<std::vector<int16_t>>& ss_out) {
    ss_out.assign(SEQ_LEN + 1, std::vector<int16_t>(D, 0));
    std::vector<int8_t> xc(D, 0), xp(D, 0), xp2(D, 0), y(D, 0);
    std::vector<int8_t> h(D, 0);
    for (int t = 0; t < SEQ_LEN; ++t) {
        xp2 = xp; xp = xc;
        q1(tokens[t], xc.data());
        q3_conv(xp2.data(), xp.data(), xc.data(), w0, w1, w2, y.data());
        // q2a_step - we only need s_new
        for (int d = 0; d < D; ++d) {
            int s = (int)ss_out[t][d] + (int)y[d];
            if (s > S_MAX) s = S_MAX; if (s < -S_MAX) s = -S_MAX;
            ss_out[t + 1][d] = (int16_t)s;
        }
        (void)h;
    }
}

struct Q4 {
    std::vector<std::vector<float>> w;  // [V][D]
    std::vector<float> b;
    Q4() {
        w.assign(V, std::vector<float>(D, 0));
        b.assign(V, 0);
        std::mt19937 rng(456);
        std::normal_distribution<float> nd(0, 0.05f);
        for (auto& row : w) for (auto& v : row) v = nd(rng);
    }
    void logits(const int16_t* state, float* out, const float* state_norm) const {
        for (int v = 0; v < V; ++v) {
            float s = b[v];
            for (int d = 0; d < D; ++d) s += w[v][d] * state_norm[d];
            out[v] = s;
        }
    }
};

void normalize_state(const int16_t* state, float* out) {
    float mean = 0;
    for (int d = 0; d < D; ++d) mean += (float)state[d];
    mean /= D;
    float var = 0;
    for (int d = 0; d < D; ++d) var += ((float)state[d] - mean) * ((float)state[d] - mean);
    var /= D;
    float stddev = std::sqrt(var + 1e-6f);
    for (int d = 0; d < D; ++d) out[d] = ((float)state[d] - mean) / stddev;
}

void softmax(const float* logits, float* probs) {
    float max_l = *std::max_element(logits, logits + V);
    float sum = 0;
    for (int v = 0; v < V; ++v) { probs[v] = std::exp(logits[v] - max_l); sum += probs[v]; }
    for (int v = 0; v < V; ++v) probs[v] /= sum;
}

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Q1 (one-hot) + Q3 + Sum + Q4 (linear) | Multi-class majority\n";
    std::cout << "================================================================\n\n";

    std::vector<float> w0(D, 0), w1(D, 0), w2(D, 0);
    std::vector<float> alpha(D, 0.99f);
    Q4 q4;

    std::mt19937 rng(789);
    std::normal_distribution<float> nd(0, 0.05f);
    for (auto& v : w0) v = nd(rng);
    for (auto& v : w1) v = nd(rng);
    for (auto& v : w2) v = nd(rng);

    std::uniform_int_distribution<int> ud(0, V - 1);
    std::vector<std::pair<std::vector<int>, int>> data;
    for (int i = 0; i < N_TRAIN; ++i) {
        std::vector<int> seq(SEQ_LEN);
        std::vector<int> counts(V, 0);
        for (int t = 0; t < SEQ_LEN; ++t) { seq[t] = ud(rng); counts[seq[t]]++; }
        int majority = 0; int mx = counts[0];
        for (int v = 1; v < V; ++v) if (counts[v] > mx) { mx = counts[v]; majority = v; }
        data.push_back({seq, majority});
    }

    std::vector<float> state_norm_buf(D);
    auto predict = [&](const std::vector<int>& seq) {
        std::vector<std::vector<int16_t>> ss;
        forward_seq2(seq, w0.data(), w1.data(), w2.data(), alpha.data(), ss);
        normalize_state(ss.back().data(), state_norm_buf.data());
        float logits[V];
        q4.logits(ss.back().data(), logits, state_norm_buf.data());
        int best = 0;
        for (int v = 1; v < V; ++v) if (logits[v] > logits[best]) best = v;
        return best;
    };

    int initial = 0;
    for (auto& p : data) if (predict(p.first) == p.second) initial++;
    std::cout << "  Initial: " << initial << "/" << data.size() << "\n\n";

    std::vector<std::vector<float>> w_grad(V, std::vector<float>(D, 0));
    std::vector<float> b_grad(V, 0);

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(data.begin(), data.end(), std::mt19937(epoch + 1));
        for (auto& row : w_grad) std::fill(row.begin(), row.end(), 0);
        std::fill(b_grad.begin(), b_grad.end(), 0);
        float total_loss = 0;
        std::vector<float> state_norm(D);
        for (auto& p : data) {
            std::vector<std::vector<int16_t>> ss;
            forward_seq2(p.first, w0.data(), w1.data(), w2.data(), alpha.data(), ss);
            normalize_state(ss.back().data(), state_norm.data());
            float logits[V];
            q4.logits(ss.back().data(), logits, state_norm.data());
            float probs[V];
            softmax(logits, probs);
            total_loss += -std::log(std::max(probs[p.second], 1e-7f));
            for (int v = 0; v < V; ++v) {
                float d_logit = probs[v] - (v == p.second ? 1.0f : 0.0f);
                for (int d = 0; d < D; ++d) w_grad[v][d] += d_logit * state_norm[d];
                b_grad[v] += d_logit;
            }
        }
        for (int v = 0; v < V; ++v) {
            for (int d = 0; d < D; ++d) {
                q4.w[v][d] -= LR * w_grad[v][d] / N_TRAIN;
                q4.w[v][d] *= (1.0f - WD);
            }
            q4.b[v] -= LR * b_grad[v] / N_TRAIN;
        }

        if ((epoch + 1) % 5 == 0 || epoch == 0 || epoch == EPOCHS - 1) {
            int c = 0;
            for (auto& pp : data) if (predict(pp.first) == pp.second) c++;
            std::cout << "  Epoch " << std::setw(3) << (epoch + 1)
                      << ": loss=" << std::fixed << std::setprecision(4) << total_loss / data.size()
                      << " acc=" << c << "/" << data.size() << " (" << std::setw(3) << c * 100 / data.size() << "%)\n";
        }
    }

    int fc = 0;
    for (auto& p : data) if (predict(p.first) == p.second) fc++;
    std::cout << "\n  Final: " << fc << "/" << data.size()
              << " (" << fc * 100 / data.size() << "%)\n";

    // Benchmark
    std::cout << "\n================================================================\n";
    std::cout << "  BENCHMARK\n";
    std::cout << "================================================================\n\n";

    std::vector<int> bench_seq(SEQ_LEN);
    for (int t = 0; t < SEQ_LEN; ++t) bench_seq[t] = ud(rng);

    for (int i = 0; i < 50; ++i) {
        std::vector<std::vector<int16_t>> ss;
        forward_seq2(bench_seq, w0.data(), w1.data(), w2.data(), alpha.data(), ss);
        if (ss.back()[0] == -1000) std::cout << "";
    }

    int iters = 1000;
    auto t0 = std::chrono::high_resolution_clock::now();
    float acc = 0;
    for (int i = 0; i < iters; ++i) {
        std::vector<std::vector<int16_t>> ss;
        forward_seq2(bench_seq, w0.data(), w1.data(), w2.data(), alpha.data(), ss);
        std::vector<float> state_norm_bench(D);
        normalize_state(ss.back().data(), state_norm_bench.data());
        float logits[V];
        q4.logits(ss.back().data(), logits, state_norm_bench.data());
        acc += logits[0];
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    if (acc == 0) std::cout << "";

    double us_per_seq = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / (double)iters;
    std::cout << "  Forward: " << std::fixed << std::setprecision(2) << us_per_seq << " µs/seq\n";
    std::cout << "  Per token: " << std::setprecision(2) << us_per_seq / SEQ_LEN << " µs\n";
    std::cout << "  Throughput: " << std::setprecision(0) << 1e6 / (us_per_seq / SEQ_LEN) << " tokens/s\n";

    return 0;
}
