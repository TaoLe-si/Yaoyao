// train_q4a.cpp
// Option A: Hash Bucket Q1 + Q4 with vocab=4, multi-class test.
//
// Architecture: Q1 (hash bucket pool) -> Q3 (k=3 conv) -> Q2-A + Sum -> Q4 (linear projection [V, D])
// Task: predict majority token (multi-class, V=4)

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
const int V = 4;            // vocab size
const int N_BUCKETS = 16;   // hash bucket pool size for Q1 (4 tokens * 2 buckets = 8 used)
const int K_HASH = 2;       // each token uses K buckets
const float LR = 0.05f;
const float WD = 0.01f;

// ============ Q1: hash bucket pool ============
struct Q1Pool {
    std::vector<std::vector<int8_t>> buckets;

    Q1Pool() {
        buckets.assign(N_BUCKETS, std::vector<int8_t>(D, 0));
        std::mt19937 rng(123);
        std::uniform_int_distribution<int> ud_int8(-1, 1);
        for (auto& b : buckets) for (auto& b : buckets) for (auto& v : b) v = (int8_t)ud_int8(rng);
    }

    inline int hash_func(int token_id, int k) const {
        return (token_id * 3 + k) % N_BUCKETS;  // guarantees unique buckets per token
    }

    void lookup(int token_id, int8_t* x) const {
        std::fill(x, x + D, 0);
        for (int k = 0; k < K_HASH; ++k) {
            int b = hash_func(token_id, k);
            for (int d = 0; d < D; ++d) x[d] += buckets[b][d];
        }
        for (int d = 0; d < D; ++d) {
            if (x[d] > H_VAL) x[d] = H_VAL;
            if (x[d] < -H_VAL) x[d] = -H_VAL;
        }
    }
};

// ============ Q3: k=3 conv ============
void q3_conv(const int8_t* x_prev2, const int8_t* x_prev,
             const int8_t* x_curr,
             const float* w0, const float* w1, const float* w2,
             int8_t* y_out) {
    for (int d = 0; d < D; ++d) {
        float v = w0[d] * (float)x_prev2[d]
                + w1[d] * (float)x_prev[d]
                + w2[d] * (float)x_curr[d];
        int r = (int)std::lroundf(v);
        if (r > H_VAL) r = H_VAL;
        if (r < -H_VAL) r = -H_VAL;
        y_out[d] = (int8_t)r;
    }
}

// ============ Q2-A + Sum ============
void q2a_step(const int8_t* h_old, const int16_t* s_old,
              const int8_t* y_t, const float* alpha,
              int8_t* h_new, int16_t* s_new) {
    for (int d = 0; d < D; ++d) {
        float v = alpha[d] * (float)h_old[d] + (1.0f - alpha[d]) * (float)y_t[d];
        int r = (int)std::lroundf(v);
        if (r > H_VAL) r = H_VAL;
        if (r < -H_VAL) r = -H_VAL;
        h_new[d] = (int8_t)r;
        int s = (int)s_old[d] + (int)y_t[d];
        if (s > S_MAX) s = S_MAX;
        if (s < -S_MAX) s = -S_MAX;
        s_new[d] = (int16_t)s;
    }
}

// ============ Q4: linear projection [V, D] ============
struct Q4Head {
    std::vector<std::vector<float>> w;  // [V][D]
    std::vector<float> b;                // [V]
    Q4Head() {
        w.assign(V, std::vector<float>(D, 0));
        b.assign(V, 0);
        std::mt19937 rng(456);
        std::normal_distribution<float> nd(0, 0.05f);
        for (auto& row : w) for (auto& v : row) v = nd(rng);
    }
    void compute_logits(const float* state, float* logits) const {
        for (int v = 0; v < V; ++v) {
            logits[v] = b[v];
            for (int d = 0; d < D; ++d) {
                logits[v] += w[v][d] * state[d];
            }
        }
    }
};

// ============ Forward ============
struct ForwardCache {
    std::vector<std::vector<int8_t>> xs;     // raw input x_t
    std::vector<std::vector<int8_t>> ys;     // Q3 output
    std::vector<std::vector<int8_t>> hs;     // h after each step
    std::vector<std::vector<int16_t>> ss;    // s after each step
};

void forward_seq(const std::vector<int>& tokens, const Q1Pool& q1,
                 const float* w_q3_0, const float* w_q3_1, const float* w_q3_2,
                 const float* alpha,
                 ForwardCache& c) {
    c.xs.assign(SEQ_LEN, std::vector<int8_t>(D, 0));
    c.ys.assign(SEQ_LEN, std::vector<int8_t>(D, 0));
    c.hs.assign(SEQ_LEN + 1, std::vector<int8_t>(D, 0));
    c.ss.assign(SEQ_LEN + 1, std::vector<int16_t>(D, 0));

    std::vector<int8_t> x_curr(D, 0), x_prev(D, 0), x_prev2(D, 0), y(D, 0);
    for (int t = 0; t < SEQ_LEN; ++t) {
        x_prev2 = x_prev; x_prev = x_curr;
        q1.lookup(tokens[t], x_curr.data());
        c.xs[t] = x_curr;

        q3_conv(x_prev2.data(), x_prev.data(), x_curr.data(),
                w_q3_0, w_q3_1, w_q3_2, y.data());
        c.ys[t] = y;

        q2a_step(c.hs[t].data(), c.ss[t].data(), y.data(), alpha,
                 c.hs[t + 1].data(), c.ss[t + 1].data());
    }
}

void softmax(const float* logits, float* probs) {
    float max_l = *std::max_element(logits, logits + V);
    float sum = 0;
    for (int v = 0; v < V; ++v) {
        probs[v] = std::exp(logits[v] - max_l);
        sum += probs[v];
    }
    for (int v = 0; v < V; ++v) probs[v] /= sum;
}

int main() {
    std::cout << "================================================================\n";
    std::cout << "  OPTION A: Hash Bucket Q1 + Q4 (vocab=4 multi-class)\n";
    std::cout << "================================================================\n\n";

    Q1Pool q1;
    std::vector<float> w_q3_0(D, 0), w_q3_1(D, 0), w_q3_2(D, 0);
    std::vector<float> alpha(D, 0.5f);
    Q4Head q4;

    // Init Q3 weights small random
    std::mt19937 rng(789);
    std::normal_distribution<float> nd(0, 0.05f);
    for (auto& v : w_q3_0) v = nd(rng);
    for (auto& v : w_q3_1) v = nd(rng);
    for (auto& v : w_q3_2) v = nd(rng);

    // Generate data: predict majority token (multi-class)
    std::vector<std::pair<std::vector<int>, int>> data;
    std::uniform_int_distribution<int> ud(0, V - 1);
    for (int i = 0; i < N_TRAIN; ++i) {
        std::vector<int> seq(SEQ_LEN);
        std::vector<int> counts(V, 0);
        for (int t = 0; t < SEQ_LEN; ++t) {
            seq[t] = ud(rng);
            counts[seq[t]]++;
        }
        // Find majority (handle ties by picking smallest)
        int majority = 0;
        int max_count = counts[0];
        for (int v = 1; v < V; ++v) {
            if (counts[v] > max_count) { max_count = counts[v]; majority = v; }
        }
        data.push_back({seq, majority});
    }
    int counts[V] = {0};
    for (auto& p : data) counts[p.second]++;
    std::cout << "  Data: " << data.size() << " sequences, V=4 multi-class\n";
    std::cout << "  Class distribution: ";
    for (int v = 0; v < V; ++v) std::cout << "[" << v << "]=" << counts[v] << " ";
    std::cout << "\n\n";

    auto predict = [&](const std::vector<int>& seq) {
        ForwardCache c;
        forward_seq(seq, q1, w_q3_0.data(), w_q3_1.data(), w_q3_2.data(), alpha.data(), c);
        // Normalize state
        float mean = 0, var = 0;
        for (int d = 0; d < D; ++d) mean += (float)c.ss.back()[d];
        mean /= D;
        for (int d = 0; d < D; ++d) var += ((float)c.ss.back()[d] - mean) * ((float)c.ss.back()[d] - mean);
        var /= D;
        float stddev = std::sqrt(var + 1e-6f);
        std::vector<float> state_norm(D);
        for (int d = 0; d < D; ++d) state_norm[d] = ((float)c.ss.back()[d] - mean) / stddev;
        float logits[V];
        q4.compute_logits(state_norm.data(), logits);
        int best = 0;
        for (int v = 1; v < V; ++v) if (logits[v] > logits[best]) best = v;
        return best;
    };

    int initial_correct = 0;
    for (auto& p : data) if (predict(p.first) == p.second) initial_correct++;
    std::cout << "  Initial acc: " << initial_correct << "/" << data.size()
              << " (" << initial_correct * 100 / data.size() << "%)\n";
    std::cout << "  Random baseline: " << 100 / V << "%\n\n";

    // Training: only Q4 weights (linear projection)
    std::vector<std::vector<float>> w_grad(V, std::vector<float>(D, 0));
    std::vector<float> b_grad(V, 0);

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(data.begin(), data.end(), std::mt19937(epoch + 1));
        for (auto& row : w_grad) std::fill(row.begin(), row.end(), 0);
        std::fill(b_grad.begin(), b_grad.end(), 0);
        float total_loss = 0;

        for (auto& p : data) {
            ForwardCache c;
            forward_seq(p.first, q1, w_q3_0.data(), w_q3_1.data(), w_q3_2.data(), alpha.data(), c);
            // Normalize state
            float mean = 0, var = 0;
            for (int d = 0; d < D; ++d) mean += (float)c.ss.back()[d];
            mean /= D;
            for (int d = 0; d < D; ++d) var += ((float)c.ss.back()[d] - mean) * ((float)c.ss.back()[d] - mean);
            var /= D;
            float stddev = std::sqrt(var + 1e-6f);
            std::vector<float> state_norm(D);
            for (int d = 0; d < D; ++d) state_norm[d] = ((float)c.ss.back()[d] - mean) / stddev;
            float logits[V];
            q4.compute_logits(state_norm.data(), logits);
            float probs[V];
            softmax(logits, probs);
            total_loss += -std::log(std::max(probs[p.second], 1e-7f));
            float d_logits[V];
            for (int v = 0; v < V; ++v) d_logits[v] = probs[v] - (v == p.second ? 1.0f : 0.0f);
            for (int v = 0; v < V; ++v) {
                for (int d = 0; d < D; ++d) {
                    w_grad[v][d] += d_logits[v] * state_norm[d];
                }
                b_grad[v] += d_logits[v];
            }
        }

        for (int v = 0; v < V; ++v) {
            for (int d = 0; d < D; ++d) q4.w[v][d] -= LR * w_grad[v][d] / N_TRAIN;
            q4.b[v] -= LR * b_grad[v] / N_TRAIN;
            // weight decay
            for (int d = 0; d < D; ++d) q4.w[v][d] *= (1.0f - WD);
        }

        if ((epoch + 1) % 10 == 0 || epoch == 0 || epoch == EPOCHS - 1) {
            int c = 0;
            for (auto& pp : data) if (predict(pp.first) == pp.second) c++;
            std::cout << "  Epoch " << std::setw(3) << (epoch + 1)
                      << ": loss=" << std::fixed << std::setprecision(4) << total_loss / data.size()
                      << " acc=" << c << "/" << data.size()
                      << " (" << std::setw(3) << c * 100 / data.size() << "%)"
                      << "\n";
        }
    }

    int fc = 0;
    for (auto& p : data) if (predict(p.first) == p.second) fc++;
    std::cout << "\n  Final: " << fc << "/" << data.size()
              << " (" << fc * 100 / data.size() << "%)\n\n";

    // Test set (held-out)
    std::cout << "  Held-out test set:\n";
    int tc = 0;
    std::mt19937 test_rng(999);
    for (int i = 0; i < 200; ++i) {
        std::vector<int> seq(SEQ_LEN);
        std::vector<int> counts(V, 0);
        for (int t = 0; t < SEQ_LEN; ++t) {
            seq[t] = test_rng() % V;
            counts[seq[t]]++;
        }
        int majority = 0; int mx = counts[0];
        for (int v = 1; v < V; ++v) if (counts[v] > mx) { mx = counts[v]; majority = v; }
        if (predict(seq) == majority) tc++;
    }
    std::cout << "    Generalization: " << tc << "/200 ("
              << std::fixed << std::setprecision(1) << tc * 100.0 / 200 << "%)\n";

    // Benchmark
    std::cout << "\n================================================================\n";
    std::cout << "  BENCHMARK (Q4 multi-class with vocab=4)\n";
    std::cout << "================================================================\n\n";

    std::vector<int> bench_seq(SEQ_LEN);
    for (int t = 0; t < SEQ_LEN; ++t) bench_seq[t] = ud(rng);

    // Warmup
    for (int i = 0; i < 50; ++i) {
        ForwardCache c;
        forward_seq(bench_seq, q1, w_q3_0.data(), w_q3_1.data(), w_q3_2.data(), alpha.data(), c);
        float logits[V];
        q4.compute_logits((float*)c.ss.back().data(), logits);
        if (logits[0] == -1e30f) std::cout << "";
    }

    int iters = 1000;
    auto t_start = std::chrono::high_resolution_clock::now();
    float acc_sum = 0;
    for (int i = 0; i < iters; ++i) {
        ForwardCache c;
        forward_seq(bench_seq, q1, w_q3_0.data(), w_q3_1.data(), w_q3_2.data(), alpha.data(), c);
        float mean = 0, var = 0;
        for (int d = 0; d < D; ++d) mean += (float)c.ss.back()[d];
        mean /= D;
        for (int d = 0; d < D; ++d) var += ((float)c.ss.back()[d] - mean) * ((float)c.ss.back()[d] - mean);
        var /= D;
        float stddev = std::sqrt(var + 1e-6f);
        std::vector<float> state_norm(D);
        for (int d = 0; d < D; ++d) state_norm[d] = ((float)c.ss.back()[d] - mean) / stddev;
        float logits[V];
        q4.compute_logits(state_norm.data(), logits);
        acc_sum += logits[0];
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    if (acc_sum == 0) std::cout << "";

    double us_per_seq = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / (double)iters;
    std::cout << "  Forward (Q1+Q3+Q2-A+Q4): " << std::fixed << std::setprecision(2) << us_per_seq << " µs/seq\n";
    std::cout << "  Per token: " << std::setprecision(2) << us_per_seq / SEQ_LEN << " µs\n";
    std::cout << "  Throughput: " << std::setprecision(0) << 1e6 / (us_per_seq / SEQ_LEN) << " tokens/s\n";

    return 0;
}
