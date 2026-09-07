// train_q4a_v2.cpp
// Hash Bucket Q1 + Linear Q4 with proper Adam optimizer.

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
const int EPOCHS = 60;
const int N_TRAIN = 500;
const int V = 4;
const int N_BUCKETS = 16;
const int K_HASH = 2;
const float LR = 0.001f;  // Adam LR

// ============ Q1: hash bucket pool with better init ============
struct Q1Pool {
    std::vector<std::vector<int8_t>> buckets;
    Q1Pool() {
        buckets.assign(N_BUCKETS, std::vector<int8_t>(D, 0));
        std::mt19937 rng(123);
        std::normal_distribution<float> nd(0, 0.3f);  // smaller std
        for (auto& b : buckets) {
            for (auto& v : b) {
                int r = (int)std::lroundf(nd(rng));
                if (r > H_VAL) r = H_VAL; if (r < -H_VAL) r = -H_VAL;
                v = (int8_t)r;
            }
        }
    }
    inline int hash_func(int token_id, int k) const {
        return (token_id * 3 + k) % N_BUCKETS;
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

struct ForwardCache {
    std::vector<std::vector<int8_t>> xs, ys, hs;
    std::vector<std::vector<int16_t>> ss;
};

void forward_seq(const std::vector<int>& tokens, const Q1Pool& q1,
                 const float* w0, const float* w1, const float* w2,
                 const float* alpha, ForwardCache& c) {
    c.xs.assign(SEQ_LEN, std::vector<int8_t>(D, 0));
    c.ys.assign(SEQ_LEN, std::vector<int8_t>(D, 0));
    c.hs.assign(SEQ_LEN + 1, std::vector<int8_t>(D, 0));
    c.ss.assign(SEQ_LEN + 1, std::vector<int16_t>(D, 0));
    std::vector<int8_t> xc(D, 0), xp(D, 0), xp2(D, 0), y(D, 0);
    for (int t = 0; t < SEQ_LEN; ++t) {
        xp2 = xp; xp = xc;
        q1.lookup(tokens[t], xc.data());
        c.xs[t] = xc;
        q3_conv(xp2.data(), xp.data(), xc.data(), w0, w1, w2, y.data());
        c.ys[t] = y;
        q2a_step(c.hs[t].data(), c.ss[t].data(), y.data(), alpha,
                 c.hs[t + 1].data(), c.ss[t + 1].data());
    }
}

// Normalize state to mean 0, std 1
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

// ============ Q4: linear projection [V, D] ============
struct Q4Head {
    std::vector<std::vector<float>> w, m, v;  // weight + Adam moments
    std::vector<float> b, mb, vb;
    Q4Head() {
        w.assign(V, std::vector<float>(D, 0));
        m.assign(V, std::vector<float>(D, 0));
        v.assign(V, std::vector<float>(D, 0));
        b.assign(V, 0); mb.assign(V, 0); vb.assign(V, 0);
        std::mt19937 rng(456);
        std::normal_distribution<float> nd(0, 0.01f);
        for (auto& row : w) for (auto& val : row) val = nd(rng);
    }
    void compute_logits(const float* state, float* logits) const {
        for (int v = 0; v < V; ++v) {
            logits[v] = b[v];
            for (int d = 0; d < D; ++d) logits[v] += w[v][d] * state[d];
        }
    }
};

void softmax(const float* logits, float* probs) {
    float max_l = *std::max_element(logits, logits + V);
    float sum = 0;
    for (int v = 0; v < V; ++v) {
        probs[v] = std::exp(logits[v] - max_l);
        sum += probs[v];
    }
    for (int v = 0; v < V; ++v) probs[v] /= sum;
}

void adam_update(std::vector<std::vector<float>>& w,
                 std::vector<std::vector<float>>& m,
                 std::vector<std::vector<float>>& v,
                 const std::vector<std::vector<float>>& grad,
                 float lr, int t) {
    float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float bc1 = 1.0f - std::pow(b1, t);
    float bc2 = 1.0f - std::pow(b2, t);
    for (int i = 0; i < (int)w.size(); ++i) {
        for (int j = 0; j < (int)w[i].size(); ++j) {
            m[i][j] = b1 * m[i][j] + (1 - b1) * grad[i][j];
            v[i][j] = b2 * v[i][j] + (1 - b2) * grad[i][j] * grad[i][j];
            float mh = m[i][j] / bc1;
            float vh = v[i][j] / bc2;
            w[i][j] -= lr * mh / (std::sqrt(vh) + eps);
        }
    }
}

void adam_update_vec(std::vector<float>& w, std::vector<float>& m, std::vector<float>& v,
                     const std::vector<float>& grad, float lr, int t) {
    float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float bc1 = 1.0f - std::pow(b1, t);
    float bc2 = 1.0f - std::pow(b2, t);
    for (int i = 0; i < (int)w.size(); ++i) {
        m[i] = b1 * m[i] + (1 - b1) * grad[i];
        v[i] = b2 * v[i] + (1 - b2) * grad[i] * grad[i];
        float mh = m[i] / bc1;
        float vh = v[i] / bc2;
        w[i] -= lr * mh / (std::sqrt(vh) + eps);
    }
}

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Q1 (Hash Bucket) + Q3 + Q2-A + Sum + Q4 (Linear) | Adam\n";
    std::cout << "================================================================\n\n";

    Q1Pool q1;
    std::vector<float> w_q3_0(D, 0), w_q3_1(D, 0), w_q3_2(D, 0);
    std::vector<float> alpha(D, 0.5f);
    Q4Head q4;

    std::mt19937 rng(789);
    std::normal_distribution<float> nd(0, 0.05f);
    for (auto& v : w_q3_0) v = nd(rng);
    for (auto& v : w_q3_1) v = nd(rng);
    for (auto& v : w_q3_2) v = nd(rng);

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

    auto predict = [&](const std::vector<int>& seq) {
        ForwardCache c;
        forward_seq(seq, q1, w_q3_0.data(), w_q3_1.data(), w_q3_2.data(), alpha.data(), c);
        std::vector<float> state_norm(D);
        normalize_state(c.ss.back().data(), state_norm.data());
        float logits[V];
        q4.compute_logits(state_norm.data(), logits);
        int best = 0;
        for (int v = 1; v < V; ++v) if (logits[v] > logits[best]) best = v;
        return best;
    };

    int initial_correct = 0;
    for (auto& p : data) if (predict(p.first) == p.second) initial_correct++;
    std::cout << "  Initial acc: " << initial_correct << "/" << data.size()
              << " (" << initial_correct * 100 / data.size() << "%)\n\n";

    int t_step = 0;
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(data.begin(), data.end(), std::mt19937(epoch + 1));
        std::vector<std::vector<float>> w_grad(V, std::vector<float>(D, 0));
        std::vector<float> b_grad(V, 0);
        float total_loss = 0;

        for (auto& p : data) {
            ForwardCache c;
            forward_seq(p.first, q1, w_q3_0.data(), w_q3_1.data(), w_q3_2.data(), alpha.data(), c);
            std::vector<float> state_norm(D);
            normalize_state(c.ss.back().data(), state_norm.data());
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

        // Average
        for (int v = 0; v < V; ++v)
            for (int d = 0; d < D; ++d) w_grad[v][d] /= N_TRAIN;
        for (int v = 0; v < V; ++v) b_grad[v] /= N_TRAIN;

        t_step++;
        adam_update(q4.w, q4.m, q4.v, w_grad, LR, t_step);
        adam_update_vec(q4.b, q4.mb, q4.vb, b_grad, LR, t_step);

        if ((epoch + 1) % 5 == 0 || epoch == 0 || epoch == EPOCHS - 1) {
            int c = 0;
            for (auto& pp : data) if (predict(pp.first) == pp.second) c++;
            std::cout << "  Epoch " << std::setw(3) << (epoch + 1)
                      << ": loss=" << std::fixed << std::setprecision(4) << total_loss / data.size()
                      << " acc=" << c << "/" << data.size()
                      << " (" << std::setw(3) << c * 100 / data.size() << "%)\n";
        }
    }

    int fc = 0;
    for (auto& p : data) if (predict(p.first) == p.second) fc++;
    std::cout << "\n  Final: " << fc << "/" << data.size()
              << " (" << fc * 100 / data.size() << "%)\n";

    // Benchmark
    std::cout << "\n================================================================\n";
    std::cout << "  BENCHMARK (Full Pipeline Q1+Q3+Q2-A+Q4)\n";
    std::cout << "================================================================\n\n";

    std::vector<int> bench_seq(SEQ_LEN);
    for (int t = 0; t < SEQ_LEN; ++t) bench_seq[t] = ud(rng);

    for (int i = 0; i < 50; ++i) {
        ForwardCache c;
        forward_seq(bench_seq, q1, w_q3_0.data(), w_q3_1.data(), w_q3_2.data(), alpha.data(), c);
        if (c.hs[SEQ_LEN][0] == -100) std::cout << "";
    }

    int iters = 1000;
    auto t_start = std::chrono::high_resolution_clock::now();
    float acc_sum = 0;
    for (int i = 0; i < iters; ++i) {
        ForwardCache c;
        forward_seq(bench_seq, q1, w_q3_0.data(), w_q3_1.data(), w_q3_2.data(), alpha.data(), c);
        std::vector<float> state_norm(D);
        normalize_state(c.ss.back().data(), state_norm.data());
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
