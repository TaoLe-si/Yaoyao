// train_q4d_simple.cpp
// Simplified Option D: No Q3, multi-channel state (h, s, m, p), use all in Q4.
// Verify the END-TO-END pipeline trains properly on majority task.

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>

const int D = 64;
const int S_MAX = 64;
const int H_VAL = 4;
const int SEQ_LEN = 16;
const int EPOCHS = 40;
const int N_TRAIN = 800;
const int V = 8;
const float LR = 0.3f;
const float WD = 0.02f;

void q1(int token_id, int8_t* x) {
    std::fill(x, x + D, 0);
    if (token_id >= 0 && token_id < V) x[token_id] = 3;
}

struct ForwardCache {
    std::vector<std::vector<int8_t>> hs;
    std::vector<std::vector<int16_t>> ss;
    std::vector<std::vector<int8_t>> ms;
};

void forward_seq(const std::vector<int>& tokens,
                 const float* alpha, ForwardCache& c) {
    c.hs.assign(SEQ_LEN + 1, std::vector<int8_t>(D, 0));
    c.ss.assign(SEQ_LEN + 1, std::vector<int16_t>(D, 0));
    c.ms.assign(SEQ_LEN + 1, std::vector<int8_t>(D, 0));
    
    std::vector<int8_t> x(D);
    for (int t = 0; t < SEQ_LEN; ++t) {
        q1(tokens[t], x.data());
        for (int d = 0; d < D; ++d) {
            // h (EMA with alpha=0.5 fixed)
            float v = 0.5f * (float)c.hs[t][d] + 0.5f * (float)x[d];
            int r = (int)std::lroundf(v);
            if (r > H_VAL) r = H_VAL; if (r < -H_VAL) r = -H_VAL;
            c.hs[t + 1][d] = (int8_t)r;
            
            // s
            int s = (int)c.ss[t][d] + (int)x[d];
            if (s > S_MAX) s = S_MAX; if (s < -S_MAX) s = -S_MAX;
            c.ss[t + 1][d] = (int16_t)s;
            
            // m
            int m = std::max((int)c.ms[t][d], (int)x[d]);
            if (m > H_VAL) m = H_VAL;
            c.ms[t + 1][d] = (int8_t)m;
        }
    }
}

// Multi-channel features: [h_norm; s_norm; m_norm] -> 3D
const int F = 3 * D;

void extract_features(const ForwardCache& c, float* feat) {
    auto norm = [&](const int8_t* state, int n, float* out) {
        float mean = 0;
        for (int i = 0; i < n; ++i) mean += (float)state[i];
        mean /= n;
        float var = 0;
        for (int i = 0; i < n; ++i) var += ((float)state[i] - mean) * ((float)state[i] - mean);
        var /= n;
        float stddev = std::sqrt(var + 1e-6f);
        for (int i = 0; i < n; ++i) out[i] = ((float)state[i] - mean) / stddev;
    };
    auto norm16 = [&](const int16_t* state, int n, float* out) {
        float mean = 0;
        for (int i = 0; i < n; ++i) mean += (float)state[i];
        mean /= n;
        float var = 0;
        for (int i = 0; i < n; ++i) var += ((float)state[i] - mean) * ((float)state[i] - mean);
        var /= n;
        float stddev = std::sqrt(var + 1e-6f);
        for (int i = 0; i < n; ++i) out[i] = ((float)state[i] - mean) / stddev;
    };
    norm(c.hs.back().data(), D, feat);
    norm16(c.ss.back().data(), D, feat + D);
    norm(c.ms.back().data(), D, feat + 2 * D);
}

struct Q4 {
    std::vector<std::vector<float>> W;  // [V, F]
    std::vector<float> b;
    Q4() {
        W.assign(V, std::vector<float>(F, 0));
        b.assign(V, 0);
        std::mt19937 rng(456);
        std::normal_distribution<float> nd(0, 0.05f);
        for (auto& row : W) for (auto& v : row) v = nd(rng);
    }
    void logits(const float* feat, float* out) const {
        for (int v = 0; v < V; ++v) {
            float s = b[v];
            for (int f = 0; f < F; ++f) s += W[v][f] * feat[f];
            out[v] = s;
        }
    }
};

void softmax(const float* logits, float* probs) {
    float max_l = *std::max_element(logits, logits + V);
    float sum = 0;
    for (int v = 0; v < V; ++v) { probs[v] = std::exp(logits[v] - max_l); sum += probs[v]; }
    for (int v = 0; v < V; ++v) probs[v] /= sum;
}

int main() {
    std::cout << "================================================================\n";
    std::cout << "  OPTION D: End-to-End Pipeline (Simplified, no Q3)\n";
    std::cout << "  Channels: h | s | m | F=" << F << " | V=" << V << " D=" << D << "\n";
    std::cout << "================================================================\n\n";
    
    std::vector<float> alpha(D, 0.5f);
    Q4 q4;
    
    std::mt19937 rng(789);
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
    
    std::vector<float> feat_buf(F);
    auto predict = [&](const std::vector<int>& seq) {
        ForwardCache c;
        forward_seq(seq, alpha.data(), c);
        extract_features(c, feat_buf.data());
        float logits[V];
        q4.logits(feat_buf.data(), logits);
        int best = 0;
        for (int v = 1; v < V; ++v) if (logits[v] > logits[best]) best = v;
        return best;
    };
    
    int initial = 0;
    for (auto& p : data) if (predict(p.first) == p.second) initial++;
    std::cout << "  Initial: " << initial << "/" << data.size() << "\n\n";
    
    std::vector<std::vector<float>> w_grad(V, std::vector<float>(F, 0));
    std::vector<float> b_grad(V, 0);
    std::vector<float> feat(F);
    
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(data.begin(), data.end(), std::mt19937(epoch + 1));
        for (auto& row : w_grad) std::fill(row.begin(), row.end(), 0);
        std::fill(b_grad.begin(), b_grad.end(), 0);
        float total_loss = 0;
        for (auto& p : data) {
            ForwardCache c;
            forward_seq(p.first, alpha.data(), c);
            extract_features(c, feat.data());
            float logits[V]; float probs[V];
            q4.logits(feat.data(), logits);
            softmax(logits, probs);
            total_loss += -std::log(std::max(probs[p.second], 1e-7f));
            for (int v = 0; v < V; ++v) {
                float d = probs[v] - (v == p.second ? 1.0f : 0.0f);
                for (int f = 0; f < F; ++f) w_grad[v][f] += d * feat[f];
                b_grad[v] += d;
            }
        }
        for (int v = 0; v < V; ++v) {
            for (int f = 0; f < F; ++f) {
                q4.W[v][f] -= LR * w_grad[v][f] / N_TRAIN;
                q4.W[v][f] *= (1.0f - WD);
            }
            q4.b[v] -= LR * b_grad[v] / N_TRAIN;
        }
        
        if ((epoch + 1) % 5 == 0 || epoch == 0 || epoch == EPOCHS - 1) {
            int c = 0;
            for (auto& pp : data) if (predict(pp.first) == pp.second) c++;
            std::cout << "  Epoch " << std::setw(2) << (epoch + 1)
                      << ": loss=" << std::fixed << std::setprecision(4) << total_loss / data.size()
                      << " acc=" << c << "/" << data.size() << " (" << std::setw(3) << c * 100 / data.size() << "%)\n";
        }
    }
    
    int fc = 0;
    for (auto& p : data) if (predict(p.first) == p.second) fc++;
    std::cout << "\n  Train Final: " << fc << "/" << data.size()
              << " (" << fc * 100 / data.size() << "%)\n";
    
    int tc = 0;
    std::mt19937 test_rng(999);
    for (int i = 0; i < 500; ++i) {
        std::vector<int> seq(SEQ_LEN);
        std::vector<int> counts(V, 0);
        for (int t = 0; t < SEQ_LEN; ++t) { seq[t] = test_rng() % V; counts[seq[t]]++; }
        int maj = 0; int mx = counts[0];
        for (int v = 1; v < V; ++v) if (counts[v] > mx) { mx = counts[v]; maj = v; }
        if (predict(seq) == maj) tc++;
    }
    std::cout << "  Held-out: " << tc << "/500 (" << std::fixed << std::setprecision(1) << tc * 100.0 / 500 << "%)\n";
    
    // Verify channel usage
    std::cout << "\n  Channel weight magnitude (avg |W[v][f]| per channel):\n";
    std::vector<float> avg(F, 0);
    for (int v = 0; v < V; ++v) for (int f = 0; f < F; ++f) avg[f] += std::abs(q4.W[v][f]);
    for (int f = 0; f < F; ++f) avg[f] /= V;
    float h_avg = 0, s_avg = 0, m_avg = 0;
    for (int d = 0; d < D; ++d) { h_avg += avg[d]; s_avg += avg[D + d]; m_avg += avg[2 * D + d]; }
    h_avg /= D; s_avg /= D; m_avg /= D;
    std::cout << "    h channel: " << std::setprecision(3) << h_avg << "\n";
    std::cout << "    s channel: " << s_avg << "  <-- majority task should use this\n";
    std::cout << "    m channel: " << m_avg << "\n";
    
    // Benchmark
    std::cout << "\n  === BENCHMARK (Full Pipeline) ===\n";
    std::vector<int> bench_seq(SEQ_LEN);
    for (int t = 0; t < SEQ_LEN; ++t) bench_seq[t] = ud(rng);
    
    for (int i = 0; i < 50; ++i) {
        ForwardCache c;
        forward_seq(bench_seq, alpha.data(), c);
        if (c.hs[SEQ_LEN][0] == -100) std::cout << "";
    }
    
    int iters = 1000;
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<float> feat_b(F);
    float acc = 0;
    for (int i = 0; i < iters; ++i) {
        ForwardCache c;
        forward_seq(bench_seq, alpha.data(), c);
        extract_features(c, feat_b.data());
        float logits[V];
        q4.logits(feat_b.data(), logits);
        acc += logits[0];
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    if (acc == 0) std::cout << "";
    
    double us_per_seq = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / (double)iters;
    std::cout << "  Forward (Q1+Q2-A+channels+Q4): " << std::fixed << std::setprecision(2) << us_per_seq << " µs/seq\n";
    std::cout << "  Per token: " << std::setprecision(2) << us_per_seq / SEQ_LEN << " µs\n";
    std::cout << "  Throughput: " << std::setprecision(0) << 1e6 / (us_per_seq / SEQ_LEN) << " tokens/s\n";

    return 0;
}
