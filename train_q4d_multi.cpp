// train_q4d_multi.cpp
// Final Option D: Multi-task pipeline with separate Q4 heads per channel.
// Task 1 (s channel): predict majority
// Task 2 (h channel): predict last token
// Verifies the full end-to-end pipeline works on multiple tasks.

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
const int EPOCHS = 50;
const int N_TRAIN = 1000;
const int V = 8;
const float LR = 0.3f;
const float WD = 0.02f;

void q1(int token_id, int8_t* x) {
    std::fill(x, x + D, 0);
    if (token_id >= 0 && token_id < V) x[token_id] = 3;
}

struct ForwardCache {
    std::vector<std::vector<int8_t>> hs;   // last x_t (with EMA smoothing)
    std::vector<std::vector<int16_t>> ss; // sum
};

void forward_seq(const std::vector<int>& tokens, ForwardCache& c) {
    c.hs.assign(SEQ_LEN + 1, std::vector<int8_t>(D, 0));
    c.ss.assign(SEQ_LEN + 1, std::vector<int16_t>(D, 0));
    std::vector<int8_t> x(D);
    for (int t = 0; t < SEQ_LEN; ++t) {
        q1(tokens[t], x.data());
        for (int d = 0; d < D; ++d) {
            float v = 0.3f * (float)c.hs[t][d] + 0.7f * (float)x[d];  // alpha=0.3 (more current)
            int r = (int)std::lroundf(v);
            if (r > H_VAL) r = H_VAL; if (r < -H_VAL) r = -H_VAL;
            c.hs[t + 1][d] = (int8_t)r;
            int s = (int)c.ss[t][d] + (int)x[d];
            if (s > S_MAX) s = S_MAX; if (s < -S_MAX) s = -S_MAX;
            c.ss[t + 1][d] = (int16_t)s;
        }
    }
}

void normalize(const int8_t* state, int n, float* out) {
    float mean = 0;
    for (int i = 0; i < n; ++i) mean += (float)state[i];
    mean /= n;
    float var = 0;
    for (int i = 0; i < n; ++i) var += ((float)state[i] - mean) * ((float)state[i] - mean);
    var /= n;
    float stddev = std::sqrt(var + 1e-6f);
    for (int i = 0; i < n; ++i) out[i] = ((float)state[i] - mean) / stddev;
}

void normalize16(const int16_t* state, int n, float* out) {
    float mean = 0;
    for (int i = 0; i < n; ++i) mean += (float)state[i];
    mean /= n;
    float var = 0;
    for (int i = 0; i < n; ++i) var += ((float)state[i] - mean) * ((float)state[i] - mean);
    var /= n;
    float stddev = std::sqrt(var + 1e-6f);
    for (int i = 0; i < n; ++i) out[i] = ((float)state[i] - mean) / stddev;
}

struct Q4 {
    std::vector<std::vector<float>> W;  // [V, D]
    std::vector<float> b;
    Q4() {
        W.assign(V, std::vector<float>(D, 0));
        b.assign(V, 0);
        std::mt19937 rng(456);
        std::normal_distribution<float> nd(0, 0.05f);
        for (auto& row : W) for (auto& v : row) v = nd(rng);
    }
    void logits(const float* feat, float* out) const {
        for (int v = 0; v < V; ++v) {
            float s = b[v];
            for (int d = 0; d < D; ++d) s += W[v][d] * feat[d];
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

struct Sample {
    std::vector<int> seq;
    int majority;
    int last;
};

int main() {
    std::cout << "================================================================\n";
    std::cout << "  OPTION D (Final): End-to-End Multi-Task Pipeline\n";
    std::cout << "  Q1 + Q2-A -> h | s -> Q4_major (s) + Q4_last (h) | V=" << V << " D=" << D << "\n";
    std::cout << "================================================================\n\n";
    
    Q4 q4_major, q4_last;
    
    std::mt19937 rng(789);
    std::uniform_int_distribution<int> ud(0, V - 1);
    std::vector<Sample> data;
    for (int i = 0; i < N_TRAIN; ++i) {
        Sample s;
        s.seq.resize(SEQ_LEN);
        std::vector<int> counts(V, 0);
        for (int t = 0; t < SEQ_LEN; ++t) { s.seq[t] = ud(rng); counts[s.seq[t]]++; }
        s.majority = 0; int mx = counts[0];
        for (int v = 1; v < V; ++v) if (counts[v] > mx) { mx = counts[v]; s.majority = v; }
        s.last = s.seq.back();
        data.push_back(s);
    }
    
    auto predict_major = [&](const std::vector<int>& seq) {
        ForwardCache c;
        forward_seq(seq, c);
        std::vector<float> s_norm(D);
        normalize16(c.ss.back().data(), D, s_norm.data());
        float logits[V];
        q4_major.logits(s_norm.data(), logits);
        int best = 0;
        for (int v = 1; v < V; ++v) if (logits[v] > logits[best]) best = v;
        return best;
    };
    auto predict_last = [&](const std::vector<int>& seq) {
        ForwardCache c;
        forward_seq(seq, c);
        std::vector<float> h_norm(D);
        normalize(c.hs.back().data(), D, h_norm.data());
        float logits[V];
        q4_last.logits(h_norm.data(), logits);
        int best = 0;
        for (int v = 1; v < V; ++v) if (logits[v] > logits[best]) best = v;
        return best;
    };
    
    int im = 0, il = 0;
    for (auto& s : data) {
        if (predict_major(s.seq) == s.majority) im++;
        if (predict_last(s.seq) == s.last) il++;
    }
    std::cout << "  Initial: majority=" << im << "/" << data.size()
              << ", last=" << il << "/" << data.size() << "\n\n";
    
    std::vector<std::vector<float>> wg_m(V, std::vector<float>(D, 0));
    std::vector<float> bg_m(V, 0);
    std::vector<std::vector<float>> wg_l(V, std::vector<float>(D, 0));
    std::vector<float> bg_l(V, 0);
    
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(data.begin(), data.end(), std::mt19937(epoch + 1));
        for (auto& r : wg_m) std::fill(r.begin(), r.end(), 0);
        for (auto& r : wg_l) std::fill(r.begin(), r.end(), 0);
        std::fill(bg_m.begin(), bg_m.end(), 0);
        std::fill(bg_l.begin(), bg_l.end(), 0);
        float total_loss = 0;
        
        std::vector<float> s_norm(D), h_norm(D);
        for (auto& s : data) {
            ForwardCache c;
            forward_seq(s.seq, c);
            normalize16(c.ss.back().data(), D, s_norm.data());
            normalize(c.hs.back().data(), D, h_norm.data());
            
            // Majority
            float lm[V]; float pm[V];
            q4_major.logits(s_norm.data(), lm);
            softmax(lm, pm);
            total_loss += -std::log(std::max(pm[s.majority], 1e-7f));
            for (int v = 0; v < V; ++v) {
                float d = pm[v] - (v == s.majority ? 1.0f : 0.0f);
                for (int d_i = 0; d_i < D; ++d_i) wg_m[v][d_i] += d * s_norm[d_i];
                bg_m[v] += d;
            }
            
            // Last
            float ll[V]; float pl[V];
            q4_last.logits(h_norm.data(), ll);
            softmax(ll, pl);
            total_loss += -std::log(std::max(pl[s.last], 1e-7f));
            for (int v = 0; v < V; ++v) {
                float d = pl[v] - (v == s.last ? 1.0f : 0.0f);
                for (int d_i = 0; d_i < D; ++d_i) wg_l[v][d_i] += d * h_norm[d_i];
                bg_l[v] += d;
            }
        }
        
        for (int v = 0; v < V; ++v) {
            for (int d_i = 0; d_i < D; ++d_i) {
                q4_major.W[v][d_i] -= LR * wg_m[v][d_i] / N_TRAIN;
                q4_major.W[v][d_i] *= (1.0f - WD);
                q4_last.W[v][d_i] -= LR * wg_l[v][d_i] / N_TRAIN;
                q4_last.W[v][d_i] *= (1.0f - WD);
            }
            q4_major.b[v] -= LR * bg_m[v] / N_TRAIN;
            q4_last.b[v] -= LR * bg_l[v] / N_TRAIN;
        }
        
        if ((epoch + 1) % 5 == 0 || epoch == 0 || epoch == EPOCHS - 1) {
            int mj = 0, ls = 0;
            for (auto& s : data) {
                if (predict_major(s.seq) == s.majority) mj++;
                if (predict_last(s.seq) == s.last) ls++;
            }
            std::cout << "  Epoch " << std::setw(2) << (epoch + 1)
                      << ": loss=" << std::fixed << std::setprecision(4) << total_loss / data.size()
                      << " | maj=" << std::setw(3) << mj * 100 / data.size() << "%"
                      << " last=" << std::setw(3) << ls * 100 / data.size() << "%"
                      << "\n";
        }
    }
    
    int fm = 0, fl = 0;
    for (auto& s : data) {
        if (predict_major(s.seq) == s.majority) fm++;
        if (predict_last(s.seq) == s.last) fl++;
    }
    std::cout << "\n  Train Final: majority=" << fm << "/" << data.size()
              << " (" << fm * 100 / data.size() << "%)"
              << ", last=" << fl << "/" << data.size()
              << " (" << fl * 100 / data.size() << "%)\n";
    
    int hm = 0, hl = 0;
    std::mt19937 test_rng(999);
    for (int i = 0; i < 500; ++i) {
        std::vector<int> seq(SEQ_LEN);
        std::vector<int> counts(V, 0);
        for (int t = 0; t < SEQ_LEN; ++t) { seq[t] = test_rng() % V; counts[seq[t]]++; }
        int maj = 0; int mx = counts[0];
        for (int v = 1; v < V; ++v) if (counts[v] > mx) { mx = counts[v]; maj = v; }
        if (predict_major(seq) == maj) hm++;
        if (predict_last(seq) == seq.back()) hl++;
    }
    std::cout << "  Held-out: majority=" << hm << "/500 (" << std::fixed << std::setprecision(1) << hm * 100.0 / 500 << "%)"
              << ", last=" << hl << "/500 (" << hl * 100.0 / 500 << "%)\n";
    
    std::cout << "\n  === BENCHMARK (Full Multi-Task Pipeline) ===\n";
    std::vector<int> bench_seq(SEQ_LEN);
    for (int t = 0; t < SEQ_LEN; ++t) bench_seq[t] = ud(rng);
    
    for (int i = 0; i < 50; ++i) {
        ForwardCache c;
        forward_seq(bench_seq, c);
        if (c.hs[SEQ_LEN][0] == -100) std::cout << "";
    }
    
    int iters = 1000;
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<float> s_norm_b(D), h_norm_b(D);
    float acc = 0;
    for (int i = 0; i < iters; ++i) {
        ForwardCache c;
        forward_seq(bench_seq, c);
        normalize16(c.ss.back().data(), D, s_norm_b.data());
        normalize(c.hs.back().data(), D, h_norm_b.data());
        float lm[V], ll[V];
        q4_major.logits(s_norm_b.data(), lm);
        q4_last.logits(h_norm_b.data(), ll);
        acc += lm[0] + ll[0];
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    if (acc == 0) std::cout << "";
    
    double us_per_seq = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / (double)iters;
    std::cout << "  Forward + 2 Q4 heads: " << std::fixed << std::setprecision(2) << us_per_seq << " µs/seq\n";
    std::cout << "  Per token: " << std::setprecision(2) << us_per_seq / SEQ_LEN << " µs\n";
    std::cout << "  Throughput: " << std::setprecision(0) << 1e6 / (us_per_seq / SEQ_LEN) << " tokens/s\n";
    
    return 0;
}
