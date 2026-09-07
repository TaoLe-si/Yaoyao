// train_q4d.cpp
// OPTION D: End-to-end pipeline with multi-channel state.
//
// Q1 (one-hot) -> Q3 (k=3 conv) -> Q2-A -> multi-channel state:
//   h: recent EMA
//   s: sum
//   m: max
//   p: parity (count % 2 sign)
// -> Q4 multi-channel linear projection
// Tasks: predict majority (uses s), predict last token (uses h)

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>

const int D = 128;
const int H_VAL = 4;
const int S_MAX = 64;
const int SEQ_LEN = 16;
const int EPOCHS = 40;
const int N_TRAIN = 1000;
const int V = 8;            // manageable vocab
const int W_MAX_Q3 = 1.0f;
const float LR = 0.3f;
const float WD = 0.02f;
const float WD_Q3 = 0.05f;

void q1(int token_id, int8_t* x) {
    std::fill(x, x + D, 0);
    if (token_id >= 0 && token_id < V) x[token_id] = 3;
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

void q2a_step(const int8_t* h_old, const int8_t* y,
              const float* alpha, int8_t* h_new) {
    for (int d = 0; d < D; ++d) {
        float v = alpha[d] * (float)h_old[d] + (1.0f - alpha[d]) * (float)y[d];
        int r = (int)std::lroundf(v);
        if (r > H_VAL) r = H_VAL; if (r < -H_VAL) r = -H_VAL;
        h_new[d] = (int8_t)r;
    }
}

struct ForwardCache {
    std::vector<std::vector<int8_t>> xs, ys, hs;
    std::vector<std::vector<int16_t>> ss;
    std::vector<std::vector<int8_t>> ms;  // max
    std::vector<std::vector<int8_t>> ps;  // parity sign
};

void forward_seq(const std::vector<int>& tokens,
                 const float* w0, const float* w1, const float* w2,
                 const float* alpha, ForwardCache& c) {
    c.xs.assign(SEQ_LEN, std::vector<int8_t>(D, 0));
    c.ys.assign(SEQ_LEN, std::vector<int8_t>(D, 0));
    c.hs.assign(SEQ_LEN + 1, std::vector<int8_t>(D, 0));
    c.ss.assign(SEQ_LEN + 1, std::vector<int16_t>(D, 0));
    c.ms.assign(SEQ_LEN + 1, std::vector<int8_t>(D, 0));
    c.ps.assign(SEQ_LEN + 1, std::vector<int8_t>(D, 0));
    
    std::vector<int8_t> xc(D, 0), xp(D, 0), xp2(D, 0), y(D, 0);
    for (int t = 0; t < SEQ_LEN; ++t) {
        xp2 = xp; xp = xc;
        q1(tokens[t], xc.data());
        c.xs[t] = xc;
        q3_conv(xp2.data(), xp.data(), xc.data(), w0, w1, w2, y.data());
        c.ys[t] = y;
        // Update all channels
        for (int d = 0; d < D; ++d) {
            // h (EMA)
            float v = alpha[d] * (float)c.hs[t][d] + (1.0f - alpha[d]) * (float)y[d];
            int r = (int)std::lroundf(v);
            if (r > H_VAL) r = H_VAL; if (r < -H_VAL) r = -H_VAL;
            c.hs[t + 1][d] = (int8_t)r;
            
            // s (sum)
            int s = (int)c.ss[t][d] + (int)y[d];
            if (s > S_MAX) s = S_MAX; if (s < -S_MAX) s = -S_MAX;
            c.ss[t + 1][d] = (int16_t)s;
            
            // m (max)
            int m = std::max((int)c.ms[t][d], (int)y[d]);
            if (m > H_VAL) m = H_VAL;
            c.ms[t + 1][d] = (int8_t)m;
            
            // p (parity: sign(cumulative product))
            int cur_p = (int)c.ps[t][d];
            // parity sign: 1 if even count, -1 if odd count
            // Initialize: 1 (no occurrence). Flip sign each time y[d] > 0.
            // But y can be 0 too. Let's say: y[d] > 0 contributes to "active" parity
            int new_p = cur_p;
            if (y[d] > 0) new_p = -cur_p;  // flip on positive contribution
            c.ps[t + 1][d] = (int8_t)new_p;
        }
    }
}

// Normalize and concatenate channels
// Feature = [h_norm; s_norm; m_norm; p_norm] (4D total)
const int F = 4 * D;

void extract_features(const ForwardCache& c, float* features) {
    // h
    float mean = 0;
    for (int d = 0; d < D; ++d) mean += (float)c.hs.back()[d];
    mean /= D;
    float var = 0;
    for (int d = 0; d < D; ++d) var += ((float)c.hs.back()[d] - mean) * ((float)c.hs.back()[d] - mean);
    var /= D;
    float stddev = std::sqrt(var + 1e-6f);
    for (int d = 0; d < D; ++d) features[d] = ((float)c.hs.back()[d] - mean) / stddev;
    
    // s
    mean = 0;
    for (int d = 0; d < D; ++d) mean += (float)c.ss.back()[d];
    mean /= D;
    var = 0;
    for (int d = 0; d < D; ++d) var += ((float)c.ss.back()[d] - mean) * ((float)c.ss.back()[d] - mean);
    var /= D;
    stddev = std::sqrt(var + 1e-6f);
    for (int d = 0; d < D; ++d) features[D + d] = ((float)c.ss.back()[d] - mean) / stddev;
    
    // m
    mean = 0;
    for (int d = 0; d < D; ++d) mean += (float)c.ms.back()[d];
    mean /= D;
    var = 0;
    for (int d = 0; d < D; ++d) var += ((float)c.ms.back()[d] - mean) * ((float)c.ms.back()[d] - mean);
    var /= D;
    stddev = std::sqrt(var + 1e-6f);
    for (int d = 0; d < D; ++d) features[2 * D + d] = ((float)c.ms.back()[d] - mean) / stddev;
    
    // p (already in {-1, +1})
    for (int d = 0; d < D; ++d) features[3 * D + d] = (float)c.ps.back()[d];
}

// Multi-task Q4: predict majority (uses s) AND predict last token (uses h)
struct MultiTaskQ4 {
    std::vector<std::vector<float>> W_major;  // [V, F]
    std::vector<float> b_major;
    std::vector<std::vector<float>> W_last;   // [V, F]
    std::vector<float> b_last;
    
    MultiTaskQ4() {
        W_major.assign(V, std::vector<float>(F, 0));
        b_major.assign(V, 0);
        W_last.assign(V, std::vector<float>(F, 0));
        b_last.assign(V, 0);
        std::mt19937 rng(456);
        std::normal_distribution<float> nd(0, 0.02f);
        for (auto& row : W_major) for (auto& v : row) v = nd(rng);
        for (auto& row : W_last) for (auto& v : row) v = nd(rng);
    }
    
    void logits_major(const float* feat, float* out) const {
        for (int v = 0; v < V; ++v) {
            float s = b_major[v];
            for (int f = 0; f < F; ++f) s += W_major[v][f] * feat[f];
            out[v] = s;
        }
    }
    void logits_last(const float* feat, float* out) const {
        for (int v = 0; v < V; ++v) {
            float s = b_last[v];
            for (int f = 0; f < F; ++f) s += W_last[v][f] * feat[f];
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
    std::cout << "  OPTION D: End-to-End Multi-Channel Pipeline\n";
    std::cout << "  Q1+Q3+Q2-A+h|s|m|p | Q4 multi-task | V=" << V << ", D=" << D << ", F=" << F << "\n";
    std::cout << "================================================================\n\n";
    
    std::vector<float> w0(D, 0), w1(D, 0), w2(D, 0);
    std::vector<float> alpha(D, 0.5f);
    MultiTaskQ4 q4;
    
    std::mt19937 rng(789);
    std::normal_distribution<float> nd(0, 0.05f);
    for (auto& v : w0) v = nd(rng);
    for (auto& v : w1) v = nd(rng);
    for (auto& v : w2) v = nd(rng);
    
    std::uniform_int_distribution<int> ud(0, V - 1);
    struct Sample {
        std::vector<int> seq;
        int majority;
        int last;
    };
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
    
    std::vector<float> feat_buf(F);
    auto predict_major = [&](const std::vector<int>& seq) {
        ForwardCache c;
        forward_seq(seq, w0.data(), w1.data(), w2.data(), alpha.data(), c);
        extract_features(c, feat_buf.data());
        float logits[V];
        q4.logits_major(feat_buf.data(), logits);
        int best = 0;
        for (int v = 1; v < V; ++v) if (logits[v] > logits[best]) best = v;
        return best;
    };
    auto predict_last = [&](const std::vector<int>& seq) {
        ForwardCache c;
        forward_seq(seq, w0.data(), w1.data(), w2.data(), alpha.data(), c);
        extract_features(c, feat_buf.data());
        float logits[V];
        q4.logits_last(feat_buf.data(), logits);
        int best = 0;
        for (int v = 1; v < V; ++v) if (logits[v] > logits[best]) best = v;
        return best;
    };
    
    int init_maj = 0, init_last = 0;
    for (auto& s : data) {
        if (predict_major(s.seq) == s.majority) init_maj++;
        if (predict_last(s.seq) == s.last) init_last++;
    }
    std::cout << "  Initial: majority=" << init_maj << "/" << data.size() 
              << ", last=" << init_last << "/" << data.size() << "\n\n";
    
    // Training
    std::vector<std::vector<float>> w_grad_m(V, std::vector<float>(F, 0));
    std::vector<float> b_grad_m(V, 0);
    std::vector<std::vector<float>> w_grad_l(V, std::vector<float>(F, 0));
    std::vector<float> b_grad_l(V, 0);
    
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(data.begin(), data.end(), std::mt19937(epoch + 1));
        for (auto& row : w_grad_m) std::fill(row.begin(), row.end(), 0);
        for (auto& row : w_grad_l) std::fill(row.begin(), row.end(), 0);
        std::fill(b_grad_m.begin(), b_grad_m.end(), 0);
        std::fill(b_grad_l.begin(), b_grad_l.end(), 0);
        float total_loss = 0;
        
        std::vector<float> feat(F);
        for (auto& s : data) {
            ForwardCache c;
            forward_seq(s.seq, w0.data(), w1.data(), w2.data(), alpha.data(), c);
            extract_features(c, feat.data());
            
            // Majority
            float logits_m[V]; float probs_m[V];
            q4.logits_major(feat.data(), logits_m);
            softmax(logits_m, probs_m);
            total_loss += -std::log(std::max(probs_m[s.majority], 1e-7f));
            for (int v = 0; v < V; ++v) {
                float d = probs_m[v] - (v == s.majority ? 1.0f : 0.0f);
                for (int f = 0; f < F; ++f) w_grad_m[v][f] += d * feat[f];
                b_grad_m[v] += d;
            }
            
            // Last
            float logits_l[V]; float probs_l[V];
            q4.logits_last(feat.data(), logits_l);
            softmax(logits_l, probs_l);
            total_loss += -std::log(std::max(probs_l[s.last], 1e-7f));
            for (int v = 0; v < V; ++v) {
                float d = probs_l[v] - (v == s.last ? 1.0f : 0.0f);
                for (int f = 0; f < F; ++f) w_grad_l[v][f] += d * feat[f];
                b_grad_l[v] += d;
            }
        }
        
        for (int v = 0; v < V; ++v) {
            for (int f = 0; f < F; ++f) {
                q4.W_major[v][f] -= LR * w_grad_m[v][f] / N_TRAIN;
                q4.W_major[v][f] *= (1.0f - WD);
                q4.W_last[v][f] -= LR * w_grad_l[v][f] / N_TRAIN;
                q4.W_last[v][f] *= (1.0f - WD);
            }
            q4.b_major[v] -= LR * b_grad_m[v] / N_TRAIN;
            q4.b_last[v] -= LR * b_grad_l[v] / N_TRAIN;
        }
        
        // Q3 weight clip
        for (int d = 0; d < D; ++d) {
            for (auto* w : {&w0, &w1, &w2}) {
                float& v = (*w)[d];
                if (v > W_MAX_Q3) v = W_MAX_Q3;
                if (v < -W_MAX_Q3) v = -W_MAX_Q3;
                v *= (1.0f - WD_Q3);
            }
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
    
    // Held-out
    std::cout << "\n  Held-out test:\n";
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
    std::cout << "    majority: " << hm << "/500 (" << std::fixed << std::setprecision(1) << hm * 100.0 / 500 << "%)\n";
    std::cout << "    last:     " << hl << "/500 (" << hl * 100.0 / 500 << "%)\n";
    
    // Benchmark
    std::cout << "\n  === BENCHMARK ===\n";
    std::vector<int> bench_seq(SEQ_LEN);
    for (int t = 0; t < SEQ_LEN; ++t) bench_seq[t] = ud(rng);
    
    for (int i = 0; i < 50; ++i) {
        ForwardCache c;
        forward_seq(bench_seq, w0.data(), w1.data(), w2.data(), alpha.data(), c);
        if (c.hs[SEQ_LEN][0] == -100) std::cout << "";
    }
    
    int iters = 1000;
    auto t0 = std::chrono::high_resolution_clock::now();
    float acc = 0;
    std::vector<float> feat_b(F);
    for (int i = 0; i < iters; ++i) {
        ForwardCache c;
        forward_seq(bench_seq, w0.data(), w1.data(), w2.data(), alpha.data(), c);
        extract_features(c, feat_b.data());
        float logits[V];
        q4.logits_major(feat_b.data(), logits);
        acc += logits[0];
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    if (acc == 0) std::cout << "";
    
    double us_per_seq = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / (double)iters;
    std::cout << "  Forward (full pipeline): " << std::fixed << std::setprecision(2) << us_per_seq << " µs/seq\n";
    std::cout << "  Per token: " << std::setprecision(2) << us_per_seq / SEQ_LEN << " µs\n";
    std::cout << "  Throughput: " << std::setprecision(0) << 1e6 / (us_per_seq / SEQ_LEN) << " tokens/s\n";
    
    return 0;
}
