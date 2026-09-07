// train_lm_e2e.cpp
// End-to-end trained LM with Q3 conv + Adam.
//
// Architecture (all trainable):
//   char_emb (int8) -> Q3 (k=3 conv, float weights) -> h, s channels -> Q4 linear
//   h[t] = alpha * h[t-1] + (1-alpha) * y[t]   (EMA)
//   s[t] = s[t-1] + x[t]                        (sum, int16)
//   logits[v] = W_h · h + W_s · s + bias
// Optimizer: Adam

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <string>
#include <map>

struct Vocab {
    std::map<char, int> char_to_id;
    std::map<int, char> id_to_char;
    Vocab() {
        std::string s = "0123456789:abcdefghijklmnopqrstuvwxyz\n ";
        for (int i = 0; i < (int)s.size(); ++i) {
            char_to_id[s[i]] = i;
            id_to_char[i] = s[i];
        }
    }
    int size() const { return (int)id_to_char.size(); }
    int encode(char c) const { return char_to_id.at(c); }
    char decode(int id) const { return id_to_char.at(id); }
    int newline_id() const { return encode('\n'); }
};

// Adam state
struct AdamState {
    std::vector<float> m, v;
    int t = 0;
};

// End-to-end LM
struct CharLM {
    int D, V;
    std::vector<std::vector<int8_t>> char_emb;    // [V, D]
    std::vector<float> q3_w0, q3_w1, q3_w2;       // [D] each
    std::vector<float> alpha;                    // [D] in [0, 1]
    std::vector<std::vector<float>> W_h, W_s;     // [V, D]
    std::vector<float> bias;                      // [V]
    
    // Adam states
    AdamState st_emb, st_q3_w0, st_q3_w1, st_q3_w2, st_alpha, st_Wh, st_Ws, st_b;
    
    CharLM(int d, int v) : D(d), V(v) {
        char_emb.assign(V, std::vector<int8_t>(D, 0));
        q3_w0.assign(D, 0); q3_w1.assign(D, 0); q3_w2.assign(D, 0);
        alpha.assign(D, 0.5f);
        W_h.assign(V, std::vector<float>(D, 0));
        W_s.assign(V, std::vector<float>(D, 0));
        bias.assign(V, 0);
        
        std::mt19937 rng(42);
        std::normal_distribution<float> nd_emb(0, 1.0f);
        std::normal_distribution<float> nd_w(0, 0.1f);
        std::normal_distribution<float> nd_alpha(0, 0.1f);
        
        for (int v = 0; v < V; ++v) for (int d = 0; d < D; ++d) {
            int r = (int)std::lroundf(nd_emb(rng));
            if (r > 4) r = 4; if (r < -4) r = -4;
            char_emb[v][d] = (int8_t)r;
            W_h[v][d] = nd_w(rng);
            W_s[v][d] = nd_w(rng);
        }
        for (int d = 0; d < D; ++d) {
            q3_w0[d] = nd_w(rng);
            q3_w1[d] = nd_w(rng);
            q3_w2[d] = nd_w(rng);
            alpha[d] = 0.3f + nd_alpha(rng) * 0.1f;
        }
        
        // Init Adam states
        st_emb.m.assign(V * D, 0); st_emb.v.assign(V * D, 0);
        st_q3_w0.m.assign(D, 0); st_q3_w0.v.assign(D, 0);
        st_q3_w1.m.assign(D, 0); st_q3_w1.v.assign(D, 0);
        st_q3_w2.m.assign(D, 0); st_q3_w2.v.assign(D, 0);
        st_alpha.m.assign(D, 0); st_alpha.v.assign(D, 0);
        st_Wh.m.assign(V * D, 0); st_Wh.v.assign(V * D, 0);
        st_Ws.m.assign(V * D, 0); st_Ws.v.assign(V * D, 0);
        st_b.m.assign(V, 0); st_b.v.assign(V, 0);
    }
    
    // Forward: compute logits and store intermediate states
    struct FwdState {
        std::vector<std::vector<int8_t>> xs;        // input embeddings
        std::vector<std::vector<int8_t>> ys;        // Q3 outputs
        std::vector<std::vector<int8_t>> hs;        // h channel
        std::vector<std::vector<int16_t>> ss;       // s channel
        std::vector<std::vector<float>> logits;
    };
    
    void forward(const std::vector<int>& input, FwdState& st) const {
        int L = input.size();
        st.xs.assign(L, std::vector<int8_t>(D, 0));
        st.ys.assign(L, std::vector<int8_t>(D, 0));
        st.hs.assign(L + 1, std::vector<int8_t>(D, 0));
        st.ss.assign(L + 1, std::vector<int16_t>(D, 0));
        st.logits.assign(L, std::vector<float>(V, 0));
        
        for (int t = 0; t < L; ++t) {
            // x[t] = char_emb[input[t]]
            const int8_t* emb = char_emb[input[t]].data();
            for (int d = 0; d < D; ++d) st.xs[t][d] = emb[d];
            
            // s[t] = s[t-1] + x[t]
            for (int d = 0; d < D; ++d) {
                int v = (int)st.ss[t][d] + (int)emb[d];
                st.ss[t + 1][d] = (int16_t)v;
            }
        }
        
        // Q3 conv + h channel
        for (int t = 0; t < L; ++t) {
            // y[t] = w_0 * x[t-2] + w_1 * x[t-1] + w_2 * x[t]
            for (int d = 0; d < D; ++d) {
                float v = 0;
                if (t >= 2) v += q3_w0[d] * (float)st.xs[t - 2][d];
                if (t >= 1) v += q3_w1[d] * (float)st.xs[t - 1][d];
                v += q3_w2[d] * (float)st.xs[t][d];
                int r = (int)std::lroundf(v);
                if (r > 4) r = 4; if (r < -4) r = -4;
                st.ys[t][d] = (int8_t)r;
            }
            // h[t] = alpha * h[t-1] + (1-alpha) * y[t]
            for (int d = 0; d < D; ++d) {
                float v = alpha[d] * (float)st.hs[t][d] + (1.0f - alpha[d]) * (float)st.ys[t][d];
                int r = (int)std::lroundf(v);
                if (r > 4) r = 4; if (r < -4) r = -4;
                st.hs[t + 1][d] = (int8_t)r;
            }
        }
        
        // Logits: W_h · h[t] + W_s · s[t]
        for (int t = 0; t < L; ++t) {
            for (int v = 0; v < V; ++v) {
                float logit = bias[v];
                for (int d = 0; d < D; ++d) {
                    logit += W_h[v][d] * (float)st.hs[t + 1][d];
                    logit += W_s[v][d] * (float)st.ss[t + 1][d];
                }
                st.logits[t][v] = logit;
            }
        }
    }
    
    // Adam update for a flat vector
    void adam_update(std::vector<float>& p, AdamState& st, const std::vector<float>& g, float lr) {
        st.t++;
        float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        float bc1 = 1.0f - std::pow(b1, st.t);
        float bc2 = 1.0f - std::pow(b2, st.t);
        for (size_t i = 0; i < p.size(); ++i) {
            st.m[i] = b1 * st.m[i] + (1 - b1) * g[i];
            st.v[i] = b2 * st.v[i] + (1 - b2) * g[i] * g[i];
            float mh = st.m[i] / bc1;
            float vh = st.v[i] / bc2;
            p[i] -= lr * mh / (std::sqrt(vh) + eps);
        }
    }
    
    // Flatten helper
    std::vector<float> flatten(const std::vector<std::vector<int8_t>>& m) const {
        std::vector<float> f(m.size() * m[0].size());
        for (size_t i = 0; i < m.size(); ++i)
            for (size_t d = 0; d < m[i].size(); ++d) f[i * m[0].size() + d] = (float)m[i][d];
        return f;
    }
    void unflatten(const std::vector<float>& f, std::vector<std::vector<int8_t>>& m) const {
        for (size_t i = 0; i < m.size(); ++i)
            for (size_t d = 0; d < m[i].size(); ++d) {
                int r = (int)std::lroundf(f[i * m[0].size() + d]);
                if (r > 4) r = 4; if (r < -4) r = -4;
                m[i][d] = (int8_t)r;
            }
    }
    std::vector<float> flatten(const std::vector<std::vector<float>>& m) const {
        std::vector<float> f(m.size() * m[0].size());
        for (size_t i = 0; i < m.size(); ++i)
            for (size_t d = 0; d < m[i].size(); ++d) f[i * m[0].size() + d] = m[i][d];
        return f;
    }
    void unflatten(const std::vector<float>& f, std::vector<std::vector<float>>& m) const {
        for (size_t i = 0; i < m.size(); ++i)
            for (size_t d = 0; d < m[i].size(); ++d) m[i][d] = f[i * m[0].size() + d];
    }
    
    float train_step(const std::vector<int>& input, const std::vector<int>& target,
                     const std::vector<int>& mask, float lr) {
        int L = input.size();
        FwdState fwd;
        forward(input, fwd);
        
        int valid = 0;
        float total_loss = 0;
        
        // Gradients
        std::vector<std::vector<float>> wh_grad(V, std::vector<float>(D, 0));
        std::vector<std::vector<float>> ws_grad(V, std::vector<float>(D, 0));
        std::vector<float> b_grad(V, 0);
        std::vector<std::vector<float>> emb_grad(V, std::vector<float>(D, 0));
        std::vector<float> q3_w0_grad(D, 0), q3_w1_grad(D, 0), q3_w2_grad(D, 0);
        std::vector<float> alpha_grad(D, 0);
        
        // d_h[t+1] and d_s[t+1] (backprop through Q2-A + Sum)
        std::vector<std::vector<float>> dh(L + 1, std::vector<float>(D, 0));
        std::vector<std::vector<float>> ds(L + 1, std::vector<float>(D, 0));
        
        for (int t = 0; t < L; ++t) {
            if (mask[t] == 0) continue;
            valid++;
            float max_l = *std::max_element(fwd.logits[t].begin(), fwd.logits[t].end());
            float sum = 0;
            std::vector<float> probs(V);
            for (int v = 0; v < V; ++v) { probs[v] = std::exp(fwd.logits[t][v] - max_l); sum += probs[v]; }
            for (int v = 0; v < V; ++v) probs[v] /= sum;
            
            int tgt = target[t];
            total_loss += -std::log(std::max(probs[tgt], 1e-7f));
            
            // d_logits[v] = probs[v] - (v == tgt ? 1 : 0)
            for (int v = 0; v < V; ++v) {
                float d = probs[v] - (v == tgt ? 1.0f : 0.0f);
                for (int d_i = 0; d_i < D; ++d_i) {
                    wh_grad[v][d_i] += d * (float)fwd.hs[t + 1][d_i];
                    ws_grad[v][d_i] += d * (float)fwd.ss[t + 1][d_i];
                    dh[t + 1][d_i] += d * W_h[v][d_i];
                    ds[t + 1][d_i] += d * W_s[v][d_i];
                }
                b_grad[v] += d;
            }
        }
        
        if (valid == 0) return 0;
        
        // Backprop through Sum: s[t+1] = s[t] + x[t]
        // d_s[t] gets contribution from s[t+1]
        for (int t = L - 1; t >= 0; --t) {
            // d_s[t] += d_s[t+1] (sum is just accumulation)
            for (int d = 0; d < D; ++d) {
                ds[t][d] += ds[t + 1][d];
                // d_x[t] += d_s[t+1]
                emb_grad[input[t]][d] += ds[t + 1][d];
            }
        }
        
        // Backprop through h: h[t+1] = alpha * h[t] + (1-alpha) * y[t]
        for (int t = L - 1; t >= 0; --t) {
            // d_alpha[d] from h[t+1]
            for (int d = 0; d < D; ++d) {
                float dh_t = dh[t + 1][d];
                alpha_grad[d] += dh_t * ((float)fwd.hs[t][d] - (float)fwd.ys[t][d]);
                dh[t][d] += dh_t * alpha[d];
            }
            // d_y[t] = dh[t+1] * (1-alpha)
            std::vector<float> dy(D);
            for (int d = 0; d < D; ++d) {
                dy[d] = dh[t + 1][d] * (1.0f - alpha[d]);
            }
            // Q3 backward: y[t] = w_0 * x[t-2] + w_1 * x[t-1] + w_2 * x[t]
            for (int d = 0; d < D; ++d) {
                if (t >= 2) {
                    q3_w0_grad[d] += dy[d] * (float)fwd.xs[t - 2][d];
                    emb_grad[input[t - 2]][d] += dy[d] * q3_w0[d];
                }
                if (t >= 1) {
                    q3_w1_grad[d] += dy[d] * (float)fwd.xs[t - 1][d];
                    emb_grad[input[t - 1]][d] += dy[d] * q3_w1[d];
                }
                q3_w2_grad[d] += dy[d] * (float)fwd.xs[t][d];
                emb_grad[input[t]][d] += dy[d] * q3_w2[d];
            }
        }
        
        // Adam updates
        auto emb_grad_flat = flatten(emb_grad);
        auto wh_grad_flat = flatten(wh_grad);
        auto ws_grad_flat = flatten(ws_grad);
        // Normalize
        for (auto& x : emb_grad_flat) x /= valid;
        for (auto& x : wh_grad_flat) x /= valid;
        for (auto& x : ws_grad_flat) x /= valid;
        for (auto& x : q3_w0_grad) x /= valid;
        for (auto& x : q3_w1_grad) x /= valid;
        for (auto& x : q3_w2_grad) x /= valid;
        for (auto& x : alpha_grad) x /= valid;
        for (auto& x : b_grad) x /= valid;
        
        auto emb_flat = flatten(char_emb);
        adam_update(emb_flat, st_emb, emb_grad_flat, lr);
        unflatten(emb_flat, char_emb);
        adam_update(q3_w0, st_q3_w0, q3_w0_grad, lr);
        adam_update(q3_w1, st_q3_w1, q3_w1_grad, lr);
        adam_update(q3_w2, st_q3_w2, q3_w2_grad, lr);
        adam_update(alpha, st_alpha, alpha_grad, lr);
        // Clamp alpha to [0, 1]
        for (auto& a : alpha) { if (a < 0) a = 0; if (a > 1) a = 1; }
        // Clip Q3 weights
        for (auto& w : q3_w0) { if (w > 2) w = 2; if (w < -2) w = -2; }
        for (auto& w : q3_w1) { if (w > 2) w = 2; if (w < -2) w = -2; }
        for (auto& w : q3_w2) { if (w > 2) w = 2; if (w < -2) w = -2; }
        adam_update(wh_grad_flat, st_Wh, wh_grad_flat, lr);
        unflatten(wh_grad_flat, W_h);
        adam_update(ws_grad_flat, st_Ws, ws_grad_flat, lr);
        unflatten(ws_grad_flat, W_s);
        adam_update(bias, st_b, b_grad, lr);
        
        return total_loss / valid;
    }
    
    std::vector<int> generate(const std::vector<int>& prefix, int max_new, int newline_id) const {
        std::vector<int> out = prefix;
        std::vector<int8_t> h(D, 0);
        std::vector<int16_t> s(D, 0);
        // Process prefix
        for (size_t t = 0; t < prefix.size(); ++t) {
            int id = prefix[t];
            const int8_t* emb = char_emb[id].data();
            for (int d = 0; d < D; ++d) s[d] = (int16_t)((int)s[d] + (int)emb[d]);
            // Q3 + h update
            std::vector<int8_t> x_p2(D, 0), x_p(D, 0), x_c(D, 0);
            for (int d = 0; d < D; ++d) x_c[d] = emb[d];
            for (int d = 0; d < D; ++d) {
                float v = 0;
                if (t >= 2) v += q3_w0[d] * (float)x_p2[d];
                if (t >= 1) v += q3_w1[d] * (float)x_p[d];
                v += q3_w2[d] * (float)x_c[d];
                int r = (int)std::lroundf(v);
                if (r > 4) r = 4; if (r < -4) r = -4;
                int8_t y_d = (int8_t)r;
                float hv = alpha[d] * (float)h[d] + (1.0f - alpha[d]) * (float)y_d;
                int hr = (int)std::lroundf(hv);
                if (hr > 4) hr = 4; if (hr < -4) hr = -4;
                h[d] = (int8_t)hr;
            }
            x_p2 = x_p;
            x_p = x_c;
        }
        for (int step = 0; step < max_new; ++step) {
            int best = 0;
            float best_score = -1e30f;
            for (int v = 0; v < V; ++v) {
                float logit = bias[v];
                for (int d = 0; d < D; ++d) {
                    logit += W_h[v][d] * (float)h[d] + W_s[v][d] * (float)s[d];
                }
                if (logit > best_score) { best_score = logit; best = v; }
            }
            out.push_back(best);
            // Update h, s
            const int8_t* emb = char_emb[best].data();
            for (int d = 0; d < D; ++d) s[d] = (int16_t)((int)s[d] + (int)emb[d]);
            // Simplified h update for generation (Q3 only sees current)
            for (int d = 0; d < D; ++d) {
                float v = q3_w2[d] * (float)emb[d];
                int r = (int)std::lroundf(v);
                if (r > 4) r = 4; if (r < -4) r = -4;
                int8_t y_d = (int8_t)r;
                float hv = alpha[d] * (float)h[d] + (1.0f - alpha[d]) * (float)y_d;
                int hr = (int)std::lroundf(hv);
                if (hr > 4) hr = 4; if (hr < -4) hr = -4;
                h[d] = (int8_t)hr;
            }
            if (best == newline_id) break;
        }
        return out;
    }
};

int main() {
    Vocab vocab;
    const int D = 32;
    const int V = vocab.size();
    
    std::cout << "================================================================\n";
    std::cout << "  E2E TRAINED LM: Q1 + Q3 + Q2-A + Sum + Q4 | D=" << D << "\n";
    std::cout << "  Optimizer: Adam, all weights trainable\n";
    std::cout << "================================================================\n\n";
    
    std::vector<std::string> pairs = {
        "0:zero\n", "1:one\n", "2:two\n", "3:three\n",
        "4:four\n", "5:five\n", "6:six\n", "7:seven\n",
        "8:eight\n", "9:nine\n"
    };
    
    std::cout << "  Training pairs:\n";
    for (auto& p : pairs) std::cout << "    \"" << p.substr(0, p.size() - 1) << "\"\n";
    std::cout << "\n";
    
    CharLM lm(D, V);
    
    int SEQ_LEN = 0;
    for (auto& p : pairs) SEQ_LEN = std::max(SEQ_LEN, (int)p.size());
    
    std::vector<std::vector<int>> train_inputs, train_targets, train_masks;
    for (auto& p : pairs) {
        std::vector<int> inp(SEQ_LEN, vocab.encode(' '));
        std::vector<int> tgt(SEQ_LEN, vocab.encode(' '));
        std::vector<int> mask(SEQ_LEN, 0);
        for (int i = 0; i < (int)p.size() - 1; ++i) {
            inp[i] = vocab.encode(p[i]);
            tgt[i] = vocab.encode(p[i + 1]);
            mask[i] = 1;
        }
        train_inputs.push_back(inp);
        train_targets.push_back(tgt);
        train_masks.push_back(mask);
    }
    
    std::cout << "  Initial generation:\n";
    for (int i = 0; i < 10; ++i) {
        std::vector<int> prefix = {vocab.encode('0' + i), vocab.encode(':')};
        auto gen = lm.generate(prefix, 8, vocab.newline_id());
        std::string result;
        for (int id : gen) result += vocab.decode(id);
        std::string result_clean = result.substr(2);
        if (result_clean.size() > 0 && result_clean.back() == '\n') result_clean.pop_back();
        std::cout << "    \"" << (char)('0' + i) << ":\" -> \"" << result_clean << "\"\n";
    }
    std::cout << "\n";
    
    int epochs = 500;
    std::cout << "  Training (" << epochs << " epochs, Adam lr=0.01)...\n\n";
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle
        std::vector<int> idx(pairs.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), std::mt19937(epoch + 1));
        float total_loss = 0;
        for (int i : idx) {
            total_loss += lm.train_step(train_inputs[i], train_targets[i], train_masks[i], 0.01f);
        }
        if ((epoch + 1) % 50 == 0 || epoch == 0 || epoch == epochs - 1) {
            std::cout << "  Epoch " << std::setw(3) << (epoch + 1)
                      << ": avg_loss=" << std::fixed << std::setprecision(4) << total_loss / train_inputs.size() << "\n";
        }
    }
    
    std::cout << "\n  ========== FINAL TEST: Input -> Output ==========\n\n";
    int correct = 0;
    for (int i = 0; i < 10; ++i) {
        std::vector<int> prefix = {vocab.encode('0' + i), vocab.encode(':')};
        auto gen = lm.generate(prefix, 8, vocab.newline_id());
        std::string result;
        for (int id : gen) result += vocab.decode(id);
        std::string result_clean = result.substr(2);
        if (result_clean.size() > 0 && result_clean.back() == '\n') result_clean.pop_back();
        
        std::string expected = pairs[i].substr(2);
        expected.pop_back();
        bool match = (result_clean == expected);
        if (match) correct++;
        std::cout << "  Input: \"" << (char)('0' + i) << ":\""
                  << "  Expected: \"" << expected << "\""
                  << "  Got: \"" << result_clean << "\""
                  << (match ? "  ✓" : "  ✗") << "\n";
    }
    
    std::cout << "\n  Correct: " << correct << "/10 (" << std::fixed << std::setprecision(1) << correct * 10.0 << "%)\n";

    return 0;
}
