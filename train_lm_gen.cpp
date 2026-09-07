// train_lm_adam.cpp
// Simple LM with Adam optimizer (no Q3, just sum channel + h channel).
//
// Improvements over previous version:
//   - Adam optimizer (handles noisy gradients)
//   - Shuffle training data each epoch
//   - Use both h (last char) and s (sum) channels
//   - Larger init for embeddings (std=1.0)

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <string>
#include <map>
#include <numeric>

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

struct AdamState {
    std::vector<float> m, v;
    int t = 0;
};

struct CharLM {
    int D, V;
    std::vector<std::vector<int8_t>> char_emb;
    std::vector<std::vector<float>> W_h, W_s;
    std::vector<float> bias;
    
    AdamState st_emb, st_Wh, st_Ws, st_b;
    
    CharLM(int d, int v) : D(d), V(v) {
        char_emb.assign(V, std::vector<int8_t>(D, 0));
        W_h.assign(V, std::vector<float>(D, 0));
        W_s.assign(V, std::vector<float>(D, 0));
        bias.assign(V, 0);
        
        std::mt19937 rng(42);
        std::normal_distribution<float> nd_emb(0, 1.0f);
        std::normal_distribution<float> nd_w(0, 0.1f);
        for (int v = 0; v < V; ++v) for (int d = 0; d < D; ++d) {
            int r = (int)std::lroundf(nd_emb(rng));
            if (r > 4) r = 4; if (r < -4) r = -4;
            char_emb[v][d] = (int8_t)r;
            W_h[v][d] = nd_w(rng);
            W_s[v][d] = nd_w(rng);
        }
        st_emb.m.assign(V * D, 0); st_emb.v.assign(V * D, 0);
        st_Wh.m.assign(V * D, 0); st_Wh.v.assign(V * D, 0);
        st_Ws.m.assign(V * D, 0); st_Ws.v.assign(V * D, 0);
        st_b.m.assign(V, 0); st_b.v.assign(V, 0);
    }
    
    void forward(const std::vector<int>& input, 
                 std::vector<std::vector<int8_t>>& h_states,
                 std::vector<std::vector<int16_t>>& s_states,
                 std::vector<std::vector<float>>& logits_out) {
        int L = input.size();
        h_states.assign(L, std::vector<int8_t>(D, 0));
        s_states.assign(L, std::vector<int16_t>(D, 0));
        logits_out.assign(L, std::vector<float>(V, 0));
        std::vector<int16_t> s(D, 0);
        for (int t = 0; t < L; ++t) {
            const int8_t* emb = char_emb[input[t]].data();
            for (int d = 0; d < D; ++d) {
                h_states[t][d] = emb[d];
                s[d] = (int16_t)((int)s[d] + (int)emb[d]);
                s_states[t][d] = s[d];
            }
            for (int v = 0; v < V; ++v) {
                float logit = bias[v];
                for (int d = 0; d < D; ++d) {
                    logit += W_h[v][d] * (float)h_states[t][d];
                    logit += W_s[v][d] * (float)s_states[t][d];
                }
                logits_out[t][v] = logit;
            }
        }
    }
    
    std::vector<float> flatten_2d(const std::vector<std::vector<int8_t>>& m) const {
        std::vector<float> f(m.size() * m[0].size());
        for (size_t i = 0; i < m.size(); ++i)
            for (size_t d = 0; d < m[i].size(); ++d) f[i * m[0].size() + d] = (float)m[i][d];
        return f;
    }
    void unflatten_2d(const std::vector<float>& f, std::vector<std::vector<int8_t>>& m) const {
        for (size_t i = 0; i < m.size(); ++i)
            for (size_t d = 0; d < m[i].size(); ++d) {
                int r = (int)std::lroundf(f[i * m[0].size() + d]);
                if (r > 4) r = 4; if (r < -4) r = -4;
                m[i][d] = (int8_t)r;
            }
    }
    std::vector<float> flatten_2d_f(const std::vector<std::vector<float>>& m) const {
        std::vector<float> f(m.size() * m[0].size());
        for (size_t i = 0; i < m.size(); ++i)
            for (size_t d = 0; d < m[i].size(); ++d) f[i * m[0].size() + d] = m[i][d];
        return f;
    }
    void unflatten_2d_f(const std::vector<float>& f, std::vector<std::vector<float>>& m) const {
        for (size_t i = 0; i < m.size(); ++i)
            for (size_t d = 0; d < m[i].size(); ++d) m[i][d] = f[i * m[0].size() + d];
    }
    
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
    
    float train_step(const std::vector<int>& input, const std::vector<int>& target,
                     const std::vector<int>& mask, float lr) {
        int L = input.size();
        std::vector<std::vector<int8_t>> h_states;
        std::vector<std::vector<int16_t>> s_states;
        std::vector<std::vector<float>> logits;
        forward(input, h_states, s_states, logits);
        
        int valid = 0;
        float total_loss = 0;
        std::vector<std::vector<float>> wh_grad(V, std::vector<float>(D, 0));
        std::vector<std::vector<float>> ws_grad(V, std::vector<float>(D, 0));
        std::vector<float> b_grad(V, 0);
        std::vector<std::vector<float>> emb_grad(V, std::vector<float>(D, 0));
        
        for (int t = 0; t < L; ++t) {
            if (mask[t] == 0) continue;
            valid++;
            float max_l = *std::max_element(logits[t].begin(), logits[t].end());
            float sum = 0;
            std::vector<float> probs(V);
            for (int v = 0; v < V; ++v) { probs[v] = std::exp(logits[t][v] - max_l); sum += probs[v]; }
            for (int v = 0; v < V; ++v) probs[v] /= sum;
            
            int tgt = target[t];
            total_loss += -std::log(std::max(probs[tgt], 1e-7f));
            
            for (int v = 0; v < V; ++v) {
                float d = probs[v] - (v == tgt ? 1.0f : 0.0f);
                for (int d_i = 0; d_i < D; ++d_i) {
                    wh_grad[v][d_i] += d * (float)h_states[t][d_i];
                    ws_grad[v][d_i] += d * (float)s_states[t][d_i];
                }
                b_grad[v] += d;
                for (int d_i = 0; d_i < D; ++d_i) {
                    emb_grad[input[t]][d_i] += d * (W_h[v][d_i] + W_s[v][d_i]);
                }
            }
        }
        
        if (valid == 0) return 0;
        
        auto wh_grad_f = flatten_2d_f(wh_grad);
        auto ws_grad_f = flatten_2d_f(ws_grad);
        auto emb_grad_f = flatten_2d_f(emb_grad);
        for (auto& x : wh_grad_f) x /= valid;
        for (auto& x : ws_grad_f) x /= valid;
        for (auto& x : emb_grad_f) x /= valid;
        for (auto& x : b_grad) x /= valid;
        
        auto wh_f = flatten_2d_f(W_h);
        auto ws_f = flatten_2d_f(W_s);
        auto emb_f = flatten_2d(char_emb);
        adam_update(wh_f, st_Wh, wh_grad_f, lr);
        unflatten_2d_f(wh_f, W_h);
        adam_update(ws_f, st_Ws, ws_grad_f, lr);
        unflatten_2d_f(ws_f, W_s);
        adam_update(emb_f, st_emb, emb_grad_f, lr);
        unflatten_2d(emb_f, char_emb);
        adam_update(bias, st_b, b_grad, lr);
        
        return total_loss / valid;
    }
    
    std::vector<int> generate(const std::vector<int>& prefix, int max_new, int newline_id) const {
        std::vector<int> out = prefix;
        std::vector<int16_t> s(D, 0);
        for (int id : prefix) {
            const int8_t* emb = char_emb[id].data();
            for (int d = 0; d < D; ++d) s[d] = (int16_t)((int)s[d] + (int)emb[d]);
        }
        for (int step = 0; step < max_new; ++step) {
            int last = out.back();
            const int8_t* emb = char_emb[last].data();
            int best = 0;
            float best_score = -1e30f;
            for (int v = 0; v < V; ++v) {
                float logit = bias[v];
                for (int d = 0; d < D; ++d) {
                    logit += W_h[v][d] * (float)emb[d] + W_s[v][d] * (float)s[d];
                }
                if (logit > best_score) { best_score = logit; best = v; }
            }
            out.push_back(best);
            const int8_t* next_emb = char_emb[best].data();
            for (int d = 0; d < D; ++d) s[d] = (int16_t)((int)s[d] + (int)next_emb[d]);
            if (best == newline_id) break;
        }
        return out;
    }
};

int main() {
    Vocab vocab;
    const int D = 32;
    const int V = vocab.size();
    
    std::vector<std::string> pairs = {
        "0:zero\n", "1:one\n", "2:two\n", "3:three\n",
        "4:four\n", "5:five\n", "6:six\n", "7:seven\n",
        "8:eight\n", "9:nine\n"
    };
    
    std::cout << "================================================================\n";
    std::cout << "  GENERALIZATION TEST: train on 0-4, test on 5-9\n";
    std::cout << "================================================================\n\n";
    
    int SEQ_LEN = 0;
    for (auto& p : pairs) SEQ_LEN = std::max(SEQ_LEN, (int)p.size());
    std::vector<std::vector<int>> ti, tt, tm;
    for (auto& p : pairs) {
        std::vector<int> inp(SEQ_LEN, vocab.encode(' '));
        std::vector<int> tgt(SEQ_LEN, vocab.encode(' '));
        std::vector<int> mask(SEQ_LEN, 0);
        for (int i = 0; i < (int)p.size() - 1; ++i) {
            inp[i] = vocab.encode(p[i]); tgt[i] = vocab.encode(p[i + 1]); mask[i] = 1;
        }
        ti.push_back(inp); tt.push_back(tgt); tm.push_back(mask);
    }
    
    CharLM lm(D, V);
    std::cout << "  Training on digits 0-4 only...\n\n";
    for (int epoch = 0; epoch < 400; ++epoch) {
        std::vector<int> idx(5);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), std::mt19937(epoch + 1));
        float total = 0;
        for (int i : idx) total += lm.train_step(ti[i], tt[i], tm[i], 0.01f);
        if ((epoch + 1) % 100 == 0 || epoch == 0 || epoch == 399) {
            std::cout << "  Epoch " << std::setw(3) << (epoch + 1)
                      << ": loss=" << std::fixed << std::setprecision(4) << total / 5 << "\n";
        }
    }
    
    std::cout << "\n  Test on TRAINED (0-4):\n";
    int train_correct = 0;
    for (int i = 0; i < 5; ++i) {
        std::vector<int> prefix = {vocab.encode('0' + i), vocab.encode(':')};
        auto gen = lm.generate(prefix, 8, vocab.newline_id());
        std::string result;
        for (int id : gen) result += vocab.decode(id);
        std::string r = result.substr(2);
        if (r.size() && r.back() == '\n') r.pop_back();
        std::string exp = pairs[i].substr(2); exp.pop_back();
        bool m = (r == exp);
        if (m) train_correct++;
        std::cout << "    " << (char)('0' + i) << ": -> \"" << r << "\" expected \"" << exp << "\" " << (m ? "OK" : "X") << "\n";
    }
    
    std::cout << "\n  Test on HELD-OUT (5-9):\n";
    int test_correct = 0;
    for (int i = 5; i < 10; ++i) {
        std::vector<int> prefix = {vocab.encode('0' + i), vocab.encode(':')};
        auto gen = lm.generate(prefix, 8, vocab.newline_id());
        std::string result;
        for (int id : gen) result += vocab.decode(id);
        std::string r = result.substr(2);
        if (r.size() && r.back() == '\n') r.pop_back();
        std::string exp = pairs[i].substr(2); exp.pop_back();
        bool m = (r == exp);
        if (m) test_correct++;
        std::cout << "    " << (char)('0' + i) << ": -> \"" << r << "\" expected \"" << exp << "\" " << (m ? "OK" : "X") << "\n";
    }
    
    std::cout << "\n  Train accuracy: " << train_correct << "/5  (" << std::fixed << std::setprecision(1) << train_correct * 20.0 << "%)\n";
    std::cout << "  Test (held-out) accuracy: " << test_correct << "/5  (" << std::fixed << std::setprecision(1) << test_correct * 20.0 << "%)\n";

    return 0;
}
