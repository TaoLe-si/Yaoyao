// train_lm_v3.cpp
// LM with int16 sum (no clamp) for better discrimination.

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
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

struct CharLM {
    int D, V;
    std::vector<std::vector<int8_t>> char_emb;    // [V, D]
    std::vector<std::vector<float>> W_h, W_s;     // [V, D]
    std::vector<float> bias;
    
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
            // h = emb[input[t]]; s += h
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
        
        for (int v = 0; v < V; ++v) {
            for (int d_i = 0; d_i < D; ++d_i) {
                W_h[v][d_i] -= lr * wh_grad[v][d_i] / valid;
                W_h[v][d_i] *= (1.0f - 0.01f);
                W_s[v][d_i] -= lr * ws_grad[v][d_i] / valid;
                W_s[v][d_i] *= (1.0f - 0.01f);
            }
            bias[v] -= lr * b_grad[v] / valid;
        }
        for (int v = 0; v < V; ++v) {
            for (int d_i = 0; d_i < D; ++d_i) {
                float new_val = (float)char_emb[v][d_i] - lr * emb_grad[v][d_i] / valid;
                int r = (int)std::lroundf(new_val);
                if (r > 4) r = 4; if (r < -4) r = -4;
                char_emb[v][d_i] = (int8_t)r;
            }
        }
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
    const int D = 128;
    const int V = vocab.size();
    
    std::cout << "================================================================\n";
    std::cout << "  BASIC LM (int16 sum) | V=" << V << " D=" << D << "\n";
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
        std::cout << "    \"" << (char)('0' + i) << ":\" -> \"" << result << "\"\n";
    }
    std::cout << "\n";
    
    int epochs = 600;
    std::cout << "  Training (" << epochs << " epochs)...\n\n";
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0;
        for (size_t i = 0; i < train_inputs.size(); ++i) {
            total_loss += lm.train_step(train_inputs[i], train_targets[i], train_masks[i], 0.1f);
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
        
        std::string expected = pairs[i].substr(2);
        expected.pop_back();
        std::string result_no_prefix = result.substr(2);  // strip "X:"
        if (result_no_prefix.size() > 0 && result_no_prefix.back() == '\n') result_no_prefix.pop_back();
        bool match = (result_no_prefix == expected);
        if (match) correct++;
        std::cout << "  Input: \"" << (char)('0' + i) << ":\""
                  << "  Expected: \"" << expected << "\""
                  << "  Got: \"" << result_no_prefix << "\""
                  << (match ? "  ✓" : "  ✗") << "\n";
    }
    
    std::cout << "\n  Correct: " << correct << "/10 (" << std::fixed << std::setprecision(1) << correct * 10.0 << "%)\n";

    return 0;
}
