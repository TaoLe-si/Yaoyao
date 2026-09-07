// train_q4c.cpp
// OPTION C: Hierarchical Softmax (HSM) Design + Implementation
//
// 2-level HSM:
//   Level 1: predict group (G logits)
//   Level 2: predict token within winning group (V/G logits)
//
// Compare with Flat Linear Q4 on the same task.

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>

const int D = 128;
const int S_MAX = 64;
const int SEQ_LEN = 16;
const int EPOCHS = 30;
const int N_TRAIN = 1000;
const int V = 50;
const int G = 10;              // 10 groups
const int TOKENS_PER_GROUP = V / G;  // 5
const float LR = 0.1f;
const float WD = 0.02f;

void q1(int token_id, int8_t* x) {
    std::fill(x, x + D, 0);
    if (token_id >= 0 && token_id < V) x[token_id] = 3;
}

void forward_seq(const std::vector<int>& tokens, std::vector<std::vector<int16_t>>& ss_out) {
    ss_out.assign(SEQ_LEN + 1, std::vector<int16_t>(D, 0));
    std::vector<int8_t> xc(D, 0);
    for (int t = 0; t < SEQ_LEN; ++t) {
        q1(tokens[t], xc.data());
        for (int d = 0; d < D; ++d) {
            int s = (int)ss_out[t][d] + (int)xc[d];
            if (s > S_MAX) s = S_MAX; if (s < -S_MAX) s = -S_MAX;
            ss_out[t + 1][d] = (int16_t)s;
        }
    }
}

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

// ============ HSM Q4 ============
// Level 1: W_g [G, D] for group logits
// Level 2: W_tok [G, V/G, D] for token logits within each group
struct HSMQ4 {
    std::vector<std::vector<float>> W_g;          // [G][D]
    std::vector<float> b_g;
    std::vector<std::vector<std::vector<float>>> W_tok;  // [G][TOKENS_PER_GROUP][D]
    std::vector<std::vector<float>> b_tok;
    
    HSMQ4() {
        W_g.assign(G, std::vector<float>(D, 0));
        b_g.assign(G, 0);
        W_tok.assign(G, std::vector<std::vector<float>>(TOKENS_PER_GROUP, std::vector<float>(D, 0)));
        b_tok.assign(G, std::vector<float>(TOKENS_PER_GROUP, 0));
        std::mt19937 rng(456);
        std::normal_distribution<float> nd(0, 0.02f);
        for (auto& row : W_g) for (auto& v : row) v = nd(rng);
        for (auto& group : W_tok) for (auto& row : group) for (auto& v : row) v = nd(rng);
    }
    
    // Returns group prediction and token prediction
    int predict(const float* state) const {
        // Level 1: group
        float group_logits[G];
        for (int g = 0; g < G; ++g) {
            float s = b_g[g];
            for (int d = 0; d < D; ++d) s += W_g[g][d] * state[d];
            group_logits[g] = s;
        }
        int best_g = 0;
        for (int g = 1; g < G; ++g) if (group_logits[g] > group_logits[best_g]) best_g = g;
        
        // Level 2: token within best_g
        float best_tok_score = -1e30f;
        int best_tok = -1;
        for (int t = 0; t < TOKENS_PER_GROUP; ++t) {
            float s = b_tok[best_g][t];
            for (int d = 0; d < D; ++d) s += W_tok[best_g][t][d] * state[d];
            if (s > best_tok_score) { best_tok_score = s; best_tok = t; }
        }
        return best_g * TOKENS_PER_GROUP + best_tok;
    }
    
    void train_step(const float* state, int target, float lr) {
        // Forward: group logits
        float group_logits[G];
        for (int g = 0; g < G; ++g) {
            float s = b_g[g];
            for (int d = 0; d < D; ++d) s += W_g[g][d] * state[d];
            group_logits[g] = s;
        }
        int target_g = target / TOKENS_PER_GROUP;
        int target_t = target % TOKENS_PER_GROUP;
        
        // Softmax for groups
        float max_g = *std::max_element(group_logits, group_logits + G);
        float sum_g = 0;
        float g_probs[G];
        for (int g = 0; g < G; ++g) { g_probs[g] = std::exp(group_logits[g] - max_g); sum_g += g_probs[g]; }
        for (int g = 0; g < G; ++g) g_probs[g] /= sum_g;
        
        // Level 2: token logits within target_g (only compute target group)
        float tok_logits[TOKENS_PER_GROUP];
        for (int t = 0; t < TOKENS_PER_GROUP; ++t) {
            float s = b_tok[target_g][t];
            for (int d = 0; d < D; ++d) s += W_tok[target_g][t][d] * state[d];
            tok_logits[t] = s;
        }
        float max_t = *std::max_element(tok_logits, tok_logits + TOKENS_PER_GROUP);
        float sum_t = 0;
        float t_probs[TOKENS_PER_GROUP];
        for (int t = 0; t < TOKENS_PER_GROUP; ++t) { t_probs[t] = std::exp(tok_logits[t] - max_t); sum_t += t_probs[t]; }
        for (int t = 0; t < TOKENS_PER_GROUP; ++t) t_probs[t] /= sum_t;
        
        // Update Level 1: gradient for W_g
        for (int g = 0; g < G; ++g) {
            float d_g = g_probs[g] - (g == target_g ? 1.0f : 0.0f);
            for (int d = 0; d < D; ++d) {
                W_g[g][d] -= lr * d_g * state[d];
                W_g[g][d] *= (1.0f - WD);
            }
            b_g[g] -= lr * d_g;
        }
        
        // Update Level 2: gradient for W_tok[target_g]
        for (int t = 0; t < TOKENS_PER_GROUP; ++t) {
            float d_t = t_probs[t] - (t == target_t ? 1.0f : 0.0f);
            for (int d = 0; d < D; ++d) {
                W_tok[target_g][t][d] -= lr * d_t * state[d];
                W_tok[target_g][t][d] *= (1.0f - WD);
            }
            b_tok[target_g][t] -= lr * d_t;
        }
    }
};

int main() {
    std::cout << "================================================================\n";
    std::cout << "  OPTION C: Hierarchical Softmax | V=" << V << ", G=" << G << ", T/G=" << TOKENS_PER_GROUP << "\n";
    std::cout << "================================================================\n\n";
    
    HSMQ4 hsm;
    
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
    
    std::vector<float> state_norm_buf(D);
    auto predict = [&](const std::vector<int>& seq) {
        std::vector<std::vector<int16_t>> ss;
        forward_seq(seq, ss);
        normalize_state(ss.back().data(), state_norm_buf.data());
        return hsm.predict(state_norm_buf.data());
    };
    
    int initial = 0;
    for (auto& p : data) if (predict(p.first) == p.second) initial++;
    std::cout << "  Initial: " << initial << "/" << data.size() << "\n\n";
    
    std::vector<float> state_norm(D);
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(data.begin(), data.end(), std::mt19937(epoch + 1));
        float total_loss = 0;
        for (auto& p : data) {
            std::vector<std::vector<int16_t>> ss;
            forward_seq(p.first, ss);
            normalize_state(ss.back().data(), state_norm.data());
            
            // Compute loss for logging
            float group_logits[G];
            for (int g = 0; g < G; ++g) {
                float s = hsm.b_g[g];
                for (int d = 0; d < D; ++d) s += hsm.W_g[g][d] * state_norm[d];
                group_logits[g] = s;
            }
            int target_g = p.second / TOKENS_PER_GROUP;
            float max_g = *std::max_element(group_logits, group_logits + G);
            float sum_g = 0;
            float g_probs[G];
            for (int g = 0; g < G; ++g) { g_probs[g] = std::exp(group_logits[g] - max_g); sum_g += g_probs[g]; }
            for (int g = 0; g < G; ++g) g_probs[g] /= sum_g;
            total_loss += -std::log(std::max(g_probs[target_g], 1e-7f));
            
            hsm.train_step(state_norm.data(), p.second, LR);
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
    
    // Held-out
    int tc = 0;
    std::mt19937 test_rng(999);
    for (int i = 0; i < 500; ++i) {
        std::vector<int> seq(SEQ_LEN);
        std::vector<int> counts(V, 0);
        for (int t = 0; t < SEQ_LEN; ++t) { seq[t] = test_rng() % V; counts[seq[t]]++; }
        int majority = 0; int mx = counts[0];
        for (int v = 1; v < V; ++v) if (counts[v] > mx) { mx = counts[v]; majority = v; }
        if (predict(seq) == majority) tc++;
    }
    std::cout << "  Held-out: " << tc << "/500 (" << std::fixed << std::setprecision(1) << tc * 100.0 / 500 << "%)\n";
    
    // Parameter count comparison
    std::cout << "\n  === PARAMETER COMPARISON ===\n";
    int linear_params = V * D + V;
    int hsm_params = G * D + G + G * TOKENS_PER_GROUP * D + G * TOKENS_PER_GROUP;
    std::cout << "  Flat Linear Q4: " << linear_params << " params (V * D)\n";
    std::cout << "  HSM 2-level:    " << hsm_params << " params (G*D + G*T*D)\n";
    std::cout << "  HSM reduction:  " << std::setprecision(1) << (1.0 - (float)hsm_params / linear_params) * 100 << "% fewer params\n\n";
    
    // Compute comparison
    std::cout << "  === COMPUTE COMPARISON ===\n";
    std::cout << "  Flat:      " << V << " * " << D << " = " << V * D << " ops per token\n";
    std::cout << "  HSM L1:    " << G << " * " << D << " = " << G * D << " ops\n";
    std::cout << "  HSM L2:    1 * " << TOKENS_PER_GROUP << " * " << D << " = " << TOKENS_PER_GROUP * D << " ops\n";
    std::cout << "  HSM total: " << (G * D + TOKENS_PER_GROUP * D) << " ops per token\n";
    std::cout << "  HSM speedup: " << std::setprecision(2) << (float)(V * D) / (G * D + TOKENS_PER_GROUP * D) << "x\n";
    
    return 0;
}
