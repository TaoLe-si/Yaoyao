// train_lm_simple.cpp
// Simplified char-level LM with trainable embeddings.
//
// Architecture (minimal but works):
//   h[t] = char_emb[input[t]]              (raw learnable embedding)
//   predict[t] = softmax(W_out · h[t])     (linear output projection)
//   Train: cross-entropy on next char
//
// Task: digit-to-word mapping.

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <string>
#include <map>
#include <immintrin.h>

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
    int pad_id() const { return encode(' '); }
};

// ============ AVX2 int8 dot ============
inline float dot_int8(const int8_t* a, const float* b, int n) {
    __m256 sumf = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) {
        __m128i va_i = _mm_loadl_epi64((__m128i*)(a + i));
        __m256 va_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(va_i));
        __m256 vb_f = _mm256_loadu_ps(b + i);
        sumf = _mm256_fmadd_ps(va_f, vb_f, sumf);
    }
    __m128 lo = _mm256_castps256_ps128(sumf);
    __m128 hi = _mm256_extractf128_ps(sumf, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

// ============ Simple char-level LM ============
struct CharLM {
    int D;
    int V;
    std::vector<std::vector<int8_t>> char_emb;  // [V, D]
    std::vector<std::vector<float>> W_out;       // [V, D]
    std::vector<float> bias;
    
    CharLM(int d, int v) : D(d), V(v) {
        char_emb.assign(V, std::vector<int8_t>(D, 0));
        W_out.assign(V, std::vector<float>(D, 0));
        bias.assign(V, 0);
        std::mt19937 rng(42);
        std::normal_distribution<float> nd(0, 0.3f);
        std::uniform_int_distribution<int> ud(-1, 1);
        // Initialize char embeddings with small random int8
        for (int v = 0; v < V; ++v) for (int d = 0; d < D; ++d) {
            int r = (int)std::lroundf(nd(rng));
            if (r > 4) r = 4; if (r < -4) r = -4;
            char_emb[v][d] = (int8_t)r;
        }
        // W_out
        for (int v = 0; v < V; ++v) for (int d = 0; d < D; ++d) {
            W_out[v][d] = nd(rng) * 0.1f;
        }
    }
    
    // Forward: compute logits for all positions
    void forward(const std::vector<int>& input, std::vector<std::vector<float>>& logits_out) const {
        int L = input.size();
        logits_out.assign(L, std::vector<float>(V, 0));
        for (int t = 0; t < L; ++t) {
            const int8_t* h = char_emb[input[t]].data();
            for (int v = 0; v < V; ++v) {
                logits_out[t][v] = bias[v] + dot_int8(h, W_out[v].data(), D);
            }
        }
    }
    
    float train_step(const std::vector<int>& input, const std::vector<int>& target, const std::vector<int>& mask, float lr) {
        int L = input.size();
        std::vector<std::vector<float>> logits;
        forward(input, logits);
        
        int valid_count = 0;
        float total_loss = 0;
        std::vector<std::vector<float>> w_grad(V, std::vector<float>(D, 0));
        std::vector<float> b_grad(V, 0);
        std::vector<std::vector<float>> emb_grad(V, std::vector<float>(D, 0));
        
        for (int t = 0; t < L; ++t) {
            if (mask[t] == 0) continue;  // skip padded positions
            valid_count++;
            
            // Softmax
            float max_l = *std::max_element(logits[t].begin(), logits[t].end());
            float sum = 0;
            std::vector<float> probs(V);
            for (int v = 0; v < V; ++v) { probs[v] = std::exp(logits[t][v] - max_l); sum += probs[v]; }
            for (int v = 0; v < V; ++v) probs[v] /= sum;
            
            int tgt = target[t];
            total_loss += -std::log(std::max(probs[tgt], 1e-7f));
            
            // Gradient
            for (int v = 0; v < V; ++v) {
                float d = probs[v] - (v == tgt ? 1.0f : 0.0f);
                for (int d = 0; d < D; ++d) w_grad[v][d] += d * (float)char_emb[input[t]][d];
                b_grad[v] += d;
            }
            for (int v = 0; v < V; ++v) {
                float d = probs[v] - (v == tgt ? 1.0f : 0.0f);
                for (int d = 0; d < D; ++d) emb_grad[input[t]][d] += d * W_out[v][d];
            }
        }
        
        if (valid_count == 0) return 0;
        
        // Update
        for (int v = 0; v < V; ++v) {
            for (int d = 0; d < D; ++d) {
                W_out[v][d] -= lr * w_grad[v][d] / valid_count;
                W_out[v][d] *= (1.0f - 0.01f);
            }
            bias[v] -= lr * b_grad[v] / valid_count;
        }
        for (int v = 0; v < V; ++v) {
            for (int d = 0; d < D; ++d) {
                float new_val = (float)char_emb[v][d] - lr * emb_grad[v][d] / valid_count;
                int r = (int)std::lroundf(new_val);
                if (r > 4) r = 4; if (r < -4) r = -4;
                char_emb[v][d] = (int8_t)r;
            }
        }
        
        return total_loss / valid_count;
    }
    
    // Generate: greedy
    std::vector<int> generate(const std::vector<int>& prefix, int max_new, int newline_id) const {
        std::vector<int> out = prefix;
        for (int step = 0; step < max_new; ++step) {
            int last = out.back();
            const int8_t* h = char_emb[last].data();
            int best = 0;
            float best_score = -1e30f;
            for (int v = 0; v < V; ++v) {
                float s = bias[v] + dot_int8(h, W_out[v].data(), D);
                if (s > best_score) { best_score = s; best = v; }
            }
            if (best == newline_id) { out.push_back(best); break; }
            out.push_back(best);
        }
        return out;
    }
};

int main() {
    Vocab vocab;
    const int D = 32;
    const int V = vocab.size();
    
    std::cout << "================================================================\n";
    std::cout << "  BASIC LM: Digit -> Word | Vocab=" << V << " D=" << D << "\n";
    std::cout << "================================================================\n\n";
    
    // Training pairs
    std::vector<std::string> pairs = {
        "0:zero\n", "1:one\n", "2:two\n", "3:three\n",
        "4:four\n", "5:five\n", "6:six\n", "7:seven\n",
        "8:eight\n", "9:nine\n"
    };
    
    std::cout << "  Training pairs:\n";
    for (auto& p : pairs) std::cout << "    \"" << p.substr(0, p.size() - 1) << "\"\n";
    std::cout << "\n";
    
    CharLM lm(D, V);
    
    // Encode training data with mask
    std::vector<std::vector<int>> train_inputs;
    std::vector<std::vector<int>> train_targets;
    std::vector<std::vector<int>> train_masks;
    int SEQ_LEN = 0;
    for (auto& p : pairs) {
        SEQ_LEN = std::max(SEQ_LEN, (int)p.size());
    }
    for (auto& p : pairs) {
        std::vector<int> inp(SEQ_LEN, vocab.pad_id());
        std::vector<int> tgt(SEQ_LEN, vocab.pad_id());
        std::vector<int> mask(SEQ_LEN, 0);
        for (int i = 0; i < (int)p.size() - 1; ++i) {
            inp[i] = vocab.encode(p[i]);
            tgt[i] = vocab.encode(p[i + 1]);
            mask[i] = 1;  // valid position: predict next char
        }
        // Don't train on the very last position (no target)
        train_inputs.push_back(inp);
        train_targets.push_back(tgt);
        train_masks.push_back(mask);
    }
    
    // Initial test
    std::cout << "  Initial generation:\n";
    for (int i = 0; i < 10; ++i) {
        std::vector<int> prefix = {vocab.encode('0' + i), vocab.encode(':')};
        auto gen = lm.generate(prefix, 8, vocab.newline_id());
        std::string result;
        for (int id : gen) result += vocab.decode(id);
        std::cout << "    Input \"" << (char)('0' + i) << ":\" -> \"" << result << "\"\n";
    }
    std::cout << "\n";
    
    // Training
    int epochs = 200;
    std::cout << "  Training (" << epochs << " epochs)...\n\n";
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0;
        for (size_t i = 0; i < train_inputs.size(); ++i) {
            total_loss += lm.train_step(train_inputs[i], train_targets[i], train_masks[i], 0.1f);
        }
        if ((epoch + 1) % 40 == 0 || epoch == 0 || epoch == epochs - 1) {
            std::cout << "  Epoch " << std::setw(3) << (epoch + 1)
                      << ": avg_loss=" << std::fixed << std::setprecision(4) << total_loss / train_inputs.size() << "\n";
        }
    }
    
    // Final test
    std::cout << "\n  ========== FINAL TEST: Input -> Output ==========\n\n";
    int correct = 0;
    for (int i = 0; i < 10; ++i) {
        std::vector<int> prefix = {vocab.encode('0' + i), vocab.encode(':')};
        auto gen = lm.generate(prefix, 8, vocab.newline_id());
        std::string result;
        for (int id : gen) result += vocab.decode(id);
        
        std::string expected = pairs[i].substr(2);
        expected.pop_back();  // remove \n
        bool match = (result.substr(0, expected.size()) == expected);
        if (match) correct++;
        std::cout << "  Input: \"" << (char)('0' + i) << ":\""
                  << "  Expected: \"" << expected << "\""
                  << "  Got: \"" << result << "\""
                  << (match ? "  ✓" : "  ✗") << "\n";
    }
    
    std::cout << "\n  Correct: " << correct << "/10 (" << std::fixed << std::setprecision(1) << correct * 10.0 << "%)\n";
    
    // Bonus: held-out addition
    std::cout << "\n  ========== BONUS: Held-out Test (arithmetic) ==========\n\n";
    // Add a few new pairs (in same session to test memorization scaling)
    std::vector<std::string> bonus = {"10:ten\n"};
    std::cout << "  After training on 0-9, can it generalize to 10?\n";
    std::vector<int> prefix10 = {vocab.encode('1'), vocab.encode('0'), vocab.encode(':')};
    auto gen10 = lm.generate(prefix10, 6, vocab.newline_id());
    std::string result10;
    for (int id : gen10) result10 += vocab.decode(id);
    std::cout << "  Input \"10:\" -> \"" << result10 << "\"  (expected \"ten\")\n\n";
    
    return 0;
}
