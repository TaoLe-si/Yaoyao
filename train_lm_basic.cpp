// train_lm_basic.cpp
// Basic Character-level Language Model
// Task: digit-to-word mapping
//   "0:zero\n", "1:one\n", ..., "9:nine\n"
// Input: digit + ":"
// Output: word for that digit + "\n"
//
// Tests: input question -> output answer

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

// ============ Optimized AVX2 int8 dot product ============
inline int32_t dot_int8_avx2(const int8_t* a, const int8_t* b, int n) {
    __m256i sum = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    int n_aligned = (n / 32) * 32;
    for (int i = 0; i < n_aligned; i += 32) {
        __m256i va = _mm256_loadu_si256((__m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((__m256i*)(b + i));
        __m256i va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
        __m256i vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_mullo_epi16(va_lo, vb_lo), ones));
        __m256i va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
        __m256i vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_mullo_epi16(va_hi, vb_hi), ones));
    }
    __m128i lo = _mm256_castsi256_si128(sum);
    __m128i hi = _mm256_extracti128_si256(sum, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
    s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
    int32_t result = _mm_cvtsi128_si32(s);
    for (int i = n_aligned; i < n; ++i) result += (int32_t)a[i] * (int32_t)b[i];
    return result;
}

// ============ Vocabulary ============
struct Vocab {
    std::map<char, int> char_to_id;
    std::map<int, char> id_to_char;
    
    Vocab() {
        std::string s = "0123456789:abcdefghijklmnopqrstuvwxyz\n";
        for (int i = 0; i < (int)s.size(); ++i) {
            char_to_id[s[i]] = i;
            id_to_char[i] = s[i];
        }
    }
    int size() const { return (int)id_to_char.size(); }
    int encode(char c) const { return char_to_id.at(c); }
    char decode(int id) const { return id_to_char.at(id); }
};

// ============ Hash Bucket Q1 (shared with Q4) ============
struct HashBucket {
    int N_BUCKETS;
    int K_HASH;
    int D;
    std::vector<std::vector<int8_t>> buckets;  // [N_BUCKETS, D]
    
    HashBucket(int n_buckets, int k_hash, int d) : N_BUCKETS(n_buckets), K_HASH(k_hash), D(d) {
        buckets.assign(N_BUCKETS, std::vector<int8_t>(D, 0));
        std::mt19937 rng(123);
        std::uniform_int_distribution<int> ud(-1, 1);
        for (auto& b : buckets) for (auto& v : b) v = (int8_t)ud(rng);
    }
    
    inline int hash_func(int token_id, int k) const {
        return (token_id * K_HASH + k) % N_BUCKETS;
    }
    
    void lookup(int token_id, int8_t* x) const {
        std::fill(x, x + D, 0);
        for (int k = 0; k < K_HASH; ++k) {
            int b = hash_func(token_id, k);
            for (int d = 0; d < D; ++d) x[d] += buckets[b][d];
        }
        for (int d = 0; d < D; ++d) {
            if (x[d] > 4) x[d] = 4;
            if (x[d] < -4) x[d] = -4;
        }
    }
};

// ============ Q3 (k=3 conv) ============
void q3_conv(const int8_t* x_p2, const int8_t* x_p, const int8_t* x_c,
             const float* w0, const float* w1, const float* w2, int8_t* y, int D) {
    for (int d = 0; d < D; ++d) {
        float v = w0[d] * (float)x_p2[d] + w1[d] * (float)x_p[d] + w2[d] * (float)x_c[d];
        int r = (int)std::lroundf(v);
        if (r > 4) r = 4; if (r < -4) r = -4;
        y[d] = (int8_t)r;
    }
}

// ============ Q2-A + Sum ============
void q2a_sum(const int8_t* h_old, const int16_t* s_old, const int8_t* y,
             const float* alpha, int8_t* h_new, int16_t* s_new, int D) {
    for (int d = 0; d < D; ++d) {
        float v = alpha[d] * (float)h_old[d] + (1.0f - alpha[d]) * (float)y[d];
        int r = (int)std::lroundf(v);
        if (r > 4) r = 4; if (r < -4) r = -4;
        h_new[d] = (int8_t)r;
        int s = (int)s_old[d] + (int)y[d];
        if (s > 64) s = 64; if (s < -64) s = -64;
        s_new[d] = (int16_t)s;
    }
}

// ============ Hash Bucket Q4 (trainable) ============
struct HashBucketQ4 {
    int N_BUCKETS;
    int K_HASH;
    int D;
    int V;
    std::vector<std::vector<float>> buckets;  // float weights (small enough)
    std::vector<float> bias;
    
    HashBucketQ4(int n_buckets, int k_hash, int d, int v)
        : N_BUCKETS(n_buckets), K_HASH(k_hash), D(d), V(v) {
        buckets.assign(N_BUCKETS, std::vector<float>(D, 0));
        bias.assign(V, 0);
        std::mt19937 rng(456);
        std::normal_distribution<float> nd(0, 0.05f);
        for (auto& b : buckets) for (auto& v : b) v = nd(rng);
    }
    
    inline int hash_func(int token_id, int k) const {
        return (token_id * K_HASH + k) % N_BUCKETS;
    }
    
    // Forward: compute logits for ALL V
    void compute_logits(const float* state, float* logits) const {
        std::vector<float> bucket_scores(N_BUCKETS, 0);
        for (int b = 0; b < N_BUCKETS; ++b) {
            int32_t s = 0;
            const float* sb = buckets[b].data();
            for (int d = 0; d < D; ++d) s += (int32_t)(state[d] * sb[d] * 1000);  // int trick? just use float
            bucket_scores[b] = 0;
            for (int d = 0; d < D; ++d) bucket_scores[b] += state[d] * sb[d];
        }
        for (int v = 0; v < V; ++v) {
            float s = bias[v];
            for (int k = 0; k < K_HASH; ++k) s += bucket_scores[hash_func(v, k)];
            logits[v] = s;
        }
    }
    
    // Faster: cache state as int8 for int8 dot
    void compute_logits_int8(const int8_t* state, float* logits) const {
        // Convert state to float first
        std::vector<float> state_f(D);
        for (int d = 0; d < D; ++d) state_f[d] = (float)state[d];
        compute_logits(state_f.data(), logits);
    }
};

// ============ Character LM ============
struct CharLM {
    int D;
    int V;
    int SEQ_LEN;
    int N_LAYERS;
    
    HashBucket q1;
    std::vector<std::vector<float>> q3_w0, q3_w1, q3_w2;  // per layer
    std::vector<std::vector<float>> alpha;                 // per layer
    HashBucketQ4 q4;
    
    CharLM(int d, int v, int seq_len, int n_layers, Vocab& vocab)
        : D(d), V(v), SEQ_LEN(seq_len), N_LAYERS(n_layers),
          q1(64, 2, d),
          q4(128, 4, d, v) {
        q3_w0.assign(N_LAYERS, std::vector<float>(D, 0));
        q3_w1.assign(N_LAYERS, std::vector<float>(D, 0));
        q3_w2.assign(N_LAYERS, std::vector<float>(D, 0));
        alpha.assign(N_LAYERS, std::vector<float>(D, 0.5f));
        std::mt19937 rng(789);
        std::normal_distribution<float> nd(0, 0.05f);
        for (int l = 0; l < N_LAYERS; ++l) {
            for (auto& w : q3_w0[l]) w = nd(rng);
            for (auto& w : q3_w1[l]) w = nd(rng);
            for (auto& w : q3_w2[l]) w = nd(rng);
        }
    }
    
    // Forward through one layer, returns sum state
    void forward_layer(const std::vector<std::vector<int8_t>>& xs_in,
                       int layer,
                       std::vector<std::vector<int8_t>>& hs_out,
                       std::vector<std::vector<int16_t>>& ss_out) const {
        int L = (int)xs_in.size();
        hs_out.assign(L + 1, std::vector<int8_t>(D, 0));
        ss_out.assign(L + 1, std::vector<int16_t>(D, 0));
        
        std::vector<int8_t> x_prev(D, 0), x_prev2(D, 0), y(D, 0);
        for (int t = 0; t < L; ++t) {
            const auto& x_curr = xs_in[t];
            q3_conv(x_prev2.data(), x_prev.data(), x_curr.data(),
                    q3_w0[layer].data(), q3_w1[layer].data(), q3_w2[layer].data(),
                    y.data(), D);
            q2a_sum(hs_out[t].data(), ss_out[t].data(), y.data(),
                    alpha[layer].data(),
                    hs_out[t + 1].data(), ss_out[t + 1].data(), D);
            x_prev2 = x_prev;
            x_prev = x_curr;
        }
    }
    
    // Process entire sequence through all layers
    std::vector<std::vector<int16_t>> forward(const std::vector<int>& tokens) const {
        // Layer 0: input is Q1(token)
        std::vector<std::vector<int8_t>> xs(SEQ_LEN, std::vector<int8_t>(D, 0));
        for (int t = 0; t < SEQ_LEN; ++t) {
            q1.lookup(tokens[t], xs[t].data());
        }
        
        std::vector<std::vector<int8_t>> hs, hs_next;
        std::vector<std::vector<int16_t>> ss, ss_next;
        forward_layer(xs, 0, hs, ss);
        
        for (int l = 1; l < N_LAYERS; ++l) {
            // Feed hs as next layer's xs
            std::vector<std::vector<int8_t>> xs_next = hs;
            forward_layer(xs_next, l, hs_next, ss_next);
            hs = hs_next;
            ss = ss_next;
        }
        return ss;  // final layer's sum state
    }
    
    // Compute logits for next token given state
    void logits(const std::vector<int16_t>& state, float* out) const {
        std::vector<float> state_f(D);
        for (int d = 0; d < D; ++d) state_f[d] = (float)state[d];
        q4.compute_logits(state_f.data(), out);
    }
    
    // Train Q4 only (frozen Q1/Q3/alpha)
    float train_step(const std::vector<int>& tokens, const std::vector<int>& targets, float lr) {
        auto ss = forward(tokens);
        std::vector<float> state_f(D);
        float total_loss = 0;
        
        // Compute loss + gradients
        std::vector<std::vector<float>> bucket_grad(q4.N_BUCKETS, std::vector<float>(D, 0));
        std::vector<float> bias_grad(V, 0);
        
        for (int t = 0; t < SEQ_LEN; ++t) {
            for (int d = 0; d < D; ++d) state_f[d] = (float)ss[t][d];
            float logits_arr[64];
            q4.compute_logits(state_f.data(), logits_arr);
            float max_l = *std::max_element(logits_arr, logits_arr + V);
            float sum = 0;
            float probs[64];
            for (int v = 0; v < V; ++v) { probs[v] = std::exp(logits_arr[v] - max_l); sum += probs[v]; }
            for (int v = 0; v < V; ++v) probs[v] /= sum;
            
            int target = targets[t];
            total_loss += -std::log(std::max(probs[target], 1e-7f));
            
            // d_logits[v] = probs[v] - (v == target ? 1 : 0)
            // For each bucket b used by token v: d_bucket[b] += d_logits[v]
            for (int v = 0; v < V; ++v) {
                float d_logit = probs[v] - (v == target ? 1.0f : 0.0f);
                for (int k = 0; k < q4.K_HASH; ++k) {
                    int b = q4.hash_func(v, k);
                    for (int d = 0; d < D; ++d) {
                        bucket_grad[b][d] += d_logit * state_f[d];
                    }
                }
                bias_grad[v] += d_logit;
            }
        }
        
        // Update Q4 weights
        for (int b = 0; b < q4.N_BUCKETS; ++b) {
            for (int d = 0; d < D; ++d) {
                q4.buckets[b][d] -= lr * bucket_grad[b][d] / SEQ_LEN;
                q4.buckets[b][d] *= (1.0f - 0.01f);  // weight decay
            }
        }
        for (int v = 0; v < V; ++v) q4.bias[v] -= lr * bias_grad[v] / SEQ_LEN;
        
        return total_loss / SEQ_LEN;
    }
    
    // Generate: greedy decode from prefix
    std::vector<int> generate(const std::vector<int>& prefix, int max_new) const {
        std::vector<int> out = prefix;
        for (int step = 0; step < max_new; ++step) {
            // Pad/truncate to SEQ_LEN
            std::vector<int> ctx(out.end() - std::min((int)out.size(), SEQ_LEN), out.end());
            while ((int)ctx.size() < SEQ_LEN) ctx.insert(ctx.begin(), 0);  // pad with 0
            
            auto ss = forward(ctx);
            float logits_arr[64];
            q4.compute_logits((float*)ss.back().data(), logits_arr);  // wait need float
            std::vector<float> state_f(D);
            for (int d = 0; d < D; ++d) state_f[d] = (float)ss.back()[d];
            q4.compute_logits(state_f.data(), logits_arr);
            
            // Argmax
            int best = 0;
            for (int v = 1; v < V; ++v) if (logits_arr[v] > logits_arr[best]) best = v;
            
            // Stop at newline
            if (best == V - 1) break;  // \n is last in vocab
            out.push_back(best);
        }
        return out;
    }
};

int main() {
    Vocab vocab;
    std::cout << "================================================================\n";
    std::cout << "  BASIC LM TEST: Input Question -> Output Answer\n";
    std::cout << "  Task: digit-to-word mapping\n";
    std::cout << "  Vocab size: " << vocab.size() << " chars\n";
    std::cout << "================================================================\n\n";
    
    // Build training data: "0:zero\n", "1:one\n", ..., "9:nine\n"
    std::vector<std::string> pairs = {
        "0:zero\n", "1:one\n", "2:two\n", "3:three\n",
        "4:four\n", "5:five\n", "6:six\n", "7:seven\n",
        "8:eight\n", "9:nine\n"
    };
    
    // Find max seq length
    int max_len = 0;
    for (auto& p : pairs) max_len = std::max(max_len, (int)p.size());
    int SEQ_LEN = 8;  // pad to 8 (longest is "3:three\n" = 8)
    
    std::cout << "  Training pairs (10 total):\n";
    for (auto& p : pairs) std::cout << "    \"" << p << "\" (len " << p.size() << ")\n";
    std::cout << "\n  Sequence length: " << SEQ_LEN << "\n\n";
    
    const int D = 64;
    const int V = vocab.size();
    const int N_LAYERS = 2;
    CharLM lm(D, V, SEQ_LEN, N_LAYERS, vocab);
    
    // Convert pairs to token sequences
    std::vector<std::vector<int>> train_tokens;
    std::vector<std::vector<int>> train_targets;
    for (auto& p : pairs) {
        std::vector<int> toks(SEQ_LEN, 0);  // pad with '0' (digit 0)
        for (int i = 0; i < (int)p.size(); ++i) toks[i] = vocab.encode(p[i]);
        // Target: shifted left, predict next char
        std::vector<int> tgts(SEQ_LEN, 0);
        for (int i = 0; i < SEQ_LEN - 1; ++i) tgts[i] = (i + 1 < (int)p.size()) ? vocab.encode(p[i + 1]) : 0;
        tgts[SEQ_LEN - 1] = 0;
        train_tokens.push_back(toks);
        train_targets.push_back(tgts);
    }
    
    // Initial test
    std::cout << "  Initial generation (before training):\n";
    for (int i = 0; i < 10; ++i) {
        std::vector<int> prefix = {vocab.encode('0' + i), vocab.encode(':')};
        auto gen = lm.generate(prefix, 6);
        std::string result;
        for (int id : gen) result += vocab.decode(id);
        std::cout << "    " << (char)('0' + i) << ": -> \"" << result << "\"\n";
    }
    std::cout << "\n";
    
    // Training loop
    int epochs = 100;
    std::cout << "  Training (" << epochs << " epochs)...\n\n";
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0;
        for (size_t i = 0; i < train_tokens.size(); ++i) {
            total_loss += lm.train_step(train_tokens[i], train_targets[i], 0.05f);
        }
        if ((epoch + 1) % 20 == 0 || epoch == 0 || epoch == epochs - 1) {
            std::cout << "  Epoch " << std::setw(3) << (epoch + 1)
                      << ": avg_loss=" << std::fixed << std::setprecision(4) << total_loss / train_tokens.size() << "\n";
        }
    }
    
    // Test
    std::cout << "\n  ========== FINAL TEST: Input -> Output ==========\n\n";
    int correct = 0;
    for (int i = 0; i < 10; ++i) {
        std::vector<int> prefix = {vocab.encode('0' + i), vocab.encode(':')};
        auto gen = lm.generate(prefix, 6);
        std::string result;
        for (int id : gen) result += vocab.decode(id);
        
        std::string expected = pairs[i].substr(2);  // word part
        // Strip \n for comparison
        std::string result_clean = result;
        for (auto& c : result_clean) if (c == '\n') c = ' ';
        
        bool match = (result.find(expected.substr(0, expected.size() - 1)) != std::string::npos);
        if (match) correct++;
        
        std::cout << "  Input: \"" << (char)('0' + i) << ":\""
                  << "  Expected: \"" << expected.substr(0, expected.size() - 1) << "\""
                  << "  Got: \"" << result << "\""
                  << (match ? "  ✓" : "  ✗") << "\n";
    }
    
    std::cout << "\n  Correct: " << correct << "/10\n";
    
    return 0;
}
