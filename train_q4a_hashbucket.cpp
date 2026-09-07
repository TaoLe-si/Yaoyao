// train_q4a_hashbucket.cpp
// OPTION A: Hash Bucket Q4 with V=4 (multi-class majority prediction)
//
// Architecture: Q1 (one-hot) + Sum + Q4 (Hash Bucket)
// Q4: For each token v, logit_v = sum over k of (bucket[hash(v,k)] . state)

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>

const int D = 32;
const int H_VAL = 4;
const int S_MAX = 64;
const int SEQ_LEN = 16;
const int EPOCHS = 40;
const int N_TRAIN = 500;
const int V = 4;
const int N_BUCKETS = 8;   // hash bucket pool size
const int K_HASH = 2;       // each token uses K buckets
const float LR = 0.05f;
const float WD = 0.05f;

// Q1: explicit one-hot per token
void q1(int token_id, int8_t* x) {
    std::fill(x, x + D, 0);
    if (token_id >= 0 && token_id < V) {
        x[token_id * 2] = 3;
        x[token_id * 2 + 1] = -3;
    }
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

// ============ Hash Bucket Q4 ============
struct HashBucketQ4 {
    std::vector<std::vector<float>> pool;  // [N_BUCKETS, D]
    std::vector<float> bias;               // [V] per-token bias
    
    HashBucketQ4() {
        pool.assign(N_BUCKETS, std::vector<float>(D, 0));
        bias.assign(V, 0);
        std::mt19937 rng(789);
        std::normal_distribution<float> nd(0, 0.05f);
        for (auto& row : pool) for (auto& v : row) v = nd(rng);
    }
    
    inline int hash_func(int token_id, int k) const {
        return token_id * K_HASH + k;  // unique per token, no collisions
    }
    
    void compute_logits(const float* state, float* logits) const {
        std::vector<float> bucket_scores(N_BUCKETS, 0);
        for (int b = 0; b < N_BUCKETS; ++b) {
            for (int d = 0; d < D; ++d) {
                bucket_scores[b] += pool[b][d] * state[d];
            }
        }
        for (int v = 0; v < V; ++v) {
            logits[v] = bias[v];
            for (int k = 0; k < K_HASH; ++k) {
                logits[v] += bucket_scores[hash_func(v, k)];
            }
        }
    }
    
    void train_step(const float* state, int target, float lr) {
        // Forward
        std::vector<float> bucket_scores(N_BUCKETS, 0);
        for (int b = 0; b < N_BUCKETS; ++b) {
            for (int d = 0; d < D; ++d) {
                bucket_scores[b] += pool[b][d] * state[d];
            }
        }
        float logits[V];
        for (int v = 0; v < V; ++v) {
            logits[v] = bias[v];
            for (int k = 0; k < K_HASH; ++k) {
                logits[v] += bucket_scores[hash_func(v, k)];
            }
        }
        float max_l = *std::max_element(logits, logits + V);
        float sum = 0;
        float probs[V];
        for (int v = 0; v < V; ++v) { probs[v] = std::exp(logits[v] - max_l); sum += probs[v]; }
        for (int v = 0; v < V; ++v) probs[v] /= sum;
        
        // Backward: update pool[b][d] for each bucket b
        std::vector<float> d_bucket(N_BUCKETS, 0);
        for (int b = 0; b < N_BUCKETS; ++b) {
            float sum_dl = 0;
            for (int v = 0; v < V; ++v) {
                for (int k = 0; k < K_HASH; ++k) {
                    if (hash_func(v, k) == b) {
                        sum_dl += probs[v] - (v == target ? 1.0f : 0.0f);
                    }
                }
            }
            d_bucket[b] = sum_dl;
        }
        for (int b = 0; b < N_BUCKETS; ++b) {
            for (int d = 0; d < D; ++d) {
                pool[b][d] -= lr * d_bucket[b] * state[d];
                pool[b][d] *= (1.0f - WD);  // weight decay
            }
        }
        for (int v = 0; v < V; ++v) {
            bias[v] -= lr * (probs[v] - (v == target ? 1.0f : 0.0f));
        }
    }
};

int main() {
    std::cout << "================================================================\n";
    std::cout << "  OPTION A: Hash Bucket Q4 | V=" << V << " multi-class\n";
    std::cout << "  Pool: " << N_BUCKETS << " x " << D << " = " << N_BUCKETS * D << " params\n";
    std::cout << "================================================================\n\n";
    
    HashBucketQ4 q4;
    
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
        float logits[V];
        q4.compute_logits(state_norm_buf.data(), logits);
        int best = 0;
        for (int v = 1; v < V; ++v) if (logits[v] > logits[best]) best = v;
        return best;
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
            
            // Compute loss
            float logits[V];
            q4.compute_logits(state_norm.data(), logits);
            float max_l = *std::max_element(logits, logits + V);
            float sum = 0;
            float probs[V];
            for (int v = 0; v < V; ++v) { probs[v] = std::exp(logits[v] - max_l); sum += probs[v]; }
            for (int v = 0; v < V; ++v) probs[v] /= sum;
            total_loss += -std::log(std::max(probs[p.second], 1e-7f));
            
            q4.train_step(state_norm.data(), p.second, LR);
        }
        
        if ((epoch + 1) % 5 == 0 || epoch == 0 || epoch == EPOCHS - 1) {
            int c = 0;
            for (auto& pp : data) if (predict(pp.first) == pp.second) c++;
            std::cout << "  Epoch " << std::setw(3) << (epoch + 1)
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
    for (int i = 0; i < 200; ++i) {
        std::vector<int> seq(SEQ_LEN);
        std::vector<int> counts(V, 0);
        for (int t = 0; t < SEQ_LEN; ++t) { seq[t] = test_rng() % V; counts[seq[t]]++; }
        int majority = 0; int mx = counts[0];
        for (int v = 1; v < V; ++v) if (counts[v] > mx) { mx = counts[v]; majority = v; }
        if (predict(seq) == majority) tc++;
    }
    std::cout << "  Held-out: " << tc << "/200 (" << std::fixed << std::setprecision(1) << tc * 100.0 / 200 << "%)\n";
    
    // Verify pool learned
    std::cout << "\n  Learned Hash Bucket Pool (key dims):\n";
    std::cout << "    Pool[0][0] (token 0 hash) = " << std::setprecision(3) << q4.pool[0][0] << "\n";
    std::cout << "    Pool[3][2] (token 1 hash) = " << std::setprecision(3) << q4.pool[3][2] << "\n";
    std::cout << "    Pool[6][4] (token 2 hash) = " << std::setprecision(3) << q4.pool[6][4] << "\n";
    std::cout << "    Pool[9 mod 8=1][6] (token 3 hash) = " << std::setprecision(3) << q4.pool[1][6] << "\n";

    return 0;
}
