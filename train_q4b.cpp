// train_q4b.cpp
// OPTION B: Top-K Validation
//
// V=50 vocab (realistic LLM size), Linear Q4.
// Measure top-1, top-3, top-5, top-10 recall after training.

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

const int D = 128;
const int S_MAX = 64;
const int SEQ_LEN = 16;
const int EPOCHS = 30;
const int N_TRAIN = 1000;
const int V = 50;           // realistic vocab
const float LR = 0.1f;
const float WD = 0.02f;

void q1(int token_id, int8_t* x) {
    std::fill(x, x + D, 0);
    if (token_id >= 0 && token_id < V) {
        x[token_id] = 3;  // one dim per token
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

struct Q4 {
    std::vector<std::vector<float>> w;  // [V][D]
    std::vector<float> b;
    Q4() {
        w.assign(V, std::vector<float>(D, 0));
        b.assign(V, 0);
        std::mt19937 rng(456);
        std::normal_distribution<float> nd(0, 0.02f);
        for (auto& row : w) for (auto& v : row) v = nd(rng);
    }
    void logits(const float* state, float* out) const {
        for (int v = 0; v < V; ++v) {
            float s = b[v];
            for (int d = 0; d < D; ++d) s += w[v][d] * state[d];
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
    std::cout << "  OPTION B: Top-K Validation | V=" << V << " multi-class\n";
    std::cout << "  Q4: Linear projection [" << V << " x " << D << "] = " << V * D << " params\n";
    std::cout << "================================================================\n\n";
    
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
    
    std::vector<float> state_norm_buf(D);
    auto predict = [&](const std::vector<int>& seq, int* top_k, int K) {
        std::vector<std::vector<int16_t>> ss;
        forward_seq(seq, ss);
        normalize_state(ss.back().data(), state_norm_buf.data());
        float logits[V];
        q4.logits(state_norm_buf.data(), logits);
        // Get top K
        std::vector<int> idx(V);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
            [&](int a, int b) { return logits[a] > logits[b]; });
        for (int k = 0; k < K; ++k) top_k[k] = idx[k];
        return top_k[0];
    };
    
    int initial = 0;
    int tmp[10];
    for (auto& p : data) if (predict(p.first, tmp, 1) == p.second) initial++;
    std::cout << "  Initial: " << initial << "/" << data.size() 
              << " (" << std::fixed << std::setprecision(1) << initial * 100.0 / data.size() << "%)\n\n";
    
    std::vector<std::vector<float>> w_grad(V, std::vector<float>(D, 0));
    std::vector<float> b_grad(V, 0);
    std::vector<float> state_norm(D);
    
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(data.begin(), data.end(), std::mt19937(epoch + 1));
        for (auto& row : w_grad) std::fill(row.begin(), row.end(), 0);
        std::fill(b_grad.begin(), b_grad.end(), 0);
        float total_loss = 0;
        for (auto& p : data) {
            std::vector<std::vector<int16_t>> ss;
            forward_seq(p.first, ss);
            normalize_state(ss.back().data(), state_norm.data());
            float logits[V];
            q4.logits(state_norm.data(), logits);
            float probs[V];
            softmax(logits, probs);
            total_loss += -std::log(std::max(probs[p.second], 1e-7f));
            for (int v = 0; v < V; ++v) {
                float d_logit = probs[v] - (v == p.second ? 1.0f : 0.0f);
                for (int d = 0; d < D; ++d) w_grad[v][d] += d_logit * state_norm[d];
                b_grad[v] += d_logit;
            }
        }
        for (int v = 0; v < V; ++v) {
            for (int d = 0; d < D; ++d) {
                q4.w[v][d] -= LR * w_grad[v][d] / N_TRAIN;
                q4.w[v][d] *= (1.0f - WD);
            }
            q4.b[v] -= LR * b_grad[v] / N_TRAIN;
        }
        
        if ((epoch + 1) % 5 == 0 || epoch == 0 || epoch == EPOCHS - 1) {
            int top1 = 0, top3 = 0, top5 = 0, top10 = 0;
            int topk[10];
            for (auto& pp : data) {
                predict(pp.first, topk, 10);
                int target = pp.second;
                if (topk[0] == target) top1++;
                for (int k = 0; k < 3; ++k) if (topk[k] == target) { top3++; break; }
                for (int k = 0; k < 5; ++k) if (topk[k] == target) { top5++; break; }
                for (int k = 0; k < 10; ++k) if (topk[k] == target) { top10++; break; }
            }
            std::cout << "  Epoch " << std::setw(2) << (epoch + 1)
                      << ": loss=" << std::fixed << std::setprecision(4) << total_loss / data.size()
                      << " | Top1=" << std::setw(3) << top1 * 100 / data.size() << "%"
                      << " Top3=" << std::setw(3) << top3 * 100 / data.size() << "%"
                      << " Top5=" << std::setw(3) << top5 * 100 / data.size() << "%"
                      << " Top10=" << std::setw(3) << top10 * 100 / data.size() << "%"
                      << "\n";
        }
    }
    
    // Final held-out test
    std::cout << "\n  === HELD-OUT TEST (500 sequences) ===\n";
    int top1 = 0, top3 = 0, top5 = 0, top10 = 0, top20 = 0;
    int topk[20];
    std::mt19937 test_rng(999);
    for (int i = 0; i < 500; ++i) {
        std::vector<int> seq(SEQ_LEN);
        std::vector<int> counts(V, 0);
        for (int t = 0; t < SEQ_LEN; ++t) { seq[t] = test_rng() % V; counts[seq[t]]++; }
        int majority = 0; int mx = counts[0];
        for (int v = 1; v < V; ++v) if (counts[v] > mx) { mx = counts[v]; majority = v; }
        predict(seq, topk, 20);
        if (topk[0] == majority) top1++;
        for (int k = 0; k < 3; ++k) if (topk[k] == majority) { top3++; break; }
        for (int k = 0; k < 5; ++k) if (topk[k] == majority) { top5++; break; }
        for (int k = 0; k < 10; ++k) if (topk[k] == majority) { top10++; break; }
        for (int k = 0; k < 20; ++k) if (topk[k] == majority) { top20++; break; }
    }
    std::cout << "  Top-1:  " << std::setw(3) << top1 * 100 / 500 << "%\n";
    std::cout << "  Top-3:  " << std::setw(3) << top3 * 100 / 500 << "%\n";
    std::cout << "  Top-5:  " << std::setw(3) << top5 * 100 / 500 << "%\n";
    std::cout << "  Top-10: " << std::setw(3) << top10 * 100 / 500 << "%\n";
    std::cout << "  Top-20: " << std::setw(3) << top20 * 100 / 500 << "%\n";
    std::cout << "  Random Top-10 baseline: " << (10.0 / V) * 100 << "%\n";
    
    return 0;
}
