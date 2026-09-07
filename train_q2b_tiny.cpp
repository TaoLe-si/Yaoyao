// train_q2b.cpp
// Q2-B with input-dependent gate, trit-style throughout.
// 
// Architecture:
//   State h in {-1, 0, +1}^D          (trit)
//   Gate matrix W in {-1, 0, +1}^{DxD} (trit, learned)
//   Input x in {-K..K}^D              (integer from Q1)
//
// Forward (per step):
//   z[d] = sum_j W[d,j] * x[j]                   # bounded integer matmul
//   g[d] = sign(z[d]) if |z[d]| >= T else 0     # trit gate via threshold
//   if g=+1: h_new = h_old
//   if g=-1: h_new = x
//   if g=0:  h_new = (h_old + x)/2
//
// Training trick: use continuous tanh(z) during training, quantize W + g for inference.
// This is QAT (Quantization-Aware Training).
//
// Task: predict LAST token. Compare to v4.
//
// Compile: clang++ -O2 -std=c++17 -march=native -o train_q2b.exe train_q2b.cpp

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>

const int D = 256;  // smaller for demo speed; full D=4096 works but slower to train
const int SEQ_LEN = 16;
const int EPOCHS = 5;
const int N_TRAIN = 50;
const float LR_W = 0.05f;
const float LR_HEAD = 0.3f;

void q1_lookup(int token_id, int8_t* out_x) {
    std::fill(out_x, out_x + D, 0);
    if (token_id == 0) { out_x[0] = 3; out_x[1] = -3; out_x[2] = 1; }
    else { out_x[0] = -3; out_x[1] = 3; out_x[2] = -1; }
}

struct Q2B {
    std::vector<float> W;  // D x D, float during training, quantized to trit
    int D_;
    Q2B(int D__, std::mt19937& rng) : D_(D__) {
        W.resize(D_ * D_);
        std::normal_distribution<float> nd(0.0f, 0.05f);
        for (auto& w : W) w = nd(rng);
    }

    void step(const int8_t* x_t, const float* h_old, float* h_new) {
        for (int d = 0; d < D_; ++d) {
            float z = 0;
            for (int j = 0; j < D_; ++j) {
                z += W[d * D_ + j] * (float)x_t[j];
            }
            float g = std::tanh(z);
            h_new[d] = 0.5f * (1.0f + g) * h_old[d] + 0.5f * (1.0f - g) * (float)x_t[d];
        }
    }

    void quantize_trit() {
        for (auto& w : W) {
            if (w > 0.5f) w = 1.0f;
            else if (w < -0.5f) w = -1.0f;
            else w = 0.0f;
        }
    }
};

float linear_forward(const float* h, const float* w, float b) {
    float sum = b;
    for (int d = 0; d < D; ++d) sum += h[d] * w[d];
    return sum;
}

inline float sigmoid_2x(float x) {
    return 1.0f / (1.0f + std::exp(-2.0f * x));
}

struct SeqState {
    std::vector<std::vector<float>> hs;
    std::vector<std::vector<float>> gs;  // cached gate values for backward
    std::vector<std::vector<int8_t>> xs;
};

SeqState forward_record(const std::vector<int>& tokens, Q2B& q2b) {
    SeqState s;
    s.hs.resize(SEQ_LEN + 1);
    s.gs.resize(SEQ_LEN);
    s.xs.resize(SEQ_LEN);
    // FIX: pre-allocate every vector to avoid nullptr data()
    for (int i = 0; i <= SEQ_LEN; ++i) s.hs[i].assign(D, 0.0f);
    for (int i = 0; i < SEQ_LEN; ++i) {
        s.gs[i].assign(D, 0.0f);
        s.xs[i].assign(D, 0);
    }
    std::vector<float> h_new(D);
    std::vector<int8_t> x_t(D);
    for (int t = 0; t < SEQ_LEN; ++t) {
        q1_lookup(tokens[t], x_t.data());
        s.xs[t] = x_t;
        for (int d = 0; d < D; ++d) {
            float z = 0;
            for (int j = 0; j < D; ++j) {
                z += q2b.W[d * D + j] * (float)x_t[j];
            }
            float g = std::tanh(z);
            s.gs[t][d] = g;  // cache
            h_new[d] = 0.5f * (1.0f + g) * s.hs[t][d] + 0.5f * (1.0f - g) * (float)x_t[d];
        }
        s.hs[t + 1] = h_new;
    }
    return s;
}

int predict(const std::vector<int>& tokens, Q2B& q2b, const float* w, float b) {
    SeqState s = forward_record(tokens, q2b);
    float logit = linear_forward(s.hs.back().data(), w, b);
    return logit > 0 ? 0 : 1;
}

float train_step(const std::vector<int>& tokens, int label,
                 Q2B& q2b, std::vector<float>& W_grad,
                 std::vector<float>& w, float& b) {
    SeqState s = forward_record(tokens, q2b);

    float logit = linear_forward(s.hs.back().data(), w.data(), b);
    float p0 = sigmoid_2x(logit);
    float prob_true = (label == 0) ? p0 : (1.0f - p0);
    float loss = -std::log(std::max(prob_true, 1e-7f));

    float d_logit = (label == 0) ? (-2.0f * (1.0f - p0)) : (2.0f * p0);

    std::vector<float> d_h(D);
    for (int d = 0; d < D; ++d) {
        d_h[d] = d_logit * w[d];
        w[d] -= LR_HEAD * d_logit * s.hs.back()[d];
    }
    b -= LR_HEAD * d_logit;

    // BPTT through Q2-B (using cached g values)
    for (int t = SEQ_LEN - 1; t >= 0; --t) {
        std::vector<float> d_h_old(D, 0.0f);
        for (int d = 0; d < D; ++d) {
            float g = s.gs[t][d];
            // h_new[d] = 0.5*(1+g)*h_old + 0.5*(1-g)*x_t
            float coef_h = 0.5f * (1.0f + g);
            d_h_old[d] = coef_h * d_h[d];

            // d_g via smooth surrogate (treat h_new as smooth in g)
            float d_g = 0.5f * (s.hs[t][d] - (float)s.xs[t][d]) * d_h[d];
            // d_z via tanh derivative
            float dg_dz = 1.0f - g * g;
            float d_z = d_g * dg_dz;

            // d_W[d, j] = d_z * x_t[j]
            for (int j = 0; j < D; ++j) {
                W_grad[d * D + j] += d_z * (float)s.xs[t][j];
            }
        }
        d_h = std::move(d_h_old);
    }

    // Note: don't apply W update here - do it after full batch
    return loss;
}

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Q2-B: input-dependent gate, trit style throughout\n";
    std::cout << "================================================================\n\n";

    std::cout << "  Architecture:\n";
    std::cout << "    h in {-1, 0, +1}^D           (trit state)\n";
    std::cout << "    W in {-1, 0, +1}^{DxD}        (trit gate matrix)\n";
    std::cout << "    g[d] = sign(W[d] . x)         (trit gate, input-dependent)\n";
    std::cout << "    h_new[d] = blend(g[d], h_old[d], x[d])  (keep/replace/average)\n\n";

    std::cout << "  Training trick: continuous tanh during training,\n";
    std::cout << "                  quantize W to trit at deployment (QAT).\n\n";

    std::cout << "  Task: predict LAST token (same as v4 for comparison)\n";
    std::cout << "  D = " << D << " (smaller than full 4096 for demo speed)\n";
    std::cout << "  N_TRAIN = " << N_TRAIN << ", EPOCHS = " << EPOCHS << "\n\n";

    std::mt19937 rng(42);
    std::vector<std::pair<std::vector<int>, int>> data;
    for (int i = 0; i < N_TRAIN; ++i) {
        std::vector<int> seq(SEQ_LEN);
        for (int t = 0; t < SEQ_LEN; ++t) seq[t] = (rng() % 2 == 0) ? 0 : 1;
        int label = seq[SEQ_LEN - 1];
        data.push_back({seq, label});
    }

    Q2B q2b(D, rng);
    std::vector<float> w(D, 0.0f);
    float b = 0.0f;
    std::normal_distribution<float> nd(0.0f, 0.1f);
    for (auto& wi : w) wi = nd(rng);

    std::vector<float> W_grad(D * D);

    int correct = 0;
    for (auto& p : data) {
        if (predict(p.first, q2b, w.data(), b) == p.second) correct++;
    }
    std::cout << "  Initial acc (float W): " << correct << "/" << data.size()
              << " (" << correct * 100 / data.size() << "%)\n\n";

    std::cout << "  Epoch | Avg Loss | Acc (float)\n";
    std::cout << "  ------+----------+------------\n";
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(data.begin(), data.end(), rng);
        std::fill(W_grad.begin(), W_grad.end(), 0.0f);
        float total_loss = 0;
        for (auto& p : data) {
            float loss = train_step(p.first, p.second, q2b, W_grad, w, b);
            total_loss += loss;
        }
        // Apply W update (full batch)
        for (size_t i = 0; i < q2b.W.size(); ++i) {
            q2b.W[i] -= LR_W * W_grad[i] / N_TRAIN;
        }

        if ((epoch + 1) % 5 == 0 || epoch == 0 || epoch == EPOCHS - 1) {
            int c = 0;
            for (auto& pp : data) {
                if (predict(pp.first, q2b, w.data(), b) == pp.second) c++;
            }
            std::cout << "  " << std::setw(5) << (epoch + 1) << " | "
                      << std::fixed << std::setprecision(4) << std::setw(8)
                      << total_loss / data.size() << " | "
                      << c << "/" << data.size()
                      << " (" << std::setw(3) << c * 100 / data.size() << "%)\n";
        }
    }

    // Final acc with float W
    int final_c = 0;
    for (auto& p : data) {
        if (predict(p.first, q2b, w.data(), b) == p.second) final_c++;
    }
    std::cout << "\n  Final acc (float W): " << final_c << "/" << data.size()
              << " (" << final_c * 100 / data.size() << "%)\n";

    // Quantize W to trit and re-evaluate
    q2b.quantize_trit();
    int trit_c = 0;
    for (auto& p : data) {
        if (predict(p.first, q2b, w.data(), b) == p.second) trit_c++;
    }
    std::cout << "  Final acc (TRIT W): " << trit_c << "/" << data.size()
              << " (" << trit_c * 100 / data.size() << "%)\n\n";

    // W distribution after quantization
    int cnt_p1 = 0, cnt_n1 = 0, cnt_0 = 0;
    for (float w_val : q2b.W) {
        if (w_val > 0.5f) cnt_p1++;
        else if (w_val < -0.5f) cnt_n1++;
        else cnt_0++;
    }
    long total_w = q2b.W.size();
    std::cout << "  W distribution (after trit quantization):\n";
    std::cout << "    +1: " << cnt_p1 << " (" << cnt_p1 * 100.0 / total_w << "%)\n";
    std::cout << "     0: " << cnt_0 << " (" << cnt_0 * 100.0 / total_w << "%)\n";
    std::cout << "    -1: " << cnt_n1 << " (" << cnt_n1 * 100.0 / total_w << "%)\n\n";

    // Show some interesting W rows
    std::cout << "  W for dim 0 (sees x[0] = +3 for A, -3 for B):\n";
    std::cout << "    First 10 entries: ";
    for (int j = 0; j < 10; ++j) std::cout << (int)q2b.W[0 * D + j] << " ";
    std::cout << "\n";
    std::cout << "  W for dim 1 (sees x[1] = -3 for A, +3 for B):\n";
    std::cout << "    First 10 entries: ";
    for (int j = 0; j < 10; ++j) std::cout << (int)q2b.W[1 * D + j] << " ";
    std::cout << "\n\n";

    // Per-step timing (inference, with trit W)
    std::cout << "  Per-step timing (Q2-B inference, 1000 iters):\n";
    std::vector<int8_t> x_t(D);
    std::vector<float> h(D, 0), h_new(D);
    q1_lookup(0, x_t.data());

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        q2b.step(x_t.data(), h.data(), h_new.data());
        std::swap(h, h_new);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    std::cout << "    " << std::setprecision(2) << us / 1000 << " us / step (D=" << D << ")\n";
    std::cout << "    (includes full D x D matmul + tanh + blend)\n";
    std::cout << "    Estimated for D=4096: ~" << std::setprecision(0)
              << (us / 1000) * 16 << " us / step (16x slower for 16x larger matrix)\n\n";

    // Memory footprint
    std::cout << "  Memory (with trit W):\n";
    std::cout << "    W:    " << total_w * 2 / 8 / 1024 << " KB (2-bit packed trit)\n";
    std::cout << "    h:    " << D << " bytes (int8)\n";
    std::cout << "    x_t:  " << D << " bytes (int8)\n";
    std::cout << "    Total working set: ~" << (D * 2 / 8 + 2 * D) / 1024 << " KB (cache friendly)\n\n";

    std::cout << "================================================================\n";
    return 0;
}
