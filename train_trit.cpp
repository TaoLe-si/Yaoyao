// train_trit.cpp
// Minimal training demo for the trit hash-bucket pool design.
//
// Task: classify words into 3 categories (animal / machine / fruit).
// Architecture:
//   word -> char-level bag-of-embeddings -> linear head -> 3 logits
//   embedding for char c = quantize(pool[hash(c)])  <- trit {-1, 0, +1}
// Training:
//   Straight-through estimator (STE) for the ternary quantization
//   Cross-entropy loss, SGD
//
// Compile: clang++ -O2 -std=c++17 -march=native -o train_trit.exe train_trit.cpp

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <utility>

// ============================================================================
// CONFIG
// ============================================================================

const int H = 256;            // number of hash buckets
const int D = 16;             // hidden dim per bucket
const int NUM_CLASSES = 3;    // animal / machine / fruit
const int EPOCHS = 120;
float LR = 0.05f;

// ============================================================================
// HELPERS
// ============================================================================

// Deterministic char -> bucket hash (multiplicative)
inline int char_hash(char c, int H) {
    return ((unsigned char)c * 2654435761u) % (unsigned)H;
}

// Float -> trit
inline int quantize(float v) {
    if (v >  0.5f) return  1;
    if (v < -0.5f) return -1;
    return 0;
}

// Softmax in-place
void softmax(std::vector<float>& x) {
    float mx = *std::max_element(x.begin(), x.end());
    float s = 0;
    for (auto& v : x) { v = std::exp(v - mx); s += v; }
    for (auto& v : x) v /= s;
}

// ============================================================================
// TRIT BUCKET POOL
//   Stores latent float weights W[H*D], quantized on every forward pass.
//   STE updates the latent values; quantization happens implicitly.
// ============================================================================

struct Pool {
    int H, D;
    std::vector<float> W;

    Pool(int H_, int D_, std::mt19937& rng) : H(H_), D(D_) {
        W.resize(H_ * D_);
        std::normal_distribution<float> nd(0.0f, 0.6f);
        for (auto& w : W) w = nd(rng);
    }

    // Forward: write quantized bucket values to out[D]
    void get_quantized(int h, int* out) const {
        for (int d = 0; d < D; ++d) {
            out[d] = quantize(W[h * D + d]);
        }
    }

    // STE backward: subtract lr*grad from latent value (clamped to avoid runaway)
    void update_ste(int h, int d, float grad) {
        float& w = W[h * D + d];
        w -= LR * grad;
        if (w >  4.0f) w =  4.0f;
        if (w < -4.0f) w = -4.0f;
    }
};

// ============================================================================
// TRAINING STEP
//   Forward: word -> bag-of-trits -> linear head -> softmax -> loss
//   Backward: cross-entropy gradient -> head -> embed -> STE -> pool
// ============================================================================

float train_step(const std::string& word, int label,
                 Pool& pool,
                 std::vector<std::vector<float>>& head_W,
                 std::vector<float>& head_b) {

    // Forward: bag-of-trits embedding
    std::vector<int> embed(D, 0);
    for (char c : word) {
        int h = char_hash(c, pool.H);
        int trit[64];
        pool.get_quantized(h, trit);
        for (int d = 0; d < D; ++d) embed[d] += trit[d];
    }

    // Linear head
    std::vector<float> logits(NUM_CLASSES, 0.0f);
    for (int c = 0; c < NUM_CLASSES; ++c) {
        for (int d = 0; d < D; ++d) logits[c] += embed[d] * head_W[c][d];
        logits[c] += head_b[c];
    }
    softmax(logits);

    // Cross-entropy loss
    float loss = -std::log(std::max(logits[label], 1e-7f));

    // Gradient w.r.t. logits (softmax + CE combined)
    std::vector<float> d_logits(NUM_CLASSES);
    for (int c = 0; c < NUM_CLASSES; ++c) {
        d_logits[c] = logits[c] - (c == label ? 1.0f : 0.0f);
    }

    // Gradient w.r.t. embedding (using current head_W, before update)
    std::vector<float> d_embed(D, 0.0f);
    for (int c = 0; c < NUM_CLASSES; ++c) {
        for (int d = 0; d < D; ++d) {
            d_embed[d] += d_logits[c] * head_W[c][d];
        }
    }

    // Update head
    for (int c = 0; c < NUM_CLASSES; ++c) {
        for (int d = 0; d < D; ++d) {
            head_W[c][d] -= LR * d_logits[c] * embed[d];
        }
        head_b[c] -= LR * d_logits[c];
    }

    // STE: pass gradient through unchanged to every bucket touched
    for (char c : word) {
        int h = char_hash(c, pool.H);
        for (int d = 0; d < D; ++d) {
            pool.update_ste(h, d, d_embed[d]);
        }
    }

    return loss;
}

// ============================================================================
// PREDICTION (inference)
// ============================================================================

int predict(const std::string& word, const Pool& pool,
            const std::vector<std::vector<float>>& head_W,
            const std::vector<float>& head_b) {

    std::vector<int> embed(D, 0);
    for (char c : word) {
        int h = char_hash(c, pool.H);
        int trit[64];
        pool.get_quantized(h, trit);
        for (int d = 0; d < D; ++d) embed[d] += trit[d];
    }

    std::vector<float> logits(NUM_CLASSES, 0.0f);
    for (int c = 0; c < NUM_CLASSES; ++c) {
        for (int d = 0; d < D; ++d) logits[c] += embed[d] * head_W[c][d];
        logits[c] += head_b[c];
    }
    return std::distance(logits.begin(),
                         std::max_element(logits.begin(), logits.end()));
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Trit hash bucket pool - minimal training demo\n";
    std::cout << "================================================================\n\n";

    std::cout << "  Task: classify words into 3 categories\n";
    std::cout << "         (animal=0, machine=1, fruit=2)\n";
    std::cout << "  Model: char-bag-of-trits + linear head\n";
    std::cout << "  Quantization:  {-1, 0, +1}, STE for backward\n";
    std::cout << "  Pool: H=" << H << " buckets x D=" << D << " dims\n";
    std::cout << "  Training: " << EPOCHS << " epochs, LR=" << LR << "\n\n";

    // ----- Training data -----
    std::vector<std::pair<std::string, int>> data = {
        // Animals
        {"cat", 0}, {"dog", 0}, {"pig", 0}, {"fox", 0}, {"owl", 0}, {"bee", 0},
        {"cow", 0}, {"rat", 0}, {"ape", 0}, {"elk", 0},
        // Machines
        {"car", 1}, {"jet", 1}, {"van", 1}, {"bot", 1}, {"cpu", 1}, {"web", 1},
        {"ram", 1}, {"hub", 1}, {"pod", 1}, {"zip", 1},
        // Fruits
        {"fig", 2}, {"pea", 2}, {"kiwi", 2}, {"lime", 2}, {"plum", 2}, {"pear", 2},
        {"date", 2}, {"berry", 2}, {"melon", 2}, {"guava", 2}
    };

    // ----- Init -----
    std::mt19937 rng(42);
    Pool pool(H, D, rng);
    std::vector<std::vector<float>> head_W(NUM_CLASSES, std::vector<float>(D, 0.0f));
    std::vector<float> head_b(NUM_CLASSES, 0.0f);
    std::normal_distribution<float> nd(0.0f, 0.3f);
    for (auto& row : head_W) for (auto& w : row) w = nd(rng);

    // ----- Initial accuracy -----
    int correct = 0;
    for (auto& [w, l] : data) if (predict(w, pool, head_W, head_b) == l) correct++;
    std::cout << "Initial accuracy: " << correct << " / " << data.size()
              << " (" << correct * 100 / data.size() << "%)\n\n";

    // ----- Train -----
    std::cout << "  Epoch | Avg Loss | Accuracy\n";
    std::cout << "  ------+----------+---------\n";
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(data.begin(), data.end(), rng);
        float total_loss = 0;
        for (auto& [w, l] : data) {
            total_loss += train_step(w, l, pool, head_W, head_b);
        }
        if ((epoch + 1) % 10 == 0) {
            int c = 0;
            for (auto& [w, l] : data) if (predict(w, pool, head_W, head_b) == l) c++;
            std::cout << "  " << std::setw(5) << (epoch + 1) << " | "
                      << std::fixed << std::setprecision(4) << std::setw(8)
                      << total_loss / data.size() << " | "
                      << c << "/" << data.size()
                      << " (" << c * 100 / data.size() << "%)\n";
        }
    }

    // ----- Final accuracy -----
    int final_correct = 0;
    for (auto& [w, l] : data) if (predict(w, pool, head_W, head_b) == l) final_correct++;
    std::cout << "\nFinal accuracy: " << final_correct << " / " << data.size()
              << " (" << final_correct * 100 / data.size() << "%)\n\n";

    // ----- Per-class breakdown -----
    std::cout << "Per-class accuracy:\n";
    const char* names[] = {"animal", "machine", "fruit  "};
    for (int c = 0; c < NUM_CLASSES; ++c) {
        int total = 0, correct = 0;
        for (auto& [w, l] : data) {
            if (l == c) {
                total++;
                if (predict(w, pool, head_W, head_b) == l) correct++;
            }
        }
        std::cout << "  " << names[c] << ": " << correct << "/" << total
                  << " (" << correct * 100 / total << "%)\n";
    }
    std::cout << "\n";

    // ----- Show learned bucket vectors for example chars -----
    std::cout << "Learned bucket vectors (quantized to trit):\n";
    std::cout << "  char  bucket | values (each = -1, 0, or +1)\n";
    std::cout << "  -----+--------+----------------------------\n";

    // Group chars by which category they appear in
    std::string animal_chars = "abcdefgijlmo-prt-w";  // mixed
    std::string showcase = "cdfgijoprtvw";  // representative
    
    for (char c : showcase) {
        int h = char_hash(c, pool.H);
        std::cout << "  '" << c << "'   " << std::setw(3) << h << "   | ";
        for (int d = 0; d < D; ++d) {
            int t = quantize(pool.W[h * D + d]);
            std::cout << std::setw(2) << t;
        }
        std::cout << "\n";
    }

    // ----- Cosine similarity between chars (to show "feature emergence") -----
    std::cout << "\nSimilarity between chars (dot product of trit vectors):\n";
    std::cout << "  Higher = more similar role in classification\n";
    std::cout << "       ";
    for (char c : showcase) std::cout << std::setw(3) << c;
    std::cout << "\n";
    for (char c1 : showcase) {
        std::cout << "  " << c1 << "   ";
        int h1 = char_hash(c1, pool.H);
        int v1[64];
        pool.get_quantized(h1, v1);
        for (char c2 : showcase) {
            int h2 = char_hash(c2, pool.H);
            int v2[64];
            pool.get_quantized(h2, v2);
            int dot = 0;
            for (int d = 0; d < D; ++d) dot += v1[d] * v2[d];
            std::cout << std::setw(3) << dot;
        }
        std::cout << "\n";
    }

    std::cout << "\n================================================================\n";
    std::cout << "  Done.\n";
    std::cout << "================================================================\n";
    return 0;
}
