// yaoyao_v21_full.cpp
// 夭夭 v21: 基于 v19 改造 - 完整版 (支持 TinyStories)
// 替换 h/s 通道为 mod 3 可逆 trit + 滚动 hash
//
// 编译: clang++ -O2 -mavx2 -mfma -fopenmp -o yaoyao_v21_full.exe yaoyao_v21_full.cpp
// 运行: yaoyao_v21_full.exe [text_path] [vocab_size=1024]
//       yaoyao_v21_full.exe D:\TaoVm\tinystories_train.txt 1024

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <random>
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <unordered_map>
#include <sstream>
#include <iomanip>
#include <map>
#include <fstream>
#include <numeric>
#include <cstdint>
#include <cassert>
#include <stdexcept>
#include <immintrin.h>  // [V21-CPU-OPT] AVX2 SIMD

// [V21-CPU-OPT] SIMD 矩阵-向量乘 (y = W·x + bias), W 是 [n, k] 浮点矩阵
// 优化: HIDDEN=192 (24 * 8), V_unit=1024 (128 * 8), 完美对齐
inline void sgl_matvec(const float* W, const float* x, const float* bias, float* y, int n, int k) {
    int k8 = k / 8;  // k 应该是 8 的倍数
    for (int i = 0; i < n; ++i) {
        __m256 sum = _mm256_setzero_ps();
        const float* row = W + i * k;
        for (int j = 0; j < k8; ++j) {
            __m256 w = _mm256_loadu_ps(row + j * 8);
            __m256 xv = _mm256_loadu_ps(x + j * 8);
            sum = _mm256_fmadd_ps(w, xv, sum);
        }
        // 水平求和 (8 个 float -> 1)
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        float dot = _mm_cvtss_f32(s);
        y[i] = dot + (bias ? bias[i] : 0.0f);
    }
}

typedef int8_t trit;
typedef uint32_t hash_t;

const hash_t INV33_MOD_2_32 = 0x3e0f83e1u;

// [V21-Phase6] 64 特征映射 (单链): h 的 8 个 4-bit nibble 循环输出
// f_i = ((h >> (4*(i%8))) & 0xF) / 15.0f - 0.5f,  i = 0..63
// 注意: HASH_FEATURES 在下方声明, 此处用 extern 声明, 实际定义在常量段之后。
inline void extract_hash_features(hash_t h, float* features);

// ============================================================================
//  [V] 数学原语 + 自动验证
// ============================================================================

inline trit mod3(int x) {
    int r = x % 3;
    if (r > 1) r -= 3;
    if (r < -1) r += 3;
    return (trit)r;
}

inline hash_t hash_forward(hash_t h, int token) {
    return ((h * 33u) + (hash_t)token);
}

inline hash_t hash_reverse(hash_t h, int token) {
    return (h - (hash_t)token) * INV33_MOD_2_32;
}

inline bool verify_mod3_closed() {
    for (int a = -3; a <= 3; a++)
    for (int b = -3; b <= 3; b++)
    for (int c = -3; c <= 3; c++) {
        if (mod3(a + b + c) < -1 || mod3(a + b + c) > 1) return false;
    }
    return true;
}

inline bool verify_mod3_reversible() {
    int h = 1, x = 2;
    int h_new = mod3(h + x);
    return mod3(h_new - x) == h;
}

inline bool verify_hash_reversible() {
    hash_t h = 5381;
    for (int i = 1; i <= 6; i++) h = hash_forward(h, i);
    for (int i = 6; i >= 1; i--) h = hash_reverse(h, i);
    return h == 5381u;
}

inline bool verify_combined_reversible() {
    const int D_T = 16;
    trit h_trit[D_T];
    hash_t h_hash = 5381;
    for (int i = 0; i < D_T; i++) h_trit[i] = (trit)(i % 3 - 1);
    trit h_trit_orig[D_T];
    memcpy(h_trit_orig, h_trit, sizeof(h_trit));
    hash_t h_hash_orig = h_hash;
    int seq[100] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                    21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
                    41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,
                    61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,
                    81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100};
    for (int t = 0; t < 100; t++) {
        for (int d = 0; d < D_T; d++)
            h_trit[d] = mod3((int)h_trit[d] + ((seq[t] >> (d % 8)) & 1 ? 1 : -1));
        h_hash = hash_forward(h_hash, seq[t]);
    }
    for (int t = 99; t >= 0; t--) {
        h_hash = hash_reverse(h_hash, seq[t]);
        for (int d = 0; d < D_T; d++)
            h_trit[d] = mod3((int)h_trit[d] - ((seq[t] >> (d % 8)) & 1 ? 1 : -1));
    }
    bool trit_ok = true;
    for (int d = 0; d < D_T; d++) if (h_trit[d] != h_trit_orig[d]) trit_ok = false;
    return trit_ok && (h_hash == h_hash_orig);
}

// ============================================================================
//  [V] 全局验证 - 编译时自动运行
// ============================================================================

bool run_math_verification() {
    std::printf("\n========================================\n");
    std::printf("  [V] 数学模型验证\n");
    std::printf("========================================\n");
    bool all = true;
    bool c = verify_mod3_closed(); all &= c;
    std::printf("  [%s] mod 3 封闭性\n", c ? "PASS" : "FAIL");
    bool r = verify_mod3_reversible(); all &= r;
    std::printf("  [%s] mod 3 可逆 (forward+reverse = id)\n", r ? "PASS" : "FAIL");
    bool hr = verify_hash_reversible(); all &= hr;
    std::printf("  [%s] hash 可逆 (33^-1 = 0x3e0f83e1)\n", hr ? "PASS" : "FAIL");
    bool cr = verify_combined_reversible(); all &= cr;
    std::printf("  [%s] 组合可逆 (trit + hash, 100 步)\n", cr ? "PASS" : "FAIL");
    std::printf("  [%s] 总体\n\n", all ? "ALL PASS" : "SOME FAIL");
    return all;
}

// ============================================================================
//  常量 (与 v19 一致)
// ============================================================================

const int D_H = 256;
const int NL_H = 2;
const int Q1_B = 128;
const int Q1_K = 16;
const int Q3_K = 5;
const int HASH_FEATURES = 64;  // [V21-Phase2] 64 features, 8 组 (Phase4 待优化)

// [V21-Phase6] 64 特征映射 (单链): h 的 8 个 4-bit nibble 循环输出
// f_i = ((h >> (4*(i%8))) & 0xF) / 15.0f - 0.5f,  i = 0..63
inline void extract_hash_features(hash_t h, float* features) {
    for (int i = 0; i < HASH_FEATURES; ++i) {
        int nibble_idx = i & 7;             // i % 8
        int shift = 4 * nibble_idx;
        hash_t nib = (h >> shift) & 0xFu;
        features[i] = ((float)nib) / 15.0f - 0.5f;
    }
}

// ============================================================================
//  Q1 (从 v19 保留, hash bucket pool)
// ============================================================================

struct Q1 {
    int B, K, D;
    std::vector<int8_t> trits;
    std::vector<float> adam_m, adam_v;
    int step = 0;
    void init(int B_, int K_, int D_, std::mt19937& rng) {
        B = B_; K = K_; D = D_;
        trits.assign((size_t)B*K*D, 0);
        adam_m.assign((size_t)B*K*D, 0.0f);
        adam_v.assign((size_t)B*K*D, 0.0f);
        std::uniform_int_distribution<int> ud(-1, 1);
        for (auto& t : trits) t = (int8_t)ud(rng);
    }
    static int hash(int id, int B) {
        uint64_t x = (uint32_t)id * 2654435761u;
        x = (x >> 16) ^ x;
        return (int)(x % (uint64_t)B);
    }
    struct Aux { int id, h; std::vector<float> weights; const float* query; };
    void forward(int id, const float* query, float* out, Aux& aux) const {
        aux.id = id; aux.h = hash(id, B); aux.query = query;
        aux.weights.assign(K, 0.0f);
        float mx = -1e9f;
        for (int k = 0; k < K; ++k) {
            float s = 0;
            for (int d = 0; d < D; ++d) s += query[d] * trits[(aux.h*K+k)*D+d];
            aux.weights[k] = s;
            if (s > mx) mx = s;
        }
        for (int k = 0; k < K; ++k) aux.weights[k] = std::exp(aux.weights[k] - mx);
        float sum = 0;
        for (int k = 0; k < K; ++k) sum += aux.weights[k];
        for (int k = 0; k < K; ++k) aux.weights[k] /= sum;
        for (int d = 0; d < D; ++d) {
            float v = 0;
            for (int k = 0; k < K; ++k) v += aux.weights[k] * trits[(aux.h*K+k)*D+d];
            out[d] = v;
        }
    }
    void backward(const Aux& aux, const float* d_out, std::vector<float>& d_trit, float* d_query) const {
        std::vector<float> d_w(K, 0.0f);
        for (int k = 0; k < K; ++k) {
            float s = 0;
            for (int d = 0; d < D; ++d) s += d_out[d] * trits[(aux.h*K+k)*D+d];
            d_w[k] = s;
        }
        float dot = 0;
        for (int k = 0; k < K; ++k) dot += aux.weights[k] * d_w[k];
        std::vector<float> d_s(K);
        for (int k = 0; k < K; ++k) d_s[k] = aux.weights[k] * (d_w[k] - dot);
        for (int k = 0; k < K; ++k) {
            float* g = &d_trit[(aux.h*K+k)*D];
            float ds = d_s[k];
            float wk = aux.weights[k];
            for (int d = 0; d < D; ++d) g[d] += d_out[d] * wk + aux.query[d] * ds;
        }
        for (int d = 0; d < D; ++d) {
            float s = 0;
            for (int k = 0; k < K; ++k) s += trits[(aux.h*K+k)*D+d] * d_s[k];
            d_query[d] += s;
        }
    }
    void adam_update(const std::vector<float>& grad, float lr, float b1, float b2, float eps) {
        step++;
        float bc1 = 1 - std::pow(b1, (float)step), bc2 = 1 - std::pow(b2, (float)step);
        for (size_t i = 0; i < trits.size(); ++i) {
            float g = grad[i]; if (g > 1) g = 1; if (g < -1) g = -1;
            adam_m[i] = b1 * adam_m[i] + (1 - b1) * g;
            adam_v[i] = b2 * adam_v[i] + (1 - b2) * g * g;
            float st = lr * (adam_m[i] / bc1) / (std::sqrt(adam_v[i] / bc2) + eps);
            float nv = (float)trits[i] - st;
            if (nv > 0.5f) trits[i] = 1;
            else if (nv < -0.5f) trits[i] = -1;
            else trits[i] = 0;
        }
    }
};

// ============================================================================
//  词汇表 (从 v19 保留)
// ============================================================================

struct Vocab {
    std::unordered_map<std::string, int> word_to_id;
    std::vector<std::string> id_to_word;
    int pad_id = 0, unk_id = 1;
    void build(const std::string& text, int max_size) {
        std::map<std::string, int> freq;
        std::string cur;
        for (char c : text) {
            if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
                if (!cur.empty()) { freq[cur]++; cur.clear(); }
            } else if (std::ispunct((unsigned char)c)) {
                if (!cur.empty()) { freq[cur]++; cur.clear(); }
                std::string s(1, c); freq[s]++;
            } else cur += c;
        }
        if (!cur.empty()) freq[cur]++;
        std::vector<std::pair<std::string, int>> v(freq.begin(), freq.end());
        std::sort(v.begin(), v.end(), [](auto& a, auto& b){ return a.second > b.second; });
        id_to_word.clear();
        id_to_word.push_back("<pad>"); id_to_word.push_back("<unk>"); id_to_word.push_back("<eos>");
        word_to_id["<pad>"] = 0; word_to_id["<unk>"] = 1; word_to_id["<eos>"] = 2;
        for (auto& p : v) {
            if ((int)id_to_word.size() >= max_size) break;
            int id = (int)id_to_word.size();
            word_to_id[p.first] = id;
            id_to_word.push_back(p.first);
        }
    }
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> tokens;
        std::string cur;
        for (char c : text) {
            if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
                if (!cur.empty()) {
                    auto it = word_to_id.find(cur);
                    tokens.push_back(it == word_to_id.end() ? unk_id : it->second);
                    cur.clear();
                }
            } else if (std::ispunct((unsigned char)c)) {
                if (!cur.empty()) {
                    auto it = word_to_id.find(cur);
                    tokens.push_back(it == word_to_id.end() ? unk_id : it->second);
                    cur.clear();
                }
                std::string s(1, c);
                auto it = word_to_id.find(s);
                tokens.push_back(it == word_to_id.end() ? unk_id : it->second);
            } else cur += std::tolower((unsigned char)c);
        }
        if (!cur.empty()) {
            auto it = word_to_id.find(cur);
            tokens.push_back(it == word_to_id.end() ? unk_id : it->second);
        }
        return tokens;
    }
};

// ============================================================================
//  [V] 模型: 替换 v19 的 h/s 通道为 mod3 + hash
// ============================================================================

struct M {
    Q1 q1;
    // 保留 v19 的 Q3, alpha, gate
    std::vector<float> q3w[Q3_K], aW, ab, gW, gb;
    std::vector<float> q3w_m[Q3_K], q3w_v[Q3_K];
    std::vector<float> aW_m, aW_v, ab_m, ab_v;
    std::vector<float> gW_m, gW_v, gb_m, gb_v;
    // 保留 Wbi
    std::vector<float> Wbi, Wbi_m, Wbi_v;
    // 新增: W (trit→vocab), W_hash (hash 特征→vocab)
    std::vector<float> W, W_m, W_v;                       // V*D
    std::vector<float> W_hash, W_hash_m, W_hash_v;        // V*HASH_FEATURES
    std::vector<float> q1_grad;
    int step = 0;
    int V_unit = 0;
    // [V21-Phase2] SwiGLU 权重
    int HIDDEN_SGL = 0;  // 在 init() 中设为 D + HASH_FEATURES
    std::vector<float> W_sgl_gate, W_sgl_gate_m, W_sgl_gate_v;   // HIDDEN × HIDDEN
    std::vector<float> b_sgl_gate, b_sgl_gate_m, b_sgl_gate_v;   // HIDDEN
    std::vector<float> W_sgl_up, W_sgl_up_m, W_sgl_up_v;         // HIDDEN × HIDDEN
    std::vector<float> b_sgl_up, b_sgl_up_m, b_sgl_up_v;         // HIDDEN
    std::vector<float> W_sgl_out, W_sgl_out_m, W_sgl_out_v;       // V × HIDDEN

    // [V21-Phase6] Raw forward/grad 接口所需的梯度缓冲
    // 父代理调用 yao_backward_raw 后, grad_* 累积 batch 内 BL 上的总梯度 (未除以 BL)
    // 调用方除以 BL 后即均值梯度, 与 yao_backward_raw 文档约定一致。
    std::vector<float> grad_Wg, grad_Wu;            // HIDDEN × HIDDEN
    std::vector<float> grad_Wout;                  // V × HIDDEN
    std::vector<float> grad_Wbi;                   // V × V (按 prev 累计: d_logits[n,v] 计入 Wbi[inp[n-1] or PAD][v])
    std::vector<float> grad_state;                  // BL × HIDDEN (诊断用, 包含 trit 与 hash 通道的反向)

    void init(std::mt19937& rng, int V) {
        const int D = D_H, NL = NL_H;
        V_unit = V;
        q1.init(Q1_B, Q1_K, D, rng);
        q1_grad.assign((size_t)Q1_B*Q1_K*D, 0.0f);
        for (int kk = 0; kk < Q3_K; ++kk) q3w[kk].assign(NL*D, 0);
        aW.assign(NL*D*D, 0); ab.assign(NL*D, 0);
        gW.assign(NL*D*D, 0); gb.assign(NL*D, 0);
        Wbi.assign(V*V, 0);
        W.assign(V*D, 0);
        W_hash.assign(V*HASH_FEATURES, 0);
        for (int kk = 0; kk < Q3_K; ++kk) {
            q3w_m[kk].assign(NL*D, 0); q3w_v[kk].assign(NL*D, 0);
        }
        aW_m.assign(NL*D*D, 0); aW_v.assign(NL*D*D, 0);
        ab_m.assign(NL*D, 0); ab_v.assign(NL*D, 0);
        gW_m.assign(NL*D*D, 0); gW_v.assign(NL*D*D, 0);
        gb_m.assign(NL*D, 0); gb_v.assign(NL*D, 0);
        Wbi_m.assign(V*V, 0); Wbi_v.assign(V*V, 0);
        W_m.assign(V*D, 0); W_v.assign(V*D, 0);
        W_hash_m.assign(V*HASH_FEATURES, 0); W_hash_v.assign(V*HASH_FEATURES, 0);
        // [V21-Phase2] SwiGLU 权重分配
        HIDDEN_SGL = D + HASH_FEATURES;  // 192
        int H = HIDDEN_SGL;
        W_sgl_gate.assign(H*H, 0); W_sgl_gate_m.assign(H*H, 0); W_sgl_gate_v.assign(H*H, 0);
        b_sgl_gate.assign(H, 0); b_sgl_gate_m.assign(H, 0); b_sgl_gate_v.assign(H, 0);
        W_sgl_up.assign(H*H, 0); W_sgl_up_m.assign(H*H, 0); W_sgl_up_v.assign(H*H, 0);
        b_sgl_up.assign(H, 0); b_sgl_up_m.assign(H, 0); b_sgl_up_v.assign(H, 0);
        W_sgl_out.assign(V*H, 0); W_sgl_out_m.assign(V*H, 0); W_sgl_out_v.assign(V*H, 0);

        // [V21-Phase6] grad 缓冲分配
        grad_Wg.assign(H*H, 0);
        grad_Wu.assign(H*H, 0);
        grad_Wout.assign(V*H, 0);
        grad_Wbi.assign(V*V, 0);
        grad_state.assign(0, 0);  // 由 raw forward 按需分配
        std::normal_distribution<float> ndw(0, 0.1f);
        for (auto& x : W) x = ndw(rng);
        for (auto& x : W_hash) x = ndw(rng) * 0.5f;
        // [V21-Q3-Test] Q3 默认设为 identity (kk=0 时=1, 其他=0)
        // 等价于不进行卷积, 直接 pass through
        // 用于测试 Q3 的实际贡献
        for (int l = 0; l < NL; ++l) {
            for (int kk = 0; kk < Q3_K; ++kk) {
                for (int d = 0; d < D; ++d) {
                    q3w[kk][l*D+d] = (kk == 0) ? 1.0f : 0.0f;
                }
            }
        }
        for (int l = 0; l < NL; ++l) {
            for (int r = 0; r < D; ++r) {
                std::vector<float> row(D); float n = 0;
                for (int k = 0; k < D; ++k) { row[k] = ((rng()&1)?1.0f:-1.0f); n += row[k]*row[k]; }
                n = std::sqrt(n);
                for (int k = 0; k < D; ++k) aW[l*D*D+r*D+k] = row[k]/n;
            }
            for (auto& x : ab) x = -1.7f;
            for (int r = 0; r < D; ++r) {
                std::vector<float> row(D); float n = 0;
                for (int k = 0; k < D; ++k) { row[k] = ((rng()&1)?1.0f:-1.0f); n += row[k]*row[k]; }
                n = std::sqrt(n);
                for (int k = 0; k < D; ++k) gW[l*D*D+r*D+k] = row[k]/n;
            }
            for (auto& x : gb) x = -1.7f;
        }
        // [V21-Phase2] SwiGLU 权重初始化
        // 复用策略: W_sgl_out[:, 0..D] = W (旧的 trit->vocab), W_sgl_out[:, D..H] 用 W_hash 填充
        // 这里 HIDDEN_SGL 已在 init 开头定义
        for (int v = 0; v < V; ++v) {
            for (int d = 0; d < D; ++d) W_sgl_out[v*HIDDEN_SGL + d] = W[v*D + d];
            for (int f = 0; f < HASH_FEATURES; ++f) W_sgl_out[v*HIDDEN_SGL + D + f] = W_hash[v*HASH_FEATURES + f];
        }
        // W_sgl_gate, W_sgl_up: 随机初始化
        std::normal_distribution<float> nds(0, 0.05f);
        for (auto& x : W_sgl_gate) x = nds(rng);
        for (auto& x : W_sgl_up) x = nds(rng);
        std::printf("  [V21-Phase2] SwiGLU initialized: HIDDEN=%d\n", HIDDEN_SGL);
    }

    // [V21] Save to binary file
    bool save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) return false;
        int magic = 0x59414F59;  // "YAOY"
        int version = 4;  // v21-Phase2 (SwiGLU)
        int v = V_unit;
        f.write((const char*)&magic, 4);
        f.write((const char*)&version, 4);
        f.write((const char*)&v, 4);
        auto wr = [&](const void* p, size_t n){ f.write((const char*)p, n); };
        // W (trit->vocab)
        wr(W.data(), W.size()*4);
        wr(W_m.data(), W_m.size()*4);
        wr(W_v.data(), W_v.size()*4);
        // W_hash
        wr(W_hash.data(), W_hash.size()*4);
        wr(W_hash_m.data(), W_hash_m.size()*4);
        wr(W_hash_v.data(), W_hash_v.size()*4);
        // Wbi
        wr(Wbi.data(), Wbi.size()*4);
        wr(Wbi_m.data(), Wbi_m.size()*4);
        wr(Wbi_v.data(), Wbi_v.size()*4);
        // Q3 conv
        for (int kk=0; kk<Q3_K; ++kk) {
            wr(q3w[kk].data(), q3w[kk].size()*4);
            wr(q3w_m[kk].data(), q3w_m[kk].size()*4);
            wr(q3w_v[kk].data(), q3w_v[kk].size()*4);
        }
        // alpha
        wr(aW.data(), aW.size()*4);
        wr(aW_m.data(), aW_m.size()*4);
        wr(aW_v.data(), aW_v.size()*4);
        wr(ab.data(), ab.size()*4);
        wr(ab_m.data(), ab_m.size()*4);
        wr(ab_v.data(), ab_v.size()*4);
        // gate
        wr(gW.data(), gW.size()*4);
        wr(gW_m.data(), gW_m.size()*4);
        wr(gW_v.data(), gW_v.size()*4);
        wr(gb.data(), gb.size()*4);
        wr(gb_m.data(), gb_m.size()*4);
        wr(gb_v.data(), gb_v.size()*4);
        // Q1
        wr(q1.trits.data(), q1.trits.size());
        wr(q1.adam_m.data(), q1.adam_m.size()*4);
        wr(q1.adam_v.data(), q1.adam_v.size()*4);
        f.write((const char*)&q1.step, 4);
        f.write((const char*)&step, 4);
        // [V21-Phase2] SwiGLU 权重
        if (HIDDEN_SGL > 0) {
            int H = HIDDEN_SGL;
            wr(W_sgl_gate.data(), W_sgl_gate.size()*4);
            wr(b_sgl_gate.data(), b_sgl_gate.size()*4);
            wr(W_sgl_up.data(), W_sgl_up.size()*4);
            wr(b_sgl_up.data(), b_sgl_up.size()*4);
            wr(W_sgl_out.data(), W_sgl_out.size()*4);
        }
        wr(W_sgl_gate_m.data(), W_sgl_gate_m.size()*4);
        wr(W_sgl_gate_v.data(), W_sgl_gate_v.size()*4);
        wr(W_sgl_up_m.data(), W_sgl_up_m.size()*4);
        wr(W_sgl_up_v.data(), W_sgl_up_v.size()*4);
        wr(W_sgl_out_m.data(), W_sgl_out_m.size()*4);
        wr(W_sgl_out_v.data(), W_sgl_out_v.size()*4);
        wr(b_sgl_gate_m.data(), b_sgl_gate_m.size()*4);
        wr(b_sgl_gate_v.data(), b_sgl_gate_v.size()*4);
        wr(b_sgl_up_m.data(), b_sgl_up_m.size()*4);
        wr(b_sgl_up_v.data(), b_sgl_up_v.size()*4);
        f.flush();
        if (!f) return false;
        std::printf("Saved v21 model to %s (step=%d)\n", path.c_str(), step);
        return true;
    }

    // [V21] Load from binary file
    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        int magic; f.read((char*)&magic, 4);
        if (magic != 0x59414F59) return false;
        int version; f.read((char*)&version, 4);
        if (version != 3 && version != 4) return false;
        int v; f.read((char*)&v, 4);
        if (v != V_unit) return false;
        auto rd = [&](void* p, size_t n){ f.read((char*)p, n); };
        rd(W.data(), W.size()*4);
        rd(W_m.data(), W_m.size()*4);
        rd(W_v.data(), W_v.size()*4);
        rd(W_hash.data(), W_hash.size()*4);
        rd(W_hash_m.data(), W_hash_m.size()*4);
        rd(W_hash_v.data(), W_hash_v.size()*4);
        rd(Wbi.data(), Wbi.size()*4);
        rd(Wbi_m.data(), Wbi_m.size()*4);
        rd(Wbi_v.data(), Wbi_v.size()*4);
        for (int kk=0; kk<Q3_K; ++kk) {
            rd(q3w[kk].data(), q3w[kk].size()*4);
            rd(q3w_m[kk].data(), q3w_m[kk].size()*4);
            rd(q3w_v[kk].data(), q3w_v[kk].size()*4);
        }
        rd(aW.data(), aW.size()*4);
        rd(aW_m.data(), aW_m.size()*4);
        rd(aW_v.data(), aW_v.size()*4);
        rd(ab.data(), ab.size()*4);
        rd(ab_m.data(), ab_m.size()*4);
        rd(ab_v.data(), ab_v.size()*4);
        rd(gW.data(), gW.size()*4);
        rd(gW_m.data(), gW_m.size()*4);
        rd(gW_v.data(), gW_v.size()*4);
        rd(gb.data(), gb.size()*4);
        rd(gb_m.data(), gb_m.size()*4);
        rd(gb_v.data(), gb_v.size()*4);
        rd(q1.trits.data(), q1.trits.size());
        rd(q1.adam_m.data(), q1.adam_m.size()*4);
        rd(q1.adam_v.data(), q1.adam_v.size()*4);
        f.read((char*)&q1.step, 4);
        f.read((char*)&step, 4);
        // [V21-Phase2] 加载 SwiGLU 权重 (version 4+) 或从 v3 升级 (复用 W, W_hash)
        if (version >= 4 && HIDDEN_SGL > 0) {
            int H = HIDDEN_SGL;
            f.read((char*)W_sgl_gate.data(), W_sgl_gate.size()*4);
            f.read((char*)b_sgl_gate.data(), b_sgl_gate.size()*4);
            f.read((char*)W_sgl_up.data(), W_sgl_up.size()*4);
            f.read((char*)b_sgl_up.data(), b_sgl_up.size()*4);
            f.read((char*)W_sgl_out.data(), W_sgl_out.size()*4);
        } else if (version == 3 && HIDDEN_SGL > 0) {
            // 从 v3 升级: 复用 W 到 W_sgl_out 前 D 列, W_hash 到 W_sgl_out 的 hash 部分
            int H = HIDDEN_SGL;
            for (int v = 0; v < V_unit; ++v) {
                for (int d = 0; d < D_H; ++d) W_sgl_out[v*H + d] = W[v*D_H + d];
                for (int f = 0; f < HASH_FEATURES; ++f) W_sgl_out[v*H + D_H + f] = W_hash[v*HASH_FEATURES + f];
            }
            std::printf("  [V21-Phase2] Upgraded from v3: W_sgl_out initialized from old W + W_hash\n");
        }
        if (!f) return false; // reject truncated mandatory parameter data
        const auto payload_end = f.tellg();
        f.seekg(0,std::ios::end); const auto file_end = f.tellg(); f.seekg(payload_end);
        const std::streamoff optimizer_bytes = (4*HIDDEN_SGL*HIDDEN_SGL + 2*V_unit*HIDDEN_SGL + 4*HIDDEN_SGL)*4;
        if (file_end != payload_end) {
            if (file_end-payload_end != optimizer_bytes) return false;
            rd(W_sgl_gate_m.data(), W_sgl_gate_m.size()*4);
            rd(W_sgl_gate_v.data(), W_sgl_gate_v.size()*4);
            rd(W_sgl_up_m.data(), W_sgl_up_m.size()*4);
            rd(W_sgl_up_v.data(), W_sgl_up_v.size()*4);
            rd(W_sgl_out_m.data(), W_sgl_out_m.size()*4);
            rd(W_sgl_out_v.data(), W_sgl_out_v.size()*4);
            rd(b_sgl_gate_m.data(), b_sgl_gate_m.size()*4);
            rd(b_sgl_gate_v.data(), b_sgl_gate_v.size()*4);
            rd(b_sgl_up_m.data(), b_sgl_up_m.size()*4);
            rd(b_sgl_up_v.data(), b_sgl_up_v.size()*4);
            if (!f) return false;
        }
        std::printf("Loaded v21 model from %s (step=%d)\n", path.c_str(), step);
        return true;
    }

    // [V21-Phase4] Warm-start: 重置 W_hash (因为 Phase 4 多链 hash features 与 Phase 2 不兼容)
    // 其他权重保留 (Phase 2 已训练好)
    void warmstart_hash(std::mt19937& rng, float scale = 0.1f, float noise = 0.02f) {
        std::normal_distribution<float> nd(0.0f, noise);
        for (auto& x : W_hash) {
            x = x * scale + nd(rng);  // 缩小 + 小扰动
        }
        // 同时重置 Adam 状态 (W_hash_m, W_hash_v) 避免旧统计污染
        std::fill(W_hash_m.begin(), W_hash_m.end(), 0.0f);
        std::fill(W_hash_v.begin(), W_hash_v.end(), 0.0f);

        // [V21-Q3-Test] 可选: 重置 Q3 为 identity (关闭 Q3 卷积)
        // 用于验证 Q3 的实际贡献
        for (int kk = 0; kk < Q3_K; ++kk) {
            for (int l = 0; l < NL_H; ++l) {
                for (int d = 0; d < D_H; ++d) {
                    q3w[kk][l*D_H + d] = (kk == 0) ? 1.0f : 0.0f;
                }
            }
        }

        std::printf("  [Warm-start] W_hash reset: scale=%.2f, noise=%.3f\n", scale, noise);
        std::printf("  [Warm-start] W_hash Adam state cleared\n");
        std::printf("  [Warm-start] Q3 reset to identity (kk=0=1, others=0)\n");
    }
};

// ============================================================================
//  [V] 验证 mod 3 单步 (运行时)
// ============================================================================
// ============================================================================
//  [V] 验证 mod 3 单步 (运行时)
// ============================================================================

inline trit step_mod3(trit prev, trit x) {
    return (trit)mod3((int)prev + (int)x);
}

// ============================================================================
//  [V] Forward 函数: 替换 v19 的 h/s
// ============================================================================

int yao_forward_raw(M&, const int*, int, int, int, int, float*, float*, float*, float*, float*, float*, hash_t*);

void yao_forward(M& m, const std::vector<int>& inp, int BATCH, int SEQ, int PAD,
             int D, int NL, int V_unit,
             std::vector<float>& x, std::vector<float>& x_prev,
             std::vector<float>& y, std::vector<float>& alpha,
             std::vector<trit>& h_trit, std::vector<hash_t>& h_hash,
             std::vector<float>& trit_features, std::vector<float>& hash_features,
             std::vector<float>& logits, std::vector<float>& probs,
             std::vector<float>& xs, std::vector<float>& ys, std::vector<float>& alphas,
             std::vector<float>& gates_v) {
    const int BL = BATCH * SEQ, H = D + HASH_FEATURES;
    std::vector<float> state(BL*H), hidden(BL*H), gp(BL*H), up(BL*H);
    if (yao_forward_raw(m, inp.data(), BL, BATCH, SEQ, PAD, logits.data(), probs.data(),
        state.data(),hidden.data(),gp.data(),up.data(),h_hash.data()) != 0)
        throw std::runtime_error("Invalid forward inputs");
    for (int n=0;n<BL;++n) {
        for (int d=0;d<D;++d) trit_features[n*D+d]=state[n*H+d];
        for (int f=0;f<HASH_FEATURES;++f) hash_features[n*HASH_FEATURES+f]=state[n*H+D+f];
    }
}

// ============================================================================
//  [V21-Phase6] Raw Forward 接口 (供父代理门禁测试使用)
//  公式: logits = Wout·hidden + Wbi[prev,:], hidden = silu(Wg·state) ⊙ (Wu·state)
//         state = [trit; hash]  (HIDDEN = D + HASH_FEATURES, 无 bias)
//  此接口: 不打印/不dump/不更新参数; 只算并保存所需激活以便 backward 使用。
//
//  调用约定:
//    inp        : [BL] int tokens, BL = BATCH * seq_len
//    PAD        : previous-token pad id (用于 t=0)
//    out_logits : [BL * V] float (已写)
//    out_probs  : [BL * V] float (已写; softmax over V)
//    out_state  : [BL * HIDDEN] float (已写)  -- 用于 dWg/dWu 的反向
//    out_hidden : [BL * HIDDEN] float (已写)  -- post silu_mul, 用于 dWout
//    out_gate_pre, out_up_pre : [BL * HIDDEN] float (已写) -- silu/gate 之前, 用于 silu_back
//    scratch_h_hash : [BATCH] hash_t (读写; 保存 batch 内每个序列累积的 hash 末态)
//    m         : 模型 (只读)
//
//  成功返回 0; inp 含非法 token (>= V_unit 或 < 0) 返回 -1。
// ============================================================================
int yao_forward_raw(M& m,
                    const int* inp, int BL, int BATCH, int seq_len, int PAD,
                    float* out_logits,    // [BL * V]
                    float* out_probs,     // [BL * V]
                    float* out_state,     // [BL * HIDDEN] (D+HASH_FEATURES)
                    float* out_hidden,    // [BL * HIDDEN]
                    float* out_gate_pre,  // [BL * HIDDEN]
                    float* out_up_pre,    // [BL * HIDDEN]
                    hash_t* scratch_h_hash // [BATCH]
                   ) {
    const int D = D_H;
    const int HASH = HASH_FEATURES;
    const int HIDDEN = D + HASH;
    const int V_unit = m.V_unit;

    // 0) token 范围检查
    for (int n = 0; n < BL; ++n) {
        int id = inp[n];
        if (id < 0 || id >= V_unit) return -1;
    }

    // 1) per-position hash 单链: h_{b,t} = h_{b,t-1} * 33 + inp[bt*SEQ + t] + 7
    //    同时 extract 64 维特征 (单链 8 个 4-bit nibble 循环输出)
    for (int b = 0; b < BATCH; ++b) {
        scratch_h_hash[b] = 5381u;
        for (int t = 0; t < seq_len; ++t) {
            int bt = b * seq_len + t;
            int id = inp[bt];
            hash_t prev_h = (t > 0) ? scratch_h_hash[b] : 5381u;
            scratch_h_hash[b] = prev_h * 33u + (hash_t)id;

            // trit 累积 (mod 3) - 与 v21 forward 一致
            int bucket = Q1::hash(id, Q1_B);
            for (int d = 0; d < D; ++d) {
                trit prev_trit = (t > 0) ? (trit)out_state[(bt - 1) * HIDDEN + d] : (trit)0;
                trit embed_trit = m.q1.trits[(bucket * Q1_K + 0) * D + d];
                out_state[bt * HIDDEN + d] = (float)step_mod3(prev_trit, embed_trit);
            }
            // hash 特征 (单链, 64 dim)
            extract_hash_features(scratch_h_hash[b], out_state + bt * HIDDEN + D);
        }
    }

    // 2) SwiGLU: gate = Wg·state (无 bias), up = Wu·state (无 bias), hidden = silu(gate) ⊙ up
    for (int n = 0; n < BL; ++n) {
        const float* state = out_state + n * HIDDEN;
        float* gp = out_gate_pre + n * HIDDEN;
        float* up = out_up_pre + n * HIDDEN;
        for (int i = 0; i < HIDDEN; ++i) {
            float gv = 0, uv = 0;
            const float* row_g = m.W_sgl_gate.data() + i * HIDDEN;
            const float* row_u = m.W_sgl_up.data()   + i * HIDDEN;
            for (int k = 0; k < HIDDEN; ++k) {
                float s = state[k];
                gv += row_g[k] * s;
                uv += row_u[k] * s;
            }
            gp[i] = gv;
            up[i] = uv;
        }
        float* hd = out_hidden + n * HIDDEN;
        for (int i = 0; i < HIDDEN; ++i) {
            float gz = gp[i];
            float sig = 1.0f / (1.0f + std::exp(-gz));
            hd[i] = gz * sig * up[i];
        }
    }

    // 3) logits = Wbi[prev, v] + Wout[v, :] · hidden
    for (int n = 0; n < BL; ++n) {
        int t = n % seq_len;
        int prev = (t > 0) ? inp[n - 1] : PAD;
        const float* hd = out_hidden + n * HIDDEN;
        for (int v = 0; v < V_unit; ++v) {
            float lv = m.Wbi[prev * V_unit + v];
            const float* row = m.W_sgl_out.data() + v * HIDDEN;
            for (int h = 0; h < HIDDEN; ++h) lv += row[h] * hd[h];
            out_logits[n * V_unit + v] = lv;
        }
    }

    // 4) softmax -> probs
    for (int n = 0; n < BL; ++n) {
        float mx = out_logits[n * V_unit];
        for (int v = 1; v < V_unit; ++v) if (out_logits[n*V_unit+v] > mx) mx = out_logits[n*V_unit+v];
        float sum = 0;
        for (int v = 0; v < V_unit; ++v) {
            float p = std::exp(out_logits[n*V_unit+v] - mx);
            out_probs[n*V_unit+v] = p; sum += p;
        }
        for (int v = 0; v < V_unit; ++v) out_probs[n*V_unit+v] /= sum;
    }
    return 0;
}

// ============================================================================
//  [V21-Phase6] Raw Backward 接口 (供父代理门禁测试使用)
//  输入: d_logits[BL, V] = probs - 1[target]  (调用方先准备, 或从 logits+targets 算)
//        state[BL, HIDDEN], hidden[BL, HIDDEN], gate_pre[BL, HIDDEN], up_pre[BL, HIDDEN]
//  输出: 累积到 m.grad_Wg, grad_Wu, grad_Wout, grad_Wbi (均为 SUM, 不除 BL)
//        调用方除以 BL 即得均值梯度, 与 Adam 接口约定一致。
//
//  返回 0 成功。inp 含非法 token 返回 -1 (不会写入 m.grad_*)。
//
//  注: trit 与 hash 参数(q1, hash) 不在公式可学习列表, 不累积梯度到它们;
//       grad_state 仍写到 m.grad_state[BL*HIDDEN] 供诊断 (q1 等聚合路径不闭合)。
// ============================================================================
int yao_backward_raw(M& m,
                     const int* inp, int BL, int BATCH, int seq_len, int PAD,
                     const float* d_logits,        // [BL, V]
                     const float* state,        // [BL, HIDDEN]
                     const float* hidden,       // [BL, HIDDEN]  post silu_mul
                     const float* gate_pre,     // [BL, HIDDEN]  pre silu_mul (= Wg·state)
                     const float* up_pre) {     // [BL, HIDDEN]  pre silu_mul (= Wu·state)
    const int D = D_H;
    const int HASH = HASH_FEATURES;
    const int HIDDEN = D + HASH;
    const int V_unit = m.V_unit;

    for (int n = 0; n < BL; ++n) {
        int id = inp[n];
        if (id < 0 || id >= V_unit) return -1;
    }
    // 清零累计 grad
    std::fill(m.grad_Wg.begin(),   m.grad_Wg.end(),   0.0f);
    std::fill(m.grad_Wu.begin(),   m.grad_Wu.end(),   0.0f);
    std::fill(m.grad_Wout.begin(), m.grad_Wout.end(), 0.0f);
    std::fill(m.grad_Wbi.begin(),  m.grad_Wbi.end(),  0.0f);
    m.grad_state.assign((size_t)BL * HIDDEN, 0.0f);

    // 1) d_hidden[n, h] = sum_v d_logits[n, v] * Wout[v, h]
    std::vector<float> d_hidden(BL * HIDDEN, 0.0f);
    for (int n = 0; n < BL; ++n) {
        for (int h = 0; h < HIDDEN; ++h) {
            float s = 0;
            for (int v = 0; v < V_unit; ++v) s += d_logits[n * V_unit + v] * m.W_sgl_out[v * HIDDEN + h];
            d_hidden[n * HIDDEN + h] = s;
        }
    }

    // 2) silu_back: d_gate_pre = d_hidden ⊙ up · silu'(g); d_up_pre = d_hidden ⊙ silu(g)
    std::vector<float> d_gate_pre(BL * HIDDEN, 0.0f), d_up_pre(BL * HIDDEN, 0.0f);
    for (int n = 0; n < BL; ++n) {
        for (int i = 0; i < HIDDEN; ++i) {
            float dh = d_hidden[n * HIDDEN + i];
            float g  = gate_pre[n * HIDDEN + i];
            float u  = up_pre[n * HIDDEN + i];
            float sig = 1.0f / (1.0f + std::exp(-g));
            float dsilg = sig * (1.0f + g * (1.0f - sig));  // 数值稳定 silu'(g)
            d_gate_pre[n * HIDDEN + i] = dh * u * dsilg;
            d_up_pre[n * HIDDEN + i]   = dh * (g * sig);
        }
    }

    // 3) grad_Wg[i, j] += sum_n d_gate_pre[n, i] * state[n, j]
    //    grad_Wu[i, j] += sum_n d_up_pre[n, i] * state[n, j]
    //    grad_Wout[v, h] += sum_n d_logits[n, v] * hidden[n, h]
    for (int i = 0; i < HIDDEN; ++i) {
        for (int j = 0; j < HIDDEN; ++j) {
            float sg = 0, su = 0;
            for (int n = 0; n < BL; ++n) {
                sg += d_gate_pre[n * HIDDEN + i] * state[n * HIDDEN + j];
                su += d_up_pre[n * HIDDEN + i] * state[n * HIDDEN + j];
            }
            m.grad_Wg[i * HIDDEN + j] = sg;
            m.grad_Wu[i * HIDDEN + j] = su;
        }
    }
    for (int v = 0; v < V_unit; ++v) {
        for (int h = 0; h < HIDDEN; ++h) {
            float s = 0;
            for (int n = 0; n < BL; ++n) s += d_logits[n * V_unit + v] * hidden[n * HIDDEN + h];
            m.grad_Wout[v * HIDDEN + h] = s;
        }
    }

    // 4) grad_Wbi[prev, v] += sum_n d_logits[n, v] where prev = (t>0 ? inp[n-1] : PAD)
    for (int n = 0; n < BL; ++n) {
        int t = n % seq_len;
        int prev = (t > 0) ? inp[n - 1] : PAD;
        for (int v = 0; v < V_unit; ++v) {
            m.grad_Wbi[prev * V_unit + v] += d_logits[n * V_unit + v];
        }
    }

    // 5) grad_state[n, j] = sum_i (d_gate_pre[n, i] * Wg[i, j] + d_up_pre[n, i] * Wu[i, j])
    for (int n = 0; n < BL; ++n) {
        for (int j = 0; j < HIDDEN; ++j) {
            float s = 0;
            for (int i = 0; i < HIDDEN; ++i) {
                s += d_gate_pre[n * HIDDEN + i] * m.W_sgl_gate[i * HIDDEN + j];
                s += d_up_pre[n * HIDDEN + i] * m.W_sgl_up[i * HIDDEN + j];
            }
            m.grad_state[n * HIDDEN + j] = s;
        }
    }

    return 0;
}

// ============================================================================
//  [V21-Phase6] 工具: 由 logits+targets 计算 d_logits = probs - 1[target]
//  probs 已是 softmax 输出, 复制 d_logits 缓冲。
// ============================================================================
inline void compute_d_logits(const float* probs, const int* targets, int BL, int V_unit, float* d_logits) {
    for (int n = 0; n < BL; ++n) {
        for (int v = 0; v < V_unit; ++v) d_logits[n * V_unit + v] = probs[n * V_unit + v];
        int t = targets[n];
        if (t >= 0 && t < V_unit) d_logits[n * V_unit + t] -= 1.0f;
    }
}

// ============================================================================
//  主程序
// ============================================================================

void apply_model_adam(M& m,float lr,float b1,float b2,float bc1,float bc2,float eps) {
    auto apply=[&](std::vector<float>& w,std::vector<float>& mom,std::vector<float>& var,const std::vector<float>& grad) {
        for(size_t i=0;i<w.size();++i) {
            if(!std::isfinite(grad[i])) throw std::runtime_error("Nonfinite gradient");
            float g=std::min(1.f,std::max(-1.f,grad[i]));
            mom[i]=b1*mom[i]+(1-b1)*g; var[i]=b2*var[i]+(1-b2)*g*g;
            w[i]=std::min(2.f,std::max(-2.f,w[i]-lr*(mom[i]/bc1)/(std::sqrt(var[i]/bc2)+eps)));
        }
    };
    apply(m.W_sgl_gate,m.W_sgl_gate_m,m.W_sgl_gate_v,m.grad_Wg);
    apply(m.W_sgl_up,m.W_sgl_up_m,m.W_sgl_up_v,m.grad_Wu);
    apply(m.W_sgl_out,m.W_sgl_out_m,m.W_sgl_out_v,m.grad_Wout);
    apply(m.Wbi,m.Wbi_m,m.Wbi_v,m.grad_Wbi);
}

int main(int argc, char** argv) {
    std::printf("========================================\n");
    std::printf("  Yaoyao v21: Reversible Chain (Full)\n");
    std::printf("========================================\n");

    if (!run_math_verification()) { std::fprintf(stderr, "Math FAIL\n"); return 1; }

    const char* path = (argc > 1) ? argv[1] : "D:\\TaoVm\\tinystories_train.txt";
    int vocab_size = (argc > 2) ? atoi(argv[2]) : 1024;
    int n_win = (argc > 3) ? atoi(argv[3]) : 500;
    int epochs = (argc > 4) ? atoi(argv[4]) : 2;
    float lr = (argc > 5) ? (float)atof(argv[5]) : 0.005f;

    std::printf("[Step 1] Loading %s ...\n", path);
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "Cannot open %s\n", path); return 1; }
    std::stringstream ss; ss << f.rdbuf();
    std::string text = ss.str();
    f.close();
    std::printf("  Loaded %zu chars\n", text.size());

    std::printf("[Step 2] Building vocab (max %d) ...\n", vocab_size);
    Vocab vocab;
    vocab.build(text.substr(0, std::min<size_t>(text.size(), 5000000)), vocab_size);
    int V_unit = (int)vocab.id_to_word.size();
    std::printf("  Vocab=%d\n", V_unit);

    std::printf("[Step 3] Encoding text ...\n");
    std::vector<int> tokens = vocab.encode(text);
    std::printf("  Encoded %zu tokens\n", tokens.size());

    std::printf("[Step 4] Initializing model ...\n");
    std::mt19937 rng(42);
    M m; m.init(rng, V_unit);

    // [V21] 增量训练: 如果 .bin 存在则 load
    const char* bin_path = (argc > 6) ? argv[6] : "D:\\TaoVm\\yaoyao_v21.bin";
    int do_warmstart = (argc > 7) ? atoi(argv[7]) : 0;
    // [V21-Phase5] argv[8] = "dump_logits" 模式: 跑一次 forward 并 dump
    bool dump_logits_mode = (argc > 8 && std::string(argv[8]) == "dump");
    bool dump_tokens_mode = (argc > 8 && std::string(argv[8]) == "dump_tokens");

    // [V21-Phase5] dump_tokens 模式: 写出 tokens 到 bin
    if (dump_tokens_mode) {
        const char* tpath = "yaoyao_v21_tokens.bin";
        FILE* ft2 = std::fopen(tpath, "wb");
        int ntok = (int)tokens.size();
        std::fwrite(&ntok, 4, 1, ft2);
        std::fwrite(tokens.data(), 4, ntok, ft2);
        std::fclose(ft2);
        std::printf("[Dump-Tokens] %d tokens -> %s\n", ntok, tpath);
        return 0;
    }

    bool loaded = false;
    if (argc > 6 || (argc > 1 && std::ifstream(bin_path).good())) {
        loaded = m.load(bin_path);
        if (loaded) {
            std::printf("  [V21] Resumed from existing model (step=%d)\n", m.step);
            // [V21-Phase4] Warm-start: 加载后扰动 W_hash (用于 Phase 4 多链 hash)
            if (do_warmstart) {
                m.warmstart_hash(rng);
                m.save(bin_path);  // 保存扰动后的模型
                std::printf("  [V21] Warm-start applied and saved\n");
            }
        }
    }
    if (!loaded) {
        std::printf("  [V21] Starting fresh model\n");
    }
    const int D = D_H, NL = NL_H;
    int SEQ = 64, BATCH = 16, BL = SEQ*BATCH;
    int PAD = vocab.pad_id;
    std::printf("  Config: V=%d D=%d NL=%d BATCH=%d SEQ=%d N_WIN=%d EPOCHS=%d LR=%.5f\n",
                V_unit, D, NL, BATCH, SEQ, n_win, epochs, lr);

    std::vector<float> x(BL*D), x_prev(D), y(BL*D), alpha(BL*D);
    std::vector<float> xs(NL*BL*D), ys(NL*BL*D), alphas(NL*BL*D), gates_v(NL*BL*D);
    std::vector<trit> h_trit(BL*D);
    std::vector<hash_t> h_hash(BATCH);
    // [V21-Phase6] 单链 hash 即可, 不再保留 h_hash1/2/3
    std::vector<float> trit_features(BL*D), hash_features(BL*HASH_FEATURES);
    std::vector<float> logits(BL*V_unit), probs(BL*V_unit), d_logits(BL*V_unit);

    float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    auto t0 = std::chrono::steady_clock::now();

    std::vector<int> all_in(n_win*BATCH*SEQ), all_tg(n_win*BATCH*SEQ);
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::uniform_int_distribution<int> udist(0, (int)tokens.size() - n_win*SEQ - SEQ - 1);
        int offset = udist(rng);
        for (int i = 0; i < n_win; ++i) for (int b = 0; b < BATCH; ++b) {
            int start = offset + i*SEQ + b*7;
            if (start + SEQ + 1 >= (int)tokens.size()) start = offset + i*SEQ;
            for (int j = 0; j < SEQ; ++j) {
                all_in[(i*BATCH+b)*SEQ+j] = tokens[start+j];
                all_tg[(i*BATCH+b)*SEQ+j] = tokens[start+j+1];
            }
        }
        float total = 0; int nb = 0;
        for (int w = 0; w < n_win; ++w) {
            std::vector<int> inpBL(BL), tgtBL(BL);
            for (int n = 0; n < BL; ++n) { inpBL[n] = all_in[w*BL+n]; tgtBL[n] = all_tg[w*BL+n]; }

            yao_forward(m, inpBL, BATCH, SEQ, PAD, D, NL, V_unit,
                    x, x_prev, y, alpha, h_trit, h_hash,
                    trit_features, hash_features, logits, probs,
                    xs, ys, alphas, gates_v);

            // [V21-Phase5] Dump logits (用于 GPU 验证)
            if (dump_logits_mode) {
                const char* dump_path = "yaoyao_v21_dump.bin";
                FILE* fd = std::fopen(dump_path, "wb");
                if (fd) {
                    int magic = 0x594F5954;  // "TOYZ" (反向 YAOY)
                    int n_inp = BL;
                    std::fwrite(&magic, 4, 1, fd);
                    std::fwrite(&n_inp, 4, 1, fd);
                    std::fwrite(inpBL.data(), 4, BL, fd);
                    std::fwrite(logits.data(), 4, BL*V_unit, fd);
                    std::fwrite(trit_features.data(), 4, BL*D, fd);
                    std::fwrite(hash_features.data(), 4, BL*HASH_FEATURES, fd);
                    std::fwrite(h_hash.data(), 4, BATCH, fd);    // [V21-Phase6] dump 单链 h_hash (BATCH)
                    std::fclose(fd);
                    std::printf("  [V21-Phase5] Dumped logits to %s (BL=%d, V=%d)\n", dump_path, BL, V_unit);
                }
                // 算 NLL loss
                float total_loss = 0;
                for (int n = 0; n < BL; ++n) {
                    float p = std::max(probs[n*V_unit + tgtBL[n]], 1e-9f);
                    total_loss += -std::log(p);
                }
                std::printf("  [V21-Phase5] CPU Loss = %.4f\n", total_loss / BL);
                std::printf("  [V21-Phase5] logits[0..10] = ");
                for (int v = 0; v < 10; ++v) std::printf("%.3f ", logits[0*V_unit+v]);
                std::printf("\n");
                std::printf("\n[V21-Phase5] Dump mode complete.\n");
                m.save(bin_path);
                return 0;
            }

            // Correctness gates live in verify_v21_cpu.cpp and independent Python tests.
            // Do not mutate forward state or print unconditional reversibility success here.

            // NLL loss
            float loss_nll = 0;
            for (int n = 0; n < BL; ++n) {
                int t = tgtBL[n];
                float p = std::max(probs[n*V_unit+t], 1e-9f);
                loss_nll += -std::log(p);
            }
            loss_nll /= BL;
            total += loss_nll; nb++; m.step++;

            // 梯度
            for (int n = 0; n < BL; ++n) {
                for (int v = 0; v < V_unit; ++v) d_logits[n*V_unit+v] = probs[n*V_unit+v];
                d_logits[n*V_unit+tgtBL[n]] -= 1.0f;
            }

            float bc1 = 1 - std::pow(b1, (float)m.step), bc2 = 1 - std::pow(b2, (float)m.step);

            // Recompute activations through the SAME forward used for inference, before updates.
            const int H = D + HASH_FEATURES;
            std::vector<float> state(BL*H), hidden(BL*H), gp(BL*H), up(BL*H);
            if (yao_forward_raw(m,inpBL.data(),BL,BATCH,SEQ,PAD,logits.data(),probs.data(),
                state.data(),hidden.data(),gp.data(),up.data(),h_hash.data()) != 0)
                throw std::runtime_error("Invalid training input");
            compute_d_logits(probs.data(),tgtBL.data(),BL,V_unit,d_logits.data());
            for (float& value : d_logits) value /= BL;
            if (yao_backward_raw(m,inpBL.data(),BL,BATCH,SEQ,PAD,d_logits.data(),
                state.data(),hidden.data(),gp.data(),up.data()) != 0)
                throw std::runtime_error("Invalid backward input");
            apply_model_adam(m,lr,b1,b2,bc1,bc2,eps);
            // Legacy W/W_hash and gate/up biases are never updated.

            if ((w+1) % 50 == 0 || w == 0) {
                float el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::printf("  ep%d win%d/%d loss=%.4f avg=%.4f (%.1fs)\n",
                            epoch+1, w+1, n_win, loss_nll, total/nb, el);
            }
        }
        std::printf("Epoch %2d/%d avg_loss=%.4f\n", epoch+1, epochs, total/nb);
    }

    // [V] 可逆性最终验证
    std::printf("\n[V] 最终可逆性验证:\n");
    {
        trit test_trit[D];
        hash_t test_hash = 5381;
        for (int d = 0; d < D; ++d) test_trit[d] = 0;
        trit test_trit_orig[D];
        memcpy(test_trit_orig, test_trit, sizeof(test_trit));
        hash_t test_hash_orig = test_hash;
        int seq[] = {3, 7, 15, 31, 63, 127, 255, 511};
        for (int i = 0; i < 8; ++i) {
            for (int d = 0; d < D; ++d) test_trit[d] = step_mod3(test_trit[d], (trit)((seq[i]>>d) & 1 ? 1 : -1));
            test_hash = hash_forward(test_hash, seq[i]);
        }
        for (int i = 7; i >= 0; --i) {
            test_hash = hash_reverse(test_hash, seq[i]);
            for (int d = 0; d < D; ++d) test_trit[d] = mod3(test_trit[d] - ((seq[i]>>d) & 1 ? 1 : -1));
        }
        bool trit_ok = true;
        for (int d = 0; d < D; ++d) if (test_trit[d] != test_trit_orig[d]) trit_ok = false;
        std::printf("  [%s] trit 完全恢复 (8 步)\n", trit_ok ? "PASS" : "FAIL");
        std::printf("  [%s] hash 完全恢复 (8 步)\n", test_hash == test_hash_orig ? "PASS" : "FAIL");
    }


    // ============================================================
    // [G] Generation Phase (从 v19 借鉴的模式)
    // ============================================================
    std::printf("\n========================================\n");
    std::printf("  [G] Generation Phase\n");
    std::printf("========================================\n\n");

    std::vector<std::string> prompts = {
        "Once upon a time",
        "The little girl",
        "He was very",
        "Lily and",
        "Today was",
        "The cat",
        "Mom said",
        "They went"
    };

    for (const auto& prompt : prompts) {
        std::vector<int> prompt_ids = vocab.encode(prompt);
        std::printf("Prompt: '%s'\n", prompt.c_str());

        std::vector<int> ids = prompt_ids;
        std::string generated = prompt;

        for (int step = 0; step < 40; ++step) {
            int L = (int)ids.size();
            std::vector<int> in2(SEQ);
            for (int i = 0; i < SEQ; ++i) {
                int idx = L - SEQ + i;
                in2[i] = (idx < 0) ? PAD : ids[idx];
            }
            std::vector<int> inBL(BL);
            for (int b = 0; b < BATCH; ++b) for (int t = 0; t < SEQ; ++t) inBL[b*SEQ+t] = in2[t];

            yao_forward(m, inBL, BATCH, SEQ, PAD, D, NL, V_unit,
                    x, x_prev, y, alpha, h_trit, h_hash,
                    trit_features, hash_features, logits, probs,
                    xs, ys, alphas, gates_v);

            int bt = (BATCH-1)*SEQ + (SEQ-1);
            float T = 0.9f;
            std::vector<float> adj_logit(V_unit);
            for (int v = 0; v < V_unit; ++v) adj_logit[v] = logits[bt*V_unit+v] / T;

            // Repetition penalty
            for (size_t back = 0; back < ids.size() && back < 6; ++back) {
                int tk = ids[ids.size()-1-back];
                if (tk >= 2 && tk < V_unit) {
                    float penalty = 3.0f * std::pow(0.65f, (float)back);
                    adj_logit[tk] -= penalty;
                }
            }

            // Top-p sampling
            std::vector<int> idx_sort(V_unit);
            std::iota(idx_sort.begin(), idx_sort.end(), 0);
            std::sort(idx_sort.begin(), idx_sort.end(),
                [&](int a, int b){ return adj_logit[a] > adj_logit[b]; });

            float mx = adj_logit[idx_sort[0]];
            std::vector<float> probs_gen(V_unit);
            float sum = 0;
            for (int v = 0; v < V_unit; ++v) {
                probs_gen[v] = std::exp(adj_logit[v] - mx);
                sum += probs_gen[v];
            }
            for (int v = 0; v < V_unit; ++v) probs_gen[v] /= sum;

            float p = 0.9f, cum = 0;
            int nuc_size = V_unit;
            for (int i = 0; i < V_unit; ++i) {
                cum += probs_gen[idx_sort[i]];
                if (cum >= p) { nuc_size = i+1; break; }
            }

            // 随机选择 nucleus 内的 token
            float r = (float)rand() / (float)RAND_MAX;
            float cumsum = 0;
            int next_token = idx_sort[nuc_size-1];
            for (int i = 0; i < nuc_size; ++i) {
                cumsum += probs_gen[idx_sort[i]];
                if (cumsum >= r) {
                    next_token = idx_sort[i];
                    break;
                }
            }

            ids.push_back(next_token);
            if (next_token >= 2 && next_token < V_unit) {
                std::string w = vocab.id_to_word[next_token];
                bool is_punct = (w.size() == 1) && std::ispunct((unsigned char)w[0]);
                if (!generated.empty() && !is_punct && generated.back() != '\n') generated += " ";
                generated += w;
            }
        }
        std::printf("  Output: '%s'\n\n", generated.c_str());
    }

    // [V21] Save model for incremental training
    m.save(bin_path);

    std::printf("\n[Done]\n");
    return 0;
}
