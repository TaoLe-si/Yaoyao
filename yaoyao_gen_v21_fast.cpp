// yaoyao_gen_v21_fast.cpp
// 夭夭 v21 推理 - 优化版 (增量状态更新, 不重算历史)
// 用法: yaoyao_gen_v21_fast.exe <model.bin> <vocab_text> <prompt> [max_tokens]

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

typedef int8_t trit;
typedef uint32_t hash_t;

const int D_H = 128;
const int NL_H = 2;
const int Q1_B = 128;
const int Q1_K = 16;
const int Q3_K = 5;
const int HASH_FEATURES = 16;
const int MAX_CTX = 64;  // 最大上下文长度

inline trit mod3(int x) {
    int r = x % 3;
    if (r > 1) r -= 3;
    if (r < -1) r += 3;
    return (trit)r;
}

inline hash_t hash_forward(hash_t h, int token) {
    return ((h * 33u) + (hash_t)token + 7u);
}

inline void extract_hash_features(hash_t h, float* features) {
    hash_t temp = h;
    for (int i = 0; i < HASH_FEATURES; i++) {
        features[i] = ((float)(temp & 0xFFFFu) / 65535.0f - 0.5f);
        temp = temp * 0x9e3779b1u + 0x1u;
    }
}

// ============================================================================
//  Q1
// ============================================================================
struct Q1 {
    int B, K, D;
    int step = 0;
    std::vector<int8_t> trits;
    std::vector<float> adam_m, adam_v;
    void init(int B_, int K_, int D_) {
        B = B_; K = K_; D = D_;
        trits.assign((size_t)B*K*D, 0);
        adam_m.assign((size_t)B*K*D, 0.0f);
        adam_v.assign((size_t)B*K*D, 0.0f);
    }
    static int hash(int id, int B) {
        uint64_t x = (uint32_t)id * 2654435761u;
        x = (x >> 16) ^ x;
        return (int)(x % (uint64_t)B);
    }
    void forward(int id, const float* query, float* out) const {
        int h = hash(id, B);
        std::vector<float> w(K);
        float mx = -1e9f;
        for (int k = 0; k < K; ++k) {
            float s = 0;
            for (int d = 0; d < D; ++d) s += query[d] * trits[(h*K+k)*D+d];
            w[k] = s;
            if (s > mx) mx = s;
        }
        for (int k = 0; k < K; ++k) w[k] = std::exp(w[k] - mx);
        float sum = 0;
        for (int k = 0; k < K; ++k) sum += w[k];
        for (int k = 0; k < K; ++k) w[k] /= sum;
        for (int d = 0; d < D; ++d) {
            float v = 0;
            for (int k = 0; k < K; ++k) v += w[k] * trits[(h*K+k)*D+d];
            out[d] = v;
        }
    }
};

// ============================================================================
//  Vocab
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
    std::string decode(const std::vector<int>& ids) const {
        std::string s;
        for (int id : ids) {
            if (id >= 2 && id < (int)id_to_word.size()) {
                if (!s.empty()) s += " ";
                s += id_to_word[id];
            }
        }
        return s;
    }
};

// ============================================================================
//  Model (load only)
// ============================================================================
struct M {
    Q1 q1;
    std::vector<float> q3w[Q3_K], aW, ab, gW, gb;
    std::vector<float> q3w_m[Q3_K], aW_m, ab_m, gW_m, gb_m;
    std::vector<float> q3w_v[Q3_K], aW_v, ab_v, gW_v, gb_v;
    std::vector<float> Wbi, Wbi_m, Wbi_v;
    std::vector<float> W, W_m, W_v;
    std::vector<float> W_hash, W_hash_m, W_hash_v;
    int step = 0;
    int V_unit = 0;

    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        int magic; f.read((char*)&magic, 4);
        if (magic != 0x59414F59) return false;
        int version; f.read((char*)&version, 4);
        if (version != 3) return false;
        int v; f.read((char*)&v, 4);
        V_unit = v;
        int V = V_unit;
        const int D = D_H, NL = NL_H;
        q1.init(Q1_B, Q1_K, D);
        for (int kk = 0; kk < Q3_K; ++kk) { q3w[kk].assign(NL*D, 0); q3w_m[kk].assign(NL*D, 0); q3w_v[kk].assign(NL*D, 0); }
        aW.assign(NL*D*D, 0); ab.assign(NL*D, 0); aW_m.assign(NL*D*D, 0); aW_v.assign(NL*D*D, 0);
        ab_m.assign(NL*D, 0); ab_v.assign(NL*D, 0);
        gW.assign(NL*D*D, 0); gb.assign(NL*D, 0); gW_m.assign(NL*D*D, 0); gW_v.assign(NL*D*D, 0);
        gb_m.assign(NL*D, 0); gb_v.assign(NL*D, 0);
        Wbi.assign(V*V, 0); Wbi_m.assign(V*V, 0); Wbi_v.assign(V*V, 0);
        W.assign(V*D, 0); W_m.assign(V*D, 0); W_v.assign(V*D, 0);
        W_hash.assign(V*HASH_FEATURES, 0); W_hash_m.assign(V*HASH_FEATURES, 0); W_hash_v.assign(V*HASH_FEATURES, 0);

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
        std::printf("[GEN] Loaded v21 model: V=%d step=%d\n", V_unit, step);
        return true;
    }
};

// ============================================================================
//  Incremental state for inference
// ============================================================================
struct InferState {
    // 状态增量
    std::vector<trit> h_trit;       // [D] 当前 trit 状态
    hash_t h_hash;                  // 当前 hash 状态
    
    // Q3 conv 需要的历史 x (Q1 输出)
    std::vector<std::vector<float>> x_history;  // [MAX_CTX][D] 最近 K 个 x
    
    // Q1 嵌入 (zero query)
    std::vector<float> x_zero;      // [D] 全 0
    std::vector<float> x_cur;       // [D] 当前 x (t 时刻)
    
    int pos;  // 当前在生成中的位置
    int V_unit;
    int D;
    
    InferState(int V, int D_) : V_unit(V), D(D_) {
        h_trit.assign(D, 0);
        h_hash = 5381u;
        x_history.assign(MAX_CTX, std::vector<float>(D, 0));
        x_zero.assign(D, 0);
        x_cur.assign(D, 0);
        pos = 0;
    }
    
    // 初始化 (用 prompt 的最后 SEQ 个 token)
    void seed(M& m, const std::vector<int>& tokens) {
        pos = 0;
        h_trit.assign(D, 0);
        h_hash = 5381u;
        
        // 计算每个 token 的 x, h_trit, h_hash
        int n = std::min((int)tokens.size(), MAX_CTX);
        int start = (int)tokens.size() - n;
        for (int i = 0; i < n; ++i) {
            int id = tokens[start + i];
            // Q1 embed
            m.q1.forward(id, x_zero.data(), x_cur.data());
            x_history[i] = x_cur;
            // h_trit update
            int bucket = Q1::hash(id, Q1_B);
            for (int d = 0; d < D; ++d) {
                trit prev = (i > 0) ? h_trit[d] : (trit)0;
                trit embed = m.q1.trits[(bucket*Q1_K + 0)*D + d];
                h_trit[d] = (trit)mod3((int)prev + (int)embed);
            }
            // h_hash update
            h_hash = hash_forward(h_hash, id);
        }
        pos = n - 1;  // 最后位置
    }
    
    // 单步 forward: 处理新 token, 输出 logits
    void forward(M& m, int new_token, float* logits_out) {
        // 1. Q1 embed for new token
        m.q1.forward(new_token, x_zero.data(), x_cur.data());
        // 更新 x_history (滑动窗口)
        for (int i = 0; i < MAX_CTX - 1; ++i) x_history[i] = x_history[i+1];
        x_history[MAX_CTX - 1] = x_cur;
        
        // 2. Q3 conv: 只算当前 y (need x[t], x[t-1], ..., x[t-5])
        std::vector<float> y_cur(D, 0);
        const int D = this->D;
        for (int l = 0; l < NL_H; ++l) {
            // Q3 conv
            for (int d = 0; d < D; ++d) {
                float v = 0;
                for (int kk = 0; kk < Q3_K; ++kk) {
                    int idx = MAX_CTX - 1 - kk;
                    v += m.q3w[kk][l*D+d] * x_history[idx][d];
                }
                if (v > 4) v = 4; if (v < -4) v = -4;
                y_cur[d] = v;
            }
            // alpha
            std::vector<float> alpha_cur(D, 0);
            for (int d = 0; d < D; ++d) {
                float z = m.ab[l*D+d];
                for (int k = 0; k < D; ++k) z += m.aW[l*D*D+d*D+k] * x_cur[k];
                float zT = z / 2.0f;
                if (zT > 20) zT = 20; if (zT < -20) zT = -20;
                alpha_cur[d] = 1.0f / (1.0f + std::exp(-zT));
            }
            // gate
            for (int d = 0; d < D; ++d) {
                float zg = m.gb[l*D+d];
                for (int k = 0; k < D; ++k) zg += m.gW[l*D*D+d*D+k] * x_cur[k];
                float zgT = zg / 2.0f;
                if (zgT > 20) zgT = 20; if (zgT < -20) zgT = -20;
                float g = 1.0f / (1.0f + std::exp(-zgT));
                y_cur[d] = y_cur[d] * g;
            }
            // Store for next layer
            x_cur = y_cur;
        }
        
        // 3. 更新 h_trit (mod 3)
        int bucket = Q1::hash(new_token, Q1_B);
        for (int d = 0; d < D; ++d) {
            trit embed = m.q1.trits[(bucket*Q1_K + 0)*D + d];
            h_trit[d] = (trit)mod3((int)h_trit[d] + (int)embed);
        }
        
        // 4. 更新 h_hash
        h_hash = hash_forward(h_hash, new_token);
        
        // 5. 提取 hash features
        std::vector<float> hash_feat(HASH_FEATURES);
        extract_hash_features(h_hash, hash_feat.data());
        
        // 6. 计算 logits (仅 V*D + V*16, 不用 BATCH*SEQ)
        const int V = V_unit;
        int prev = (pos > 0) ? new_token : 0;  // 仅用当前 prev
        for (int v = 0; v < V; ++v) {
            float lv = m.Wbi[prev*V + v];
            for (int d = 0; d < D; ++d) lv += m.W[v*D+d] * (float)h_trit[d];
            for (int f = 0; f < HASH_FEATURES; ++f) lv += m.W_hash[v*HASH_FEATURES+f] * hash_feat[f];
            logits_out[v] = lv;
        }
        pos++;
    }
};

// ============================================================================
//  Main
// ============================================================================
int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("Usage: yaoyao_gen_v21_fast.exe <model.bin> <vocab_text> <prompt> [max_tokens=40]\n");
        return 1;
    }
    std::string model_path = argv[1];
    std::string text_path = argv[2];
    std::string prompt = argv[3];
    int max_tokens = (argc > 4) ? atoi(argv[4]) : 40;
    const int D = D_H;

    // 加载 vocab
    std::printf("[GEN] Loading vocab from %s\n", text_path.c_str());
    std::ifstream f(text_path, std::ios::binary);
    std::stringstream ss; ss << f.rdbuf();
    std::string text = ss.str();
    f.close();
    Vocab vocab;
    vocab.build(text.substr(0, std::min<size_t>(text.size(), 5000000)), 1024);
    int V_unit = (int)vocab.id_to_word.size();
    int PAD = vocab.pad_id;
    std::printf("[GEN] Vocab=%d\n", V_unit);

    // 加载模型
    M m;
    if (!m.load(model_path)) { std::fprintf(stderr, "Failed to load model\n"); return 1; }

    // 准备
    std::vector<int> prompt_ids = vocab.encode(prompt);
    std::vector<int> ids = prompt_ids;
    
    // 初始化增量状态
    InferState state(V_unit, D);
    state.seed(m, prompt_ids);
    
    std::printf("\nPrompt: %s\n", prompt.c_str());
    
    std::vector<float> logits(V_unit);
    auto t0 = std::chrono::steady_clock::now();
    double total_step_ms = 0;
    
    int generated_count = 0;
    for (int step = 0; step < max_tokens; ++step) {
        auto t1 = std::chrono::steady_clock::now();
        
        int prev_token = ids.back();
        state.forward(m, prev_token, logits.data());
        
        // Sampling
        float T = 0.9f;
        std::vector<float> adj_logit(V_unit);
        for (int v = 0; v < V_unit; ++v) adj_logit[v] = logits[v] / T;
        for (size_t back = 0; back < ids.size() && back < 6; ++back) {
            int tk = ids[ids.size()-1-back];
            if (tk >= 2 && tk < V_unit) adj_logit[tk] -= 3.0f * std::pow(0.65f, (float)back);
        }
        std::vector<int> idx_sort(V_unit);
        std::iota(idx_sort.begin(), idx_sort.end(), 0);
        std::sort(idx_sort.begin(), idx_sort.end(),
            [&](int a, int b){ return adj_logit[a] > adj_logit[b]; });
        float mx = adj_logit[idx_sort[0]];
        std::vector<float> probs_gen(V_unit);
        float sum = 0;
        for (int v = 0; v < V_unit; ++v) { probs_gen[v] = std::exp(adj_logit[v] - mx); sum += probs_gen[v]; }
        for (int v = 0; v < V_unit; ++v) probs_gen[v] /= sum;
        float p = 0.9f, cum = 0;
        int nuc_size = V_unit;
        for (int i = 0; i < V_unit; ++i) { cum += probs_gen[idx_sort[i]]; if (cum >= p) { nuc_size = i+1; break; } }
        float r = (float)rand() / (float)RAND_MAX;
        float cumsum = 0;
        int next_token = idx_sort[nuc_size-1];
        for (int i = 0; i < nuc_size; ++i) {
            cumsum += probs_gen[idx_sort[i]];
            if (cumsum >= r) { next_token = idx_sort[i]; break; }
        }
        ids.push_back(next_token);
        generated_count++;
        
        auto t2 = std::chrono::steady_clock::now();
        double step_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        total_step_ms += step_ms;
        
        if (next_token == 2) break;
    }
    auto t_end = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t0).count();
    
    std::string generated = vocab.decode(ids);
    std::printf("Output: %s\n\n", generated.c_str());
    std::printf("=== Performance ===\n");
    std::printf("Total time: %.1f ms\n", total_ms);
    std::printf("Tokens generated: %d\n", generated_count);
    std::printf("Avg per token: %.2f ms\n", total_step_ms / std::max(1, generated_count));
    std::printf("Throughput: %.2f tok/s\n", generated_count * 1000.0 / total_ms);
    std::printf("Speedup: ~%.0fx vs slow version\n", 125.0 / (total_step_ms / std::max(1, generated_count)));
    return 0;
}
