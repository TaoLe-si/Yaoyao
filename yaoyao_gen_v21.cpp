// yaoyao_gen_v21.cpp
// 夭夭 v21 inference server - 基于 mod3 + hash 可逆架构
// 用法: yaoyao_gen_v21.exe --server <model.bin> <text_for_vocab>
//       yaoyao_gen_v21.exe <model.bin> <text_for_vocab> <prompt> [max_tokens]

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
//  Q1 (简化版, inference only)
// ============================================================================
struct Q1 {
    int B, K, D;
    std::vector<int8_t> trits;
    std::vector<float> adam_m, adam_v;
    int step = 0;
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
        // 分配空间
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
//  Forward (inference only)
// ============================================================================
void forward_step(M& m, const std::vector<int>& inp, int BATCH, int SEQ, int PAD,
                  int D, int NL, int V_unit,
                  std::vector<float>& x, std::vector<float>& x_prev,
                  std::vector<float>& y, std::vector<float>& alpha,
                  std::vector<trit>& h_trit, std::vector<hash_t>& h_hash,
                  std::vector<float>& trit_features, std::vector<float>& hash_features,
                  std::vector<float>& logits, std::vector<float>& probs,
                  std::vector<float>& xs, std::vector<float>& ys,
                  std::vector<float>& alphas, std::vector<float>& gates_v) {
    int BL = BATCH * SEQ;

    // Q1 嵌入
    for (int d = 0; d < D; ++d) x_prev[d] = 0.0f;
    for (int t = 0; t < BL; ++t) {
        int id = inp[t];
        m.q1.forward(id, x_prev.data(), x.data() + t*D);
        for (int d = 0; d < D; ++d) x_prev[d] = 0.0f;
    }

    for (int l = 0; l < NL; ++l) {
        for (int b = 0; b < BATCH; ++b) for (int t = 0; t < SEQ; ++t) {
            int bt = b*SEQ+t;
            for (int d = 0; d < D; ++d) {
                float v = 0;
                for (int kk = 0; kk < Q3_K; ++kk) {
                    if (t >= kk) v += m.q3w[kk][l*D+d] * x[(b*SEQ+t-kk)*D+d];
                }
                if (v > 4) v = 4; if (v < -4) v = -4;
                y[bt*D+d] = v;
            }
        }
        for (int b = 0; b < BATCH; ++b) for (int t = 0; t < SEQ; ++t) {
            int bt = b*SEQ+t;
            for (int d = 0; d < D; ++d) {
                float z = m.ab[l*D+d];
                for (int k = 0; k < D; ++k) z += m.aW[l*D*D+d*D+k] * x[bt*D+k];
                float zT = z / 2.0f;
                if (zT > 20) zT = 20; if (zT < -20) zT = -20;
                alpha[bt*D+d] = 1.0f / (1.0f + std::exp(-zT));
            }
        }
        for (int b = 0; b < BATCH; ++b) for (int t = 0; t < SEQ; ++t) {
            int bt = b*SEQ+t;
            for (int d = 0; d < D; ++d) {
                float zg = m.gb[l*D+d];
                for (int k = 0; k < D; ++k) zg += m.gW[l*D*D+d*D+k] * x[bt*D+k];
                float zgT = zg / 2.0f;
                if (zgT > 20) zgT = 20; if (zgT < -20) zgT = -20;
                float g = 1.0f / (1.0f + std::exp(-zgT));
                gates_v[l*BL*D+bt*D+d] = g;
                y[bt*D+d] = y[bt*D+d] * g;
            }
        }
        if (l == NL - 1) {
            for (int b = 0; b < BATCH; ++b) {
                h_hash[b] = 5381u;
                for (int d = 0; d < D; ++d) h_trit[b*D + d] = 0;
            }
            for (int b = 0; b < BATCH; ++b) for (int t = 0; t < SEQ; ++t) {
                int bt = b*SEQ + t;
                int id = inp[bt];
                int bucket = Q1::hash(id, Q1_B);
                for (int d = 0; d < D; ++d) {
                    trit prev_trit = (t > 0) ? trit_features[(b*SEQ+t-1)*D + d] : (trit)0;
                    trit embed_trit = m.q1.trits[(bucket*Q1_K + 0)*D + d];
                    trit_features[bt*D + d] = (trit)mod3((int)prev_trit + (int)embed_trit);
                }
                hash_t prev_hash = (t > 0) ? h_hash[b] : 5381u;
                h_hash[b] = hash_forward(prev_hash, id);
            }
            for (int b = 0; b < BATCH; ++b) for (int t = 0; t < SEQ; ++t) {
                int bt = b*SEQ + t;
                extract_hash_features(h_hash[b], hash_features.data() + bt*HASH_FEATURES);
            }
        }
    }

    // 预测头
    for (int b = 0; b < BATCH; ++b) for (int t = 0; t < SEQ; ++t) {
        int bt = b*SEQ+t;
        int prev = (t > 0) ? inp[(b*SEQ+t-1)] : PAD;
        for (int v = 0; v < V_unit; ++v) {
            float lv = m.Wbi[prev*V_unit + v];
            for (int d = 0; d < D; ++d) lv += m.W[v*D+d] * (float)trit_features[bt*D+d];
            for (int f = 0; f < HASH_FEATURES; ++f) lv += m.W_hash[v*HASH_FEATURES+f] * hash_features[bt*HASH_FEATURES+f];
            logits[bt*V_unit + v] = lv;
        }
    }

    for (int n = 0; n < BL; ++n) {
        float mx = logits[n*V_unit];
        for (int v = 1; v < V_unit; ++v) if (logits[n*V_unit+v] > mx) mx = logits[n*V_unit+v];
        float sum = 0;
        for (int v = 0; v < V_unit; ++v) { probs[n*V_unit+v] = std::exp(logits[n*V_unit+v] - mx); sum += probs[n*V_unit+v]; }
        for (int v = 0; v < V_unit; ++v) probs[n*V_unit+v] /= sum;
    }
}

// ============================================================================
//  Server Mode
// ============================================================================
void run_server(M& m, Vocab& vocab, int BATCH, int SEQ, int D, int NL, int V_unit, int PAD) {
    std::printf("Server ready: yaoyao_gen_v21 listening on stdin (BATCH=%d SEQ=%d V=%d)\n", BATCH, SEQ, V_unit);
    std::printf("[Server] Reading prompts from stdin, format: PROMPT <text>\n");
    std::fflush(stdout);

    std::vector<float> x(BATCH*SEQ*D), x_prev(D), y(BATCH*SEQ*D), alpha(BATCH*SEQ*D);
    std::vector<float> xs(NL*BATCH*SEQ*D), ys(NL*BATCH*SEQ*D), alphas(NL*BATCH*SEQ*D), gates_v(NL*BATCH*SEQ*D);
    std::vector<trit> h_trit(BATCH*SEQ*D);
    std::vector<hash_t> h_hash(BATCH);
    std::vector<float> trit_features(BATCH*SEQ*D), hash_features(BATCH*SEQ*HASH_FEATURES);
    std::vector<float> logits(BATCH*SEQ*V_unit), probs(BATCH*SEQ*V_unit);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        if (line.substr(0, 6) == "PROMPT") {
            std::string prompt = line.substr(6);
            while (!prompt.empty() && (prompt[0] == ' ' || prompt[0] == '\t')) prompt = prompt.substr(1);

            std::vector<int> prompt_ids = vocab.encode(prompt);
            std::vector<int> ids = prompt_ids;
            int max_tokens = 40;

            // 解析 MAXTOKENS
            size_t space_pos = prompt.find(' ');
            // 简化: 总是生成 40 tokens

            auto t_total_start = std::chrono::steady_clock::now();
            int n_gen = 0;
            for (int step = 0; step < max_tokens; ++step) {
                auto t_step_start = std::chrono::steady_clock::now();
                int L = (int)ids.size();
                std::vector<int> in2(SEQ);
                for (int i = 0; i < SEQ; ++i) {
                    int idx = L - SEQ + i;
                    in2[i] = (idx < 0) ? PAD : ids[idx];
                }
                std::vector<int> inBL(BATCH*SEQ);
                for (int b = 0; b < BATCH; ++b) for (int t = 0; t < SEQ; ++t) inBL[b*SEQ+t] = in2[t];

                forward_step(m, inBL, BATCH, SEQ, PAD, D, NL, V_unit,
                             x, x_prev, y, alpha, h_trit, h_hash,
                             trit_features, hash_features, logits, probs,
                             xs, ys, alphas, gates_v);

                int bt = (BATCH-1)*SEQ + (SEQ-1);
                float T = 0.9f;
                std::vector<float> adj_logit(V_unit);
                for (int v = 0; v < V_unit; ++v) adj_logit[v] = logits[bt*V_unit+v] / T;
                for (size_t back = 0; back < ids.size() && back < 6; ++back) {
                    int tk = ids[ids.size()-1-back];
                    if (tk >= 2 && tk < V_unit) {
                        adj_logit[tk] -= 3.0f * std::pow(0.65f, (float)back);
                    }
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
                n_gen++;

                auto t_step_end = std::chrono::steady_clock::now();
                double step_ms = std::chrono::duration<double, std::milli>(t_step_end - t_step_start).count();

                // 输出 token (SSE 风格, yaoyao_api.py 解析)
                std::string token_word = (next_token >= 2 && next_token < V_unit) ? vocab.id_to_word[next_token] : "";
                std::printf("TOKEN %s %.1f\n", token_word.c_str(), step_ms);
                std::fflush(stdout);

                if (next_token == 2) break;  // EOS
            }
            std::printf("DONE\n");
            std::fflush(stdout);
        } else if (line == "QUIT") {
            break;
        }
    }
}

// ============================================================================
//  Main
// ============================================================================
int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("Usage:\n");
        std::printf("  Server: yaoyao_gen_v21.exe --server <model.bin> <vocab_text>\n");
        std::printf("  Test:   yaoyao_gen_v21.exe <model.bin> <vocab_text> <prompt>\n");
        return 1;
    }

    const int D = D_H, NL = NL_H;
    int BATCH = 16, SEQ = 64;
    int V_unit = 0;
    int PAD = 0;

    if (std::string(argv[1]) == "--server") {
        // Server mode
        std::string model_path = argv[2];
        std::string text_path = argv[3];

        std::printf("[GEN] Loading text for vocab: %s\n", text_path.c_str());
        std::ifstream f(text_path, std::ios::binary);
        std::stringstream ss; ss << f.rdbuf();
        std::string text = ss.str();
        f.close();
        std::printf("[GEN] Loaded %zu chars\n", text.size());

        Vocab vocab;
        vocab.build(text.substr(0, std::min<size_t>(text.size(), 5000000)), 1024);
        V_unit = (int)vocab.id_to_word.size();
        PAD = vocab.pad_id;
        std::printf("[GEN] Vocab=%d\n", V_unit);

        M m;
        if (!m.load(model_path)) { std::fprintf(stderr, "Failed to load model\n"); return 1; }

        run_server(m, vocab, BATCH, SEQ, D, NL, V_unit, PAD);
    } else {
        // Test mode
        std::string model_path = argv[1];
        std::string text_path = argv[2];
        std::string prompt = (argc > 3) ? argv[3] : "Once upon a time";

        std::printf("[GEN] Loading vocab from %s\n", text_path.c_str());
        std::ifstream f(text_path, std::ios::binary);
        std::stringstream ss; ss << f.rdbuf();
        std::string text = ss.str();
        f.close();

        Vocab vocab;
        vocab.build(text.substr(0, std::min<size_t>(text.size(), 5000000)), 1024);
        V_unit = (int)vocab.id_to_word.size();
        PAD = vocab.pad_id;

        M m;
        if (!m.load(model_path)) { std::fprintf(stderr, "Failed to load model\n"); return 1; }

        std::vector<float> x(BATCH*SEQ*D), x_prev(D), y(BATCH*SEQ*D), alpha(BATCH*SEQ*D);
        std::vector<float> xs(NL*BATCH*SEQ*D), ys(NL*BATCH*SEQ*D), alphas(NL*BATCH*SEQ*D), gates_v(NL*BATCH*SEQ*D);
        std::vector<trit> h_trit(BATCH*SEQ*D);
        std::vector<hash_t> h_hash(BATCH);
        std::vector<float> trit_features(BATCH*SEQ*D), hash_features(BATCH*SEQ*HASH_FEATURES);
        std::vector<float> logits(BATCH*SEQ*V_unit), probs(BATCH*SEQ*V_unit);

        std::vector<int> prompt_ids = vocab.encode(prompt);
        std::vector<int> ids = prompt_ids;
        std::printf("\nPrompt: %s\n", prompt.c_str());

        int max_tokens = (argc > 4) ? atoi(argv[4]) : 40;
        auto t0 = std::chrono::steady_clock::now();
        double total_step_ms = 0;

        for (int step = 0; step < max_tokens; ++step) {
            auto t1 = std::chrono::steady_clock::now();
            int L = (int)ids.size();
            std::vector<int> in2(SEQ);
            for (int i = 0; i < SEQ; ++i) {
                int idx = L - SEQ + i;
                in2[i] = (idx < 0) ? PAD : ids[idx];
            }
            std::vector<int> inBL(BATCH*SEQ);
            for (int b = 0; b < BATCH; ++b) for (int t = 0; t < SEQ; ++t) inBL[b*SEQ+t] = in2[t];

            forward_step(m, inBL, BATCH, SEQ, PAD, D, NL, V_unit,
                         x, x_prev, y, alpha, h_trit, h_hash,
                         trit_features, hash_features, logits, probs,
                         xs, ys, alphas, gates_v);

            int bt = (BATCH-1)*SEQ + (SEQ-1);
            float T = 0.9f;
            std::vector<float> adj_logit(V_unit);
            for (int v = 0; v < V_unit; ++v) adj_logit[v] = logits[bt*V_unit+v] / T;
            for (size_t back = 0; back < ids.size() && back < 6; ++back) {
                int tk = ids[ids.size()-1-back];
                if (tk >= 2 && tk < V_unit) {
                    adj_logit[tk] -= 3.0f * std::pow(0.65f, (float)back);
                }
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
        std::printf("Tokens generated: %d\n", (int)ids.size() - (int)prompt_ids.size());
        std::printf("Avg per token: %.2f ms\n", total_step_ms / std::max(1, (int)ids.size() - (int)prompt_ids.size()));
        std::printf("Throughput: %.2f tok/s\n", (ids.size() - prompt_ids.size()) * 1000.0 / total_ms);
    }
    return 0;
}
