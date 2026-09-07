// 夭夭 Yaoyao v0.2 - OMP parallel batch + file logging
// Logs to D:\TaoMa\yaoyao_train.log (overwrite each run)

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <string>
#include <map>
#include <numeric>
#include <fstream>
#include <sstream>
#include <ctime>
#include <omp.h>

static std::ofstream g_log;

void LOG(const std::string& s) {
    std::cout << s << std::flush;
    if (g_log.is_open()) g_log << s << std::flush;
}

struct Vocab {
    std::map<char, int> c2i;
    std::map<int, int> i2code;
    int pad_id = 0, unk_id = 1;
    void load(const std::string& path) {
        std::ifstream f(path);
        int V; f >> V;
        for (int i = 0; i < V; ++i) {
            int code; f >> code;
            i2code[i] = code;
            if (code >= 0 && code < 128) c2i[(char)code] = i;
        }
    }
    int size() const { return (int)i2code.size(); }
    int encode(char c) const { auto it = c2i.find(c); return it == c2i.end() ? unk_id : it->second; }
    std::string decode(int id) const {
        auto it = i2code.find(id);
        if (it == i2code.end()) return "?";
        if (it->second == -1) return "<pad>";
        if (it->second == -2) return "<unk>";
        if (it->second >= 0 && it->second < 128) return std::string(1, (char)it->second);
        return "?";
    }
};

struct Yaoyao {
    int D, V, N_LAYERS;
    std::vector<std::vector<int8_t>> char_emb;
    std::vector<std::vector<float>> W_h, W_s;
    std::vector<float> bias;
    
    std::vector<float> emb_m, emb_v;
    std::vector<float> Wh_m, Wh_v, Ws_m, Ws_v, b_m, b_v;
    int adam_t = 0;
    
    Yaoyao(int d, int v, int n_layers, unsigned seed = 42) : D(d), V(v), N_LAYERS(n_layers) {
        std::ostringstream o;
        o << "  init V=" << V << " D=" << D << " L=" << N_LAYERS << " threads=" << omp_get_max_threads() << "\n";
        LOG(o.str());
        char_emb.assign(V, std::vector<int8_t>(D, 0));
        std::mt19937 rng(seed);
        std::normal_distribution<float> nde(0, 0.5f), ndw(0, 0.1f);
        for (int v = 0; v < V; ++v) for (int d = 0; d < D; ++d) {
            int r = (int)std::lroundf(nde(rng));
            if (r > 4) r = 4; if (r < -4) r = -4;
            char_emb[v][d] = (int8_t)r;
        }
        W_h.assign(V, std::vector<float>(D, 0));
        W_s.assign(V, std::vector<float>(D, 0));
        bias.assign(V, 0);
        for (int v = 0; v < V; ++v) for (int d = 0; d < D; ++d) {
            W_h[v][d] = ndw(rng); W_s[v][d] = ndw(rng);
        }
        emb_m.assign(V * D, 0); emb_v.assign(V * D, 0);
        Wh_m.assign(V * D, 0); Wh_v.assign(V * D, 0);
        Ws_m.assign(V * D, 0); Ws_v.assign(V * D, 0);
        b_m.assign(V, 0); b_v.assign(V, 0);
    }
    
    struct ForwardCache {
        std::vector<std::vector<int8_t>> hs;
        std::vector<std::vector<int16_t>> ss;
        std::vector<std::vector<float>> logits;
    };
    
    void forward(const std::vector<int>& input, ForwardCache& cache) const {
        int L = input.size();
        cache.hs.assign(L + 1, std::vector<int8_t>(D, 0));
        cache.ss.assign(L + 1, std::vector<int16_t>(D, 0));
        cache.logits.assign(L, std::vector<float>(V, 0));
        std::vector<int8_t> h(D);
        std::vector<int16_t> s(D, 0);
        for (int t = 0; t < L; ++t) {
            const int8_t* e = char_emb[input[t]].data();
            for (int d = 0; d < D; ++d) h[d] = e[d];
            for (int d = 0; d < D; ++d) s[d] = (int16_t)((int)s[d] + (int)h[d]);
            cache.hs[t + 1] = h;
            cache.ss[t + 1] = s;
        }
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < L; ++t) {
            for (int v = 0; v < V; ++v) {
                float L_v = bias[v];
                for (int d = 0; d < D; ++d) {
                    L_v += W_h[v][d] * (float)cache.hs[t + 1][d];
                    L_v += W_s[v][d] * (float)cache.ss[t + 1][d];
                }
                cache.logits[t][v] = L_v;
            }
        }
    }
    
    void adam_update(std::vector<float>& p, std::vector<float>& m, std::vector<float>& v,
                     const std::vector<float>& g, float lr) {
        adam_t++;
        float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        float bc1 = 1.0f - std::pow(b1, (float)adam_t);
        float bc2 = 1.0f - std::pow(b2, (float)adam_t);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)p.size(); ++i) {
            m[i] = b1 * m[i] + (1 - b1) * g[i];
            v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
            p[i] -= lr * (m[i] / bc1) / (std::sqrt(v[i] / bc2) + eps);
        }
    }
    
    float train_batch(const std::vector<std::vector<int>>& inputs,
                      const std::vector<std::vector<int>>& targets,
                      const std::vector<std::vector<int>>& masks, float lr) {
        int B = inputs.size();
        std::vector<ForwardCache> caches(B);
        std::vector<float> batch_loss(B, 0);
        std::vector<int> valid_count(B, 0);
        
        #pragma omp parallel for schedule(dynamic)
        for (int b = 0; b < B; ++b) {
            forward(inputs[b], caches[b]);
            int L = inputs[b].size();
            int valid = 0;
            float local_loss = 0;
            for (int t = 0; t < L; ++t) {
                if (masks[b][t] == 0) continue;
                valid++;
                float mx = *std::max_element(caches[b].logits[t].begin(), caches[b].logits[t].end());
                float sm = 0;
                std::vector<float> probs(V);
                for (int v = 0; v < V; ++v) { probs[v] = std::exp(caches[b].logits[t][v] - mx); sm += probs[v]; }
                for (int v = 0; v < V; ++v) probs[v] /= sm;
                int tgt = targets[b][t];
                local_loss += -std::log(std::max(probs[tgt], 1e-7f));
            }
            batch_loss[b] = local_loss;
            valid_count[b] = valid;
        }
        
        float total_loss = 0;
        int total_valid = 0;
        for (int b = 0; b < B; ++b) { total_loss += batch_loss[b]; total_valid += valid_count[b]; }
        if (total_valid == 0) return 0;
        
        std::vector<std::vector<float>> wh_g(V, std::vector<float>(D, 0));
        std::vector<std::vector<float>> ws_g(V, std::vector<float>(D, 0));
        std::vector<float> b_g(V, 0);
        std::vector<std::vector<float>> emb_g(V, std::vector<float>(D, 0));
        
        for (int b = 0; b < B; ++b) {
            int L = inputs[b].size();
            for (int t = 0; t < L; ++t) {
                if (masks[b][t] == 0) continue;
                float mx = *std::max_element(caches[b].logits[t].begin(), caches[b].logits[t].end());
                float sm = 0;
                std::vector<float> probs(V);
                for (int v = 0; v < V; ++v) { probs[v] = std::exp(caches[b].logits[t][v] - mx); sm += probs[v]; }
                for (int v = 0; v < V; ++v) probs[v] /= sm;
                int tgt = targets[b][t];
                for (int v = 0; v < V; ++v) {
                    float d = probs[v] - (v == tgt ? 1.0f : 0.0f);
                    for (int d_i = 0; d_i < D; ++d_i) {
                        wh_g[v][d_i] += d * (float)caches[b].hs[t + 1][d_i];
                        ws_g[v][d_i] += d * (float)caches[b].ss[t + 1][d_i];
                        emb_g[inputs[b][t]][d_i] += d * (W_h[v][d_i] + W_s[v][d_i]);
                    }
                    b_g[v] += d;
                }
            }
        }
        for (int v = 0; v < V; ++v) {
            for (int d = 0; d < D; ++d) {
                wh_g[v][d] /= total_valid;
                ws_g[v][d] /= total_valid;
                emb_g[v][d] /= total_valid;
            }
            b_g[v] /= total_valid;
        }
        
        std::vector<float> emb_p(V * D);
        for (int v = 0; v < V; ++v)
            for (int d = 0; d < D; ++d) emb_p[v * D + d] = (float)char_emb[v][d];
        std::vector<float> emb_gf(V * D);
        for (int v = 0; v < V; ++v)
            for (int d = 0; d < D; ++d) emb_gf[v * D + d] = emb_g[v][d];
        adam_update(emb_p, emb_m, emb_v, emb_gf, lr);
        for (int v = 0; v < V; ++v)
            for (int d = 0; d < D; ++d) {
                int r = (int)std::lroundf(emb_p[v * D + d]);
                if (r > 4) r = 4; if (r < -4) r = -4;
                char_emb[v][d] = (int8_t)r;
            }
        std::vector<float> wh_p(V * D), ws_p(V * D);
        std::vector<float> wh_gf(V * D), ws_gf(V * D);
        for (int v = 0; v < V; ++v)
            for (int d = 0; d < D; ++d) {
                wh_p[v * D + d] = W_h[v][d]; ws_p[v * D + d] = W_s[v][d];
                wh_gf[v * D + d] = wh_g[v][d]; ws_gf[v * D + d] = ws_g[v][d];
            }
        adam_update(wh_p, Wh_m, Wh_v, wh_gf, lr);
        adam_update(ws_p, Ws_m, Ws_v, ws_gf, lr);
        for (int v = 0; v < V; ++v)
            for (int d = 0; d < D; ++d) {
                W_h[v][d] = wh_p[v * D + d];
                W_s[v][d] = ws_p[v * D + d];
            }
        adam_update(bias, b_m, b_v, b_g, lr);
        return total_loss / total_valid;
    }
    
    std::string generate(const std::string& prompt, int max_new, Vocab& vocab) {
        std::vector<int> out;
        for (char c : prompt) out.push_back(vocab.encode(c));
        std::vector<int16_t> s(D, 0);
        for (int id : out) {
            const int8_t* e = char_emb[id].data();
            for (int d = 0; d < D; ++d) s[d] = (int16_t)((int)s[d] + (int)e[d]);
        }
        for (int step = 0; step < max_new; ++step) {
            int last = out.back();
            const int8_t* e = char_emb[last].data();
            int best = 0; float bs = -1e30f;
            for (int v = 0; v < V; ++v) {
                float L = bias[v];
                for (int d = 0; d < D; ++d) L += W_h[v][d] * (float)e[d] + W_s[v][d] * (float)s[d];
                if (L > bs) { bs = L; best = v; }
            }
            if (best == vocab.pad_id) break;
            out.push_back(best);
            const int8_t* ne = char_emb[best].data();
            for (int d = 0; d < D; ++d) s[d] = (int16_t)((int)s[d] + (int)ne[d]);
        }
        std::string result;
        for (int id : out) result += vocab.decode(id);
        return result;
    }
};

int main() {
    g_log.open("D:\\TaoVm\\yaoyao_train.log", std::ios::trunc);
    {
        time_t now = time(nullptr);
        std::ostringstream o;
        o << "=== 夭夭 Yaoyao v0.2 log started " << ctime(&now);
        LOG(o.str());
    }
    Vocab vocab;
    vocab.load("D:\\TaoVm\\vocab.txt");
    {
        std::ostringstream o;
        o << "  V=" << vocab.size() << " threads=" << omp_get_max_threads() << "\n";
        LOG(o.str());
    }
    
    const int D = 64;
    const int V = vocab.size();
    const int N_LAYERS = 1;
    
    LOG("  loading text...\n");
    std::ifstream ft("D:\\TaoVm\\tinystories_train.txt");
    std::stringstream ss; ss << ft.rdbuf();
    std::string text = ss.str();
    std::vector<int> tokens;
    tokens.reserve(text.size());
    for (char c : text) tokens.push_back(vocab.encode(c));
    {
        std::ostringstream o;
        o << "  tokens=" << tokens.size() << "\n";
        LOG(o.str());
    }
    
    const int SEQ_LEN = 64;
    const int N_WINDOWS = 5000;
    const int BATCH_SIZE = 32;
    {
        std::ostringstream o;
        o << "  N_WINDOWS=" << N_WINDOWS << " batch=" << BATCH_SIZE << "\n";
        LOG(o.str());
    }
    
    std::vector<std::vector<int>> all_in(N_WINDOWS), all_tg(N_WINDOWS), all_ms(N_WINDOWS);
    for (int i = 0; i < N_WINDOWS; ++i) {
        int start = i * SEQ_LEN;
        all_in[i].assign(SEQ_LEN, 0);
        all_tg[i].assign(SEQ_LEN, 0);
        all_ms[i].assign(SEQ_LEN, 0);
        for (int j = 0; j < SEQ_LEN; ++j) {
            all_in[i][j] = tokens[start + j];
            if (start + j + 1 < (int)tokens.size()) {
                all_tg[i][j] = tokens[start + j + 1];
                all_ms[i][j] = 1;
            }
        }
    }
    
    Yaoyao model(D, V, N_LAYERS);
    auto gen = [&](const std::string& prompt, int len) {
        return model.generate(prompt, len, vocab);
    };
    {
        std::ostringstream o;
        o << "\n  Initial: \"" << gen("Once upon a time", 80) << "\"\n";
        LOG(o.str());
    }
    
    int epochs = 10;
    LOG("\n  Training 10 epochs Adam lr=0.01...\n\n");
    auto t0 = std::chrono::steady_clock::now();
    auto last_log = t0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<int> idx(N_WINDOWS);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), std::mt19937(epoch + 1));
        float total = 0;
        int n_batches = 0;
        for (int i = 0; i < N_WINDOWS; i += BATCH_SIZE) {
            std::vector<std::vector<int>> bi(BATCH_SIZE), bt(BATCH_SIZE), bm(BATCH_SIZE);
            int actual_bs = 0;
            for (int j = 0; j < BATCH_SIZE && i + j < N_WINDOWS; ++j) {
                bi[j] = all_in[idx[i + j]];
                bt[j] = all_tg[idx[i + j]];
                bm[j] = all_ms[idx[i + j]];
                actual_bs = j + 1;
            }
            bi.resize(actual_bs); bt.resize(actual_bs); bm.resize(actual_bs);
            total += model.train_batch(bi, bt, bm, 0.01f);
            n_batches++;
            // Log progress every 10 batches or every 5 seconds
            auto now = std::chrono::steady_clock::now();
            double since = std::chrono::duration<double>(now - last_log).count();
            if (n_batches % 10 == 0 || since > 5.0) {
                std::ostringstream o;
                o << "    epoch " << (epoch + 1) << "/" << epochs
                  << " batch " << n_batches << "/" << (N_WINDOWS / BATCH_SIZE)
                  << " loss=" << std::fixed << std::setprecision(4) << total / n_batches
                  << " elapsed=" << std::setprecision(1) << std::chrono::duration<double>(now - t0).count() << "s\r";
                LOG(o.str());
                last_log = now;
            }
        }
        std::ostringstream o;
        o << "\n  Epoch " << (epoch + 1) << " avg_loss=" << std::fixed << std::setprecision(4)
          << total / n_batches << "  \"" << gen("Once upon a time", 80) << "\"\n";
        LOG(o.str());
    }
    auto t1 = std::chrono::steady_clock::now();
    {
        std::ostringstream o;
        o << "\n  Total: " << std::fixed << std::setprecision(1)
          << std::chrono::duration<double>(t1 - t0).count() << "s, "
          << std::setprecision(0) << (N_WINDOWS * epochs) / std::chrono::duration<double>(t1 - t0).count() << " windows/s\n";
        LOG(o.str());
    }
    LOG("=== done ===\n");
    g_log.close();
    return 0;
}
