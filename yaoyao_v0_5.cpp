// 夭夭 Yaoyao v0.5 - Larger + Dropout + Cosine LR
// D=256, L=6, dropout=0.1, cosine LR schedule, more data

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
    float dropout;  // dropout rate
    std::vector<std::vector<int8_t>> char_emb;
    std::vector<std::vector<float>> q3_w0, q3_w1, q3_w2;
    std::vector<std::vector<float>> alpha;
    std::vector<std::vector<float>> W_h, W_s;
    std::vector<float> bias;
    
    std::vector<float> emb_m, emb_v;
    std::vector<std::vector<float>> q3w0_m, q3w0_v, q3w1_m, q3w1_v, q3w2_m, q3w2_v;
    std::vector<std::vector<float>> alpha_m, alpha_v;
    std::vector<float> Wh_m, Wh_v, Ws_m, Ws_v, b_m, b_v;
    int adam_t = 0;
    
    Yaoyao(int d, int v, int n_layers, float drop, unsigned seed = 42)
        : D(d), V(v), N_LAYERS(n_layers), dropout(drop) {
        std::ostringstream o;
        o << "  init V=" << V << " D=" << D << " L=" << N_LAYERS << " dropout=" << dropout
          << " threads=" << omp_get_max_threads() << "\n";
        LOG(o.str());
        char_emb.assign(V, std::vector<int8_t>(D, 0));
        std::mt19937 rng(seed);
        std::normal_distribution<float> nde(0, 0.5f), ndw(0, 0.1f);
        for (int v = 0; v < V; ++v) for (int d = 0; d < D; ++d) {
            int r = (int)std::lroundf(nde(rng));
            if (r > 4) r = 4; if (r < -4) r = -4;
            char_emb[v][d] = (int8_t)r;
        }
        q3_w0.assign(N_LAYERS, std::vector<float>(D, 0));
        q3_w1.assign(N_LAYERS, std::vector<float>(D, 0));
        q3_w2.assign(N_LAYERS, std::vector<float>(D, 0));
        alpha.assign(N_LAYERS, std::vector<float>(D, 0.3f));
        for (int l = 0; l < N_LAYERS; ++l) {
            for (int d = 0; d < D; ++d) {
                q3_w0[l][d] = ndw(rng) * 0.3f;
                q3_w1[l][d] = ndw(rng) * 0.3f;
                q3_w2[l][d] = ndw(rng) * 0.3f;
                alpha[l][d] = 0.3f + ndw(rng) * 0.05f;
            }
        }
        W_h.assign(V, std::vector<float>(D, 0));
        W_s.assign(V, std::vector<float>(D, 0));
        bias.assign(V, 0);
        for (int v = 0; v < V; ++v) for (int d = 0; d < D; ++d) {
            W_h[v][d] = ndw(rng); W_s[v][d] = ndw(rng);
        }
        emb_m.assign(V * D, 0); emb_v.assign(V * D, 0);
        q3w0_m.assign(N_LAYERS, std::vector<float>(D, 0)); q3w0_v.assign(N_LAYERS, std::vector<float>(D, 0));
        q3w1_m.assign(N_LAYERS, std::vector<float>(D, 0)); q3w1_v.assign(N_LAYERS, std::vector<float>(D, 0));
        q3w2_m.assign(N_LAYERS, std::vector<float>(D, 0)); q3w2_v.assign(N_LAYERS, std::vector<float>(D, 0));
        alpha_m.assign(N_LAYERS, std::vector<float>(D, 0)); alpha_v.assign(N_LAYERS, std::vector<float>(D, 0));
        Wh_m.assign(V * D, 0); Wh_v.assign(V * D, 0);
        Ws_m.assign(V * D, 0); Ws_v.assign(V * D, 0);
        b_m.assign(V, 0); b_v.assign(V, 0);
    }
    
    struct LayerStates {
        std::vector<std::vector<int8_t>> xs;
        std::vector<std::vector<int8_t>> ys;
        std::vector<std::vector<int8_t>> hs;
        std::vector<std::vector<int16_t>> ss;
    };
    struct ForwardCache {
        std::vector<LayerStates> layers;
        std::vector<std::vector<float>> logits;
        std::vector<std::vector<char>> dropout_mask;  // [L][D]
    };
    
    // Forward with optional dropout
    void forward(const std::vector<int>& input, ForwardCache& cache, bool training = true,
                 std::mt19937* rng = nullptr) const {
        int L = input.size();
        cache.logits.assign(L, std::vector<float>(V, 0));
        cache.layers.assign(N_LAYERS, LayerStates());
        if (training) cache.dropout_mask.assign(N_LAYERS, std::vector<char>(D, 0));
        cache.layers[0].xs.assign(L, std::vector<int8_t>(D, 0));
        for (int t = 0; t < L; ++t) {
            const int8_t* e = char_emb[input[t]].data();
            for (int d = 0; d < D; ++d) cache.layers[0].xs[t][d] = e[d];
        }
        for (int l = 0; l < N_LAYERS; ++l) {
            cache.layers[l].ys.assign(L, std::vector<int8_t>(D, 0));
            cache.layers[l].hs.assign(L + 1, std::vector<int8_t>(D, 0));
            cache.layers[l].ss.assign(L + 1, std::vector<int16_t>(D, 0));
            // Build dropout mask for this layer
            if (training) {
                std::uniform_real_distribution<float> ud(0, 1);
                for (int d = 0; d < D; ++d) {
                    cache.dropout_mask[l][d] = (ud(*rng) > dropout) ? 1 : 0;
                }
            }
            for (int t = 0; t < L; ++t) {
                for (int d = 0; d < D; ++d) {
                    float v = 0;
                    if (t >= 2) v += q3_w0[l][d] * (float)cache.layers[l].xs[t - 2][d];
                    if (t >= 1) v += q3_w1[l][d] * (float)cache.layers[l].xs[t - 1][d];
                    v += q3_w2[l][d] * (float)cache.layers[l].xs[t][d];
                    // Apply dropout to Q3 output (scale by 1/(1-dropout) for inverted dropout)
                    if (training && cache.dropout_mask[l][d] == 0) v = 0;
                    else if (training) v /= (1.0f - dropout);
                    int r = (int)std::lroundf(v);
                    if (r > 4) r = 4; if (r < -4) r = -4;
                    cache.layers[l].ys[t][d] = (int8_t)r;
                }
            }
            #pragma omp parallel for schedule(static)
            for (int d = 0; d < D; ++d) {
                int16_t s = 0;
                int8_t h = 0;
                for (int t = 0; t < L; ++t) {
                    int y = (int)cache.layers[l].ys[t][d];
                    s = (int16_t)((int)s + y);
                    float hv = alpha[l][d] * (float)h + (1.0f - alpha[l][d]) * (float)y;
                    int hr = (int)std::lroundf(hv);
                    if (hr > 4) hr = 4; if (hr < -4) hr = -4;
                    h = (int8_t)hr;
                    cache.layers[l].ss[t + 1][d] = s;
                    cache.layers[l].hs[t + 1][d] = h;
                }
            }
            if (l + 1 < N_LAYERS) {
                cache.layers[l + 1].xs.assign(L, std::vector<int8_t>(D, 0));
                for (int t = 0; t < L; ++t)
                    cache.layers[l + 1].xs[t] = cache.layers[l].ys[t];
            }
        }
        const auto& last_hs = cache.layers[N_LAYERS - 1].hs;
        const auto& last_ss = cache.layers[N_LAYERS - 1].ss;
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < L; ++t) {
            for (int v = 0; v < V; ++v) {
                float L_v = bias[v];
                for (int d = 0; d < D; ++d) {
                    L_v += W_h[v][d] * (float)last_hs[t + 1][d];
                    L_v += W_s[v][d] * (float)last_ss[t + 1][d];
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
                      const std::vector<std::vector<int>>& masks, float lr,
                      std::mt19937* rng) {
        int B = inputs.size();
        std::vector<ForwardCache> caches(B);
        std::vector<float> batch_loss(B, 0);
        std::vector<int> valid_count(B, 0);
        
        #pragma omp parallel for schedule(dynamic)
        for (int b = 0; b < B; ++b) {
            forward(inputs[b], caches[b], true, rng);
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
        std::vector<std::vector<float>> q3w0_g(N_LAYERS, std::vector<float>(D, 0));
        std::vector<std::vector<float>> q3w1_g(N_LAYERS, std::vector<float>(D, 0));
        std::vector<std::vector<float>> q3w2_g(N_LAYERS, std::vector<float>(D, 0));
        std::vector<std::vector<float>> alpha_g(N_LAYERS, std::vector<float>(D, 0));
        
        for (int b = 0; b < B; ++b) {
            int L = inputs[b].size();
            std::vector<std::vector<std::vector<float>>> dy_l(N_LAYERS,
                std::vector<std::vector<float>>(L, std::vector<float>(D, 0)));
            std::vector<std::vector<std::vector<float>>> dh(N_LAYERS,
                std::vector<std::vector<float>>(L + 1, std::vector<float>(D, 0)));
            std::vector<std::vector<std::vector<float>>> ds(N_LAYERS,
                std::vector<std::vector<float>>(L + 1, std::vector<float>(D, 0)));
            
            for (int t = 0; t < L; ++t) {
                if (masks[b][t] == 0) continue;
                float mx = *std::max_element(caches[b].logits[t].begin(), caches[b].logits[t].end());
                float sm = 0;
                std::vector<float> probs(V);
                for (int v = 0; v < V; ++v) { probs[v] = std::exp(caches[b].logits[t][v] - mx); sm += probs[v]; }
                for (int v = 0; v < V; ++v) probs[v] /= sm;
                int tgt = targets[b][t];
                const auto& h_now = caches[b].layers[N_LAYERS - 1].hs[t + 1];
                const auto& s_now = caches[b].layers[N_LAYERS - 1].ss[t + 1];
                for (int v = 0; v < V; ++v) {
                    float d = probs[v] - (v == tgt ? 1.0f : 0.0f);
                    for (int d_i = 0; d_i < D; ++d_i) {
                        wh_g[v][d_i] += d * (float)h_now[d_i];
                        ws_g[v][d_i] += d * (float)s_now[d_i];
                        dh[N_LAYERS - 1][t + 1][d_i] += d * W_h[v][d_i];
                        ds[N_LAYERS - 1][t + 1][d_i] += d * W_s[v][d_i];
                    }
                    b_g[v] += d;
                }
            }
            
            for (int l = N_LAYERS - 1; l >= 0; --l) {
                for (int t = L - 1; t >= 0; --t) {
                    for (int d = 0; d < D; ++d) {
                        ds[l][t][d] += ds[l][t + 1][d];
                        dy_l[l][t][d] += ds[l][t + 1][d] + dh[l][t + 1][d] * (1.0f - alpha[l][d]);
                        alpha_g[l][d] += dh[l][t + 1][d] * ((float)caches[b].layers[l].hs[t][d] - (float)caches[b].layers[l].ys[t][d]);
                        dh[l][t][d] += dh[l][t + 1][d] * alpha[l][d];
                        // Dropout: zero gradient where masked
                        if (caches[b].dropout_mask[l][d] == 0) {
                            dy_l[l][t][d] = 0;
                        }
                    }
                }
                if (l > 0) {
                    for (int t = 0; t < L; ++t)
                        for (int d = 0; d < D; ++d)
                            dy_l[l - 1][t][d] += dy_l[l][t][d];
                }
            }
            for (int l = 0; l < N_LAYERS; ++l) {
                for (int t = 0; t < L; ++t) {
                    for (int d = 0; d < D; ++d) {
                        float dy = dy_l[l][t][d];
                        if (t >= 2) q3w0_g[l][d] += dy * (float)caches[b].layers[l].xs[t - 2][d];
                        if (t >= 1) q3w1_g[l][d] += dy * (float)caches[b].layers[l].xs[t - 1][d];
                        q3w2_g[l][d] += dy * (float)caches[b].layers[l].xs[t][d];
                    }
                }
            }
            for (int t = 0; t < L; ++t) {
                for (int d = 0; d < D; ++d) {
                    float dy = dy_l[0][t][d];
                    if (t >= 2) emb_g[inputs[b][t - 2]][d] += dy * q3_w0[0][d];
                    if (t >= 1) emb_g[inputs[b][t - 1]][d] += dy * q3_w1[0][d];
                    emb_g[inputs[b][t]][d] += dy * q3_w2[0][d];
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
        for (int l = 0; l < N_LAYERS; ++l)
            for (int d = 0; d < D; ++d) {
                q3w0_g[l][d] /= total_valid;
                q3w1_g[l][d] /= total_valid;
                q3w2_g[l][d] /= total_valid;
                alpha_g[l][d] /= total_valid;
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
        for (int l = 0; l < N_LAYERS; ++l) {
            adam_update(q3_w0[l], q3w0_m[l], q3w0_v[l], q3w0_g[l], lr);
            adam_update(q3_w1[l], q3w1_m[l], q3w1_v[l], q3w1_g[l], lr);
            adam_update(q3_w2[l], q3w2_m[l], q3w2_v[l], q3w2_g[l], lr);
            adam_update(alpha[l], alpha_m[l], alpha_v[l], alpha_g[l], lr);
            for (int d = 0; d < D; ++d) {
                if (q3_w0[l][d] > 2) q3_w0[l][d] = 2; if (q3_w0[l][d] < -2) q3_w0[l][d] = -2;
                if (q3_w1[l][d] > 2) q3_w1[l][d] = 2; if (q3_w1[l][d] < -2) q3_w1[l][d] = -2;
                if (q3_w2[l][d] > 2) q3_w2[l][d] = 2; if (q3_w2[l][d] < -2) q3_w2[l][d] = -2;
                if (alpha[l][d] < 0) alpha[l][d] = 0; if (alpha[l][d] > 1) alpha[l][d] = 1;
            }
        }
        return total_loss / total_valid;
    }
    
    std::string generate(const std::string& prompt, int max_new, Vocab& vocab,
                         int top_k = 10, float temperature = 0.8) {
        std::vector<int> out;
        for (char c : prompt) out.push_back(vocab.encode(c));
        std::mt19937 rng(123);
        for (int step = 0; step < max_new; ++step) {
            ForwardCache gen_cache;
            forward(out, gen_cache, false);
            const auto& last_h = gen_cache.layers[N_LAYERS - 1].hs[out.size()];
            const auto& last_s = gen_cache.layers[N_LAYERS - 1].ss[out.size()];
            std::vector<float> logits(V);
            float max_l = -1e30f;
            for (int v = 0; v < V; ++v) {
                float L = bias[v];
                for (int d = 0; d < D; ++d) L += W_h[v][d] * (float)last_h[d] + W_s[v][d] * (float)last_s[d];
                logits[v] = L;
                if (L > max_l) max_l = L;
            }
            for (auto& l : logits) l = (l - max_l) / temperature;
            std::vector<int> idx(V);
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                              [&](int a, int b) { return logits[a] > logits[b]; });
            float mx = logits[idx[0]];
            std::vector<float> probs(top_k);
            float sum = 0;
            for (int k = 0; k < top_k; ++k) { probs[k] = std::exp(logits[idx[k]] - mx); sum += probs[k]; }
            for (int k = 0; k < top_k; ++k) probs[k] /= sum;
            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            int chosen = idx[dist(rng)];
            if (chosen == vocab.pad_id) break;
            out.push_back(chosen);
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
        o << "=== 夭夭 Yaoyao v0.5 log started " << ctime(&now);
        LOG(o.str());
    }
    Vocab vocab;
    vocab.load("D:\\TaoVm\\vocab.txt");
    {
        std::ostringstream o;
        o << "  V=" << vocab.size() << " threads=" << omp_get_max_threads() << "\n";
        LOG(o.str());
    }
    
    // v0.5 config: D=256, L=6, dropout=0.1
    const int D = 256;
    const int V = vocab.size();
    const int N_LAYERS = 6;
    const float dropout = 0.1f;
    
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
    const int N_WINDOWS = 30000;  // 6x more data
    const int BATCH_SIZE = 32;
    {
        std::ostringstream o;
        o << "  N_WINDOWS=" << N_WINDOWS << " batch=" << BATCH_SIZE << "\n";
        LOG(o.str());
    }
    
    std::vector<std::vector<int>> all_in(N_WINDOWS), all_tg(N_WINDOWS), all_ms(N_WINDOWS);
    for (int i = 0; i < N_WINDOWS; ++i) {
        int start = (i * SEQ_LEN) % (tokens.size() - SEQ_LEN - 1);
        all_in[i].assign(SEQ_LEN, 0);
        all_tg[i].assign(SEQ_LEN, 0);
        all_ms[i].assign(SEQ_LEN, 0);
        for (int j = 0; j < SEQ_LEN; ++j) {
            all_in[i][j] = tokens[start + j];
            all_tg[i][j] = tokens[start + j + 1];
            all_ms[i][j] = 1;
        }
    }
    
    Yaoyao model(D, V, N_LAYERS, dropout);
    std::mt19937 rng(42);
    auto gen = [&](const std::string& prompt, int len, int k = 10, float t = 0.8) {
        return model.generate(prompt, len, vocab, k, t);
    };
    {
        std::ostringstream o;
        for (auto& prompt : std::vector<std::string>{"Once upon a time", "Lily and Tom", "The cat sat"}) {
            o << "  Initial greedy [\"" << prompt << "\"]: \"" << gen(prompt, 60, 1, 1.0) << "\"\n";
            o << "  Initial top-10 [\"" << prompt << "\"]: \"" << gen(prompt, 60, 10, 0.8) << "\"\n";
        }
        o << "\n";
        LOG(o.str());
    }
    
    int epochs = 12;
    LOG("  Training 12 epochs cosine LR 0.005 -> 0.0001...\n\n");
    auto t0 = std::chrono::steady_clock::now();
    auto last_log = t0;
    float best_loss = 1e30f;
    int no_improve = 0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Cosine LR
        float lr = 0.0001f + 0.5f * (0.005f - 0.0001f) * (1.0f + std::cos(3.14159f * epoch / (epochs - 1)));
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
            total += model.train_batch(bi, bt, bm, lr, &rng);
            n_batches++;
            auto now = std::chrono::steady_clock::now();
            double since = std::chrono::duration<double>(now - last_log).count();
            if (n_batches % 20 == 0 || since > 5.0) {
                std::ostringstream o;
                o << "    epoch " << (epoch + 1) << "/" << epochs << " lr=" << std::fixed << std::setprecision(4) << lr
                  << " batch " << n_batches << "/" << (N_WINDOWS / BATCH_SIZE)
                  << " loss=" << std::setprecision(4) << total / n_batches
                  << " elapsed=" << std::setprecision(1) << std::chrono::duration<double>(now - t0).count() << "s\r";
                LOG(o.str());
                last_log = now;
            }
        }
        std::ostringstream o;
        o << "\n  Epoch " << (epoch + 1) << " avg_loss=" << std::fixed << std::setprecision(4)
          << total / n_batches << "\n";
        for (auto& prompt : std::vector<std::string>{"Once upon a time", "Lily and Tom", "The cat sat"}) {
            o << "    greedy [\"" << prompt << "\"]: \"" << gen(prompt, 50, 1, 1.0) << "\"\n";
            o << "    top-10 [\"" << prompt << "\"]: \"" << gen(prompt, 50, 10, 0.8) << "\"\n";
        }
        o << "\n";
        LOG(o.str());
        // Early stopping check
        float avg = total / n_batches;
        if (avg < best_loss) {
            best_loss = avg;
            no_improve = 0;
        } else {
            no_improve++;
            if (no_improve >= 3) {
                std::ostringstream eo;
                eo << "  >>> Early stop at epoch " << (epoch + 1) << " (no improve for 3 epochs)\n";
                LOG(eo.str());
                break;
            }
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    {
        std::ostringstream o;
        o << "\n  Total: " << std::fixed << std::setprecision(1)
          << std::chrono::duration<double>(t1 - t0).count() << "s\n";
        LOG(o.str());
    }
    LOG("=== done ===\n");
    g_log.close();
    return 0;
}
