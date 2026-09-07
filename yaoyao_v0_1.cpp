// 夭夭 Yaoyao v0.1 - TinyStories character-level LM
// Multi-layer Q1+Q3+Q2-A+Sum+Q4, Adam optimizer.

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
    std::vector<std::vector<float>> q3_w0, q3_w1, q3_w2;
    std::vector<std::vector<float>> alpha;
    std::vector<std::vector<float>> W_h, W_s;
    std::vector<float> bias;
    
    std::vector<float> emb_m, emb_v;
    std::vector<std::vector<float>> q3w0_m, q3w0_v, q3w1_m, q3w1_v, q3w2_m, q3w2_v;
    std::vector<std::vector<float>> alpha_m, alpha_v;
    std::vector<float> Wh_m, Wh_v, Ws_m, Ws_v, b_m, b_v;
    int adam_t = 0;
    
    Yaoyao(int d, int v, int n_layers, unsigned seed = 42) : D(d), V(v), N_LAYERS(n_layers) {
        std::cout << "  init V=" << V << " D=" << D << " L=" << N_LAYERS << std::endl; std::cout.flush();
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
                q3_w0[l][d] = ndw(rng) * 0.5f;
                q3_w1[l][d] = ndw(rng) * 0.5f;
                q3_w2[l][d] = ndw(rng) * 0.5f;
                alpha[l][d] = 0.2f + ndw(rng) * 0.05f;
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
        std::cout << "  init done" << std::endl; std::cout.flush();
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
    };
    
    void forward(const std::vector<int>& input, ForwardCache& cache) const {
        int L = input.size();
        cache.logits.assign(L, std::vector<float>(V, 0));
        cache.layers.assign(N_LAYERS, LayerStates());
        cache.layers[0].xs.assign(L, std::vector<int8_t>(D, 0));
        for (int t = 0; t < L; ++t) {
            const int8_t* e = char_emb[input[t]].data();
            for (int d = 0; d < D; ++d) cache.layers[0].xs[t][d] = e[d];
        }
        for (int l = 0; l < N_LAYERS; ++l) {
            cache.layers[l].ys.assign(L, std::vector<int8_t>(D, 0));
            cache.layers[l].hs.assign(L + 1, std::vector<int8_t>(D, 0));
            cache.layers[l].ss.assign(L + 1, std::vector<int16_t>(D, 0));
            for (int t = 0; t < L; ++t) {
                for (int d = 0; d < D; ++d) {
                    float v = 0;
                    if (t >= 2) v += q3_w0[l][d] * (float)cache.layers[l].xs[t - 2][d];
                    if (t >= 1) v += q3_w1[l][d] * (float)cache.layers[l].xs[t - 1][d];
                    v += q3_w2[l][d] * (float)cache.layers[l].xs[t][d];
                    int r = (int)std::lroundf(v);
                    if (r > 4) r = 4; if (r < -4) r = -4;
                    cache.layers[l].ys[t][d] = (int8_t)r;
                }
                for (int d = 0; d < D; ++d) {
                    int16_t sv = (int16_t)((int)cache.layers[l].ss[t][d] + (int)cache.layers[l].ys[t][d]);
                    cache.layers[l].ss[t + 1][d] = sv;
                    float hv = alpha[l][d] * (float)cache.layers[l].hs[t][d]
                             + (1.0f - alpha[l][d]) * (float)cache.layers[l].ys[t][d];
                    int hr = (int)std::lroundf(hv);
                    if (hr > 4) hr = 4; if (hr < -4) hr = -4;
                    cache.layers[l].hs[t + 1][d] = (int8_t)hr;
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
        for (size_t i = 0; i < p.size(); ++i) {
            m[i] = b1 * m[i] + (1 - b1) * g[i];
            v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
            p[i] -= lr * (m[i] / bc1) / (std::sqrt(v[i] / bc2) + eps);
        }
    }
    
    float train_step(const std::vector<int>& input, const std::vector<int>& target,
                     const std::vector<int>& mask, float lr) {
        int L = input.size();
        ForwardCache cache;
        forward(input, cache);
        
        int valid = 0;
        float total_loss = 0;
        std::vector<std::vector<float>> wh_g(V, std::vector<float>(D, 0));
        std::vector<std::vector<float>> ws_g(V, std::vector<float>(D, 0));
        std::vector<float> b_g(V, 0);
        std::vector<std::vector<float>> emb_g(V, std::vector<float>(D, 0));
        std::vector<std::vector<float>> q3w0_g(N_LAYERS, std::vector<float>(D, 0));
        std::vector<std::vector<float>> q3w1_g(N_LAYERS, std::vector<float>(D, 0));
        std::vector<std::vector<float>> q3w2_g(N_LAYERS, std::vector<float>(D, 0));
        std::vector<std::vector<float>> alpha_g(N_LAYERS, std::vector<float>(D, 0));
        std::vector<std::vector<std::vector<float>>> dy_l(N_LAYERS,
            std::vector<std::vector<float>>(L, std::vector<float>(D, 0)));
        std::vector<std::vector<std::vector<float>>> dh(N_LAYERS,
            std::vector<std::vector<float>>(L + 1, std::vector<float>(D, 0)));
        std::vector<std::vector<std::vector<float>>> ds(N_LAYERS,
            std::vector<std::vector<float>>(L + 1, std::vector<float>(D, 0)));
        
        for (int t = 0; t < L; ++t) {
            if (mask[t] == 0) continue;
            valid++;
            float mx = *std::max_element(cache.logits[t].begin(), cache.logits[t].end());
            float sm = 0;
            std::vector<float> probs(V);
            for (int v = 0; v < V; ++v) { probs[v] = std::exp(cache.logits[t][v] - mx); sm += probs[v]; }
            for (int v = 0; v < V; ++v) probs[v] /= sm;
            int tgt = target[t];
            total_loss += -std::log(std::max(probs[tgt], 1e-7f));
            const auto& h_now = cache.layers[N_LAYERS - 1].hs[t + 1];
            const auto& s_now = cache.layers[N_LAYERS - 1].ss[t + 1];
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
        if (valid == 0) return 0;
        
        for (int l = N_LAYERS - 1; l >= 0; --l) {
            for (int t = L - 1; t >= 0; --t) {
                for (int d = 0; d < D; ++d) {
                    ds[l][t][d] += ds[l][t + 1][d];
                    dy_l[l][t][d] += ds[l][t + 1][d] + dh[l][t + 1][d] * (1.0f - alpha[l][d]);
                    alpha_g[l][d] += dh[l][t + 1][d] * ((float)cache.layers[l].hs[t][d] - (float)cache.layers[l].ys[t][d]);
                    dh[l][t][d] += dh[l][t + 1][d] * alpha[l][d];
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
                    if (t >= 2) q3w0_g[l][d] += dy * (float)cache.layers[l].xs[t - 2][d];
                    if (t >= 1) q3w1_g[l][d] += dy * (float)cache.layers[l].xs[t - 1][d];
                    q3w2_g[l][d] += dy * (float)cache.layers[l].xs[t][d];
                }
            }
        }
        for (int t = 0; t < L; ++t) {
            for (int d = 0; d < D; ++d) {
                float dy = dy_l[0][t][d];
                if (t >= 2) emb_g[input[t - 2]][d] += dy * q3_w0[0][d];
                if (t >= 1) emb_g[input[t - 1]][d] += dy * q3_w1[0][d];
                emb_g[input[t]][d] += dy * q3_w2[0][d];
            }
        }
        
        for (int v = 0; v < V; ++v) {
            for (int d = 0; d < D; ++d) {
                wh_g[v][d] /= valid;
                ws_g[v][d] /= valid;
                emb_g[v][d] /= valid;
            }
            b_g[v] /= valid;
        }
        for (int l = 0; l < N_LAYERS; ++l)
            for (int d = 0; d < D; ++d) {
                q3w0_g[l][d] /= valid;
                q3w1_g[l][d] /= valid;
                q3w2_g[l][d] /= valid;
                alpha_g[l][d] /= valid;
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
        return total_loss / valid;
    }
};

int main() {
    std::cout << "  夭夭 Yaoyao v0.1 starting..." << std::endl;
    Vocab vocab;
    vocab.load("D:\\TaoVm\\vocab.txt");
    std::cout << "  V=" << vocab.size() << std::endl;
    
    const int D = 64;
    const int V = vocab.size();
    const int N_LAYERS = 4;
    
    std::cout << "  loading text..." << std::endl;
    std::ifstream ft("D:\\TaoVm\\tinystories_train.txt");
    std::stringstream ss; ss << ft.rdbuf();
    std::string text = ss.str();
    std::cout << "  Train chars: " << text.size() << std::endl;
    
    std::vector<int> tokens;
    tokens.reserve(text.size());
    for (char c : text) tokens.push_back(vocab.encode(c));
    std::cout << "  Tokens: " << tokens.size() << std::endl;
    
    const int SEQ_LEN = 64;
    const int N_WINDOWS = std::min((int)((tokens.size() - 1) / SEQ_LEN), 5000);
    std::cout << "  Building " << N_WINDOWS << " windows of length " << SEQ_LEN << std::endl;
    std::vector<std::vector<int>> inputs(N_WINDOWS), targets(N_WINDOWS), masks(N_WINDOWS);
    for (int i = 0; i < N_WINDOWS; ++i) {
        int start = i * SEQ_LEN;
        inputs[i].assign(SEQ_LEN, 0);
        targets[i].assign(SEQ_LEN, 0);
        masks[i].assign(SEQ_LEN, 0);
        for (int j = 0; j < SEQ_LEN; ++j) {
            inputs[i][j] = tokens[start + j];
            if (start + j + 1 < (int)tokens.size()) {
                targets[i][j] = tokens[start + j + 1];
                masks[i][j] = 1;
            }
        }
    }
    
    Yaoyao model(D, V, N_LAYERS);
    
    auto generate_text = [&](const std::string& prompt, int len) {
        std::vector<int> out;
        for (char c : prompt) out.push_back(vocab.encode(c));
        for (int step = 0; step < len; ++step) {
            Yaoyao::ForwardCache gen_cache;
            model.forward(out, gen_cache);
            const auto& last_h = gen_cache.layers[N_LAYERS - 1].hs[out.size()];
            const auto& last_s = gen_cache.layers[N_LAYERS - 1].ss[out.size()];
            int best = 0; float bs = -1e30f;
            for (int v = 0; v < V; ++v) {
                float L = model.bias[v];
                for (int d = 0; d < D; ++d) L += model.W_h[v][d] * (float)last_h[d] + model.W_s[v][d] * (float)last_s[d];
                if (L > bs) { bs = L; best = v; }
            }
            if (best == vocab.pad_id) break;
            out.push_back(best);
        }
        std::string result;
        for (int id : out) result += vocab.decode(id);
        return result;
    };
    
    std::cout << "\n  Initial generation:" << std::endl;
    std::cout << "    \"" << generate_text("Once upon a time", 100) << "\"" << std::endl;
    
    int epochs = 10;
    std::cout << "\n  Training " << epochs << " epochs, Adam lr=0.005..." << std::endl;
    auto t0 = std::chrono::steady_clock::now();
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<int> idx(N_WINDOWS);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), std::mt19937(epoch + 1));
        float total = 0;
        for (int i = 0; i < N_WINDOWS; ++i) {
            total += model.train_step(inputs[idx[i]], targets[idx[i]], masks[idx[i]], 0.005f);
            if (i % 500 == 0) {
                std::cout << "    epoch " << (epoch + 1) << " step " << i << "/" << N_WINDOWS
                          << " loss=" << std::fixed << std::setprecision(4) << total / (i + 1) << "\r" << std::flush;
            }
        }
        std::cout << "\n  Epoch " << (epoch + 1) << " avg_loss=" << std::fixed << std::setprecision(4)
                  << total / N_WINDOWS << std::endl;
        std::cout << "    \"" << generate_text("Once upon a time", 100) << "\"" << std::endl;
    }
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\n  Total: " << std::fixed << std::setprecision(1) << sec << "s, "
              << std::fixed << std::setprecision(0) << (N_WINDOWS * epochs) / sec << " windows/s" << std::endl;
    
    return 0;
}
