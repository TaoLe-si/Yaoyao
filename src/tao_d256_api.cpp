#include <array>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <cctype>
#include <numeric>
#include <cmath>

#include "self_decoding_node.hpp"
#include "native_reader_checkpoint.hpp"
#include "tao_state_gates.hpp"

struct Q1 {
    int B = 128, K = 16, D = 256;
    std::vector<int8_t> trits;
    static int hash(int id, int B_) {
        uint64_t x = (uint32_t)id * 2654435761u;
        x = (x >> 16) ^ x;
        return int(x % (uint64_t)B_);
    }
    const int8_t* code(int id) const {
        return trits.data() + size_t(hash(id, B)) * K * D;
    }
    bool load(int B_, int K_, int D_, const std::vector<int8_t>& tr) {
        B = B_; K = K_; D = D_;
        if ((int)tr.size() != B * K * D) return false;
        trits = tr;
        return true;
    }
};

struct Vocab {
    std::unordered_map<std::string,int> w2i;
    std::vector<std::string> i2w;
    int pad_id = 0, unk_id = 1, eos_id = 2;
    void build(const std::string& text, int max_size) {
        i2w.clear(); w2i.clear();
        i2w.push_back("<pad>"); i2w.push_back("<unk>"); i2w.push_back("<eos>");
        w2i["<pad>"] = 0; w2i["<unk>"] = 1; w2i["<eos>"] = 2;
        // Match training tokenizer ordering, including frequency ties.
        std::map<std::string,int> freq_local;
        std::string cur;
        auto flush = [&](){ if (cur.empty()) return; ++freq_local[cur]; cur.clear(); };
        for (size_t i = 0; i < text.size(); ++i) {
            unsigned char ch = (unsigned char)text[i];
            if (std::isspace(ch)) { flush(); continue; }
            if (std::ispunct(ch)) { flush(); std::string one(1, (char)ch); ++freq_local[one]; continue; }
            cur.push_back((char)ch);
        }
        flush();
        std::vector<std::pair<std::string,int>> items;
        for (auto& kv : freq_local) items.push_back(kv);
        std::sort(items.begin(), items.end(), [](auto& a, auto& b){ return a.second > b.second; });
        for (auto& p : items) {
            if ((int)i2w.size() >= max_size) break;
            if (w2i.find(p.first) != w2i.end()) continue;
            int id = (int)i2w.size();
            w2i[p.first] = id;
            i2w.push_back(p.first);
        }
    }
    int encode_word(const std::string& w) const {
        auto it = w2i.find(w);
        return it == w2i.end() ? unk_id : it->second;
    }
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        std::string cur;
        auto flush = [&](){ if (cur.empty()) return; ids.push_back(encode_word(cur)); cur.clear(); };
        for (size_t i = 0; i < text.size(); ++i) {
            unsigned char ch = (unsigned char)text[i];
            if (std::isspace(ch)) { flush(); continue; }
            if (std::ispunct(ch)) {
                flush();
                std::string one(1, (char)ch);
                ids.push_back(encode_word(one));
                continue;
            }
            cur.push_back((char)std::tolower(ch));
        }
        flush();
        return ids;
    }
};

struct FixedBoundariesModel {
    static constexpr int V = 1024, D = 256, HASH = 64, HIDDEN = D + HASH, B = 128, K = 16;
    int step = 0;
    std::vector<int8_t> q1;
    std::vector<float> W_sgl_gate, b_sgl_gate;
    std::vector<float> W_sgl_up,   b_sgl_up;
    std::vector<float> W_sgl_out;
    std::vector<float> Wbi;

    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        int magic; f.read((char*)&magic, 4);
        int version; f.read((char*)&version, 4);
        int v; f.read((char*)&v, 4);
        if (magic != 0x59414F59 || version != 4 || v != V) return false;
        auto skip = [&](size_t n){ f.ignore((std::streamsize)n); };
        skip(size_t(V) * D * 4 * 3);
        skip(size_t(V) * HASH * 4 * 3);
        Wbi.assign(V * V, 0);
        f.read((char*)Wbi.data(), Wbi.size() * sizeof(float));
        skip(size_t(V) * V * sizeof(float) * 2);
        skip(5 * 2 * D * 4 * 3);
        skip(2 * D * D * 4 * 3 + 2 * D * 4 * 3);
        skip(2 * D * D * 4 * 3 + 2 * D * 4 * 3);
        q1.assign(B * K * D, 0);
        f.read((char*)q1.data(), q1.size());
        skip(B * K * D * 4 * 2);
        int q1_step; f.read((char*)&q1_step, 4);
        f.read((char*)&step, 4);
        W_sgl_gate.assign(HIDDEN * HIDDEN, 0);
        b_sgl_gate.assign(HIDDEN, 0);
        W_sgl_up.assign(HIDDEN * HIDDEN, 0);
        b_sgl_up.assign(HIDDEN, 0);
        W_sgl_out.assign(V * HIDDEN, 0);
        f.read((char*)W_sgl_gate.data(), W_sgl_gate.size() * 4);
        f.read((char*)b_sgl_gate.data(), b_sgl_gate.size() * 4);
        f.read((char*)W_sgl_up.data(), W_sgl_up.size() * 4);
        f.read((char*)b_sgl_up.data(), b_sgl_up.size() * 4);
        f.read((char*)W_sgl_out.data(), W_sgl_out.size() * 4);
        return (bool)f;
    }
};

struct ServerCore {
    FixedBoundariesModel model;
    std::vector<int8_t> gates;
    std::vector<int8_t> reader_sel; // 17*256 ternary selectors from RTS1
    Vocab vocab;
    Q1 q1;
    std::vector<uint8_t> state_gates = std::vector<uint8_t>(1024 * 15, 0);

    void load_state_gates(const char *path, const char *model_path, const char *parent_path,
                          const char *reader_path, const char *tcg_path,
                          const char *train_path, const char *heldout_path) {
        FixedBoundariesModel parent;
        if (!parent.load(parent_path)) throw std::runtime_error("state gate parent load");
        tds::validate_transfer(model.q1.data(), parent.q1.data(), model.q1.size(),
                              model.Wbi.data(), parent.Wbi.data(), model.Wbi.size() * sizeof(float));
        tds::Dependencies dep;
        dep.model = file_sha256(model_path); dep.parent = file_sha256(parent_path);
        dep.reader = file_sha256(reader_path); dep.tcg = file_sha256(tcg_path);
        dep.train = file_sha256(train_path); dep.heldout = file_sha256(heldout_path);
        dep.q1 = tds::memory_sha256(model.q1.data(), model.q1.size());
        dep.wbi = tds::memory_sha256(model.Wbi.data(), model.Wbi.size() * sizeof(float));
        auto frozen = tds::load_tcg(tcg_path, dep.parent, dep.train, dep.reader);
        auto selectors = tds::load_reader(reader_path, dep.train, dep.heldout);
        if (frozen != gates || selectors != reader_sel) throw std::runtime_error("loaded dependency mismatch");
        auto artifact = tds::Artifact::load(path, dep);
        tds::validate_union(artifact.codes, frozen);
        state_gates = std::move(artifact.codes);
    }

    bool load_reader(const std::string& path, const std::string& tokens_path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f || f.tellg() != std::streamoff(4484)) return false;
        f.seekg(0);
        std::vector<unsigned char> b(4484);
        if (!f.read((char*)b.data(), 4484)) return false;
        auto get=[&](int o){uint32_t x=0;for(int j=0;j<4;++j)x|=uint32_t(b[o+j])<<(8*j);return x;};
        if (get(0)!=0x31535452u||get(4)!=1||get(8)!=16||get(12)!=256) return false;
        if (get(4480)!=reader_crc32(b.data(),4480)) return false;
        if (!tokens_path.empty()) {
            auto sha=file_sha256(tokens_path.c_str());
            if (!std::equal(sha.begin(),sha.end(),b.begin()+64)) return false;
        }
        reader_sel.assign(17*256,0);
        for (int k=0;k<=16;++k) for (int d=0;d<256;++d) {
            if (b[128+k*256+d]>2) return false;
            reader_sel[k*256+d]=int8_t(b[128+k*256+d])-1;
        }
        return true;
    }

    bool load(const std::string& model_path,
              const std::string& gate_path,
              const std::string& reader_path,
              const std::string& text_path,
              const std::string& tokens_path) {
        if (!model.load(model_path)) return false;
        if (!q1.load(128, 16, 256, model.q1)) return false;

        std::ifstream gf(gate_path, std::ios::binary | std::ios::ate);
        if (!gf || gf.tellg() != std::streamoff(15480)) return false;
        gf.seekg(0);
        std::vector<unsigned char> pack(15480);
        if (!gf.read((char*)pack.data(), 15480)) return false;
        auto get = [&](int o) {
            uint32_t x = 0;
            for (int j = 0; j < 4; ++j) x |= uint32_t(pack[o + j]) << (j * 8);
            return x;
        };
        if (get(0) != 0x31474354u || get(4) != 1 || get(8) != 1024 || get(12) != 256 || get(16) != 16) return false;
        if (get(15476) != reader_crc32(pack.data(), 15476)) return false;
        gates.assign(1024 * 15, 0);
        for (int i = 0; i < 1024 * 15; ++i) {
            int q = int(pack[116 + i]) - 8;
            if (q != 0 && std::abs(q) != 1 && std::abs(q) != 2 && std::abs(q) != 4 && std::abs(q) != 8) return false;
            gates[i] = (int8_t)q;
        }
        for (int id = 0; id < 1024; ++id) {
            int count = 0, sum = 0;
            for (int k = 0; k < 15; ++k) {
                int q = gates[id * 15 + k];
                count += q != 0;
                sum += std::abs(q);
            }
            if (count > 2 || sum > 8) return false;
        }

        std::ifstream tf(text_path, std::ios::binary);
        std::stringstream ss; ss << tf.rdbuf();
        std::string text = ss.str();
        vocab.build(text.substr(0, std::min<size_t>(text.size(), 5000000)), 1024);
        if (!load_reader(reader_path, tokens_path)) return false;
        return true;
    }

    static int sample_top_p(const std::vector<float>& logits, float T, float p, std::mt19937& rng) {
        const int V = (int)logits.size();
        if (V == 0 || !std::isfinite(T) || T < 0 || !std::isfinite(p) || p <= 0 || p > 1)
            throw std::invalid_argument("invalid sampling parameters");
        if (T == 0)
            return int(std::max_element(logits.begin(), logits.end()) - logits.begin());
        std::vector<int> idx(V);
        std::iota(idx.begin(), idx.end(), 0);
        float mx = logits[0];
        for (int v = 1; v < V; ++v) if (logits[v] > mx) mx = logits[v];
        std::vector<float> pr(V);
        double sum = 0;
        for (int v = 0; v < V; ++v) { pr[v] = std::exp((logits[v] - mx) / T); sum += pr[v]; }
        for (int v = 0; v < V; ++v) pr[v] = float(pr[v] / sum);
        std::sort(idx.begin(), idx.end(), [&](int a, int b){ return pr[a] > pr[b]; });
        float cum = 0; int nuc = V;
        for (int i = 0; i < V; ++i) { cum += pr[idx[i]]; if (cum >= p) { nuc = i + 1; break; } }
        std::uniform_real_distribution<float> ud(0, 1);
        float target = ud(rng) * cum, acc = 0;
        int chosen = idx[nuc - 1];
        for (int i = 0; i < nuc; ++i) { acc += pr[idx[i]]; if (acc >= target) { chosen = idx[i]; break; } }
        return chosen;
    }

    void forward_window(const std::vector<int>& win, std::vector<float>& logits) {
        const int V = FixedBoundariesModel::V, D = FixedBoundariesModel::D;
        const int HASH = FixedBoundariesModel::HASH, HIDDEN = FixedBoundariesModel::HIDDEN;
        const int SEQ = 64;
        if (win.size() != SEQ) throw std::invalid_argument("expected 64-token window");
        for (int id : win) if (id < 0 || id >= V) throw std::invalid_argument("invalid token ID");
        logits.resize(V);
        std::vector<float> state(HIDDEN), gate_pre(HIDDEN), up_pre(HIDDEN), hidden(HIDDEN), hist(V), mixed(D);
        SelfDecodingNode<16, 256> node;
        for (int t = 0; t < SEQ; ++t) node.push(win[t], [&](int x){ return q1.code(x); });
        // Reader mixture over exactly recoverable chain states (inverse_reader.hpp semantics).
        std::fill(mixed.begin(), mixed.end(), 0.0f);
        auto view = node;
        for (int k = 0;; ++k) {
            float scale = tds::distance_scale(state_gates.data(), win.back(), k);
            for (int d = 0; d < D; ++d) {
                int w = reader_sel[k * D + d];
                if (w == 1) mixed[d] += scale * view.trit[d];
                else if (w == -1) mixed[d] -= scale * view.trit[d];
            }
            if (!view.count) break;
            view.pop([&](int x){ return q1.code(x); });
        }
        for (int d = 0; d < D; ++d) state[d] = mixed[d];
        for (int i = 0; i < HASH; ++i) {
            int nibble_idx = i & 7;
            uint32_t nib = (node.hash >> (4 * nibble_idx)) & 0xFu;
            state[D + i] = (float)nib / 15.0f - 0.5f;
        }
        for (int h = 0; h < HIDDEN; ++h) {
            float acc = 0;
            const float* row = model.W_sgl_gate.data() + h * HIDDEN;
            for (int k = 0; k < HIDDEN; ++k) acc += row[k] * state[k];
            gate_pre[h] = acc + model.b_sgl_gate[h];
        }
        for (int h = 0; h < HIDDEN; ++h) {
            float acc = 0;
            const float* row = model.W_sgl_up.data() + h * HIDDEN;
            for (int k = 0; k < HIDDEN; ++k) acc += row[k] * state[k];
            up_pre[h] = acc + model.b_sgl_up[h];
        }
        for (int h = 0; h < HIDDEN; ++h) {
            float g = gate_pre[h];
            g = g / (1.0f + std::exp(-g));
            hidden[h] = g * up_pre[h];
        }
        int id_now = win[SEQ - 1];
        int prev = win[SEQ - 2];
        std::fill(hist.begin(), hist.end(), 0.0f);
        for (int k = 1; k <= 15; ++k) {
            int q = gates[id_now * 15 + (k - 1)];
            if (!q) continue;
            float scale = float(q) * (1.0f / 32.0f);
            const float* row = model.Wbi.data() + win[SEQ - 1 - k] * V;
            for (int v = 0; v < V; ++v) hist[v] += scale * row[v];
        }
        for (int v = 0; v < V; ++v) {
            float lv = model.Wbi[prev * V + v] + hist[v];
            const float* row = model.W_sgl_out.data() + v * HIDDEN;
            float acc = 0;
            for (int h = 0; h < HIDDEN; ++h) acc += row[h] * hidden[h];
            logits[v] = lv + acc;
        }
    }

    void generate(const std::string& prompt, int max_tokens, float T, float p,
                  const std::function<void(double, double, const std::string&, int)>& cb_step) {
        std::vector<int> ids = vocab.encode(prompt);
        if (ids.empty()) ids.push_back(vocab.unk_id);
        while (!ids.empty() && ids.back() == vocab.eos_id) ids.pop_back();
        if (ids.empty()) ids.push_back(vocab.unk_id);
        std::mt19937 rng(unsigned(std::chrono::steady_clock::now().time_since_epoch().count()) ^ 0x9e3779b9u);
        const int V = FixedBoundariesModel::V;
        const int D = FixedBoundariesModel::D;
        const int HASH = FixedBoundariesModel::HASH;
        const int HIDDEN = FixedBoundariesModel::HIDDEN;
        const int SEQ = 64;

        std::vector<float> logits(V);
        auto total_begin = std::chrono::steady_clock::now();
        auto step_begin = total_begin;

        for (int step = 0; step < max_tokens; ++step) {
            // Sliding 64-token window with per-window fresh state, exactly as training windows.
            std::vector<int> win(SEQ, vocab.pad_id);
            size_t L = ids.size() < (size_t)SEQ ? ids.size() : (size_t)SEQ;
            std::copy(ids.end() - L, ids.end(), win.begin() + (SEQ - (int)L));
            forward_window(win, logits);
            // Production repetition penalty over the last six tokens.
            for (int back = 0; back < 6 && back < (int)ids.size(); ++back) {
                int tk = ids[ids.size() - 1 - back];
                if (tk >= 2 && tk < V) logits[tk] -= 3.0f * std::pow(0.65f, (float)back);
            }
            logits[vocab.eos_id] = -1e9f;
            logits[vocab.unk_id] = -1e9f;
            int next = sample_top_p(logits, T, p, rng);
            std::string word = (next >= 0 && next < (int)vocab.i2w.size()) ? vocab.i2w[next] : std::string();
            for (auto& ch : word) if (ch == ' ' || ch == '\t') ch = (char)0x01;
            ids.push_back(next);

            auto step_end = std::chrono::steady_clock::now();
            double step_ms = std::chrono::duration<double, std::milli>(step_end - step_begin).count();
            double total_ms = std::chrono::duration<double, std::milli>(step_end - total_begin).count();
            cb_step(step_ms, total_ms, word, next);
            step_begin = step_end;
            if (next == vocab.eos_id) break;
        }
    }
};

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    if (argc < 5) {
        std::fprintf(stderr, "Usage: tao_d256_api.exe --server <model.bin> <gates.tcg> <reader.rts1> <vocab_text> [train_tokens.bin]\n");
        return 1;
    }
    ServerCore core;
    if (argc < 6) {
        std::fprintf(stderr, "load failed: need model gates reader vocab_text [tokens]\n");
        return 2;
    }
    if (!core.load(argv[2], argv[3], argv[4], argv[5], argc > 6 ? argv[6] : "")) {
        std::fprintf(stderr, "load failed\n");
        return 2;
    }
    if (argc > 7) {
        if (argc != 10) return 2;
        try {
            core.load_state_gates(argv[7], argv[2], argv[8], argv[4], argv[3], argv[6], argv[9]);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "state gate load failed: %s\n", e.what());
            return 2;
        }
    }
    std::fprintf(stdout, "Server ready vocab=%zu step=%d state_gates=%zu\n", core.vocab.i2w.size(), core.model.step,
                 size_t(std::count_if(core.state_gates.begin(), core.state_gates.end(), [](uint8_t c){return c != 0;})));
    std::fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "QUIT") break;
        // Field-count-aware protocol:
        //   4 fields: prompt|max_tokens|T|p            (direct/testing)
        //   5 fields: prompt|stream|max_tokens|T|p     (yaoyao_api.py)
        std::vector<std::string> fields;
        size_t pos = 0;
        while (true) {
            size_t nxt = line.find('|', pos);
            if (nxt == std::string::npos) { fields.push_back(line.substr(pos)); break; }
            fields.push_back(line.substr(pos, nxt - pos));
            pos = nxt + 1;
        }
        if (fields.size() != 4 && fields.size() != 5) {
            std::fprintf(stdout, "ERROR bad request line: [%s] fields=%zu\n", line.c_str(), fields.size());
            std::fflush(stdout);
            continue;
        }
        std::string prompt = fields[0];
        int max_tokens; float T, p;
        if (fields.size() >= 5) {
            max_tokens = std::atoi(fields[2].c_str());
            T = (float)std::atof(fields[3].c_str());
            p = (float)std::atof(fields[4].c_str());
        } else {
            max_tokens = std::atoi(fields[1].c_str());
            T = (float)std::atof(fields[2].c_str());
            p = (float)std::atof(fields[3].c_str());
        }
        if (max_tokens < 1 || max_tokens > 4096 || !std::isfinite(T) || T < 0 ||
            !std::isfinite(p) || p <= 0 || p > 1) {
            std::fprintf(stdout, "ERROR invalid generation parameters\n");
            continue;
        }
        core.generate(prompt, max_tokens, T, p, [](double step_ms, double total_ms, const std::string& w, int id){
            std::fprintf(stdout, "TOKEN %d %.3f %.3f %s\n", id, step_ms, total_ms, w.c_str());
            std::fflush(stdout);
        });
        std::fprintf(stdout, "DONE\n");
        std::fflush(stdout);
    }
    return 0;
}