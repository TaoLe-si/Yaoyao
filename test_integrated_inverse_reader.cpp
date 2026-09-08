#define main legacy_main
#include "yaoyao_v21_full.cpp"
#undef main
#include "inverse_reader.hpp"
#include <cassert>
#include <iostream>
#include <chrono>
volatile double checksum = 0;
using Node = SelfDecodingNode<16, 256>;
using Reader = InverseReader<16, 256>;
// Reuses production weights and exact gate/up/output formulas, incremental context.
void head(M &m, const std::array<float, 256> &trit, uint32_t hash, int prev,
          std::vector<float> &z) {
    float s[320], h[320];
    for (int d = 0; d < 256; ++d)
        s[d] = trit[d];
    extract_hash_features(hash, s + 256);
    for (int i = 0; i < 320; ++i) {
        float g = 0, u = 0;
        for (int j = 0; j < 320; ++j) {
            g += m.W_sgl_gate[i * 320 + j] * s[j];
            u += m.W_sgl_up[i * 320 + j] * s[j];
        }
        h[i] = g / (1 + std::exp(-g)) * u;
    }
    for (int v = 0; v < 1024; ++v) {
        float x = m.Wbi[prev * 1024 + v];
        for (int i = 0; i < 320; ++i)
            x += m.W_sgl_out[v * 320 + i] * h[i];
        z[v] = x;
    }
}
#ifndef INVERSE_READER_HEAD_ONLY
int main() {
    M m;
    std::mt19937 rng(42);
    m.init(rng, 1024);
    if (!m.load("experiments/d256_nibble64_baseline/fresh.bin"))
        return 2;
    auto code = [&](int id) { return m.q1.trits.data() + Q1::hash(id, 128) * 16 * 256; };
    std::vector<int> ids(64);
    for (auto &x : ids)
        x = int(rng() % 1024);
    std::vector<float> z(1024), all(64 * 1024), p(64 * 1024), s(64 * 320), h(64 * 320), g(64 * 320),
        u(64 * 320);
    uint32_t hash;
    assert(!yao_forward_raw(m, ids.data(), 64, 1, 64, 0, all.data(), p.data(), s.data(), h.data(),
                            g.data(), u.data(), &hash));
    Node n;
    Reader reader;
    float error = 0;
    std::vector<Node> prefixes{n};
    for (int t = 0; t < 64; ++t) {
        n.push(ids[t], code);
        prefixes.push_back(n);
        head(m, reader.read(n, code), n.hash, t ? ids[t - 1] : 0, z);
        for (int v = 0; v < 1024; ++v)
            error = std::max(error, std::abs(z[v] - all[t * 1024 + v]));
    }
    assert(error < 1e-5);
    std::cout << "Identity-reader vs production full-prefix logits max_abs=" << error << " PASS\n";
    for (auto &row : reader.selectors)
        for (auto &x : row)
            x = int8_t(int(rng() % 3) - 1);
    auto actual = reader.read(n, code);
    for (int d = 0; d < 256; ++d) {
        float expected = 0;
        for (int k = 0; k <= 16; ++k)
            expected += reader.selectors[k][d] * prefixes[64 - k].trit[d];
        assert(actual[d] == expected);
    }
    std::cout << "Nonidentity selectors match saved-prefix oracle exactly PASS\n";
    // Same incremental head and input path in both timing variants, warmed7 medians.
    for (int mode = 0; mode < 2; ++mode) {
        std::vector<double> times;
        for (int r = 0; r < 7; ++r) {
            Node live;
            auto begin = std::chrono::steady_clock::now();
            for (int t = 0; t < 2048; ++t) {
                live.push(ids[t % 64], code);
                std::array<float, 256> f{};
                if (mode)
                    f = reader.read(live, code);
                else
                    for (int d = 0; d < 256; ++d)
                        f[d] = live.trit[d];
                head(m, f, live.hash, t ? ids[(t - 1) % 64] : 0, z);
                checksum += z[t % 1024];
            }
            times.push_back(
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin)
                    .count() /
                2048);
        }
        std::sort(times.begin(), times.end());
        std::cout << (mode ? "inverse_reader" : "direct_state")
                  << " incremental_us_per_token=" << times[3] << "\n";
    }
}
#endif
