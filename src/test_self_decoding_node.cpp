#define main legacy_main
#include "yaoyao_v21_full.cpp"
#undef main
#include "self_decoding_node.hpp"
#include <cassert>
#include <iostream>

template <size_t K> void test(M &m) {
    using N = SelfDecodingNode<K, 256>;
    auto code = [&](int id) { return m.q1.trits.data() + (Q1::hash(id, 128) * 16) * 256; };
    std::mt19937 rng(73);
    size_t checks = 0;
    for (int kind = 0; kind < 4; ++kind) {
        N n;
        std::vector<int> ids;
        std::vector<N> states{n};
        for (size_t i = 0; i < 2 * K + 9; ++i) {
            int x = kind == 0   ? 0
                    : kind == 1 ? 1023
                    : kind == 2 ? int(i % 1024)
                                : int(rng() % 1024);
            ids.push_back(x);
            n.push(x, code);
            states.push_back(n);
            auto saved = n.bytes();
            auto cursor = N::load(saved);
            assert(cursor.bytes() == saved);
            for (size_t j = 0; j < std::min(K, ids.size()); ++j) {
                assert(cursor.pop(code) == ids[ids.size() - 1 - j]);
                const auto &expected = states[ids.size() - 1 - j];
                assert(cursor.trit == expected.trit && cursor.hash == expected.hash);
                ++checks;
            }
            try {
                cursor.pop(code);
                assert(false);
            } catch (const std::out_of_range &) {
            }
            assert(n.bytes() == saved);
        }
        auto saved = n.bytes();
        try {
            n.push(1024, code);
            assert(false);
        } catch (const std::invalid_argument &) {
        }
        assert(saved == n.bytes());
        auto corrupt = saved;
        corrupt.pop_back();
        try {
            N::load(corrupt);
            assert(false);
        } catch (const std::invalid_argument &) {
        }
        corrupt = saved;
        corrupt.back() = 3;
        try {
            N::load(corrupt);
            assert(false);
        } catch (const std::invalid_argument &) {
        }
    }
    std::cout << "K=" << K << " exact comparisons=" << checks << " sizeof(node)=" << sizeof(N)
              << " PASS\n";
}
int main() {
    M m;
    std::mt19937 rng(42);
    m.init(rng, 1024);
    if (!m.load("tao_fixed_step50002.bin"))
        return 2;
    test<1>(m);
    test<4>(m);
    test<7>(m);
    test<16>(m);
    test<32>(m);
    test<64>(m);
    for (uint32_t x = 0; x < 1024; ++x)
        for (uint32_t h : {0u, 1u, 5381u, 0xffffffffu, 0x80000000u})
            assert((SelfDecodingNode<16, 256>::undo_hash(h, x) == uint32_t((h - x) * 0x3e0f83e1u)));
    std::cout << "Production Q1 integration, cross-word packing, in-memory serialization and "
                 "shift-add inverse PASS\n";
}
