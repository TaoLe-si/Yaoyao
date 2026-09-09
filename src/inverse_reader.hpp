#pragma once
#include "self_decoding_node.hpp"
#include <array>
// Learnable per-distance ternary selectors. No attention, KV, or state multiply.
// Exact discrete coordinate fitting is available separately; no STE derivative claim.
template <size_t K, size_t D> struct InverseReader {
    std::array<std::array<int8_t, D>, K + 1> selectors{};
    InverseReader() {
        selectors[0].fill(1);
    } // exact current-state identity initialization
    template <class Code>
    std::array<float, D> read(const SelfDecodingNode<K, D> &live, const Code &code) const {
        std::array<float, D> out{};
        auto c = live;
        for (size_t distance = 0;; ++distance) {
            for (size_t d = 0; d < D; ++d) {
                auto w = selectors[distance][d];
                if (w == 1)
                    out[d] += c.trit[d];
                else if (w == -1)
                    out[d] -= c.trit[d];
                else if (w != 0)
                    throw std::invalid_argument("selector not ternary");
            }
            if (!c.count)
                break;
            c.pop(code);
        }
        return out;
    }
};
