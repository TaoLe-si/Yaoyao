#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "tao_ternary.hpp"

// Fixed-width reference, V=1024. No attention, no external token tape.
template <size_t K, size_t D> struct SelfDecodingNode {
    static_assert(K > 0 && K <= 65535, "invalid horizon");
    static constexpr size_t Words = (10 * K + 63) / 64;
    std::array<uint64_t, Words> payload{};
    std::array<int8_t, D> trit{};
    uint32_t hash = 5381;
    uint16_t count = 0;
    static int8_t mod3(int x) {
        return tao::ternary::mod3(x);
    }
    static uint32_t undo_hash(uint32_t h, uint32_t x) {
        // (1+2^5)^-1 = sum_{j=0..6}(-2^5)^j mod2^32.
        uint32_t y = h - x;
        return y - (y << 5) + (y << 10) - (y << 15) + (y << 20) - (y << 25) + (y << 30);
    }
    template <class Code> void push(int token, const Code &code) {
        if (token < 0 || token >= 1024)
            throw std::invalid_argument("token");
        uint64_t carry = uint64_t(token);
        for (size_t i = 0; i < Words; ++i) {
            uint64_t next = payload[i] >> 54;
            payload[i] = (payload[i] << 10) | carry;
            carry = next;
        }
        if constexpr ((10 * K) % 64)
            payload[Words - 1] &= (uint64_t(1) << ((10 * K) % 64)) - 1;
        const auto *e = code(token);
        for (size_t d = 0; d < D; ++d)
            trit[d] = mod3(int(trit[d]) + e[d]);
        hash = (hash << 5) + hash + uint32_t(token);
        if (count < K)
            ++count;
    }
    // Call on a copy to leave live node intact. After pop this is a suffix cursor,
    // not the historical full-window node (older evicted tokens cannot reappear).
    template <class Code> int pop(const Code &code) {
        if (!count)
            throw std::out_of_range("history exhausted");
        int token = int(payload[0] & 1023);
        for (size_t i = 0; i < Words; ++i)
            payload[i] = (payload[i] >> 10) | (i + 1 < Words ? payload[i + 1] << 54 : 0);
        const auto *e = code(token);
        for (size_t d = 0; d < D; ++d)
            trit[d] = mod3(int(trit[d]) - e[d]);
        hash = undo_hash(hash, uint32_t(token));
        --count;
        return token;
    }
    // Explicit little-endian format; never serialize struct padding.
    std::vector<uint8_t> bytes() const {
        std::vector<uint8_t> b;
        auto put = [&](uint64_t v, size_t n) {
            for (size_t i = 0; i < n; ++i)
                b.push_back(uint8_t(v >> (8 * i)));
        };
        put(0x314E4453, 4);
        put(K, 2);
        put(D, 4);
        put(count, 2);
        put(hash, 4);
        for (auto w : payload)
            put(w, 8);
        for (auto t : trit)
            b.push_back(uint8_t(int(t) + 1));
        return b;
    }
    static SelfDecodingNode load(const std::vector<uint8_t> &b) {
        if (b.size() != 16 + Words * 8 + D)
            throw std::invalid_argument("length");
        size_t off = 0;
        auto get = [&](size_t n) {
            uint64_t v = 0;
            for (size_t i = 0; i < n; ++i)
                v |= uint64_t(b[off++]) << (8 * i);
            return v;
        };
        if (get(4) != 0x314E4453 || get(2) != K || get(4) != D)
            throw std::invalid_argument("format");
        SelfDecodingNode s;
        s.count = uint16_t(get(2));
        s.hash = uint32_t(get(4));
        if (s.count > K)
            throw std::invalid_argument("count");
        for (auto &w : s.payload)
            w = get(8);
        for (auto &t : s.trit) {
            auto v = get(1);
            if (v > 2)
                throw std::invalid_argument("trit");
            t = int8_t(int(v) - 1);
        }
        for (size_t bit = 10 * s.count; bit < Words * 64; ++bit)
            if ((s.payload[bit / 64] >> (bit % 64)) & 1)
                throw std::invalid_argument("unused bits");
        return s;
    }
};
