#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#include <array>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#pragma comment(lib, "bcrypt.lib")
inline std::array<unsigned char, 32> file_sha256(const char *path) {
    BCRYPT_ALG_HANDLE alg = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    auto check = [](NTSTATUS s) {
        if (s < 0)
            throw std::runtime_error("BCrypt SHA256 failure");
    };
    check(BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0));
    try {
        check(BCryptCreateHash(alg, &hash, nullptr, 0, nullptr, 0, 0));
        std::ifstream in(path, std::ios::binary);
        if (!in)
            throw std::runtime_error("hash input missing");
        char buffer[65536];
        while (in) {
            in.read(buffer, sizeof(buffer));
            auto n = in.gcount();
            if (n)
                check(BCryptHashData(hash, (PUCHAR)buffer, ULONG(n), 0));
        }
        if (!in.eof())
            throw std::runtime_error("hash read failed");
        std::array<unsigned char, 32> result;
        check(BCryptFinishHash(hash, result.data(), 32, 0));
        BCryptDestroyHash(hash);
        BCryptCloseAlgorithmProvider(alg, 0);
        return result;
    } catch (...) {
        if (hash)
            BCryptDestroyHash(hash);
        BCryptCloseAlgorithmProvider(alg, 0);
        throw;
    }
}
// Integrity check for accidental corruption, not an authentication signature.
inline uint32_t reader_crc32(const unsigned char *data, size_t size) {
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
        }
    }
    return ~crc;
}

struct BoundReader {
    std::array<unsigned char, 32> model{}, corpus{};
    uint32_t next = 0;
    std::vector<int8_t> a = std::vector<int8_t>(17 * 256, 0);
    BoundReader() {
        for (int d = 0; d < 256; ++d)
            a[d] = 1;
    }
    std::vector<unsigned char> bytes() const {
        std::vector<unsigned char> b;
        auto put = [&](uint32_t x) {
            for (int j = 0; j < 4; ++j)
                b.push_back((x >> (8 * j)) & 255);
        };
        // IDB3: fixed first65-token contract, coordinate cursor and trailing CRC32.
        for (uint32_t x : {0x33424449u, 3u, 16u, 256u, 1024u, 64u, 1u, next})
            put(x);
        b.insert(b.end(), model.begin(), model.end());
        b.insert(b.end(), corpus.begin(), corpus.end());
        for (auto x : a) {
            if (x < -1 || x > 1)
                throw std::runtime_error("invalid selector");
            b.push_back(x + 1);
        }
        put(reader_crc32(b.data(), b.size()));
        return b;
    }
    static BoundReader load(const char *path, const std::array<unsigned char, 32> &model,
                            const std::array<unsigned char, 32> &corpus) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f || f.tellg() != std::streamoff(100 + 17 * 256))
            throw std::runtime_error("checkpoint size");
        f.seekg(0);
        std::vector<unsigned char> b(100 + 17 * 256);
        if (!f.read((char *)b.data(), b.size()))
            throw std::runtime_error("checkpoint read");
        uint32_t saved_crc = 0;
        for (int j = 0; j < 4; ++j) {
            saved_crc |= uint32_t(b[b.size() - 4 + j]) << (8 * j);
        }
        if (saved_crc != reader_crc32(b.data(), b.size() - 4)) {
            throw std::runtime_error("checkpoint integrity");
        }
        size_t off = 0;
        auto get = [&]() {
            uint32_t x = 0;
            for (int j = 0; j < 4; ++j)
                x |= uint32_t(b[off++]) << (8 * j);
            return x;
        };
        for (uint32_t expected : {0x33424449u, 3u, 16u, 256u, 1024u, 64u, 1u})
            if (get() != expected)
                throw std::runtime_error("checkpoint contract");
        BoundReader r;
        r.next = get();
        for (auto &x : r.model)
            x = b[off++];
        for (auto &x : r.corpus)
            x = b[off++];
        if (r.model != model || r.corpus != corpus)
            throw std::runtime_error("model/corpus binding mismatch");
        for (auto &x : r.a) {
            auto v = b[off++];
            if (v > 2)
                throw std::runtime_error("checkpoint selector");
            x = int8_t(v) - 1;
        }
        return r;
    }
    void save_new(const char *path) const {
        auto b = bytes();
        HANDLE h = CreateFileA(path, GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL,
                               nullptr);
        if (h == INVALID_HANDLE_VALUE)
            throw std::runtime_error("refuse overwrite or create failure");
        DWORD written = 0;
        bool ok = WriteFile(h, b.data(), DWORD(b.size()), &written, nullptr) && written == b.size();
        if (ok)
            ok = FlushFileBuffers(h) != 0;
        CloseHandle(h);
        if (!ok)
            throw std::runtime_error("checkpoint write failure");
    }
};