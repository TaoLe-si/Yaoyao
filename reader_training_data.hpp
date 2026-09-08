#pragma once
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

// Each sequence includes its own next-token target. No per-token resampling.
struct ReaderBatch {
    std::vector<int> input;
    std::vector<int> target;
};
inline ReaderBatch read_reader_batch(const char *path, uint64_t offset, unsigned sequences) {
    if (sequences == 0 || sequences > 64) {
        throw std::invalid_argument("sequences must be 1..64");
    }
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
        throw std::runtime_error("missing corpus");
    const auto bytes = in.tellg();
    in.seekg(0);
    uint32_t count = 0;
    if (!in.read(reinterpret_cast<char *>(&count), 4) ||
        uint64_t(bytes) != 4 + uint64_t(count) * 4) {
        throw std::runtime_error("invalid corpus length");
    }
    const uint64_t needed = uint64_t(sequences) * 65;
    if (offset > count || needed > count - offset) {
        throw std::out_of_range("corpus batch bounds");
    }
    std::vector<int> tokens(needed);
    in.seekg(std::streamoff(4 + 4 * offset));
    if (!in.read(reinterpret_cast<char *>(tokens.data()), needed * 4)) {
        throw std::runtime_error("truncated batch");
    }
    for (int token : tokens) {
        if (token < 0 || token >= 1024)
            throw std::runtime_error("invalid token");
    }
    ReaderBatch batch;
    for (unsigned b = 0; b < sequences; ++b) {
        for (unsigned t = 0; t < 64; ++t) {
            batch.input.push_back(tokens[b * 65 + t]);
            batch.target.push_back(tokens[b * 65 + t + 1]);
        }
    }
    return batch;
}
