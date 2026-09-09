#pragma once
#include "language_data_contract.hpp"
#include <cstdint>
namespace tao::data {
// FNV-1a deterministic grouping fingerprint; non-cryptographic, not a dedup proof.
inline uint64_t fingerprint(const std::vector<Message>&m){uint64_t h=14695981039346656037ull;auto byte=[&](unsigned char c){h^=c;h*=1099511628211ull;};auto number=[&](uint64_t v){for(int i=0;i<8;++i)byte((v>>(8*i))&255);};number(m.size());for(const auto&x:m){byte(x.assistant?1:0);number(x.utf8.size());for(unsigned char c:x.utf8)byte(c);}return h;}
enum class Partition { train,validation,test };
inline Partition partition(uint64_t h){auto bucket=h%10000;return bucket<9800?Partition::train:bucket<9900?Partition::validation:Partition::test;}
}
