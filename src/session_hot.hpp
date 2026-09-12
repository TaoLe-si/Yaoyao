#pragma once
#include "dual_state_cpu.hpp"
#include <immintrin.h>
#include <vector>
namespace tao::dual {
// Software-keep the fixed session slice in L2. Vocab-head weights are streamed
// with NTA on the GEMV side so they do not evict this working set.
inline void prefetch_bytes_l2(const void* p, size_t n) {
    const char* c = static_cast<const char*>(p);
    const char* e = c + n;
    for (; c < e; c += 64) _mm_prefetch(c, _MM_HINT_T1);
}
inline void prefetch_layer_l2(const LayerState& z) {
    const char* a = reinterpret_cast<const char*>(z.s.data());
    const char* b = reinterpret_cast<const char*>(z.m.data() + z.m.size());
    if (b > a) prefetch_bytes_l2(a, size_t(b - a));
    else {
        prefetch_bytes_l2(z.s.data(), z.s.size() * sizeof(float));
        prefetch_bytes_l2(z.m.data(), z.m.size() * sizeof(float));
    }
}
inline void prefetch_session_l2(const std::vector<LayerState>& st) {
    for (const auto& z : st) prefetch_layer_l2(z);
}
}
