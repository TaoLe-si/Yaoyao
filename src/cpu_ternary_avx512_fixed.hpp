#pragma once
#include "cpu_ternary_avx2.hpp"
// Fixed ISA type: no mutable selector; definition lives in the AVX512-only TU.
struct CpuTernaryRows512Fixed : CpuTernaryRows {
 using CpuTernaryRows::CpuTernaryRows;
 float dot(size_t row,const float* x) const;
};
