#pragma once
#include "cpu_ternary_avx2.hpp"
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstdint>
// 固化内核：2-bit 打包常驻 + AVX2 寄存器内展开。
//
// 原 AVX-512 VNNI 路径已移除，理由是实测：
//   AVX2  : 1 线程 53.1 | 4 线程 98.7 | 16 线程 106.3 tok/s
//   VNNI  : 1 线程 68.5 | 4 线程 93.5 | 16 线程  96.7 tok/s
// 单线程 +29% 但多线程 -9%。Zen4 的 512-bit 通路是 256-bit 双泵，
// 且 quantize_input 每 token 要多两遍遍历 x。而把常驻从 1 字节/权重
// 降到 2-bit 直接砍掉 75% 的字节流量 —— 在带宽墙上这比内核位宽重要得多。
struct PipelineRows:CpuTernaryRows{
    using CpuTernaryRows::CpuTernaryRows;
    bool vnni_=false;        // 保留成员以兼容既有调用点；打包内核不需要 int8 量化
    mutable std::vector<uint8_t> xu_;
    mutable std::vector<int32_t> rowsum_;
    mutable const float* qc_x_=nullptr;
    mutable float sx_=1.0f;

    void build_rowsum(){
        rowsum_.resize(rows);
        for(size_t r=0;r<rows;++r){
            int32_t s=0;
            for(size_t j=0;j<cols;++j)s+=int32_t(at(r,j));
            rowsum_[r]=s;
        }
    }
    void quantize_input(const float* x)const{qc_x_=x;}
    // 单行点积走打包内核。
    float dot(size_t row,const float*x)const{return CpuTernaryRows::dot(row,x);}
};
