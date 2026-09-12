#pragma once
#include "cpu_ternary_avx2.hpp"
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
// 固化内核：2-bit 打包常驻，两条内核按矩阵规模分档。
//
// 为什么要有 VNNI 分档：解码的瓶颈已经从内存转移到了指令吞吐。
//   2-bit 打包后每 token 只需 29.8 MB，带宽墙是 1081 tps，实测只有 130 tps（12%）。
//   AVX2 float 内核每 32 个权重约 33 条指令（解包 13 + 转换/乘加 20），约 1 MAC/指令。
//   VNNI 的 vpdpbusd 一条指令做 64 次 int8 乘加。
// 之前实测 VNNI 在多线程下输给 AVX2（106.3 -> 96.7），那是在 int8 常驻、
// 带宽受限的前提下；现在字节数已降 4 倍，指令数成为主约束，所以重新启用。
//
// 代价：VNNI 需要把输入 x 量化成 int8（xu_ = round(x/sx)+128），是有损的，
// 因此会产生极小的数值偏差。AVX2 打包内核保持逐位精确。默认只对
// rows*cols >= 1e6 的大矩阵走 VNNI（见 set_vnni），小矩阵仍走精确路径。
#if defined(__AVX512F__) || defined(TAO_HAS_AVX512)
#define TAO_AVX512_KERNEL 1
#endif
struct PipelineRows:CpuTernaryRows{
    using CpuTernaryRows::CpuTernaryRows;
    bool vnni_=false;
    mutable std::vector<uint8_t> xu_;        // 量化后的输入（uint8，偏移 128）
    mutable std::vector<int32_t> rowsum_;    // 每行权重之和，用于减掉 128 偏移
    mutable const float* qc_x_=nullptr;
    mutable float sx_=1.0f;

    // 16 个打包字节（64 个 2-bit 码）-> 64 个 int8，值域 {-1,0,1}。
    // 复用 unpack32 两次再拼成 512 位，解包过程完全在寄存器内。
    static inline __m512i unpack64(const uint8_t* p){
#ifdef TAO_AVX512_KERNEL
        const __m256i lo=unpack32(p);
        const __m256i hi=unpack32(p+8);
        return _mm512_inserti64x4(_mm512_castsi256_si512(lo),hi,1);
#else
        (void)p;
        return _mm512_setzero_si512();
#endif
    }

    void build_rowsum(){
        rowsum_.resize(rows);
        for(size_t r=0;r<rows;++r){
            int32_t s=0;
            for(size_t j=0;j<cols;++j)s+=int32_t(at(r,j));
            rowsum_[r]=s;
        }
    }
    void quantize_input(const float* x)const{
        float mx=0;
        for(size_t j=0;j<cols;++j){const float v=x[j]<0.0f?-x[j]:x[j];if(v>mx)mx=v;}
        sx_=mx>0.0f?mx/127.0f:1.0f;
        xu_.resize(cols);
        for(size_t j=0;j<cols;++j){
            int v=int(std::lround(double(x[j])/double(sx_)));
            if(v>127)v=127;else if(v<-128)v=-128;
            xu_[j]=uint8_t(v+128);
        }
        qc_x_=x;
    }
#ifdef TAO_AVX512_KERNEL
    // y[r] = scale[r] * sx * ( sum_j q[r][j]*xu[j] - 128*rowsum[r] )
    // 累加在 int32 内：cols 最大 16384，|xu|<=255、|q|<=1，远不溢出。
    float dot_vnni(size_t row,const float*)const{
        const uint8_t* p=q.data()+row*stride();
        const size_t nb=cols/64;
        __m512i acc=_mm512_setzero_si512();
        size_t j=0;
        for(size_t b=0;b<nb;++b){
            const __m512i xv=_mm512_loadu_si512(reinterpret_cast<const void*>(xu_.data()+j));
            acc=_mm512_dpbusd_epi32(acc,xv,unpack64(p+16*b));
            j+=64;
        }
        alignas(64)int32_t lane[16];
        _mm512_store_si512(reinterpret_cast<void*>(lane),acc);
        int64_t s=0;for(int k=0;k<16;++k)s+=lane[k];
        int64_t tot=s-128LL*int64_t(rowsum_[row]);
        for(;j<cols;++j)tot+=int64_t(at(row,j))*int64_t(int(xu_[j])-128);
        return float(tot)*sx_*scale[row];
    }
#endif
    float dot(size_t row,const float*x)const{
#ifdef TAO_AVX512_KERNEL
        if(vnni_)return dot_vnni(row,x);
#endif
        return CpuTernaryRows::dot(row,x);
    }
};
