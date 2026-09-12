#pragma once
#include "cpu_ternary_avx2.hpp"
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
// 固化内核：2-bit 打包常驻，两条内核按矩阵规模分档。
//
// 【为什么 VNNI 分档】2-bit 打包后每 token 只需 29.8 MB，带宽墙约 1081 tps，而实测
//   远低于此 —— 瓶颈已从内存转到指令吞吐。AVX2 float 内核每 32 个权重约 33 条指令
//   （解包 13 + 转换/乘加 20），约 1 MAC/指令；VNNI 的 vpdpbusd 一条指令做 64 次
//   int8 乘加。代价是输入必须量化成 int8，有损。小矩阵仍走逐位精确的 AVX2 路径。
//
// 【量化本身曾是最大单项】实测 profile 显示 s 相位（4 个矩阵、共 30.7M MAC/步）
//   耗时是 read 相位（2 个矩阵、同样 30.7M MAC/步）的 2.6 倍。原因不在矩阵乘法：
//   group<> 会对组内每个矩阵各调用一次 quantize_input，而 s 相位的 4 个矩阵只有
//   2 个不同输入（candidate.x/gate.x 共用 xn，candidate.s/gate.s 共用 s），于是
//   xn 和 s 各被量化两遍；read 的两个矩阵输入不同，没有重复。叠加原实现逐元素
//   std::lround(double) + double 除法，量化开销足以盖过 MAC 开销。
//   两处修复：(1) quantize_input 全面向量化；(2) 组内相同输入只量化一次，
//   其余矩阵通过 xuSrc_/sxSrc_ 指向同一份结果（见下）。
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
    // 组内复用：同一 group<> 调用中若多个矩阵共享同一个输入向量，只对第一个
    // 调用 quantize_input，其余把 xuSrc_/sxSrc_ 指过去。指向的对象（源矩阵的
    // xu_/sx_）是模型成员，生命周期长于任何一次前向，因此安全。
    mutable const std::vector<uint8_t>* xuSrc_=nullptr;
    mutable const float* sxSrc_=nullptr;

    // 16 个打包字节（64 个 2-bit 码）-> 64 个 int8，值域 {-1,0,1}。
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
    // 向量化量化：原实现是逐元素 std::lround(double) + double 除法。
    // cvtps_epi32 用就近取偶，与 lround 的“就近取远离零”仅在恰好 .5 处不同，
    // 而 .5 在量化前的浮点乘法结果里几乎不出现；实测 NLL 变化在小数第四位以外。
    void quantize_input(const float* x)const{
        const size_t n=cols;
        size_t j=0;
        const __m256 absmask=_mm256_set1_ps(-0.0f);
        __m256 vmax=_mm256_setzero_ps();
        for(;j+8<=n;j+=8)
            vmax=_mm256_max_ps(vmax,_mm256_andnot_ps(absmask,_mm256_loadu_ps(x+j)));
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp,vmax);
        float mx=0.0f;
        for(int k=0;k<8;++k)if(tmp[k]>mx)mx=tmp[k];
        for(;j<n;++j){const float v=x[j]<0.0f?-x[j]:x[j];if(v>mx)mx=v;}
        sx_=mx>0.0f?mx/127.0f:1.0f;
        const float inv=1.0f/sx_;
        xu_.resize(n);
        const __m256 vinv=_mm256_set1_ps(inv),vhi=_mm256_set1_ps(127.0f),vlo=_mm256_set1_ps(-128.0f);
        const __m256i vbias=_mm256_set1_epi32(128);
        j=0;
        for(;j+8<=n;j+=8){
            __m256 v=_mm256_mul_ps(_mm256_loadu_ps(x+j),vinv);
            v=_mm256_max_ps(v,vlo);v=_mm256_min_ps(v,vhi);
            __m256i iv=_mm256_add_epi32(_mm256_cvtps_epi32(v),vbias);
            alignas(32) int32_t t[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(t),iv);
            for(int k=0;k<8;++k)xu_[j+k]=uint8_t(t[k]);
        }
        for(;j<n;++j){
            int v=int(std::lround(double(x[j])/double(sx_)));
            if(v>127)v=127;else if(v<-128)v=-128;
            xu_[j]=uint8_t(v+128);
        }
        xuSrc_=&xu_;sxSrc_=&sx_;
        qc_x_=x;
    }
    bool has_input()const{return xuSrc_!=nullptr;}
    void share_input_from(const PipelineRows&src)const{xuSrc_=src.xuSrc_;sxSrc_=src.sxSrc_;}
#ifdef TAO_AVX512_KERNEL
    // y[r] = scale[r] * sx * ( sum_j q[r][j]*xu[j] - 128*rowsum[r] )
    // 累加在 int32 内：cols 最大 16384，|xu|<=255、|q|<=1，远不溢出。
    float dot_vnni(size_t row,const float*)const{
        const std::vector<uint8_t>&X=*xuSrc_;
        const uint8_t* p=q.data()+row*stride();
        const size_t nb=cols/64;
        __m512i acc=_mm512_setzero_si512();
        size_t j=0;
        for(size_t b=0;b<nb;++b){
            const __m512i xv=_mm512_loadu_si512(reinterpret_cast<const void*>(X.data()+j));
            acc=_mm512_dpbusd_epi32(acc,xv,unpack64(p+16*b));
            j+=64;
        }
        alignas(64)int32_t lane[16];
        _mm512_store_si512(reinterpret_cast<void*>(lane),acc);
        int64_t s=0;for(int k=0;k<16;++k)s+=lane[k];
        int64_t tot=s-128LL*int64_t(rowsum_[row]);
        for(;j<cols;++j)tot+=int64_t(at(row,j))*int64_t(int(X[j])-128);
        return float(tot)*(*sxSrc_)*scale[row];
    }
#endif
    float dot(size_t row,const float*x)const{
#ifdef TAO_AVX512_KERNEL
        if(vnni_&&xuSrc_)return dot_vnni(row,x);
#endif
        return CpuTernaryRows::dot(row,x);
    }
};
