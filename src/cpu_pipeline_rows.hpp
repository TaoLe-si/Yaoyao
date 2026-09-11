#pragma once
#include "cpu_ternary_avx2.hpp"
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
// 固化内核：默认 AVX2 int8 点积；输出头（>=1e6 MAC）走 VNNI。
// 已删除的旧路径：float 权重展开、AVX-512 浮点核、多累加器、2-bit 位平面、kfold。
#if defined(__AVX512F__) || defined(TAO_HAS_AVX512)
#define TAO_AVX512_KERNEL 1
#endif
struct PipelineRows:CpuTernaryRows{
    using CpuTernaryRows::CpuTernaryRows;
    bool vnni_=false;        // VNNI int8 内核（运行期开关，按矩阵规模分档）
    // y[r] = alpha[r] * sx * ( sum_j q[r][j]*xu[j] - 128*rowsum[r] )
    //   xu[j] = round(x[j]/sx) + 128 in [0,255]
    //   rowsum[r] = sum_j q[r][j]
    mutable std::vector<uint8_t> xu_;
    mutable std::vector<int32_t> rowsum_;
    mutable const float* qc_x_=nullptr;
    mutable float sx_=1.0f;

    void build_rowsum(){
        rowsum_.resize(rows);
        for(size_t r=0;r<rows;++r){
            const int8_t* p=q.data()+r*cols;int32_t s=0;
            for(size_t j=0;j<cols;++j)s+=p[j];
            rowsum_[r]=s;
        }
    }
    void quantize_input(const float* x)const{
        float mx=0;for(size_t j=0;j<cols;++j){float v=x[j]<0?-x[j]:x[j];if(v>mx)mx=v;}
        sx_=mx>0.0f?mx/127.0f:1.0f;
        xu_.resize(cols);
        for(size_t j=0;j<cols;++j){
            int v=(int)std::lround(x[j]/sx_);
            if(v>127)v=127;else if(v<-128)v=-128;
            xu_[j]=(uint8_t)(v+128);
        }
        qc_x_=x;
    }
#ifdef TAO_AVX512_KERNEL
    float dot_vnni(size_t row,const float* x)const{
        (void)x;
        const int8_t* p=q.data()+row*cols;
        __m512i acc=_mm512_setzero_si512();size_t j=0;
        for(;j+64<=cols;j+=64){
            __m512i xv=_mm512_loadu_si512(reinterpret_cast<const void*>(xu_.data()+j));
            __m512i wv=_mm512_loadu_si512(reinterpret_cast<const void*>(p+j));
            acc=_mm512_dpbusd_epi32(acc,xv,wv);
        }
        alignas(64)int32_t lane[16];_mm512_store_si512(reinterpret_cast<void*>(lane),acc);
        int64_t s=0;for(int k=0;k<16;++k)s+=lane[k];
        int64_t tot=s-128LL*int64_t(rowsum_[row]);
        for(;j<cols;++j)tot+=int64_t(p[j])*(int64_t(xu_[j])-128);
        return float(tot)*sx_*scale[row];
    }
#endif

    float dot(size_t row,const float*x)const{
#ifdef TAO_AVX512_KERNEL
        if(vnni_)return dot_vnni(row,x);
#endif
        __m256 sum=_mm256_setzero_ps(),alpha=_mm256_set1_ps(scale[row]);size_t j=0;const int8_t*p=q.data()+row*cols;
        for(;j+16<=cols;j+=16){auto w0=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p+j)))),alpha);auto w1=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p+j+8)))),alpha);auto a=_mm256_mul_ps(w0,_mm256_loadu_ps(x+j));auto b=_mm256_mul_ps(w1,_mm256_loadu_ps(x+j+8));sum=_mm256_add_ps(sum,a);sum=_mm256_add_ps(sum,b);}for(;j+8<=cols;j+=8){__m128i bytes=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p+j));__m256 weights=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes)),alpha);sum=_mm256_add_ps(sum,_mm256_mul_ps(weights,_mm256_loadu_ps(x+j)));}alignas(32)float lane[8];_mm256_store_ps(lane,sum);float z=0;for(float v:lane)z+=v;for(;j<cols;++j)z+=(float(p[j])*scale[row])*x[j];return z;}
};
