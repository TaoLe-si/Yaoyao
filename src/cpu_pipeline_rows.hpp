#pragma once
#include "cpu_ternary_avx2.hpp"
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
// 固化内核：默认 AVX2 int8×float；输出头（>=1e6 MAC）走 int8 VNNI 公式。
// AVX-512 有则用 vpdpbusd；否则 AVX2 maddubs（三值权重无饱和，与 VNNI 整数式相同）。
#if defined(__AVX512F__) || defined(TAO_HAS_AVX512)
#define TAO_AVX512_KERNEL 1
#endif
struct PipelineRows:CpuTernaryRows{
    using CpuTernaryRows::CpuTernaryRows;
    bool vnni_=false;
    // y[r] = alpha[r] * sx * ( sum_j q[r][j]*xu[j] - 128*rowsum[r] )
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
        float mx=0;
        size_t j=0;
        __m256 vm=_mm256_setzero_ps();
        const __m256 sign=_mm256_set1_ps(-0.0f);
        for(;j+8<=cols;j+=8){
            __m256 v=_mm256_andnot_ps(sign,_mm256_loadu_ps(x+j));
            vm=_mm256_max_ps(vm,v);
        }
        alignas(32) float lane[8];_mm256_store_ps(lane,vm);
        for(int t=0;t<8;++t)if(lane[t]>mx)mx=lane[t];
        for(;j<cols;++j){float v=x[j]<0?-x[j]:x[j];if(v>mx)mx=v;}
        sx_=mx>0.0f?mx/127.0f:1.0f;
        xu_.resize(cols);
        const float inv=1.0f/sx_;
        j=0;
        const __m256 vinv=_mm256_set1_ps(inv);
        const __m256 vlo=_mm256_set1_ps(-128.0f),vhi=_mm256_set1_ps(127.0f);
        const __m256i bias=_mm256_set1_epi32(128);
        for(;j+8<=cols;j+=8){
            __m256 qv=_mm256_max_ps(vlo,_mm256_min_ps(vhi,_mm256_mul_ps(_mm256_loadu_ps(x+j),vinv)));
            __m256i qi=_mm256_add_epi32(_mm256_cvtps_epi32(qv),bias);
            __m128i packed=_mm_packus_epi32(_mm256_castsi256_si128(qi),_mm256_extracti128_si256(qi,1));
            packed=_mm_packus_epi16(packed,packed);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(xu_.data()+j),packed);
        }
        for(;j<cols;++j){
            int v=(int)std::lround(x[j]*inv);
            if(v>127)v=127;else if(v<-128)v=-128;
            xu_[j]=(uint8_t)(v+128);
        }
        qc_x_=x;
    }

    static int32_t hsum_i32(__m256i acc){
        __m128i lo=_mm256_castsi256_si128(acc);
        __m128i hi=_mm256_extracti128_si256(acc,1);
        __m128i s=_mm_add_epi32(lo,hi);
        s=_mm_add_epi32(s,_mm_shuffle_epi32(s,0x4E));
        s=_mm_add_epi32(s,_mm_shuffle_epi32(s,0xB1));
        return _mm_cvtsi128_si32(s);
    }
#ifdef TAO_AVX512_KERNEL
    float dot_vnni_avx512(size_t row)const{
        const int8_t* p=q.data()+row*cols;
        const uint8_t* xu=xu_.data();
        __m512i acc=_mm512_setzero_si512();size_t j=0;
        for(;j+64<=cols;j+=64)
            acc=_mm512_dpbusd_epi32(acc,
                _mm512_loadu_si512(reinterpret_cast<const void*>(xu+j)),
                _mm512_loadu_si512(reinterpret_cast<const void*>(p+j)));
        alignas(64)int32_t lane[16];_mm512_store_si512(reinterpret_cast<void*>(lane),acc);
        int64_t tot=0;for(int k=0;k<16;++k)tot+=lane[k];
        for(;j<cols;++j)tot+=int64_t(p[j])*int64_t(xu[j]);
        return float(tot-128LL*int64_t(rowsum_[row]))*sx_*scale[row];
    }
#endif
    float dot_vnni_avx2(size_t row)const{
        const int8_t* p=q.data()+row*cols;
        const uint8_t* xu=xu_.data();
        __m256i acc=_mm256_setzero_si256();size_t j=0;
        const __m256i one=_mm256_set1_epi16(1);
        for(;j+32<=cols;j+=32){
            __m256i prod=_mm256_maddubs_epi16(
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xu+j)),
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+j)));
            acc=_mm256_add_epi32(acc,_mm256_madd_epi16(prod,one));
        }
        int64_t tot=hsum_i32(acc);
        for(;j<cols;++j)tot+=int64_t(p[j])*int64_t(xu[j]);
        return float(tot-128LL*int64_t(rowsum_[row]))*sx_*scale[row];
    }
    void vnni4(size_t row,float* y)const{
        const int8_t* p0=q.data()+row*cols;
        const int8_t* p1=p0+cols,*p2=p1+cols,*p3=p2+cols;
        const uint8_t* xu=xu_.data();
        __m256i a0=_mm256_setzero_si256(),a1=a0,a2=a0,a3=a0;
        const __m256i one=_mm256_set1_epi16(1);
        size_t j=0;
        for(;j+32<=cols;j+=32){
            __m256i xv=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(xu+j));
            a0=_mm256_add_epi32(a0,_mm256_madd_epi16(_mm256_maddubs_epi16(xv,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p0+j))),one));
            a1=_mm256_add_epi32(a1,_mm256_madd_epi16(_mm256_maddubs_epi16(xv,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p1+j))),one));
            a2=_mm256_add_epi32(a2,_mm256_madd_epi16(_mm256_maddubs_epi16(xv,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p2+j))),one));
            a3=_mm256_add_epi32(a3,_mm256_madd_epi16(_mm256_maddubs_epi16(xv,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p3+j))),one));
        }
        int64_t t0=hsum_i32(a0),t1=hsum_i32(a1),t2=hsum_i32(a2),t3=hsum_i32(a3);
        for(;j<cols;++j){
            const int xj=int(xu[j]);
            t0+=int64_t(p0[j])*xj;t1+=int64_t(p1[j])*xj;t2+=int64_t(p2[j])*xj;t3+=int64_t(p3[j])*xj;
        }
        y[0]=float(t0-128LL*int64_t(rowsum_[row]))*sx_*scale[row];
        y[1]=float(t1-128LL*int64_t(rowsum_[row+1]))*sx_*scale[row+1];
        y[2]=float(t2-128LL*int64_t(rowsum_[row+2]))*sx_*scale[row+2];
        y[3]=float(t3-128LL*int64_t(rowsum_[row+3]))*sx_*scale[row+3];
    }
#ifdef TAO_AVX512_KERNEL
    void vnni8(size_t row,float* y)const{
        const int8_t* p0=q.data()+row*cols;
        const int8_t* p1=p0+cols,*p2=p1+cols,*p3=p2+cols;
        const int8_t* p4=p3+cols,*p5=p4+cols,*p6=p5+cols,*p7=p6+cols;
        if(row+8<rows){
            const char* n=reinterpret_cast<const char*>(p0)+size_t(8)*cols;
            _mm_prefetch(n,_MM_HINT_NTA);
            _mm_prefetch(n+cols,_MM_HINT_NTA);
            _mm_prefetch(n+2*cols,_MM_HINT_NTA);
            _mm_prefetch(n+3*cols,_MM_HINT_NTA);
            _mm_prefetch(n+4*cols,_MM_HINT_NTA);
            _mm_prefetch(n+5*cols,_MM_HINT_NTA);
            _mm_prefetch(n+6*cols,_MM_HINT_NTA);
            _mm_prefetch(n+7*cols,_MM_HINT_NTA);
        }
        const uint8_t* xu=xu_.data();
        __m512i a0=_mm512_setzero_si512(),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
        size_t j=0;
        for(;j+64<=cols;j+=64){
            __m512i xv=_mm512_loadu_si512(reinterpret_cast<const void*>(xu+j));
            a0=_mm512_dpbusd_epi32(a0,xv,_mm512_loadu_si512(reinterpret_cast<const void*>(p0+j)));
            a1=_mm512_dpbusd_epi32(a1,xv,_mm512_loadu_si512(reinterpret_cast<const void*>(p1+j)));
            a2=_mm512_dpbusd_epi32(a2,xv,_mm512_loadu_si512(reinterpret_cast<const void*>(p2+j)));
            a3=_mm512_dpbusd_epi32(a3,xv,_mm512_loadu_si512(reinterpret_cast<const void*>(p3+j)));
            a4=_mm512_dpbusd_epi32(a4,xv,_mm512_loadu_si512(reinterpret_cast<const void*>(p4+j)));
            a5=_mm512_dpbusd_epi32(a5,xv,_mm512_loadu_si512(reinterpret_cast<const void*>(p5+j)));
            a6=_mm512_dpbusd_epi32(a6,xv,_mm512_loadu_si512(reinterpret_cast<const void*>(p6+j)));
            a7=_mm512_dpbusd_epi32(a7,xv,_mm512_loadu_si512(reinterpret_cast<const void*>(p7+j)));
        }
        auto finish=[&](size_t rr,__m512i acc,const int8_t* p){
            alignas(64)int32_t lane[16];_mm512_store_si512(reinterpret_cast<void*>(lane),acc);
            int64_t tot=0;for(int k=0;k<16;++k)tot+=lane[k];
            for(size_t t=j;t<cols;++t)tot+=int64_t(p[t])*int64_t(xu[t]);
            return float(tot-128LL*int64_t(rowsum_[rr]))*sx_*scale[rr];
        };
        y[0]=finish(row,a0,p0);y[1]=finish(row+1,a1,p1);y[2]=finish(row+2,a2,p2);y[3]=finish(row+3,a3,p3);
        y[4]=finish(row+4,a4,p4);y[5]=finish(row+5,a5,p5);y[6]=finish(row+6,a6,p6);y[7]=finish(row+7,a7,p7);
    }
#endif

    static __m256 w8(const int8_t* p,__m256 alpha){
        return _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)))),alpha);
    }
    static float hsum_exact(__m256 sum){
        alignas(32)float lane[8];_mm256_store_ps(lane,sum);float z=0;for(float v:lane)z+=v;return z;
    }
    float dot_float_row(size_t row,const float* x)const{
        __m256 sum=_mm256_setzero_ps(),alpha=_mm256_set1_ps(scale[row]);
        size_t j=0;const int8_t* p=q.data()+row*cols;
        for(;j+16<=cols;j+=16){
            sum=_mm256_add_ps(sum,_mm256_mul_ps(w8(p+j,alpha),_mm256_loadu_ps(x+j)));
            sum=_mm256_add_ps(sum,_mm256_mul_ps(w8(p+j+8,alpha),_mm256_loadu_ps(x+j+8)));
        }
        for(;j+8<=cols;j+=8)
            sum=_mm256_add_ps(sum,_mm256_mul_ps(w8(p+j,alpha),_mm256_loadu_ps(x+j)));
        float z=hsum_exact(sum);
        for(;j<cols;++j)z+=(float(p[j])*scale[row])*x[j];
        return z;
    }
    void float4(size_t row,const float* x,float* y)const{
        const int8_t* p0=q.data()+row*cols;
        const int8_t* p1=p0+cols,*p2=p1+cols,*p3=p2+cols;
        const __m256 a0=_mm256_set1_ps(scale[row]),a1=_mm256_set1_ps(scale[row+1]);
        const __m256 a2=_mm256_set1_ps(scale[row+2]),a3=_mm256_set1_ps(scale[row+3]);
        __m256 s0=_mm256_setzero_ps(),s1=s0,s2=s0,s3=s0;
        size_t j=0;
        for(;j+16<=cols;j+=16){
            __m256 x0=_mm256_loadu_ps(x+j),x1=_mm256_loadu_ps(x+j+8);
            s0=_mm256_add_ps(s0,_mm256_mul_ps(w8(p0+j,a0),x0)); s0=_mm256_add_ps(s0,_mm256_mul_ps(w8(p0+j+8,a0),x1));
            s1=_mm256_add_ps(s1,_mm256_mul_ps(w8(p1+j,a1),x0)); s1=_mm256_add_ps(s1,_mm256_mul_ps(w8(p1+j+8,a1),x1));
            s2=_mm256_add_ps(s2,_mm256_mul_ps(w8(p2+j,a2),x0)); s2=_mm256_add_ps(s2,_mm256_mul_ps(w8(p2+j+8,a2),x1));
            s3=_mm256_add_ps(s3,_mm256_mul_ps(w8(p3+j,a3),x0)); s3=_mm256_add_ps(s3,_mm256_mul_ps(w8(p3+j+8,a3),x1));
        }
        for(;j+8<=cols;j+=8){
            __m256 xv=_mm256_loadu_ps(x+j);
            s0=_mm256_add_ps(s0,_mm256_mul_ps(w8(p0+j,a0),xv));
            s1=_mm256_add_ps(s1,_mm256_mul_ps(w8(p1+j,a1),xv));
            s2=_mm256_add_ps(s2,_mm256_mul_ps(w8(p2+j,a2),xv));
            s3=_mm256_add_ps(s3,_mm256_mul_ps(w8(p3+j,a3),xv));
        }
        y[0]=hsum_exact(s0);y[1]=hsum_exact(s1);y[2]=hsum_exact(s2);y[3]=hsum_exact(s3);
        for(;j<cols;++j){
            y[0]+=(float(p0[j])*scale[row])*x[j];
            y[1]+=(float(p1[j])*scale[row+1])*x[j];
            y[2]+=(float(p2[j])*scale[row+2])*x[j];
            y[3]+=(float(p3[j])*scale[row+3])*x[j];
        }
    }

    float dot(size_t row,const float* x)const{
        if(vnni_&&xu_.size()==cols){
#ifdef TAO_AVX512_KERNEL
            return dot_vnni_avx512(row);
#else
            return dot_vnni_avx2(row);
#endif
        }
        return dot_float_row(row,x);
    }
    void gemv_rows(size_t begin,size_t end,const float* x,float* y)const{
        size_t r=begin;
        if(vnni_&&xu_.size()==cols){
#ifdef TAO_AVX512_KERNEL
            for(;r+8<=end;r+=8)vnni8(r,y+r);
#endif
            for(;r+4<=end;r+=4)vnni4(r,y+r);
            for(;r<end;++r)y[r]=dot(r,x);
            return;
        }
        for(;r+4<=end;r+=4)float4(r,x,y+r);
        for(;r<end;++r)y[r]=dot_float_row(r,x);
    }
};
