#pragma once
#include <immintrin.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <cstdint>
// 三值权重以 2-bit 打包常驻：每字节 4 个权重。
//   码 0 -> 0    码 1 -> +1    码 2 -> -1    码 3 -> 填充（按 0 处理）
//
// 为什么必须打包：CPU 解码是硬带宽受限的，tps = 内存带宽 / 每 token 常驻字节。
//   1 字节/权重：119,033,986 权重 -> 119.0 MB/token
//   2-bit 打包 ：同模型           ->  29.8 MB/token
// 在 18.68 GB/s 的实测带宽墙下，这直接把上限从 157 tps 抬到 628 tps。
//
// 展开必须在寄存器内完成。任何"先解包到中间缓冲再算"的写法都会把 29.8 MB
// 重新变成 119 MB 的读写流量，等于没改。
struct CpuTernaryRows {
    size_t rows,cols;
    std::vector<uint8_t> q;        // 每行 stride() 字节，而不是 cols 字节
    std::vector<float> scale;

    static size_t stride_for(size_t c){return (c+3)/4;}
    size_t stride() const {return stride_for(cols);}
    size_t packed_bytes() const {return q.size();}

    static int8_t sym(unsigned code){static const int8_t t[4]={0,1,-1,0};return t[code&3u];}
    // 标量取权重（仅非热点回退路径使用）。
    int8_t at(size_t row,size_t j) const{
        const unsigned b=q[row*stride()+(j>>2)];
        return sym((b>>((j&3u)*2u))&3u);
    }

    CpuTernaryRows(const std::vector<float>&w,size_t r,size_t c)
        :rows(r),cols(c),q(stride_for(c)*r,0),scale(r,1){
        if(!r||!c||w.size()!=r*c)throw std::runtime_error("ternary shape");
        const size_t st=stride_for(c);
        for(size_t i=0;i<r;++i){
            float a=0;
            for(size_t j=0;j<c;++j){
                const float v=w[i*c+j];
                if(!std::isfinite(v))throw std::runtime_error("nonfinite weight");
                if(v!=0){
                    if(a&&a!=std::abs(v))throw std::runtime_error("not exact row ternary");
                    a=std::abs(v);
                    q[i*st+(j>>2)]|=uint8_t((v>0?1u:2u)<<((j&3u)*2u));
                }
            }
            scale[i]=a?a:1;
        }
    }

    // 8 个打包字节（32 个码）-> 32 个 int8，值域 {-1,0,1}。
    // 位平面分离（4 个 AND/SHIFT）-> 交错回码的天然顺序 -> pshufb 查表转符号。
    static inline __m256i unpack32(const uint8_t* p){
        const __m128i m=_mm_set1_epi8(3);
        const __m128i x=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
        const __m128i c0=_mm_and_si128(x,m);
        const __m128i c1=_mm_and_si128(_mm_srli_epi16(x,2),m);
        const __m128i c2=_mm_and_si128(_mm_srli_epi16(x,4),m);
        const __m128i c3=_mm_and_si128(_mm_srli_epi16(x,6),m);
        const __m128i A=_mm_unpacklo_epi8(c0,c1);
        const __m128i B=_mm_unpacklo_epi8(c2,c3);
        const __m128i C=_mm_unpacklo_epi16(A,B);   // 码 0..15
        const __m128i D=_mm_unpackhi_epi16(A,B);   // 码 16..31
        const __m256i codes=_mm256_set_m128i(D,C);
        const __m256i lut=_mm256_setr_epi8(0,1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                           0,1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0);
        return _mm256_shuffle_epi8(lut,codes);
    }

    // 把一个 32 权重块累加进 sum。
    inline void mac32(__m256& sum,const uint8_t* p,const float* x,const __m256 alpha)const{
        const __m256i w=unpack32(p);
        const __m128i lo=_mm256_castsi256_si128(w);
        const __m128i hi=_mm256_extracti128_si256(w,1);
        const __m256 f0=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
        const __m256 f1=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo,8)));
        const __m256 f2=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
        const __m256 f3=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi,8)));
        sum=_mm256_add_ps(sum,_mm256_mul_ps(_mm256_mul_ps(f0,alpha),_mm256_loadu_ps(x+0)));
        sum=_mm256_add_ps(sum,_mm256_mul_ps(_mm256_mul_ps(f1,alpha),_mm256_loadu_ps(x+8)));
        sum=_mm256_add_ps(sum,_mm256_mul_ps(_mm256_mul_ps(f2,alpha),_mm256_loadu_ps(x+16)));
        sum=_mm256_add_ps(sum,_mm256_mul_ps(_mm256_mul_ps(f3,alpha),_mm256_loadu_ps(x+24)));
    }

    float dot(size_t row,const float*x)const{
        const __m256 alpha=_mm256_set1_ps(scale[row]);
        const uint8_t* p=q.data()+row*stride();
        const size_t nb=cols/32;
        size_t j=0;
        __m256 sum=_mm256_setzero_ps();
        // 一个 32 权重块 = 8 个打包字节（4 权重/字节）。
        for(size_t b=0;b<nb;++b){mac32(sum,p+8*b,x+j,alpha);j+=32;}
        alignas(32)float lane[8];
        _mm256_store_ps(lane,sum);
        float z=0;for(float v:lane)z+=v;
        for(;j<cols;++j)z+=(float(at(row,j))*scale[row])*x[j];
        return z;
    }

    void matvec_four_rows(const float* x,float* y)const{
        const size_t st=stride();
        const size_t nb=cols/32;
        size_t row=0;
        for(;row+4<=rows;row+=4){
            const __m256 a0=_mm256_set1_ps(scale[row]);
            const __m256 a1=_mm256_set1_ps(scale[row+1]);
            const __m256 a2=_mm256_set1_ps(scale[row+2]);
            const __m256 a3=_mm256_set1_ps(scale[row+3]);
            const uint8_t* p0=q.data()+row*st;
            const uint8_t* p1=p0+st;const uint8_t* p2=p1+st;const uint8_t* p3=p2+st;
            __m256 s0=_mm256_setzero_ps(),s1=s0,s2=s0,s3=s0;
            size_t j=0;
            for(size_t b=0;b<nb;++b){
                mac32(s0,p0+8*b,x+j,a0);
                mac32(s1,p1+8*b,x+j,a1);
                mac32(s2,p2+8*b,x+j,a2);
                mac32(s3,p3+8*b,x+j,a3);
                j+=32;
            }
            alignas(32)float lane[4][8];
            _mm256_store_ps(lane[0],s0);_mm256_store_ps(lane[1],s1);
            _mm256_store_ps(lane[2],s2);_mm256_store_ps(lane[3],s3);
            for(size_t r=0;r<4;++r){
                float z=0;for(float v:lane[r])z+=v;
                for(size_t k=j;k<cols;++k)z+=(float(at(row+r,k))*scale[row+r])*x[k];
                y[row+r]=z;
            }
        }
        for(;row<rows;++row)y[row]=dot(row,x);
    }
};
