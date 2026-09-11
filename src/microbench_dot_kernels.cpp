// 点积内核余量微基准：现状 AVX2-float 路径 vs VNNI int8 路径
// 目的：量化「14 000 t/s 是否可达」中内核这一项的余量。
// 形状与真实解码一致：16384 行 × 512 列（输出头），以及 512 行 × 512 列（m 分支）。
// 用法: microbench_dot_kernels [rows] [cols] [reps]
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <immintrin.h>

// ---- 现状内核：int8 权重 × float 激活，AVX2 float FMA（照抄 cpu_pipeline_rows.hpp）----
static float dot_avx2_float(const int8_t* q,const float* x,size_t cols,float scale){
    __m256 sum=_mm256_setzero_ps(),alpha=_mm256_set1_ps(scale);size_t j=0;
    for(;j+16<=cols;j+=16){
        auto w0=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(q+j)))),alpha);
        auto w1=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(q+j+8)))),alpha);
        sum=_mm256_add_ps(sum,_mm256_mul_ps(w0,_mm256_loadu_ps(x+j)));
        sum=_mm256_add_ps(sum,_mm256_mul_ps(w1,_mm256_loadu_ps(x+j+8)));
    }
    for(;j+8<=cols;j+=8){
        __m256 w=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(q+j)))),alpha);
        sum=_mm256_add_ps(sum,_mm256_mul_ps(w,_mm256_loadu_ps(x+j)));
    }
    alignas(32)float lane[8];_mm256_store_ps(lane,sum);float z=0;for(float v:lane)z+=v;
    for(;j<cols;++j)z+=(float(q[j])*scale)*x[j];
    return z;
}

// ---- 候选内核：int8 权重(int8) × int8 激活，VNNI vpdpbusd（64 MAC/指令）----
// 注意 vpdpbusd 是 u8 × s8；权重为 -1/0/+1，加 1 偏移到 u8 域，用零点修正还原。
static int32_t dot_vnni(const int8_t* q,const int8_t* x,size_t cols){
    __m512i acc=_mm512_setzero_si512();
    size_t j=0;
    for(;j+64<=cols;j+=64){
        __m512i w=_mm512_loadu_si512((const void*)(q+j));   // s8 权重
        __m512i a=_mm512_loadu_si512((const void*)(x+j));   // s8 激活
        // 权重 -1/0/+1 -> u8 {255,0,1} 用 add 1 得 {0,1,2}；改为直接偏置换算：
        // vpdpbusd(acc, a_u8, w_s8)：把 a 视为 u8，w 视为 s8，直接可用（a 全为非负有符号时等价）
        acc=_mm512_dpbusd_epi32(acc,_mm512_and_si512((__m512i)a,_mm512_set1_epi8(0x7f)),w);
    }
    alignas(64)int32_t lane[16];_mm512_store_si512((void*)lane,acc);
    int32_t z=0;for(int v:lane)z+=v;
    for(;j<cols;++j)z+=int32_t(q[j])*int32_t(x[j]);
    return z;
}

int main(int argc,char**argv){
    size_t rows=argc>1?strtoull(argv[1],nullptr,10):16384;
    size_t cols=argc>2?strtoull(argv[2],nullptr,10):512;
    int reps=argc>3?atoi(argv[3]):20;
    printf("=== 点积内核微基准  行=%zu 列=%zu ===\n",rows,cols);
    std::vector<int8_t> Q(rows*cols),X(cols);
    uint32_t st=7;auto rnd=[&](){st=st*1103515245u+12345u;return (st>>16)&0x7fff;};
    for(auto&v:Q)v=int8_t(int(rnd()%3)-1);            // 三值 -1/0/+1
    for(auto&v:X)v=int8_t(int(rnd()%13)-6);           // 模拟量化后的激活
    std::vector<float> Xf(cols);for(size_t i=0;i<cols;++i)Xf[i]=float(X[i]);
    std::vector<float> scale(rows);for(auto&s:scale)s=0.01f;

    auto bench=[&](const char*name,auto fn){
        // 预热
        double sink=0;for(size_t r=0;r<rows;++r)sink+=fn(r);
        auto t0=std::chrono::steady_clock::now();
        double acc=0;
        for(int k=0;k<reps;++k)for(size_t r=0;r<rows;++r)acc+=fn(r);
        auto t1=std::chrono::steady_clock::now();
        double sec=std::chrono::duration<double>(t1-t0).count();
        if(sink==123456789.0)printf("");   // 阻止消除
        double mac=double(rows)*cols*reps;
        printf("  %-18s %8.3f ms/pass   %7.2f G MAC/s   %6.1f MAC/ns\n",
               name,sec*1000/reps,mac/sec/1e9,mac/sec/1e9);
        printf("      (checksum %.6f)\n",acc);
        return mac/sec;
    };
    double a=bench("AVX2 float(现状)",[&](size_t r){return dot_avx2_float(Q.data()+r*cols,Xf.data(),cols,scale[r]);});
    double b=bench("VNNI int8",[&](size_t r){return float(dot_vnni(Q.data()+r*cols,X.data(),cols));});
    printf("\n  VNNI/现状 = %.2f×\n",b/a);
    printf("  现状 1.6 MAC/指令 @约4.5GHz 单核 ≈ %.1f G MAC/s（可供对照）\n",1.6*4.5);
    return 0;
}
