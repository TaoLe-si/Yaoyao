// 累加器依赖链假说。
//
// 现有内核每 16 权重：
//     sum = add_ps(sum, a);      // 依赖
//     sum = add_ps(sum, b);      // 依赖
// 单条 sum 链，vaddps 延迟约 3-4 周期 => 每 16 权重约 8 周期 => 2 MAC/周期。
// 实测单线程 8.60 G MAC/s，单核 boost 约 4.5 GHz => 1.91 MAC/周期。**吻合。**
//
// 若成立，用多个独立累加器即可解除依赖链，理论可达 8 MAC/周期。
//
// 数值性质：改变求和**顺序**，不改变函数语义 => 不需重训，但会改变 checksum。
// 与 §27 的 AVX-512 同类；区别是纯 AVX2，**没有全核频率惩罚**。
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <vector>
#include <cstring>
#include <immintrin.h>

// 现有实现（单累加器）
static float dot_1acc(const int8_t* p,const float* x,size_t cols,float alpha){
    __m256 sum=_mm256_setzero_ps(),a8=_mm256_set1_ps(alpha);size_t j=0;
    for(;j+16<=cols;j+=16){
        auto w0=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(p+j)))),a8);
        auto w1=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(p+j+8)))),a8);
        sum=_mm256_add_ps(sum,_mm256_mul_ps(w0,_mm256_loadu_ps(x+j)));
        sum=_mm256_add_ps(sum,_mm256_mul_ps(w1,_mm256_loadu_ps(x+j+8)));
    }
    alignas(32)float l[8];_mm256_store_ps(l,sum);float z=0;for(float v:l)z+=v;
    for(;j<cols;++j)z+=(float(p[j])*alpha)*x[j];
    return z;
}
// 2 累加器
static float dot_2acc(const int8_t* p,const float* x,size_t cols,float alpha){
    __m256 s0=_mm256_setzero_ps(),s1=_mm256_setzero_ps(),a8=_mm256_set1_ps(alpha);size_t j=0;
    for(;j+16<=cols;j+=16){
        auto w0=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(p+j)))),a8);
        auto w1=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(p+j+8)))),a8);
        s0=_mm256_add_ps(s0,_mm256_mul_ps(w0,_mm256_loadu_ps(x+j)));
        s1=_mm256_add_ps(s1,_mm256_mul_ps(w1,_mm256_loadu_ps(x+j+8)));
    }
    s0=_mm256_add_ps(s0,s1);
    alignas(32)float l[8];_mm256_store_ps(l,s0);float z=0;for(float v:l)z+=v;
    for(;j<cols;++j)z+=(float(p[j])*alpha)*x[j];
    return z;
}
// 4 累加器（每轮 32 权重）
static float dot_4acc(const int8_t* p,const float* x,size_t cols,float alpha){
    __m256 s0=_mm256_setzero_ps(),s1=_mm256_setzero_ps(),s2=_mm256_setzero_ps(),s3=_mm256_setzero_ps();
    __m256 a8=_mm256_set1_ps(alpha);size_t j=0;
    for(;j+32<=cols;j+=32){
        auto w0=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(p+j)))),a8);
        auto w1=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(p+j+8)))),a8);
        auto w2=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(p+j+16)))),a8);
        auto w3=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(p+j+24)))),a8);
        s0=_mm256_add_ps(s0,_mm256_mul_ps(w0,_mm256_loadu_ps(x+j)));
        s1=_mm256_add_ps(s1,_mm256_mul_ps(w1,_mm256_loadu_ps(x+j+8)));
        s2=_mm256_add_ps(s2,_mm256_mul_ps(w2,_mm256_loadu_ps(x+j+16)));
        s3=_mm256_add_ps(s3,_mm256_mul_ps(w3,_mm256_loadu_ps(x+j+24)));
    }
    s0=_mm256_add_ps(_mm256_add_ps(s0,s1),_mm256_add_ps(s2,s3));
    alignas(32)float l[8];_mm256_store_ps(l,s0);float z=0;for(float v:l)z+=v;
    for(;j<cols;++j)z+=(float(p[j])*alpha)*x[j];
    return z;
}
// 8 累加器（每轮 64 权重）
static float dot_8acc(const int8_t* p,const float* x,size_t cols,float alpha){
    __m256 s[8];for(auto&v:s)v=_mm256_setzero_ps();
    __m256 a8=_mm256_set1_ps(alpha);size_t j=0;
    for(;j+64<=cols;j+=64){
        for(int k=0;k<8;++k){
            auto w=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(p+j+8*k)))),a8);
            s[k]=_mm256_add_ps(s[k],_mm256_mul_ps(w,_mm256_loadu_ps(x+j+8*k)));
        }
    }
    for(int k=1;k<8;++k)s[0]=_mm256_add_ps(s[0],s[k]);
    alignas(32)float l[8];_mm256_store_ps(l,s[0]);float z=0;for(float v:l)z+=v;
    for(;j<cols;++j)z+=(float(p[j])*alpha)*x[j];
    return z;
}

int main(){
    const size_t rows=4096,cols=512;
    std::vector<int8_t> q(rows*cols);std::vector<float> scale(rows),x(cols);
    unsigned seed=777;auto nx=[&](){seed=seed*1664525u+1013904223u;return seed;};
    for(auto&v:q)v=(int8_t)((nx()>>16)%3)-1;
    for(auto&v:scale)v=0.001f+0.0001f*((nx()>>16)%7);
    for(auto&v:x)v=(nx()>>8)*(1.0f/16777216.0f)-0.5f;

    printf("=== %zux%zu ===\n",rows,cols);
    double mx=0;for(size_t r=0;r<rows;++r){
        float a=dot_1acc(q.data()+r*cols,x.data(),cols,scale[r]);
        for(auto f:{dot_2acc,dot_4acc,dot_8acc}){
            float b=f(q.data()+r*cols,x.data(),cols,scale[r]);
            mx=std::max(mx,std::fabs(double(a)-double(b))/std::max(1e-30,std::fabs(double(a))));}}
    printf("  多累加器 vs 单累加器 最大相对差: %.3e\n\n",mx);

    auto bench=[&](const char*n,auto fn,int reps){
        float sink=0;auto t0=std::chrono::steady_clock::now();
        for(int k=0;k<reps;++k)for(size_t r=0;r<rows;++r)sink+=fn(q.data()+r*cols,x.data(),cols,scale[r]);
        auto t1=std::chrono::steady_clock::now();
        double sec=std::chrono::duration<double>(t1-t0).count();
        if(sink==123456789.0f)printf("");
        double g=double(rows)*cols*reps/sec/1e9;
        printf("  %-14s %7.2f G MAC/s  %5.2fx\n",n,g,g/8.60);
        return g;
    };
    printf("--- 单线程（此环境下稳定，§28.4）---\n");
    const int R=20;
    bench("1 累加器(现状)",dot_1acc,R);
    bench("2 累加器",dot_2acc,R);
    bench("4 累加器",dot_4acc,R);
    bench("8 累加器",dot_8acc,R);
    return 0;
}
