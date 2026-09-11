// AVX2 vs AVX-512 三元内核。
//
// 动机：现有内核 1.14 MAC/指令、实测 3.46 指令/周期 —— 已撞前端发射宽度。
// AVX-512 把向量宽度翻倍 => 同样权重所需指令数减半 => 若确为前端受限，应近 2x。
//
// 数值性质（与 fast_act 截然不同）：
//   AVX-512 版只是把浮点**累加分组**改了（16 lane 累加器 vs 8 lane），
//   每个元素的 IEEE 运算完全相同，相对误差 ~1e-7。
//   => 不改变模型的**函数语义**，不需要重训。（仍需实测确认影响可忽略。）
//
// 访问模式：每个 rep 完整扫一遍矩阵（真实推理每 token 只扫一遍），
//   避免上一轮 float 权重那样「重复访问同一矩阵导致缓存驻留」的误判。
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <vector>
#include <cstring>
#include <immintrin.h>

static float dot_avx2(const int8_t* p,const float* x,size_t cols,float alpha){
    __m256 sum=_mm256_setzero_ps(),a8=_mm256_set1_ps(alpha);size_t j=0;
    for(;j+16<=cols;j+=16){
        auto w0=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(p+j)))),a8);
        auto w1=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(p+j+8)))),a8);
        sum=_mm256_add_ps(sum,_mm256_mul_ps(w0,_mm256_loadu_ps(x+j)));
        sum=_mm256_add_ps(sum,_mm256_mul_ps(w1,_mm256_loadu_ps(x+j+8)));
    }
    for(;j+8<=cols;j+=8)
        sum=_mm256_add_ps(sum,_mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(p+j)))),a8),_mm256_loadu_ps(x+j)));
    alignas(32)float lane[8];_mm256_store_ps(lane,sum);
    float z=0;for(float v:lane)z+=v;
    for(;j<cols;++j)z+=(float(p[j])*alpha)*x[j];
    return z;
}

static float dot_avx512(const int8_t* p,const float* x,size_t cols,float alpha){
    __m512 sum=_mm512_setzero_ps(),a16=_mm512_set1_ps(alpha);size_t j=0;
    for(;j+32<=cols;j+=32){
        auto w0=_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(p+j)))),a16);
        auto w1=_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(p+j+16)))),a16);
        sum=_mm512_add_ps(sum,_mm512_mul_ps(w0,_mm512_loadu_ps(x+j)));
        sum=_mm512_add_ps(sum,_mm512_mul_ps(w1,_mm512_loadu_ps(x+j+16)));
    }
    for(;j+16<=cols;j+=16)
        sum=_mm512_add_ps(sum,_mm512_mul_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(p+j)))),a16),_mm512_loadu_ps(x+j)));
    alignas(64)float lane[16];_mm512_store_ps(lane,sum);
    float z=0;for(float v:lane)z+=v;
    for(;j<cols;++j)z+=(float(p[j])*alpha)*x[j];
    return z;
}

int main(int argc,char**argv){
    size_t rows=argc>1?strtoull(argv[1],nullptr,10):512;
    size_t cols=argc>2?strtoull(argv[2],nullptr,10):512;
    int reps=argc>3?atoi(argv[3]):40;
    std::vector<int8_t> q(rows*cols);
    std::vector<float> scale(rows),x(cols);
    unsigned seed=999;
    auto nx=[&](){seed=seed*1664525u+1013904223u;return seed;};
    for(auto&v:q)v=(int8_t)((nx()>>16)%3)-1;
    for(auto&v:scale)v=0.001f+0.0001f*((nx()>>16)%7);
    for(auto&v:x)v=(nx()>>8)*(1.0f/16777216.0f)-0.5f;

    printf("=== 行=%zu 列=%zu  权重 %.2f MB ===\n",rows,cols,double(rows)*cols/1e6);
    // 数值差异
    double maxrel=0; size_t worst=0;
    for(size_t r=0;r<rows;++r){
        float a=dot_avx2(q.data()+r*cols,x.data(),cols,scale[r]);
        float b=dot_avx512(q.data()+r*cols,x.data(),cols,scale[r]);
        double d=std::fabs(double(a)-double(b))/std::max(1e-30,std::fabs(double(a)));
        if(d>maxrel){maxrel=d;worst=r;}
    }
    printf("  AVX2 vs AVX512 最大相对差: %.3e (行 %zu)\n",maxrel,worst);

    auto bench=[&](const char*n,auto fn){
        float sink=0;
        auto t0=std::chrono::steady_clock::now();
        for(int k=0;k<reps;++k)fn(sink);
        auto t1=std::chrono::steady_clock::now();
        double sec=std::chrono::duration<double>(t1-t0).count();
        double mac=double(rows)*cols*reps;
        if(sink==123456789.0f)printf("");
        printf("  %-14s %8.3f ms/pass  %7.2f G MAC/s\n",n,sec*1000/reps,mac/sec/1e9);
        return mac/sec/1e9;
    };
    double a=bench("AVX2",[&](float&s){for(size_t r=0;r<rows;++r)s+=dot_avx2(q.data()+r*cols,x.data(),cols,scale[r]);});
    double b=bench("AVX-512",[&](float&s){for(size_t r=0;r<rows;++r)s+=dot_avx512(q.data()+r*cols,x.data(),cols,scale[r]);});
    printf("  => AVX-512 / AVX2 = %.2fx\n",b/a);
    return 0;
}
