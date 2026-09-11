// 预计算 float 权重内核 vs 现有 int8 三元内核。
// 动机：现有 dot 每 16 权重 14 条指令（cvtepi8->cvtepi32->cvtdq2ps->mul alpha->mul x->add），
// 其中 int8->float 转换是纯开销。q[j]*scale[row] 是每行常量，可在装载时预计算。
//
// 关键：为保持**逐位相同**，预计算版必须沿用 mul+add 而非 FMA
//（FMA 不round中间乘积，会改变结果）。且累加顺序必须一致。
//
// 风险：float32 权重是 int8 的 4 倍字节。小矩阵可忽略，大矩阵可能撞带宽墙。
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <vector>
#include <cstring>
#include <immintrin.h>

// 现有实现（与 cpu_pipeline_rows.hpp 的 PipelineRows::dot 完全一致）
static float dot_ref(const int8_t* p, const float* x, size_t cols, float alpha){
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

// 预计算版（单行）：同样的 mul+add 顺序，只是权重已展开为 float
static float dot_pre(const float* w, const float* x, size_t cols){
    __m256 sum=_mm256_setzero_ps();size_t j=0;
    for(;j+16<=cols;j+=16){
        sum=_mm256_add_ps(sum,_mm256_mul_ps(_mm256_loadu_ps(w+j),_mm256_loadu_ps(x+j)));
        sum=_mm256_add_ps(sum,_mm256_mul_ps(_mm256_loadu_ps(w+j+8),_mm256_loadu_ps(x+j+8)));
    }
    for(;j+8<=cols;j+=8)sum=_mm256_add_ps(sum,_mm256_mul_ps(_mm256_loadu_ps(w+j),_mm256_loadu_ps(x+j)));
    alignas(32)float lane[8];_mm256_store_ps(lane,sum);
    float z=0;for(float v:lane)z+=v;
    for(;j<cols;++j)z+=w[j]*x[j];
    return z;
}

// 预计算版（4 行同时）：每 4 行摊薄一次 x 加载
static void dot_pre4(const float* w,size_t cols,const float* x,float* out,size_t r0){
    __m256 s0=_mm256_setzero_ps(),s1=_mm256_setzero_ps(),s2=_mm256_setzero_ps(),s3=_mm256_setzero_ps();
    const float* w0=w+(r0+0)*cols,*w1=w+(r0+1)*cols,*w2=w+(r0+2)*cols,*w3=w+(r0+3)*cols;
    for(size_t j=0;j+8<=cols;j+=8){
        __m256 xv=_mm256_loadu_ps(x+j);
        s0=_mm256_add_ps(s0,_mm256_mul_ps(_mm256_loadu_ps(w0+j),xv));
        s1=_mm256_add_ps(s1,_mm256_mul_ps(_mm256_loadu_ps(w1+j),xv));
        s2=_mm256_add_ps(s2,_mm256_mul_ps(_mm256_loadu_ps(w2+j),xv));
        s3=_mm256_add_ps(s3,_mm256_mul_ps(_mm256_loadu_ps(w3+j),xv));
    }
    alignas(32)float l0[8],l1[8],l2[8],l3[8];
    _mm256_store_ps(l0,s0);_mm256_store_ps(l1,s1);_mm256_store_ps(l2,s2);_mm256_store_ps(l3,s3);
    float a0=0,a1=0,a2=0,a3=0;
    for(int k=0;k<8;++k){a0+=l0[k];a1+=l1[k];a2+=l2[k];a3+=l3[k];}
    out[r0]=a0;out[r0+1]=a1;out[r0+2]=a2;out[r0+3]=a3;
}

int main(int argc,char**argv){
    size_t rows=argc>1?strtoull(argv[1],nullptr,10):512;
    size_t cols=argc>2?strtoull(argv[2],nullptr,10):512;
    int reps=argc>3?atoi(argv[3]):40;
    std::vector<int8_t> q(rows*cols);
    std::vector<float> scale(rows),x(cols);
    unsigned seed=12345;
    auto rnd=[&](){seed=seed*1664525u+1013904223u;return (int)((seed>>16)%3)-1;};
    for(auto&v:q)v=(int8_t)rnd();
    for(auto&v:scale)v=0.001f+0.0001f*rnd();
    for(auto&v:x)v=((seed=seed*1664525u+1013904223u)>>8)*(1.0f/16777216.0f)-0.5f;

    std::vector<float> wf(rows*cols);
    for(size_t r=0;r<rows;++r)for(size_t c=0;c<cols;++c)wf[r*cols+c]=float(q[r*cols+c])*scale[r];

    // 1) 逐位相等检查
    size_t diff=0;
    for(size_t r=0;r<rows;++r){
        float a=dot_ref(q.data()+r*cols,x.data(),cols,scale[r]);
        float b=dot_pre(wf.data()+r*cols,x.data(),cols);
        if(std::memcmp(&a,&b,4)!=0)++diff;
    }
    printf("=== 行=%zu 列=%zu ===\n",rows,cols);
    printf("  逐位相等检查: %zu / %zu 行不同  %s\n",diff,rows,diff==0?"✓ 完全一致":"✗ 有差异");

    std::vector<float> out(rows);
    auto bench=[&](const char*name,auto fn){
        float sink=0;
        auto t0=std::chrono::steady_clock::now();
        for(int k=0;k<reps;++k)fn(sink);
        auto t1=std::chrono::steady_clock::now();
        double sec=std::chrono::duration<double>(t1-t0).count();
        double mac=double(rows)*cols*reps;
        if(sink==123456789.0f)printf("");
        printf("  %-22s %8.3f ms/pass  %7.2f G MAC/s\n",name,sec*1000/reps,mac/sec/1e9);
        return mac/sec/1e9;
    };
    double a=bench("现有 int8 三元",[&](float&s){for(size_t r=0;r<rows;++r)s+=dot_ref(q.data()+r*cols,x.data(),cols,scale[r]);});
    double b=bench("预计算 float 单行",[&](float&s){for(size_t r=0;r<rows;++r)s+=dot_pre(wf.data()+r*cols,x.data(),cols);});
    double c=bench("预计算 float 4行",[&](float&s){for(size_t r=0;r+4<=rows;r+=4){dot_pre4(wf.data(),cols,x.data(),out.data(),r);}for(size_t r=0;r<rows;++r)s+=out[r];});
    printf("  倍数: 单行 %.2fx   4行 %.2fx\n",b/a,c/a);
    printf("  字节/行: int8 %zu   float %zu (%.1fx)\n",cols,cols*4,4.0);
    printf("  总字节: int8 %.2f MB   float %.2f MB\n",double(rows)*cols/1e6,double(rows)*cols*4/1e6);
    return 0;
}
