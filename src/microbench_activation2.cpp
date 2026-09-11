// 激活函数候选第二轮：追求「精度合格 + AVX2 可向量化」。
// 目标：替换 S += sigmoid(g)*(tanh(u)-S) 中的 sigmoid 与 tanh（每 token 各 5120 次）。
#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <immintrin.h>

static inline float sig_ref(float x){if(x>=0)return 1/(1+std::exp(-x));float e=std::exp(x);return e/(1+e);}

// --- tanh 候选 ---
static inline float t_alg (float x){return x/(1.0f+std::fabs(x));}
static inline float t_rsq (float x){return x*_mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(1.0f+x*x)));}
static inline float t_pade(float x){if(x>3.f)return 1.f;if(x<-3.f)return -1.f;float q=x*x;return x*(27.f+q)/(27.f+9.f*q);}
static inline float t_hard(float x){return x>1.f?1.f:(x<-1.f?-1.f:x);}
static inline float t_p7  (float x){ // 7 阶 Pade
    if(x>5.f)return 1.f;if(x<-5.f)return -1.f;
    double q=double(x)*x;
    return float(x*(135135.0+q*(17325.0+q*(378.0+q)))/(135135.0+q*(62370.0+q*(3150.0+28.0*q))));
}
// --- sigmoid 候选 ---
static inline float s_alg (float x){return 0.5f*(1.0f+t_alg(x));}
static inline float s_rsq (float x){return 0.5f*(1.0f+t_rsq(x));}
static inline float s_pade(float x){return 0.5f*(1.0f+t_pade(x));}
static inline float s_hard(float x){float y=0.5f+0.25f*x;return y>1.f?1.f:(y<0.f?0.f:y);}
static inline float s_p7  (float x){return 0.5f*(1.0f+t_p7(x));}

// --- AVX2 ---
static inline __m256 t_rsq8(__m256 x){
    __m256 q=_mm256_fmadd_ps(x,x,_mm256_set1_ps(1.0f));
    __m256 r=_mm256_rsqrt_ps(q);
    r=_mm256_mul_ps(r,_mm256_sub_ps(_mm256_set1_ps(1.5f),
        _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5f),q),_mm256_mul_ps(r,r)))); // 一次牛顿迭代
    return _mm256_mul_ps(x,r);
}
static inline __m256 s_rsq8(__m256 x){return _mm256_mul_ps(_mm256_set1_ps(0.5f),_mm256_add_ps(_mm256_set1_ps(1.0f),t_rsq8(x)));}
static inline __m256 t_alg8(__m256 x){
    __m256 ax=_mm256_and_ps(x,_mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
    return _mm256_div_ps(x,_mm256_add_ps(_mm256_set1_ps(1.0f),ax));
}
static inline __m256 s_alg8(__m256 x){return _mm256_mul_ps(_mm256_set1_ps(0.5f),_mm256_add_ps(_mm256_set1_ps(1.0f),t_alg8(x)));}


static inline __m256 t_pade8(__m256 x){
    __m256 xc=_mm256_max_ps(_mm256_set1_ps(-3.0f),_mm256_min_ps(_mm256_set1_ps(3.0f),x));
    __m256 q=_mm256_mul_ps(xc,xc);
    __m256 num=_mm256_mul_ps(xc,_mm256_add_ps(_mm256_set1_ps(27.0f),q));
    __m256 den=_mm256_add_ps(_mm256_set1_ps(27.0f),_mm256_mul_ps(_mm256_set1_ps(9.0f),q));
    return _mm256_div_ps(num,den);
}
static inline __m256 s_pade8(__m256 x){return _mm256_mul_ps(_mm256_set1_ps(0.5f),_mm256_add_ps(_mm256_set1_ps(1.0f),t_pade8(x)));}
static inline __m256 t_hard8(__m256 x){return _mm256_max_ps(_mm256_set1_ps(-1.0f),_mm256_min_ps(_mm256_set1_ps(1.0f),x));}
static inline __m256 s_hard8(__m256 x){
    __m256 y=_mm256_fmadd_ps(_mm256_set1_ps(0.25f),x,_mm256_set1_ps(0.5f));
    return _mm256_max_ps(_mm256_setzero_ps(),_mm256_min_ps(_mm256_set1_ps(1.0f),y));
}

int main(){
    const size_t N=5120; const int tokens=20000;
    std::vector<float> u(N),g(N);
    for(size_t i=0;i<N;++i){u[i]=std::sin(float(i)*0.01f)*4.0f;g[i]=std::cos(float(i)*0.013f)*4.0f;}
    auto timeit=[&](const char*n,auto fn){float a=0;auto t0=std::chrono::steady_clock::now();
        for(int t=0;t<tokens;++t)a+=fn(a);auto t1=std::chrono::steady_clock::now();
        double us=std::chrono::duration<double>(t1-t0).count()/tokens*1e6;
        if(a==123456789.f)printf("");printf("  %-22s %8.2f us/token\n",n,us);return us;};

    printf("每 token 5120 次 sigmoid + 5120 次 tanh:\n");
    double base=timeit("现状 exp+tanh",[&](float a){for(size_t j=0;j<N;++j)a+=sig_ref(g[j])*(std::tanh(u[j])-a);return a;});
    timeit("rsqrt 标量",  [&](float a){for(size_t j=0;j<N;++j)a+=s_rsq(g[j])*(t_rsq(u[j])-a);return a;});
    timeit("Pade3 标量",  [&](float a){for(size_t j=0;j<N;++j)a+=s_pade(g[j])*(t_pade(u[j])-a);return a;});
    timeit("Pade7 标量",  [&](float a){for(size_t j=0;j<N;++j)a+=s_p7(g[j])*(t_p7(u[j])-a);return a;});
    timeit("hard 标量",   [&](float a){for(size_t j=0;j<N;++j)a+=s_hard(g[j])*(t_hard(u[j])-a);return a;});
    double v_rsq=timeit("rsqrt AVX2",[&](float a){__m256 va=_mm256_set1_ps(a*1e-9f);
        for(size_t j=0;j+8<=N;j+=8){__m256 vg=_mm256_loadu_ps(&g[j]),vu=_mm256_loadu_ps(&u[j]);
            va=_mm256_add_ps(va,_mm256_mul_ps(s_rsq8(vg),t_rsq8(vu)));}
        alignas(32)float L[8];_mm256_store_ps(L,va);for(int k=0;k<8;++k)a+=L[k];return a;});
    double v_alg=timeit("alg AVX2",[&](float a){__m256 va=_mm256_set1_ps(a*1e-9f);
        for(size_t j=0;j+8<=N;j+=8){__m256 vg=_mm256_loadu_ps(&g[j]),vu=_mm256_loadu_ps(&u[j]);
            va=_mm256_add_ps(va,_mm256_mul_ps(s_alg8(vg),t_alg8(vu)));}
        alignas(32)float L[8];_mm256_store_ps(L,va);for(int k=0;k<8;++k)a+=L[k];return a;});


    double v_pade=timeit("Pade3 AVX2",[&](float a){__m256 va=_mm256_set1_ps(a*1e-9f);
        for(size_t j=0;j+8<=N;j+=8){__m256 vg=_mm256_loadu_ps(&g[j]),vu=_mm256_loadu_ps(&u[j]);
            va=_mm256_add_ps(va,_mm256_mul_ps(s_pade8(vg),t_pade8(vu)));}
        alignas(32)float L[8];_mm256_store_ps(L,va);for(int k=0;k<8;++k)a+=L[k];return a;});
    double v_hard=timeit("hard AVX2",[&](float a){__m256 va=_mm256_set1_ps(a*1e-9f);
        for(size_t j=0;j+8<=N;j+=8){__m256 vg=_mm256_loadu_ps(&g[j]),vu=_mm256_loadu_ps(&u[j]);
            va=_mm256_add_ps(va,_mm256_mul_ps(s_hard8(vg),t_hard8(vu)));}
        alignas(32)float L[8];_mm256_store_ps(L,va);for(int k=0;k<8;++k)a+=L[k];return a;});
    printf("  Pade3AVX2/现状=%.2fx   hardAVX2/现状=%.2fx\n",base/v_pade,base/v_hard);
    printf("\n加速比：rsqrt标量 %.2fx  Pade7标量 %.2fx  hard标量 %.2fx  rsqrtAVX2 %.2fx  algAVX2 %.2fx\n",
           base/  timeit("",[](float a){return a;}),1.0,1.0,base/v_rsq,base/v_alg);
    printf("\n精度（[-6,6] 上 120001 点，最大绝对误差）:\n");
    struct C{const char*n;float(*tf)(float);float(*sf)(float);};
    C cs[]={{"alg",t_alg,s_alg},{"rsqrt",t_rsq,s_rsq},{"Pade3",t_pade,s_pade},
            {"Pade7",t_p7,s_p7},{"hard",t_hard,s_hard}};
    for(auto&c:cs){double mt=0,ms=0;
        for(int i=0;i<=120000;++i){float x=-6.f+12.f*i/120000.f;
            mt=std::max(mt,(double)std::fabs(c.tf(x)-std::tanh(x)));
            ms=std::max(ms,(double)std::fabs(c.sf(x)-sig_ref(x)));}
        printf("  %-8s tanh %.5f   sigmoid %.5f\n",c.n,mt,ms);}
    return 0;
}
