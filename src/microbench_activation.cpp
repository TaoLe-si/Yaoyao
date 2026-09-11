// 门控循环更新 S += sigmoid(g)*(tanh(u)-S) 的激活函数候选：速度 + 精度。
// 真实调用量：每 token 5120 次 sigmoid + 5120 次 tanh（8 层 × (128 s + 512 m)）。
#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <immintrin.h>

static inline float sig_ref(float x){if(x>=0)return 1/(1+std::exp(-x));float e=std::exp(x);return e/(1+e);}
static inline float tanh_ref(float x){return std::tanh(x);}

// 候选 1：代数式（无 exp）。tanh 型 x/(1+|x|)，sigmoid 型 0.5*(1+x/(1+|x|))
static inline float tanh_alg(float x){return x/(1.0f+std::fabs(x));}
static inline float sig_alg(float x){return 0.5f*(1.0f+tanh_alg(x));}

// 候选 2：tanh 的 Padé 近似 x(27+x^2)/(27+9x^2)，需夹紧
static inline float tanh_pade(float x){
    if(x> 3.0f)return 1.0f; if(x<-3.0f)return -1.0f;
    float x2=x*x; return x*(27.0f+x2)/(27.0f+9.0f*x2);
}
static inline float sig_pade(float x){return 0.5f*(1.0f+tanh_pade(x));}

// 候选 3：AVX2 向量化的代数式
static inline __m256 tanh_alg8(__m256 x){
    __m256 ax=_mm256_and_ps(x,_mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
    return _mm256_div_ps(x,_mm256_add_ps(_mm256_set1_ps(1.0f),ax));
}
static inline __m256 sig_alg8(__m256 x){
    return _mm256_mul_ps(_mm256_set1_ps(0.5f),_mm256_add_ps(_mm256_set1_ps(1.0f),tanh_alg8(x)));
}

int main(){
    const size_t N=5120;          // 每 token 真实调用次数
    const int tokens=20000;
    std::vector<float> u(N),g(N);
    for(size_t i=0;i<N;++i){u[i]=std::sin(float(i)*0.01f)*4.0f;g[i]=std::cos(float(i)*0.013f)*4.0f;}

    auto timeit=[&](const char*name,auto fn){
        float acc=0;
        auto t0=std::chrono::steady_clock::now();
        for(int t=0;t<tokens;++t)acc+=fn(acc);
        auto t1=std::chrono::steady_clock::now();
        double us=std::chrono::duration<double>(t1-t0).count()/tokens*1e6;
        if(acc==123456789.0f)printf("");
        printf("  %-26s %9.2f us/token\n",name,us);
        return us;
    };

    printf("每 token 5120 次 sigmoid + 5120 次 tanh:\n");
    double base=timeit("现状 std::exp + std::tanh",[&](float a){for(size_t j=0;j<N;++j)a+=sig_ref(g[j])*(tanh_ref(u[j])-a);return a;});
    double alg =timeit("代数式 标量",[&](float a){for(size_t j=0;j<N;++j)a+=sig_alg(g[j])*(tanh_alg(u[j])-a);return a;});
    double pade=timeit("Pade 标量",[&](float a){for(size_t j=0;j<N;++j)a+=sig_pade(g[j])*(tanh_pade(u[j])-a);return a;});
    double vec =timeit("代数式 AVX2 向量化",[&](float a){
        __m256 va=_mm256_set1_ps(a*1e-9f);
        for(size_t j=0;j+8<=N;j+=8){
            __m256 vg=_mm256_loadu_ps(&g[j]),vu=_mm256_loadu_ps(&u[j]);
            va=_mm256_add_ps(va,_mm256_mul_ps(sig_alg8(vg),tanh_alg8(vu)));
        }
        alignas(32)float lane[8];_mm256_store_ps(lane,va);
        for(int k=0;k<8;++k)a+=lane[k];return a;});

    printf("\n相对现状加速：代数式 %.2fx  Pade %.2fx  AVX2 %.2fx\n",base/alg,base/pade,base/vec);

    // 精度
    printf("\n精度（对真实 tanh / sigmoid，在 [-6,6] 上 100001 点）：\n");
    double mt=0,ms=0,mt2=0,ms2=0,rt=0,rs=0;
    for(int i=0;i<=100000;++i){
        float x=-6.0f+12.0f*i/100000.0f;
        double e1=std::fabs(tanh_alg(x)-std::tanh(x)), e2=std::fabs(tanh_pade(x)-std::tanh(x));
        double f1=std::fabs(sig_alg(x)-sig_ref(x)),  f2=std::fabs(sig_pade(x)-sig_ref(x));
        mt=std::max(mt,e1);ms=std::max(ms,f1);mt2=std::max(mt2,e2);ms2=std::max(ms2,f2);
        rt+=e1;rs+=f1;
    }
    printf("  代数式: tanh 最大误差 %.4f (平均 %.4f)   sigmoid 最大误差 %.4f (平均 %.4f)\n",mt,rt/100001,ms,rs/100001);
    printf("  Pade  : tanh 最大误差 %.4f              sigmoid 最大误差 %.4f\n",mt2,ms2);
    return 0;
}
