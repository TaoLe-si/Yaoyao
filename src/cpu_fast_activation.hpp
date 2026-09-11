#pragma once
// 廉价门控激活：替换 S += sigmoid(g) * (tanh(u) - S) 中的 sigmoid 与 tanh。
//
// 动机（见 docs/architecture-proposals/02-cpu-decode-throughput.md §22）：
// 该状态更新是标量循环、每元素一次 std::exp 与一次 std::tanh，**完全串行**。
// 每 token 5120 次，实测 53.5 µs/token，占 8 线程单 token（369 µs）的 14.5%，
// 并且是「不可并行」部分的主体 —— 它把绝对吞吐天花板锁在约 13 000 tps。
//
// 本文件用 tanh 的三阶 Padé 近似：
//     tanh(x) ≈ x(27 + x^2) / (27 + 9x^2)，|x|>3 时夹紧到 ±1
//     sigmoid(x) = 0.5 * (1 + tanh(x))
// 在 [-6,6] 上 tanh 最大绝对误差 0.0235（实测 120001 点），
// AVX2 向量化后 1.08 µs/token，比标量 std::exp+std::tanh 快 49.5×。
//
// 注意：这**改变数值**，因此默认关闭（TAO_FAST_ACT 或运行期开关）。
// 开启后 checksum 必然变化 —— 它属于「架构升级」，必须在共享/新激活上重新训练。
#include <cstddef>
#ifdef TAO_CPU_AVX2
#include <immintrin.h>
#endif

namespace tao::dual {

// 标量参考实现（无 AVX2 时使用；也是 AVX2 路径的语义定义）
inline float fast_tanh(float x){
    if(x> 3.0f)return  1.0f;
    if(x<-3.0f)return -1.0f;
    const float q=x*x;
    return x*(27.0f+q)/(27.0f+9.0f*q);
}
inline float fast_sigmoid(float x){return 0.5f*(1.0f+fast_tanh(x));}

// 就地把 v 中每个元素做 tanh 近似
inline void fast_tanh_vec(float* v,size_t n){
#ifdef TAO_CPU_AVX2
    const __m256 one=_mm256_set1_ps(1.0f),lo=_mm256_set1_ps(-3.0f),hi=_mm256_set1_ps(3.0f);
    const __m256 c27=_mm256_set1_ps(27.0f),c9=_mm256_set1_ps(9.0f);
    size_t j=0;
    for(;j+8<=n;j+=8){
        __m256 x=_mm256_loadu_ps(v+j);
        x=_mm256_max_ps(lo,_mm256_min_ps(hi,x));
        __m256 q=_mm256_mul_ps(x,x);
        __m256 num=_mm256_mul_ps(x,_mm256_add_ps(c27,q));
        __m256 den=_mm256_add_ps(c27,_mm256_mul_ps(c9,q));
        _mm256_storeu_ps(v+j,_mm256_div_ps(num,den));
    }
    for(;j<n;++j)v[j]=fast_tanh(v[j]);
#else
    for(size_t j=0;j<n;++j)v[j]=fast_tanh(v[j]);
#endif
}

// 门控状态更新： state[j] += sigmoid(gate[j]) * (cand[j] - state[j])
// cand 先就地被 tanh 近似覆盖（调用方不应再使用它）。
// 元素之间互不依赖，故任意向量宽度都得到相同结果（便于将来加宽）。
inline void fast_gated_update(float* state,float* cand,const float* gate,size_t n){
#ifdef TAO_CPU_AVX2
    const __m256 half=_mm256_set1_ps(0.5f),one=_mm256_set1_ps(1.0f);
    const __m256 lo=_mm256_set1_ps(-3.0f),hi=_mm256_set1_ps(3.0f);
    const __m256 c27=_mm256_set1_ps(27.0f),c9=_mm256_set1_ps(9.0f);
    size_t j=0;
    for(;j+8<=n;j+=8){
        __m256 u=_mm256_loadu_ps(cand+j);
        u=_mm256_max_ps(lo,_mm256_min_ps(hi,u));
        __m256 q=_mm256_mul_ps(u,u);
        __m256 tu=_mm256_div_ps(_mm256_mul_ps(u,_mm256_add_ps(c27,q)),
                                _mm256_add_ps(c27,_mm256_mul_ps(c9,q)));
        __m256 a=_mm256_loadu_ps(gate+j);
        a=_mm256_max_ps(lo,_mm256_min_ps(hi,a));
        __m256 qa=_mm256_mul_ps(a,a);
        __m256 ta=_mm256_div_ps(_mm256_mul_ps(a,_mm256_add_ps(c27,qa)),
                                _mm256_add_ps(c27,_mm256_mul_ps(c9,qa)));
        __m256 sg=_mm256_mul_ps(half,_mm256_add_ps(one,ta));
        __m256 st=_mm256_loadu_ps(state+j);
        _mm256_storeu_ps(state+j,_mm256_add_ps(st,_mm256_mul_ps(sg,_mm256_sub_ps(tu,st))));
    }
    for(;j<n;++j)state[j]+=fast_sigmoid(gate[j])*(fast_tanh(cand[j])-state[j]);
#else
    for(size_t j=0;j<n;++j)state[j]+=fast_sigmoid(gate[j])*(fast_tanh(cand[j])-state[j]);
#endif
}

} // namespace tao::dual
