// 量 m 状态更新循环（sigmoid + std::tanh）的纯成本。
// 真实调用量：8 层 × c.m(512) = 4096 次/ token。
#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>
#include <cstdint>

static float sigmoid(float x){if(x>=0)return 1/(1+std::exp(-x));float e=std::exp(x);return e/(1+e);}

int main(){
    const size_t N=4096;              // 每 token 的真实调用次数
    const int tokens=20000;
    std::vector<float> v(N),g(N),m(N);
    for(size_t i=0;i<N;++i){v[i]=std::sin(float(i))*2.0f;g[i]=std::cos(float(i))*2.0f;m[i]=0.1f;}

    // 1) 现状：标量循环
    auto t0=std::chrono::steady_clock::now();
    for(int t=0;t<tokens;++t)
        for(size_t j=0;j<N;++j) m[j]+=sigmoid(g[j])*(std::tanh(v[j])-m[j]);
    auto t1=std::chrono::steady_clock::now();
    double scalar=std::chrono::duration<double>(t1-t0).count()/tokens*1e6;

    // 2) 只算 tanh（看 sigmoid 占多少）
    t0=std::chrono::steady_clock::now();
    float acc=0;
    for(int t=0;t<tokens;++t)
        for(size_t j=0;j<N;++j) acc+=std::tanh(v[j]);
    t1=std::chrono::steady_clock::now();
    double onlytanh=std::chrono::duration<double>(t1-t0).count()/tokens*1e6;

    // 3) 无超越函数的同形状循环（下界参考）
    t0=std::chrono::steady_clock::now();
    for(int t=0;t<tokens;++t)
        for(size_t j=0;j<N;++j) m[j]+=g[j]*(v[j]-m[j]);
    t1=std::chrono::steady_clock::now();
    double plain=std::chrono::duration<double>(t1-t0).count()/tokens*1e6;

    printf("每 token 4096 次更新:\n");
    printf("  现状（sigmoid+tanh） %8.3f us/token   (%.1f ns/元素)\n",scalar,scalar*1000/N);
    printf("  仅 tanh              %8.3f us/token\n",onlytanh);
    printf("  无超越函数           %8.3f us/token\n",plain);
    printf("  ⇒ 超越函数开销       %8.3f us/token\n",scalar-plain);
    printf("\n参照：当前 8 线程单 token 约 369-393 us (2651-2765 tps)\n");
    printf("  ⇒ 占 %.1f%%\n",(scalar-plain)/380*100);
    printf("  (acc=%.1f，防止优化掉)\n",acc);
    return 0;
}
