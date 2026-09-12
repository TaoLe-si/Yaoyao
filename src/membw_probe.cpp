// membw_probe.cpp -- 顺序读/写内存带宽探针。
// 解码是带宽墙限制的，所以任何"模型变大后解码多快"的回答都必须以本机
// 实测带宽为分母，而不是以理论峰值或别的机器的数字。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
int main(int argc,char**argv){
    const size_t N = argc>1 ? (size_t)strtoull(argv[1],nullptr,10) : (size_t)(768ull<<20);
    std::vector<char> buf(N);
    std::memset(buf.data(),1,N);
    double bestR=0;
    for(int rep=0;rep<4;++rep){
        volatile unsigned long long sum=0;
        auto t0=std::chrono::steady_clock::now();
        for(size_t i=0;i<N;i+=64) sum+=buf[i];          // 每 64B 读 1 字节 => 走满每条 cache line
        auto t1=std::chrono::steady_clock::now();
        double s=std::chrono::duration<double>(t1-t0).count();
        double g=double(N)/s/1e9; if(g>bestR)bestR=g;
    }
    double bestW=0;
    for(int rep=0;rep<4;++rep){
        auto t0=std::chrono::steady_clock::now();
        std::memset(buf.data(),rep+2,N);
        auto t1=std::chrono::steady_clock::now();
        double s=std::chrono::duration<double>(t1-t0).count();
        double g=double(N)/s/1e9; if(g>bestW)bestW=g;
    }
    std::printf("MEMBW read=%.2f GB/s write=%.2f GB/s (N=%zu MB)\n",bestR,bestW,N>>20);
    return 0;
}
