#include "cpu_row_parallel_pool.hpp"
#include <cstdio>
#include <chrono>
using namespace tao::dual;
int main(){
    const bool spin=std::getenv("TAO_POOL_SPIN")!=nullptr;
    printf("空分发纯 fork-join 屏障开销  TAO_POOL_SPIN=%s\n",spin?"1":"0");
    printf("线程    空分发us     每行us\n");
    for(unsigned nt : {1u,2u,4u,8u}){
        CpuRowParallelPool pool(nt);
        pool.set_minimum_dispatch_cost(0);
        for(int i=0;i<300;++i)pool.run(size_t(nt)*64,1,[](size_t,size_t,size_t){});
        const int N=30000;
        auto t0=std::chrono::steady_clock::now();
        for(int i=0;i<N;++i)pool.run(size_t(nt)*64,1,[](size_t,size_t b,size_t e){
            volatile size_t s=0;for(size_t k=b;k<e;++k)s+=k;});
        auto t1=std::chrono::steady_clock::now();
        double us=std::chrono::duration<double>(t1-t0).count()*1e6/N;
        printf("%4u %11.3f %9.4f\n",nt,us,us/(nt*64));
    }
    return 0;
}