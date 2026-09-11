// 对比「自旋池」与「裸 std::thread」在同一内核、同一工作量下的多线程效率。
// 动机：§45.3 发现模型层栈 8 线程 31.6 G MAC/s，而裸线程微基准 44.78 G MAC/s。
#include "cpu_row_parallel_pool.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <thread>
#include <atomic>
#include <immintrin.h>
using namespace tao::dual;

static const size_t ROWS=512, COLS=512, NDISP=16;   // 16 次 dispatch，每次 512x512

static float g_w[ROWS*COLS], g_x[COLS], g_out[ROWS];
struct Row{ const float* w; const float* x; float* o; };

static void kernel(const Row& R,size_t begin,size_t end){
    for(size_t r=begin;r<end;++r){
        const float* p=R.w+r*COLS; __m256 s=_mm256_setzero_ps(); size_t j=0;
        for(;j+8<=COLS;j+=8)s=_mm256_add_ps(s,_mm256_mul_ps(_mm256_loadu_ps(p+j),_mm256_loadu_ps(R.x+j)));
        alignas(32) float lane[8]; _mm256_store_ps(lane,s);
        float z=0; for(float v:lane)z+=v; for(;j<COLS;++j)z+=p[j]*R.x[j];
        R.o[r]=z;
    }
}
static double now(){ return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc,char**argv){
    for(size_t i=0;i<ROWS*COLS;++i)g_w[i]=float((int(i*2654435761u%7))-3);
    for(size_t j=0;j<COLS;++j)g_x[j]=float(j%13)-6.0f;
    const unsigned T=argc>1?unsigned(atoi(argv[1])):8;
    const int REP=argc>2?atoi(argv[2]):40;
    const double MAC=double(ROWS)*COLS*NDISP;
    Row R{g_w,g_x,g_out};
    // ---- A: 自旋池 ----
    double bestA=1e30;
    { CpuRowParallelPool pool(T); pool.set_minimum_dispatch_cost(0);
      for(int rep=0;rep<REP;++rep){ double t0=now();
        for(size_t d=0;d<NDISP;++d) pool.run(ROWS,COLS,[&](size_t,size_t b,size_t e){kernel(R,b,e);});
        double dt=now()-t0; if(dt<bestA)bestA=dt; } }
    // ---- B: 裸持久线程 + 自旋栅栏 ----
    double bestB=1e30;
    { std::atomic<unsigned> gen{0}, done{0}; std::atomic<bool> stop{false};
      std::vector<std::thread> th; const unsigned W=T>1?T-1:0;
      for(unsigned k=1;k<T;++k) th.emplace_back([&,k]{
          unsigned seen=0;
          size_t b=ROWS*k/T, e=ROWS*(k+1)/T;
          while(true){ while(gen.load(std::memory_order_acquire)==seen){ if(stop.load(std::memory_order_relaxed))return; _mm_pause(); }
              seen=gen.load(std::memory_order_relaxed); if(stop.load(std::memory_order_relaxed))return;
              kernel(R,b,e); done.fetch_add(1,std::memory_order_release); } });
      for(int rep=0;rep<REP;++rep){ double t0=now();
        for(size_t d=0;d<NDISP;++d){ done.store(0,std::memory_order_relaxed);
            gen.fetch_add(1,std::memory_order_release);
            kernel(R,0,ROWS/T);
            while(done.load(std::memory_order_acquire)!=W)_mm_pause(); }
        double dt=now()-t0; if(dt<bestB)bestB=dt; }
      stop.store(true); gen.fetch_add(1,std::memory_order_release); for(auto&t:th)t.join(); }
    std::printf("T=%u 工作=%.2f M MAC x %d 次\n",T,MAC/1e6,NDISP);
    std::printf("  自旋池    %7.3f ms  %7.2f G MAC/s\n",bestA*1e3,MAC/bestA/1e9);
    std::printf("  裸线程    %7.3f ms  %7.2f G MAC/s\n",bestB*1e3,MAC/bestB/1e9);
    std::printf("  池/裸线程 %7.3f x\n",bestB/bestA);
    return 0;
}
