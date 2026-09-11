// 多线程内核扩展：把「内核本身不扩展」与「模型的分发/串行开销」分开。
//
// 已知：8 线程解码有效吞吐 47.1 G MAC/s（14.68M / 0.3116ms）。
// 单线程内核约 13-14.6 G MAC/s => 8 线程理想 117 G，频率修正(÷1.346) 86.7 G。
// 效率仅 54%。本测试回答：损失在内核，还是在模型？
//
// 同时测「每核工作集」对吞吐的影响：
//   R3+R4 权重 14.68 MB / 8 核 = 1.84 MB/核 > L2(1 MB) => 走 L3
//   若 2-bit 打包 => 0.46 MB/核 < L2 => 可能显著变快
//   这决定了 §五死路清单里「位打包 0×」的结论是否需要订正。
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <vector>
#include <thread>
#include <atomic>
#include <cstring>
#include <immintrin.h>

static float dot_row(const int8_t* p,const float* x,size_t cols,float alpha){
    __m256 sum=_mm256_setzero_ps(),a8=_mm256_set1_ps(alpha);size_t j=0;
    for(;j+16<=cols;j+=16){
        auto w0=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(p+j)))),a8);
        auto w1=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(p+j+8)))),a8);
        sum=_mm256_add_ps(sum,_mm256_mul_ps(w0,_mm256_loadu_ps(x+j)));
        sum=_mm256_add_ps(sum,_mm256_mul_ps(w1,_mm256_loadu_ps(x+j+8)));
    }
    alignas(32)float lane[8];_mm256_store_ps(lane,sum);
    float z=0;for(float v:lane)z+=v;
    for(;j<cols;++j)z+=(float(p[j])*alpha)*x[j];
    return z;
}

// 2-bit 打包内核：每字节存 4 个三元值（0,1,2 => -1,0,+1）。
// 注意：解包后逐位精确 —— 值完全相同，只是存储格式不同。
static float dot_row_packed(const uint8_t* pk,const float* x,size_t cols,float alpha){
    __m256 sum=_mm256_setzero_ps(),a8=_mm256_set1_ps(alpha),one=_mm256_set1_ps(1.0f);
    size_t j=0;
    for(;j+8<=cols;j+=8){
        uint32_t w=0;size_t nb=(cols-j)>=4?2:1;
        if(nb==2){w=pk[j/4]|(uint32_t(pk[j/4+1])<<8);}
        else w=pk[j/4];
        alignas(32)int32_t q[8];
        for(int k=0;k<8;++k)q[k]=(int32_t)((w>>(2*k))&3u)-1;
        __m256 wv=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_load_si256((const __m256i*)q)),a8);
        sum=_mm256_add_ps(sum,_mm256_mul_ps(wv,_mm256_loadu_ps(x+j)));
    }
    (void)one;
    alignas(32)float lane[8];_mm256_store_ps(lane,sum);
    float z=0;for(float v:lane)z+=v;
    for(;j<cols;++j)z+=(float(((pk[j/4]>>(2*(j%4)))&3u)-1)*alpha)*x[j];
    return z;
}

#include <immintrin.h>
static float dot_512(const int8_t* p,const float* x,size_t cols,float alpha){
    __m512 sum=_mm512_setzero_ps(),a16=_mm512_set1_ps(alpha);size_t j=0;
    for(;j+32<=cols;j+=32){
        auto w0=_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(p+j)))),a16);
        auto w1=_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(p+j+16)))),a16);
        sum=_mm512_add_ps(sum,_mm512_mul_ps(w0,_mm512_loadu_ps(x+j)));
        sum=_mm512_add_ps(sum,_mm512_mul_ps(w1,_mm512_loadu_ps(x+j+16)));
    }
    alignas(64)float lane[16];_mm512_store_ps(lane,sum);
    float z=0;for(float v:lane)z+=v;
    for(;j<cols;++j)z+=(float(p[j])*alpha)*x[j];
    return z;
}

int main(){
    const size_t rows=4096,cols=512;
    std::vector<int8_t> q(rows*cols);
    std::vector<float> scale(rows),x(cols);
    unsigned seed=4242;auto nx=[&](){seed=seed*1664525u+1013904223u;return seed;};
    for(auto&v:q)v=(int8_t)((nx()>>16)%3)-1;
    for(auto&v:scale)v=0.001f+0.0001f*((nx()>>16)%7);
    for(auto&v:x)v=(nx()>>8)*(1.0f/16777216.0f)-0.5f;
    std::vector<uint8_t> pk(rows*cols/4+1);
    for(size_t i=0;i<rows*cols;++i){
        uint32_t v=uint32_t(q[i]+1);
        pk[i/4]|=uint8_t(v<<(2*(i%4)));
    }
    printf("=== 内核多线程扩展（%zux%zu，权重 %.2f MB int8 / %.2f MB 2bit）===\n",
           rows,cols,double(rows)*cols/1e6,double(rows)*cols/4/1e6);

    // 逐位精确性
    size_t diff=0;
    for(size_t r=0;r<rows;r+=97){
        float a=dot_row(q.data()+r*cols,x.data(),cols,scale[r]);
        float b=dot_row_packed(pk.data()+r*cols/4,x.data(),cols,scale[r]);
        if(std::memcmp(&a,&b,4)!=0)++diff;
    }
    printf("  2-bit 打包逐位相等: %zu 处不同 (抽样 %zu 行)  %s\n",diff,rows/97+1,diff==0?"✓":"✗");

    auto run=[&](int nt,bool packed,int reps,bool k512=false){
        std::atomic<size_t> sink{0};
        std::vector<std::thread> ts;
        auto t0=std::chrono::steady_clock::now();
        for(int t=0;t<nt;++t)ts.emplace_back([&,t]{
            size_t lo=rows*t/nt,hi=rows*(t+1)/nt;float s=0;
            for(int k=0;k<reps;++k)
                for(size_t r=lo;r<hi;++r)
                    s+= packed?dot_row_packed(pk.data()+r*cols/4,x.data(),cols,scale[r])
                              : (k512?dot_512(q.data()+r*cols,x.data(),cols,scale[r])
                                     :dot_row(q.data()+r*cols,x.data(),cols,scale[r]));
            if(s==123456789.0f)sink+=1;
        });
        for(auto&v:ts)v.join();
        auto t1=std::chrono::steady_clock::now();
        double sec=std::chrono::duration<double>(t1-t0).count();
        return double(rows)*cols*reps/sec/1e9;
    };
    const int reps=12;
    printf("\n  线程   int8 G/s   2bit G/s   AVX512 G/s   512/int8\n");
    double base=0;
    for(int nt:{1,2,4,8}){
        double a=run(nt,false,reps),b=run(nt,true,reps),c=run(nt,false,reps,true);
        if(nt==1)base=a;
        printf("  %4d %9.2f %10.2f %12.2f %10.2fx   (int8 扩展 %.2fx)\n",nt,a,b,c,c/a,a/base);
    }
    return 0;
}
