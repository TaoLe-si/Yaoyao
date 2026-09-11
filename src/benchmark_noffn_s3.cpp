#define NOMINMAX
#define TAO_INPUT_SCALE
#define TAO_CPU_AVX2
// Build with /DTAO_NO_PHASE_TIMING for an uninstrumented, apples-to-apples
// throughput comparison against the frozen baseline binary.
#ifndef TAO_NO_PHASE_TIMING
#define TAO_PHASE_TIMING
#endif
#include "tokenizer_file.hpp"
#include "greedy_resident_lifecycle.hpp"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <atomic>
#include <immintrin.h>
#include <cstdlib>
#include <new>
#include <malloc.h>
// 分配器三模式 A/B：plain(malloc/free) / count(计数，有原子争用) / pool(线程本地空闲链表)
// 目的：验证「每 token 176 次堆分配的跨线程争用是串行瓶颈」这一假说。
static std::atomic<size_t> g_alloc_n{0},g_alloc_b{0};
static int g_alloc_mode=0;   // 0=plain 1=count 2=pool
static thread_local void* tl_free[65];
static inline void* xalloc(size_t n){
    if(g_alloc_mode==1){g_alloc_n.fetch_add(1,std::memory_order_relaxed);g_alloc_b.fetch_add(n,std::memory_order_relaxed);}
    if(g_alloc_mode==2&&n>=64&&n<=4096){
        size_t c=(n+63)>>6;
        void* p=tl_free[c];
        if(p){tl_free[c]=*(void**)p;return p;}
    }
    void* p=std::malloc(n?n:1);if(!p)throw std::bad_alloc();return p;
}
static inline void xfree(void* p){
    if(!p)return;
    if(g_alloc_mode==2){size_t n=_msize(p);if(n>=64&&n<=4096){size_t c=(n+63)>>6;*(void**)p=tl_free[c];tl_free[c]=p;return;}}
    std::free(p);
}
void* operator new(size_t n){return xalloc(n);}
void* operator new[](size_t n){return xalloc(n);}
void operator delete(void*p)noexcept{xfree(p);}
void operator delete[](void*p)noexcept{xfree(p);}
void operator delete(void*p,size_t)noexcept{xfree(p);}
void operator delete[](void*p,size_t)noexcept{xfree(p);}
#include <string>
// S3 diagnostic: thread sweep + phase breakdown.
// Numerics are invariant under thread count, so every configuration must
// reproduce the frozen baseline checksum. Instrumented timings are diagnostic
// and carry stopwatch overhead; the uninstrumented NORMAL/FORCED numbers above
// them are the throughput evidence.
// 频率探针：依赖 mulsd 链（Zen4 延迟 3 周期），占空比 2ms/10ms，其余 sleep 让出核心。
struct FreqProbe{
    std::atomic<bool> stop{false};
    std::thread th;
    double ghz=0;int windows=0;
    void start(){th=std::thread([this]{
        using C=std::chrono::steady_clock;
        __m128d v=_mm_set_sd(1.0),c=_mm_set_sd(1.0000000001);
        double sum=0;int n=0;
        while(!stop.load(std::memory_order_relaxed)){
            auto a=C::now();uint64_t it=0;
            do{ for(int k=0;k<400;++k)v=_mm_mul_sd(v,c); it+=400; }
            while(std::chrono::duration<double>(C::now()-a).count()<0.002);
            double dt=std::chrono::duration<double>(C::now()-a).count();
            sum+=3.0*double(it)/dt/1e9;++n;
            std::this_thread::sleep_for(std::chrono::milliseconds(8));
        }
        ghz=n?sum/n:0;windows=n;
        if(_mm_cvtsd_f64(v)==12345.0)ghz=0;   // 防优化
    });}
    void finish(){stop.store(true,std::memory_order_relaxed);if(th.joinable())th.join();}
};
int main(int argc,char**argv){
    using namespace tao::dual;
    using Clock=std::chrono::steady_clock;
    if(const char* am=std::getenv("TAO_ALLOC"))g_alloc_mode=(std::string(am)=="count")?1:((std::string(am)=="pool")?2:0);
    auto seconds=[](auto a){return std::chrono::duration<double>(Clock::now()-a).count();};
    const char* path=argc>1?argv[1]:"build/noffn_fresh/step_1512/final.dsb";
    // argv[2] is a comma-separated thread list; a token of 0 expands to 1,2,4,8.
    // Interleaving repeats inside one process averages out machine drift.
    const char* spec=argc>2?argv[2]:"2";
    const size_t mincost=argc>3?size_t(strtoull(argv[3],nullptr,10)):262144;
    unsigned list[64];int n=0;
    for(const char* p=spec;*p&&n<64;){
        char* e=nullptr;long v=strtol(p,&e,10);
        if(e==p)break;
        if(v<=0){if(n+4<=64){list[n++]=1;list[n++]=2;list[n++]=4;list[n++]=8;}}
        else list[n++]=unsigned(v);
        p=e;if(*p==',')++p;
    }
    if(!n){list[n++]=2;}
    std::string hash;
    auto tok=tao::text::load_tokenizer("build/formal_tokenizer.bbp",hash);
    auto begin=Clock::now();
    GreedyPipelineGroupedModel model(read_compact_bundle(path,hash));
    printf("S3 LOAD db=%s load_seconds=%.6f resident_bytes=%zu layers=%u d=%u s=%u m=%u vocab=%u\n",
        path,seconds(begin),model.weight_bytes(),model.c.layers,model.c.d,model.c.s,model.c.m,model.c.vocab);
    auto prompt=tok.encode("你好，请用一句话介绍自己。");
    GreedyResidentSession session(model);
    // argv[4] = 层共享串（如 "0,0,0,0,4,4,4,4"）；给了就启用 R3+R4，
    // 使相位分解对应真正的最优配置（原基准跑的是未共享模型）。
    for(int k=0;k<n;++k){
        const unsigned T=list[k];
        // 关键：每次设线程数之前重新应用层共享（与 bench_interleaved 的 apply() 顺序一致）。
        if(argc>4){
            model.set_layer_share(GreedyPipelineGroupedModel::parseLayerShare(argv[4],model.c.layers));
            model.set_reuse_mv(true);
        }
        // 深度默认 2 层（applyDefaultArch）。仅当显式设置 TAO_LAYER_LIMIT 时覆盖。
        model.set_cpu_threads(T);
        if(argc>4&&k==0)printf("S3 LAYERSHARE share=%s reuse_mv=1\n",argv[4]);
        printf("S3 DUMP threads=%u %s\n",T,model.config_dump().c_str());
        model.set_row_parallel_minimum(mincost);
        for(int i=0;i<3;i++){session.reset();session.reply(prompt);}
        double total=0,ttft=0,decode=0;size_t decoded=0;int empty=0;
        for(int i=0;i<20;i++){session.reset();begin=Clock::now();auto r=session.reply(prompt);
            total+=seconds(begin);ttft+=r.ttft;decode+=r.decode_seconds;if(r.ids.size()>1)decoded+=r.ids.size()-1;empty+=r.ids.empty();}
        printf("S3 threads=%u NORMAL trials=20 prompt_tokens=%zu mean_reply_ms=%.6f mean_ttft_ms=%.6f decode_steps=%zu decode_seconds=%.6f tps=%.3f empty=%d\n",
            T,prompt.size(),total*1000/20,ttft*1000/20,decoded,decode,decode>0?decoded/decode:0,empty);
        printf("S3 threads=%u min_dispatch_cost=%zu\n",T,model.row_parallel_minimum());
        uint32_t sum=0;
        for(int round=0;round<5;round++){
            auto state=model.initial();
            model.advance(256,state);model.advance(257,state);
            for(auto t:prompt)model.advance(t,state);
            model.advance(259,state);
            begin=Clock::now();uint32_t next=model.greedy_step(258,state);double first=seconds(begin);
            FreqProbe fp;const bool probe=(round%2==1);if(probe)fp.start();
            const size_t a0=g_alloc_n.load(),b0=g_alloc_b.load();
            begin=Clock::now();for(int i=0;i<512;i++)next=model.greedy_step(next,state);double elapsed=seconds(begin);
            const size_t a1=g_alloc_n.load(),b1=g_alloc_b.load();
            if(probe)fp.finish();
            if(round==0)sum=next; else if(next!=sum)sum=0;
            printf("S3 threads=%u FORCED round=%d steps=512 seconds=%.6f tps=%.3f ms_per_step=%.6f first_prediction_ms=%.6f checksum=%u\n",
                T,round,elapsed,512/elapsed,elapsed*1000/512,first*1000,next);
            printf("S3 threads=%u ALLOC round=%d per_token=%.1f bytes_per_token=%.0f\n",T,round,double(a1-a0)/512.0,double(b1-b0)/512.0);
            if(probe)printf("S3 threads=%u FREQ round=%d probe_ghz=%.4f windows=%d\n",T,round,fp.ghz,fp.windows);
        }
        // Phase breakdown on a dedicated 512-step pass.
#ifdef TAO_PHASE_TIMING
        model.reset_phases();
        auto state=model.initial();
        model.advance(256,state);model.advance(257,state);
        for(auto t:prompt)model.advance(t,state);
        model.advance(259,state);
        begin=Clock::now();uint32_t next=model.greedy_step(258,state);
        for(int i=0;i<512;i++)next=model.greedy_step(next,state);
        double wall=seconds(begin);
        double norm=model.ms_norm*1000/512,ss=model.ms_s*1000/512,mm=model.ms_m*1000/512,
               rd=model.ms_read*1000/512,hd=model.ms_head*1000/512;
        double body=norm+ss+mm+rd;
        printf("S3 threads=%u PHASE per_token_ms norm=%.6f s_update=%.6f m_update=%.6f read_residual=%.6f head=%.6f body_total=%.6f instrumented_wall_ms=%.6f checksum=%u\n",
            T,norm,ss,mm,rd,hd,body,wall*1000/512,next);
        printf("S3 threads=%u SUBPHASE per_token_ms group=%.6f add_lookup=%.6f norm=%.6f residual=%.6f\n",T,model.ms_grp*1000/512,model.ms_addop*1000/512,model.ms_normf*1000/512,(model.ms_s+model.ms_m+model.ms_read-model.ms_grp-model.ms_addop-model.ms_normf)*1000/512);
        printf("S3 threads=%u PRELOOKUP per_token_ms pre=%.6f dispatch=%.6f\n",T,model.ms_pre*1000/512,(model.ms_grp-model.ms_pre)*1000/512);
        printf("S3 threads=%u PHASE share_of_body norm=%.4f s_update=%.4f m_update=%.4f read_residual=%.4f head_over_body=%.4f accounted=%.4f\n",
            T,norm/body,ss/body,mm/body,rd/body,hd/body,(body+hd)/(wall*1000/512));
#else
        printf("S3 threads=%u PHASE instrumentation_disabled\n",T);
#endif
    }
    model.set_cpu_threads(2);
    auto state=model.initial();
    model.advance(256,state);model.advance(257,state);
    for(auto t:prompt)model.advance(t,state);
    model.advance(259,state);
    uint32_t next=model.greedy_step(258,state);
    for(int i=0;i<512;i++)next=model.greedy_step(next,state);
    printf("S3 RESTORED threads=2 checksum=%u\n",next);
    return 0;
}
