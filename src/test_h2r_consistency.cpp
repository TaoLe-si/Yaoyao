#define NOMINMAX
#define TAO_INPUT_SCALE
#define TAO_CPU_AVX2
#include "dual_state_initialization.hpp"
#include "dual_model_bundle.hpp"
#include "greedy_pipeline_grouped_model.hpp"
#include <cstdio>
#include <cmath>
#include <string>
using namespace tao::dual;
static const char* TOK="34463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333";
static void project(CpuModel&m,const Config&c){
    for(auto&t:schema(c)){ if(!t.ternary)continue; auto&w=m.w.at(t.name);
        for(uint32_t r=0;r<t.rows;++r){ float sc=0; for(uint32_t j=0;j<t.cols;++j)sc=std::max(sc,std::abs(w[size_t(r)*t.cols+j]));
            if(sc==0)sc=1; for(uint32_t j=0;j<t.cols;++j){ float v=w[size_t(r)*t.cols+j]; w[size_t(r)*t.cols+j]= v>0?sc:v<0?-sc:0.f; } } }
}
int main(){
    setvbuf(stdout,NULL,_IONBF,0);
    Config c; c.layers=8; c.d=512; c.s=128; c.m=512; c.dk=64; c.vocab=16384;
    auto model=initialize(c,20260911);
    project(model,c);
    try{std::filesystem::remove("build/h2r_consistency.dsb");}catch(...){}
    save_bundle(model,"build/h2r_consistency.dsb",TOK);
    GreedyPipelineGroupedModel opt(read_compact_bundle("build/h2r_consistency.dsb",TOK));
    // --- 1) 参考实现 vs 优化实现 ---
    auto sr=model.initial(), so=opt.initial();
    opt.set_cpu_threads(2);
    double worst=0; double scale=0;
    for(int i=0;i<24;++i){
        uint32_t t=uint32_t(261+(i*911)%16000);
        auto a=model.step(t,sr), b=opt.step(t,so);
        for(size_t j=0;j<a.size();++j){ double d=std::abs(double(a[j])-b[j]); if(d>worst)worst=d; if(std::abs(double(a[j]))>scale)scale=std::abs(double(a[j])); }
    }
    printf("[一致性] 参考(CpuModel) vs 优化(GreedyPipelineGroupedModel) 24 步 logits 最大绝对差 = %.3e  (量级 %.3f)\n",worst,scale);
    // --- 2) 线程不变性：同一 token 序列的生成结果必须逐位一致 ---
    unsigned threads[]={1,2,3,4,8,16};
    std::string ref; unsigned refT=0; bool first=true, allsame=true;
    for(unsigned th:threads){
        GreedyPipelineGroupedModel m(read_compact_bundle("build/h2r_consistency.dsb",TOK));
        m.set_cpu_threads(th);
        auto s=m.initial(); uint64_t h=1469598103934665603ull;
        for(int i=0;i<64;++i){ uint32_t tok=m.greedy_step(261+i*97,s); h^=tok; h*=1099511628211ull; }
        char buf[64]; std::snprintf(buf,sizeof buf,"%016llx",(unsigned long long)h);
        printf("  threads=%-3u checksum=%s\n",th,buf);
        if(first){ref=buf;refT=th;first=false;} else if(ref!=buf)allsame=false;
    }
    printf("[线程不变性] 全部 %zu 种线程数逐位一致: %s\n",sizeof(threads)/sizeof(threads[0]),allsame?"YES":"**NO**");
    // --- 3) 状态形状 ---
    auto s=opt.initial();
    printf("[状态] per layer s=%zu M=%zu   总 %zu B\n",s[0].s.size(),s[0].m.size(),c.layers*4*(s[0].s.size()+s[0].m.size()));
    return allsame?0:1;
}
