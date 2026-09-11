// 同进程交错 A/B 基准：一次加载，进程内切换所有配置，多轮交错，报告中位数 + IQR。
// 动机见 docs/architecture-proposals/02-cpu-decode-throughput.md §10/§15/§18/§19：
//   (a) 跨进程比较不可靠（同配置测得 710/842/870/932/987/1119/1178 tps）；
//   (b) set_cpu_threads(1) 与 (2) 曾经是同一配置，所以必须打印 config_dump() 自证。
//
// 用法: bench_interleaved <checkpoint> [rounds] [steps]
#define NOMINMAX
#define TAO_INPUT_SCALE
#define TAO_CPU_AVX2
#include "tokenizer_file.hpp"
#include "greedy_resident_lifecycle.hpp"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>

using namespace tao::dual;
using Clock=std::chrono::steady_clock;
static double secs(Clock::time_point a){return std::chrono::duration<double>(Clock::now()-a).count();}

struct Cfg {
    std::string name;
    bool share;        // 层共享（2 组 0,0,0,0,4,4,4,4）
    bool reuse_mv;     // R4
    size_t mincost;
    unsigned threads;
    bool fast_act=false;
    bool float_w=false;
    bool avx512=false;
    bool macc=false;
    bool vnni=false;
};

int main(int argc,char**argv){
    const char* path=argc>1?argv[1]:"build/arch_A_ds3/step_150/final.dsb";
    int rounds=argc>2?atoi(argv[2]):7;
    int steps=argc>3?atoi(argv[3]):512;
    std::string hash;
    auto tok=tao::text::load_tokenizer("build/formal_tokenizer.bbp",hash);
    GreedyPipelineGroupedModel model(read_compact_bundle(path,hash));
    const uint32_t L=model.c.layers;
    printf("LOAD db=%s resident_bytes=%zu layers=%u\n",path,model.weight_bytes(),L);
    const auto prompt=tok.encode("你好，请用一句话介绍自己。");

    const std::vector<uint32_t> ident=GreedyPipelineGroupedModel::identitySource(L);
    const std::vector<uint32_t> share2=GreedyPipelineGroupedModel::parseLayerShare("0,0,0,0,4,4,4,4",L);
    if(share2==ident)printf("WARN 层共享向量未生效（层数 %u 与向量长度不符）\n",L);

    std::vector<Cfg> cfgs;
    for(unsigned t: {1u,2u,4u,8u})
        cfgs.push_back({"R3+R4_t"+std::to_string(t),true,true,1,t,false,false,false,false,false});
    cfgs.push_back({"R0_t8",false,false,262144u,8u,false,false,false,false,false});
    cfgs.push_back({"R3_t8",true,false,262144u,8u,false,false,false,false,false});
    cfgs.push_back({"R3+R4_mc262k_t8",true,true,262144u,8u,false,false,false,false,false});
    cfgs.push_back({"R3+R4_t8_fastact",true,true,1u,8u,true,false,false,false,false});
    cfgs.push_back({"R3+R4_t8_floatw",true,true,1u,8u,false,true,false,false,false});
    cfgs.push_back({"R3+R4_t8_act+floatw",true,true,1u,8u,true,true,false,false,false});
    cfgs.push_back({"R3+R4_t8_avx512",true,true,1u,8u,false,false,true,false,false});
    cfgs.push_back({"R3+R4_t8_act+512",true,true,1u,8u,true,false,true,false,false});
    cfgs.push_back({"R3+R4_t1_avx512",true,true,1u,1u,false,false,true,false,false});
    cfgs.push_back({"R3+R4_t8_macc",true,true,1u,8u,false,false,false,true,false});
    cfgs.push_back({"R3+R4_t8_act+macc",true,true,1u,8u,true,false,false,true,false});
    cfgs.push_back({"R3+R4_t1_macc",true,true,1u,1u,false,false,false,true,false});
    cfgs.push_back({"R3+R4_t8_vnni",true,true,1u,8u,false,false,false,false,true});
    cfgs.push_back({"R3+R4_t8_act+vnni",true,true,1u,8u,true,false,false,false,true});
    cfgs.push_back({"R3+R4_t1_vnni",true,true,1u,1u,false,false,false,false,true});
    cfgs.push_back({"R3+R4_t1_fastact",true,true,1u,1u,true,false,false,false,false});

    if(const char* flt=std::getenv("TAO_BENCH_FILTER")){
        std::vector<Cfg> keep;std::string f(flt);
        for(auto&c:cfgs)if(f.find(c.name)!=std::string::npos)keep.push_back(c);
        cfgs=keep;printf("filter -> %zu configs\n",cfgs.size());
    }
    std::vector<std::vector<double>> samples(cfgs.size());
    std::vector<uint32_t> cks(cfgs.size(),0);
    auto apply=[&](const Cfg&c){
        model.set_layer_share(c.share?share2:ident);
        model.set_reuse_mv(c.reuse_mv);
        model.set_layer_limit(0);
        model.set_cpu_threads(c.threads);
        model.set_row_parallel_minimum(c.mincost);
        model.set_fast_act(c.fast_act);
        model.set_float_weights(c.float_w);
        model.set_avx512(c.avx512);
        model.set_multi_acc(c.macc);
        model.set_vnni(c.vnni);
    };
    // 预热
    for(auto&c:cfgs){
        apply(c);
        auto st=model.initial();
        model.advance(256,st);model.advance(257,st);
        for(auto t:prompt)model.advance(t,st);
        model.advance(259,st);
        uint32_t nx=model.greedy_step(258,st);
        for(int i=0;i<64;++i)nx=model.greedy_step(nx,st);
    }
    // 交错多轮
    for(int r=0;r<rounds;++r)for(size_t ci=0;ci<cfgs.size();++ci){
        const Cfg&c=cfgs[ci];
        apply(c);
        auto st=model.initial();
        model.advance(256,st);model.advance(257,st);
        for(auto t:prompt)model.advance(t,st);
        model.advance(259,st);
        uint32_t nx=model.greedy_step(258,st);
        auto t0=Clock::now();
        for(int i=0;i<steps;++i)nx=model.greedy_step(nx,st);
        double e=secs(t0);
        samples[ci].push_back(steps/e);
        if(r==0)printf("VERIFY %-22s %s\n",c.name.c_str(),model.config_dump().c_str());
        cks[ci]=nx;
    }
    printf("\n%-22s %9s %9s %9s %9s %9s %10s\n","config","median","Q1","Q3","IQR","min","token");
    for(size_t ci=0;ci<cfgs.size();++ci){
        auto v=samples[ci];std::sort(v.begin(),v.end());
        size_t n=v.size();
        printf("%-22s %9.1f %9.1f %9.1f %9.1f %9.1f %10u\n",
               cfgs[ci].name.c_str(),v[n/2],v[n/4],v[(3*n)/4],v[(3*n)/4]-v[n/4],v[0],cks[ci]);
    }
    printf("\n判据：效应需 > 2*IQR 才认定为真实（见 §18.7）\n");
    for(size_t ci=0;ci<cfgs.size();++ci)for(size_t cj=ci+1;cj<cfgs.size();++cj){
        auto a=samples[ci],b=samples[cj];
        std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());
        size_t n=a.size();
        double ma=a[n/2],mb=b[n/2];
        double thr=2*std::max(a[(3*n)/4]-a[n/4],b[(3*n)/4]-b[n/4]);
        double d=std::fabs(ma-mb);
        if(d>thr)printf("  真实: %-18s %8.1f  vs  %-18s %8.1f   Δ=%.1f (2IQR=%.1f)\n",
                        cfgs[ci].name.c_str(),ma,cfgs[cj].name.c_str(),mb,d,thr);
    }
    return 0;
}
