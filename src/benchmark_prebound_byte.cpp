#define TAO_CPU_AVX2
#define TAO_INPUT_SCALE
#include "dual_model_bundle.hpp"
#include "prebound_byte_cpu_model.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <algorithm>

static volatile float sink=0;
static void equal(const tao::dual::Vec& a,const tao::dual::Vec& b) {
    if(a.size()!=b.size())throw std::runtime_error("compare shape");
    for(size_t j=0;j<a.size();++j)
        if(!std::isfinite(a[j])||!std::isfinite(b[j])||std::memcmp(&a[j],&b[j],sizeof(float)))
            throw std::runtime_error("bitwise logits/state mismatch");
}
template<class M> double timing(M& m,unsigned token,int steps) {
    auto state=m.initial();
    auto start=std::chrono::steady_clock::now();
    for(int i=0;i<steps;++i) {
        // Bind reference: baseline temporary lifetime extends; scratch result is
        // borrowed without copying/allocating. Both destroyed before next step.
        const auto& y=m.step(token,state);
        token=unsigned(std::max_element(y.begin(),y.end())-y.begin());sink=y[token];
    }
    return steps/std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
}
int main(int argc,char** argv) {
    try {
        using namespace tao::dual;
        if(argc>4)throw std::runtime_error("usage: benchmark_prebound_byte [bundle [tokenizer_sha256 [steps]]]");
        const char* path=argc>1?argv[1]:"build/yaoyao_graph_step_360.dsb";
        const char* hash=argc>2?argv[2]:"34463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333";
        int steps=argc>3?std::stoi(argv[3]):128;
        if(steps<=0)throw std::runtime_error("positive steps required");
        auto src=load_bundle(path,hash);ByteCpuModel byte(src);src.w.clear();
        byte.fast=true;byte.four_rows=false;
        PreboundByteCpuModel bound(byte);
        unsigned first=byte.c.vocab>256?256:0,token=first;
        auto a=byte.initial(),b=bound.initial();
        for(int i=0;i<32;++i) {
            const auto x=byte.step(token,a);const auto& y=bound.step(token,b);
            equal(x,y);
            for(size_t l=0;l<a.size();++l){equal(a[l].s,b[l].s);equal(a[l].m,b[l].m);}
            token=unsigned(std::max_element(x.begin(),x.end())-x.begin());
            if(token!=unsigned(std::max_element(y.begin(),y.end())-y.begin()))throw std::runtime_error("argmax mismatch");
        }
        // Reset states and exercise different inputs to detect stale scratch.
        a=byte.initial();b=bound.initial();
        for(unsigned i=0;i<8;++i) {
            token=unsigned((uint64_t(i)*997+17)%byte.c.vocab);
            const auto x=byte.step(token,a);const auto& y=bound.step(token,b);equal(x,y);
            for(size_t l=0;l<a.size();++l){equal(a[l].s,b[l].s);equal(a[l].m,b[l].m);}
        }
        std::printf("CHECK prebound bitwise logits/states/argmax greedy32 reset_forced8 PASS shared_weight_bytes=%zu scratch_bytes=%zu\n",bound.weight_bytes(),bound.scratch_bytes());
        timing(byte,first,4);timing(bound,first,4);
        for(int trial=0;trial<4;++trial) {
            double bt,pt;
            if(trial%2){pt=timing(bound,first,steps);bt=timing(byte,first,steps);}
            else{bt=timing(byte,first,steps);pt=timing(bound,first,steps);}
            std::printf("FULL_GREEDY trial=%d steps=%d byte_tps=%.3f prebound_tps=%.3f ratio=%.4f\n",trial,steps,bt,pt,pt/bt);
        }
        return 0;
    }catch(const std::exception& e){std::printf("FAIL %s\n",e.what());return 1;}
}
