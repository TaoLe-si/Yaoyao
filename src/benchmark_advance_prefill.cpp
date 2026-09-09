#define NOMINMAX
#define TAO_INPUT_SCALE
#define TAO_CPU_AVX2
#include "advance_byte_cpu_model.hpp"
#include "tokenizer_file.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <algorithm>
static volatile float sink=0;
static void equal(const tao::dual::Vec& a,const tao::dual::Vec& b){
    if(a.size()!=b.size())throw std::runtime_error("shape");
    for(size_t j=0;j<a.size();++j)if(!std::isfinite(a[j])||!std::isfinite(b[j])||std::memcmp(&a[j],&b[j],4))throw std::runtime_error("bitwise mismatch");
}
static void states(const std::vector<tao::dual::LayerState>& a,const std::vector<tao::dual::LayerState>& b){
    if(a.size()!=b.size())throw std::runtime_error("state shape");
    for(size_t l=0;l<a.size();++l){equal(a[l].s,b[l].s);equal(a[l].m,b[l].m);}
}
int main(int argc,char** argv){try{
    using namespace tao::dual;
    if(argc>2)throw std::runtime_error("usage: benchmark_advance_prefill [step360 bundle]");
    std::string hash;auto tokenizer=tao::text::load_tokenizer("build/formal_tokenizer.bbp",hash);
    AdvanceByteCpuModel candidate(read_compact_bundle(argc>1?argv[1]:"build/yaoyao_graph_step_360.dsb",hash));
    // Same immutable weight storage and byte kernel, qualified base step baseline.
    const ByteCpuModel& baseline=candidate;
    const std::string prompt="Please explain why the sky is blue, using simple words and a short example.";
    auto encoded=tokenizer.encode(prompt);
    auto tokens=encoded;tokens.insert(tokens.begin(),257);tokens.insert(tokens.begin(),256);tokens.push_back(259);tokens.push_back(258);
    auto a=baseline.initial(),b=candidate.initial();
    for(size_t i=0;i<tokens.size();++i){
        auto x=baseline.step(tokens[i],a);
        if(i+1==tokens.size()){auto y=candidate.step(tokens[i],b);equal(x,y);}
        else candidate.advance(tokens[i],b);
        states(a,b);
    }
    // Continuing conversation: no BOS. Then reset and repeat fresh semantics.
    for(int pass=0;pass<2;++pass){
        if(pass==1){a=baseline.initial();b=candidate.initial();}
        size_t start=pass==0?1:0;
        for(size_t i=start;i<tokens.size();++i){
            auto x=baseline.step(tokens[i],a);
            if(i+1==tokens.size()){auto y=candidate.step(tokens[i],b);equal(x,y);}
            else candidate.advance(tokens[i],b);
            states(a,b);
        }
    }
    std::printf("CHECK prefill fresh/continuation/reset every-token states and final logits bitwise PASS prompt_tokens=%zu total_fresh_tokens=%zu\n",encoded.size(),tokens.size());
    auto measure=[&](bool advance){
        double seconds=0;
        for(int repeat=0;repeat<4;++repeat){
            auto state=baseline.initial(); // excluded from prompt latency
            auto begin=std::chrono::steady_clock::now();
            Vec logits;
            for(size_t i=0;i<tokens.size();++i){
                if(advance&&i+1<tokens.size())candidate.advance(tokens[i],state);
                else logits=advance?candidate.step(tokens[i],state):baseline.step(tokens[i],state);
            }
            seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
            sink=logits[0];
        }
        return seconds*1000/4;
    };
    measure(false);measure(true);
    for(int trial=0;trial<4;++trial){double full,advance;
        if(trial%2){advance=measure(true);full=measure(false);}else{full=measure(false);advance=measure(true);}
        std::printf("PREFILL_ONLY trial=%d repetitions=4 baseline_ms=%.6f advance_ms=%.6f latency_ratio=%.4f\n",trial,full,advance,advance/full);
    }
    std::puts("Timing excludes tokenization/load/state allocation and generation; includes final assistant-role output head. No generated-token speedup claim.");return 0;
}catch(const std::exception& e){std::printf("FAIL %s\n",e.what());return 1;}}
