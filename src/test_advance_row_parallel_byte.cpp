#define NOMINMAX
#define TAO_INPUT_SCALE
#define TAO_CPU_AVX2
#include "tokenizer_file.hpp"
#include "advance_byte_cpu_model.hpp"
#include "advance_row_parallel_byte_cpu_model.hpp"
#include <iostream>
#include <cstring>
#include <type_traits>
using namespace tao::dual;
static void equal(const Vec&a,const Vec&b){
    if(a.size()!=b.size())throw std::runtime_error("vector shape mismatch");
    for(size_t i=0;i<a.size();++i)if(!std::isfinite(a[i])||!std::isfinite(b[i])||std::memcmp(&a[i],&b[i],sizeof(float)))
        throw std::runtime_error("nonfinite/bit mismatch element "+std::to_string(i));
}
static void states(const std::vector<LayerState>&a,const std::vector<LayerState>&b){
    if(a.size()!=b.size())throw std::runtime_error("layer count mismatch");
    for(size_t i=0;i<a.size();++i){equal(a[i].s,b[i].s);equal(a[i].m,b[i].m);}
}
int main(){try{
    static_assert(!std::is_copy_constructible<AdvanceRowParallelByteCpuModel>::value,"pool must not copy");
    static_assert(!std::is_move_constructible<AdvanceRowParallelByteCpuModel>::value,"pool must not move");
    const char* identity="34463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333";
    std::string hash;auto tokenizer=tao::text::load_tokenizer("build/formal_tokenizer.bbp",hash);
    if(hash!=identity)throw std::runtime_error("tokenizer identity mismatch");
    const char* path="build/yaoyao_graph_step_360.dsb";
    AdvanceByteCpuModel baseline(read_compact_bundle(path,hash));
    AdvanceRowParallelByteCpuModel candidate(read_compact_bundle(path,hash));
    // Loader temporaries are gone: both models must own every weight buffer.
    auto a=baseline.initial(),b=candidate.initial();bool fresh=true;
    auto reset=[&]{a=baseline.initial();b=candidate.initial();fresh=true;states(a,b);};
    auto advance=[&](unsigned t){baseline.advance(t,a);candidate.advance(t,b);states(a,b);};
    auto step=[&](unsigned t){auto x=baseline.step(t,a),y=candidate.step(t,b);equal(x,y);states(a,b);return x;};
    // forced_end: -1 real greedy, 259/260 inject terminal branch, 0 force cap tail.
    auto request=[&](const char* label,const std::string&prompt,int forced_end){
        if(fresh){advance(256);fresh=false;}
        advance(257);for(auto t:tokenizer.encode(prompt))advance(t);advance(259);
        auto logits=step(258);int end=-1;unsigned count=0;
        for(int i=0;i<64;++i){
            for(unsigned t:{256u,257u,258u})logits[t]=-std::numeric_limits<float>::infinity();
            unsigned t=unsigned(std::max_element(logits.begin(),logits.end())-logits.begin());
            if(forced_end==259||forced_end==260)t=unsigned(forced_end);
            if(forced_end==0)t=unsigned('a'); // deterministically reach advance-only cap tail
            if(t==259||t==260){advance(t);end=int(t);if(t==260)reset();break;}
            ++count;if(i==63)advance(t);else logits=step(t);
        }
        if(end<0)advance(259);
        states(a,b);
        // Probe final logits on copies, without changing live continuation state.
        auto probe_a=a,probe_b=b;
        equal(baseline.step(258,probe_a),candidate.step(258,probe_b));states(probe_a,probe_b);
        std::cout<<"PASS case="<<label<<" full_states_logits_bitwise=1 tokens="<<count<<" end="<<end<<" fresh="<<fresh<<std::endl;
    };
    request("fresh_real","Hello",-1);
    // A forced EOT establishes a nonfresh state even if previous real reply was EOS.
    request("eot_tail","Hi",259);
    request("continuation_real","Thanks",-1);
    reset();request("explicit_reset_real","Hello",-1);
    request("cap_tail","",0);
    request("continuation_after_cap","",259);
    request("eos_reset","",260);
    request("fresh_after_eos","",259);
    std::cout<<"PASS all_cases=8 threads=2 threshold=262144 loads_per_model=1 benchmark=0 deployed=0"<<std::endl;
    return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}
catch(...){std::cerr<<"FAIL unknown exception"<<std::endl;return 1;}}
