#define NOMINMAX
#define TAO_CPU_AVX2
#define TAO_INPUT_SCALE
#include "affinity_pipeline_grouped_model.hpp"
#include "pipeline_grouped_model.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif
using namespace tao::dual;
static void cpuid(unsigned leaf,unsigned sub,unsigned& a,unsigned& b,unsigned& c,unsigned& d){
#ifdef _MSC_VER
 int v[4];__cpuidex(v,int(leaf),int(sub));a=v[0];b=v[1];c=v[2];d=v[3];
#else
 __cpuid_count(leaf,sub,a,b,c,d);
#endif
}
static bool supported(){
 unsigned a,b,c,d;cpuid(0,0,a,b,c,d);if(a<7)return false;
 cpuid(1,0,a,b,c,d);
 if((c&((1u<<26)|(1u<<27)|(1u<<28)))!=((1u<<26)|(1u<<27)|(1u<<28)))return false;
 unsigned long long xcr0;
#ifdef _MSC_VER
 xcr0=_xgetbv(0);
#else
 unsigned lo,hi;__asm__ volatile("xgetbv":"=a"(lo),"=d"(hi):"c"(0));xcr0=(static_cast<unsigned long long>(hi)<<32)|lo;
#endif
 if((xcr0&6)!=6)return false; // XMM/YMM
 cpuid(7,0,a,b,c,d);
 const unsigned mask=(1u<<5); // AVX2 only, same pipeline kernel
 return (b&mask)==mask;
}
static void equal(const Vec&a,const Vec&b){
 if(a.size()!=b.size())throw std::runtime_error("shape mismatch");
 for(size_t i=0;i<a.size();++i)if(!std::isfinite(a[i])||!std::isfinite(b[i])||std::memcmp(&a[i],&b[i],sizeof(float)))throw std::runtime_error("nonfinite or bitwise mismatch at "+std::to_string(i));
}
static void states(const std::vector<LayerState>&a,const std::vector<LayerState>&b){
 if(a.size()!=b.size())throw std::runtime_error("state shape");
 for(size_t l=0;l<a.size();++l){equal(a[l].s,b[l].s);equal(a[l].m,b[l].m);}
}
static unsigned greedy(const Vec&v){return unsigned(std::max_element(v.begin(),v.end())-v.begin());}
struct Trial {double seconds;std::vector<unsigned> tokens;};
template<class M> static Trial bench(const M&model,const char*name){
 auto warm=model.initial();unsigned wt=256;for(int i=0;i<8;++i)wt=greedy(model.step(wt,warm));
 auto state=model.initial();unsigned token=256;Trial result;result.tokens.reserve(128);
 const auto start=std::chrono::steady_clock::now();
 for(int i=0;i<128;++i){token=greedy(model.step(token,state));result.tokens.push_back(token);}
 result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 std::printf("RUN backend=%s generated_positions=128 seconds=%.9f tps=%.3f final_token=%u\n",name,result.seconds,128/result.seconds,token);std::fflush(stdout);return result;
}
// CPU-only, no process affinity or priority setters. Build x64 /O2 /arch:AVX2
// /fp:precise /std:c++17 /EHsc; no AVX512 kernel TU or LTCG needed.
int main(int argc,char**argv){try{
 if(!supported()){std::puts("SKIP AVX2 or OS XMM/YMM unavailable");return 77;}
 if(argc!=3)throw std::runtime_error("usage: benchmark_pipeline_affinity bundle tokenizer_hash");
 const char* path=argv[1];const char* hash=argv[2];
 std::puts("CPU ONLY: LP0+LP2 are topology candidates, NOT least-busy cores; concurrent host load remains a confounder.");
 // Eligibility before model I/O; worker rechecks its own inherited restrictions.
 affinity_distinct(0,2);affinity_eligible(0);affinity_eligible(2);
 const PipelineGroupedModel baseline(read_compact_bundle(path,hash));
 const AffinityPipelineGroupedModel candidate(read_compact_bundle(path,hash));
 std::printf("GUARD threads=2 threshold=%zu same_pipeline_kernel=1 weight_bytes=%zu\n",CpuRowParallelExecutor::minimum_elements,candidate.weight_bytes());
 // Keep baseline unpinned throughout; save exact reference states/logits first.
 std::vector<Vec> logits;std::vector<std::vector<LayerState>> snapshots;
 std::vector<unsigned> inputs;auto a=baseline.initial();unsigned token=256;
 for(int i=0;i<32;++i){inputs.push_back(token);auto x=baseline.step(token,a);token=greedy(x);logits.push_back(std::move(x));snapshots.push_back(a);}
 candidate.enable_affinity();
 auto b=candidate.initial(),advance=candidate.initial();
 for(int i=0;i<32;++i){auto y=candidate.step(inputs[i],b);equal(logits[i],y);states(snapshots[i],b);candidate.advance(inputs[i],advance);states(snapshots[i],advance);}
 candidate.disable_affinity();
 std::puts("CHECK full_logits_all_states_32_bitwise_exact=PASS advance_states_32_bitwise_exact=PASS caller_worker_restore=PASS");
 auto pinned=[&](){candidate.enable_affinity();auto t=bench(candidate,"pipeline_pinned_LP0_LP2");candidate.disable_affinity();return t;};
 Trial base[2],cand[2];
 base[0]=bench(baseline,"pipeline_unpinned");cand[0]=pinned();
 cand[1]=pinned();base[1]=bench(baseline,"pipeline_unpinned");
 for(int i=0;i<2;++i)if(base[i].tokens!=cand[i].tokens||base[i].tokens!=base[0].tokens)throw std::runtime_error("generated trace mismatch");
 double bs=base[0].seconds+base[1].seconds,cs=cand[0].seconds+cand[1].seconds;
 std::printf("SUMMARY pairs=2 positions_per_trial=128 warmup_each=8 generated_trace_exact=PASS baseline_aggregate_tps=%.3f pinned_aggregate_tps=%.3f ratio=%.6f restored=1 deployed=0\n",256/bs,256/cs,bs/cs);
 return 0;
}catch(const AffinitySkip&e){std::fprintf(stderr,"SKIP affinity experiment (fail closed): %s\n",e.what());return 77;}
 catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
