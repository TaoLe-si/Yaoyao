#define NOMINMAX
#define TAO_CPU_AVX2
#define TAO_INPUT_SCALE
#include "advance_row_parallel_byte_cpu_model.hpp"
#include "advance_stage_grouped_byte_cpu_model.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
using namespace tao::dual;
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
 auto state=model.initial();unsigned token=256;Trial result;result.tokens.reserve(128);
 const auto start=std::chrono::steady_clock::now();
 for(int i=0;i<128;++i){token=greedy(model.step(token,state));result.tokens.push_back(token);}
 result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 std::printf("RUN backend=%s generated_positions=128 seconds=%.9f tps=%.3f final_token=%u\n",name,result.seconds,128/result.seconds,token);std::fflush(stdout);return result;
}
int main(int argc,char**argv){try{
 // Compile AVX2 with strict FP, matching the existing baseline build.
 if(argc>3)throw std::runtime_error("usage: benchmark_advance_stage_grouped [bundle [tokenizer_hash]]");
 const char*path=argc>1?argv[1]:"build/yaoyao_graph_step_360.dsb";
 const char*hash=argc>2?argv[2]:"34463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333";
 std::printf("GUARD PASS bundle=%s hash=%s threads=2 threshold=%zu\n",path,hash,CpuRowParallelExecutor::minimum_elements);std::fflush(stdout);
 const AdvanceRowParallelByteCpuModel baseline(read_compact_bundle(path,hash));
 const AdvanceStageGroupedByteCpuModel candidate(read_compact_bundle(path,hash));
 auto a=baseline.initial(),b=candidate.initial(),advance=candidate.initial();unsigned token=256;
 for(int i=0;i<32;++i){
  auto x=baseline.step(token,a),y=candidate.step(token,b);equal(x,y);states(a,b);
  candidate.advance(token,advance);states(a,advance);token=greedy(x);
 }
 std::puts("CHECK full_logits_all_states_32_bitwise_exact=PASS advance_states_32_bitwise_exact=PASS");std::fflush(stdout);
 // Exactly two alternating pairs, independent resets, forced 128 positions (no EOS exit).
 Trial base[2],cand[2];
 base[0]=bench(baseline,"compact_avx2_2thread");cand[0]=bench(candidate,"stage_grouped_avx2_2thread");
 cand[1]=bench(candidate,"stage_grouped_avx2_2thread");base[1]=bench(baseline,"compact_avx2_2thread");
 for(int i=0;i<2;++i)if(base[i].tokens!=cand[i].tokens||base[i].tokens!=base[0].tokens)throw std::runtime_error("generated trace mismatch");
 double bs=base[0].seconds+base[1].seconds,cs=cand[0].seconds+cand[1].seconds;
 std::printf("SUMMARY pairs=2 positions_per_trial=128 generated_trace_exact=PASS baseline_aggregate_tps=%.3f candidate_aggregate_tps=%.3f ratio=%.6f deployed=0\n",256/bs,256/cs,bs/cs);
 return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
