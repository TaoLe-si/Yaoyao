#define NOMINMAX
#define TAO_CPU_AVX2
#define TAO_INPUT_SCALE
#include "pipeline_grouped_unroll4_model.hpp"
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
 if((xcr0&6)!=6)return false; // XMM/YMM only
 cpuid(7,0,a,b,c,d);
 const unsigned mask=(1u<<5); // AVX2 only
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
 auto state=model.initial();unsigned token=256;Trial result;result.tokens.reserve(128);
 const auto start=std::chrono::steady_clock::now();
 for(int i=0;i<128;++i){token=greedy(model.step(token,state));result.tokens.push_back(token);}
 result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 std::printf("RUN backend=%s generated_positions=128 seconds=%.9f tps=%.3f final_token=%u\n",name,result.seconds,128/result.seconds,token);std::fflush(stdout);return result;
}
// One untimed tiny correctness pass, NOT a kernel performance sweep.
static void tiny_tail_checks(){
 for(size_t n : {size_t(1),size_t(7),size_t(8),size_t(15),size_t(16),size_t(17),size_t(31),size_t(32),size_t(33),size_t(47),size_t(48),size_t(55),size_t(63),size_t(64),size_t(65)}){
  Vec w(3*n),x(n);
  for(size_t j=0;j<n;++j)x[j]=float(int(j%11)-5)*0.1234567f;
  for(size_t row=0;row<3;++row)for(size_t j=0;j<n;++j)w[row*n+j]=float(int((j+row)%3)-1)*(row==0?0.13f:row==1?1.75f:0.0f);
  PipelineRows a(w,3,n);PipelineRowsUnroll4 b(w,3,n);
  for(size_t row=0;row<3;++row)equal(Vec{a.dot(row,x.data())},Vec{b.dot(row,x.data())});
 }
 std::puts("CHECK tiny_tail_once_bitwise_exact=PASS");
}
int main(int argc,char**argv){try{
 // Compile AVX2 /fp:strict (or -fno-fast-math -ffp-contract=off), no LTO.
 if(!supported()){std::puts("SKIP AVX2 or OS XCR0 unavailable");return 77;}
 if(argc!=1)throw std::runtime_error("no arguments: frozen step360 and tokenizer identity only");
  tiny_tail_checks();
 const char*path="build/yaoyao_graph_step_360.dsb";
 const char*hash="34463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333";
 std::printf("GUARD PASS bundle=%s hash=%s threads=2 threshold=%zu\n",path,hash,CpuRowParallelExecutor::minimum_elements);std::fflush(stdout);
 const PipelineGroupedModel baseline(read_compact_bundle(path,hash));
 const PipelineGroupedUnroll4Model candidate(read_compact_bundle(path,hash));
 auto a=baseline.initial(),b=candidate.initial(),advance=candidate.initial(),baseline_advance=baseline.initial();unsigned token=256;
 for(int i=0;i<32;++i){
  auto x=baseline.step(token,a),y=candidate.step(token,b);equal(x,y);states(a,b);
  baseline.advance(token,baseline_advance);candidate.advance(token,advance);states(a,baseline_advance);states(a,advance);token=greedy(x);
 }
 std::puts("CHECK full_logits_all_states_32_bitwise_exact=PASS advance_states_32_bitwise_exact=PASS");std::fflush(stdout);
 // Exactly two alternating pairs, independent resets, forced 128 positions (no EOS exit).
 Trial base[2],cand[2];
 base[0]=bench(baseline,"pipeline_unroll2_avx2_2thread");cand[0]=bench(candidate,"pipeline_unroll4_avx2_2thread");
 cand[1]=bench(candidate,"pipeline_unroll4_avx2_2thread");base[1]=bench(baseline,"pipeline_unroll2_avx2_2thread");
 for(int i=0;i<2;++i)if(base[i].tokens!=cand[i].tokens||base[i].tokens!=base[0].tokens)throw std::runtime_error("generated trace mismatch");
 double bs=base[0].seconds+base[1].seconds,cs=cand[0].seconds+cand[1].seconds;
 std::printf("SUMMARY pairs=2 positions_per_trial=128 generated_trace_exact=PASS baseline_aggregate_tps=%.3f candidate_aggregate_tps=%.3f ratio=%.6f deployed=0\n",256/bs,256/cs,bs/cs);
 return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
