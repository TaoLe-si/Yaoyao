#define NOMINMAX
#define TAO_CPU_AVX2
#define TAO_INPUT_SCALE
#include "pipeline_grouped_model.hpp"
#include "greedy_pipeline_grouped_model.hpp"
#include "advance_stage_grouped_byte_cpu_model.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
using namespace tao::dual;
static void require(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
static void equal(const Vec&a,const Vec&b){
 require(a.size()==b.size(),"vector shape");
 for(size_t i=0;i<a.size();++i)require(std::isfinite(a[i])&&std::isfinite(b[i])&&std::memcmp(&a[i],&b[i],sizeof(float))==0,"scalar/state bits mismatch");
}
static void states(const std::vector<LayerState>&a,const std::vector<LayerState>&b){
 require(a.size()==b.size(),"state count");for(size_t i=0;i<a.size();++i){equal(a[i].s,b[i].s);equal(a[i].m,b[i].m);}
}
// Actual resident reference: full vector finite pass, role masking, max_element.
static uint32_t scan(Vec logits){
 for(float z:logits)if(!std::isfinite(z))throw std::runtime_error("nonfinite");
 for(unsigned t:{256u,257u,258u})if(t<logits.size())logits[t]=-std::numeric_limits<float>::infinity();
 return uint32_t(std::max_element(logits.begin(),logits.end())-logits.begin());
}
static CompactBundleData fixture(unsigned width){
 CompactBundleData b;b.c={1,width,9,17,23,263};
 for(const auto&t:schema(b.c)){
  if(t.ternary){CompactBundleData::Matrix m;m.q.resize(t.elements());m.scale.resize(t.rows);
   for(size_t i=0;i<m.q.size();++i)m.q[i]=int8_t(int((i*17+3)%3)-1);
   for(size_t i=0;i<m.scale.size();++i)m.scale[i]=0.0078125f*float(1+i%3);
   b.matrices.emplace(t.name,std::move(m));
  }else{Vec v(t.elements());for(size_t i=0;i<v.size();++i)v[i]=t.name.find("norm")!=std::string::npos?1.0f:float(int(i%7)-3)*0.00390625f;b.vectors.emplace(t.name,std::move(v));}
 }
 return b;
}
static void head_check(GreedyPipelineGroupedModel& m,const Vec& hidden){
 auto logits=m.linear("embedding",m.norm(hidden,"final.norm"),m.c.vocab);m.add(logits,m.w.at("vocab.bias"));
 Vec observed(logits.size());std::vector<unsigned> visits(logits.size());
 auto token=m.greedy_head_observe(hidden,[&](size_t row,float value){observed[row]=value;++visits[row];});
 equal(logits,observed);for(auto n:visits)require(n==1,"row not visited exactly once");require(token==scan(logits),"head token mismatch");
}
static void unit_tests(){
 // 263*996=261948 is serial; 263*997=262211 triggers two tasks.
 for(unsigned width:{7u,8u,16u,23u,996u,997u,1031u}){
  PipelineGroupedModel ref(fixture(width));GreedyPipelineGroupedModel m(fixture(width));
  for(bool fast:{true,false}){
   ref.fast=fast;m.fast=fast;Vec hidden(width);for(size_t i=0;i<hidden.size();++i)hidden[i]=float(int(i%19)-9)*0.0625f;head_check(m,hidden);
   auto a=ref.initial(),b=m.initial();unsigned token=256;
   for(unsigned i=0;i<8;++i){auto v=ref.step(token,a);auto t=m.greedy_step(token,b);require(t==scan(v),"normal token mismatch");states(a,b);token=t;}
   auto& bias=m.w.at("vocab.bias");auto saved=bias;
   // Zero hidden makes every dot exactly zero; ties cross the worker split.
   std::fill(hidden.begin(),hidden.end(),0.0f);std::fill(bias.begin(),bias.end(),0.0f);
   bias[256]=bias[257]=bias[258]=100;head_check(m,hidden);require(m.greedy_head(hidden)==0,"lowest tie/exclusion");
   bias[1]=bias[262]=5;head_check(m,hidden);require(m.greedy_head(hidden)==1,"cross-partition tie");
   bias[262]=6;head_check(m,hidden);require(m.greedy_head(hidden)==262,"worker half unique winner");
   // Empty serial slot must never act as a zero-valued candidate.
   std::fill(bias.begin(),bias.end(),-8.0f);bias[262]=-4.0f;
   head_check(m,hidden);require(m.greedy_head(hidden)==262,"negative-only winner");
   for(unsigned row:{0u,256u,257u,258u,262u})for(float bad:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()}){
    float old=bias[row];bias[row]=bad;std::vector<unsigned> visits(bias.size());bool rejected=false;
    try{m.greedy_head_observe(hidden,[&](size_t r,float){++visits[r];});}catch(const std::runtime_error&e){rejected=std::string(e.what())=="nonfinite";}
    require(rejected,"nonfinite not rejected");for(auto n:visits)require(n==1,"nonfinite scan incomplete");bias[row]=old;
   }
   bias=saved;
  }
 }
 std::puts("UNIT PASS scalar_bits_all_rows tails serial_parallel fast_scalar ties exclusions nonfinite_all_rows states_tokens");std::fflush(stdout);
}
struct Trial{double seconds;std::vector<uint32_t> tokens;std::vector<LayerState> state;};
template<class M,class Step>static Trial bench(const M&m,const char*name,Step step){
 Trial t;t.state=m.initial();t.tokens.reserve(128);unsigned token=256;
 auto start=std::chrono::steady_clock::now();
 for(unsigned i=0;i<128;++i){token=step(token,t.state);t.tokens.push_back(token);}
 t.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 std::printf("RUN backend=%s positions=128 seconds=%.9f tps=%.3f final_token=%u\n",name,t.seconds,128/t.seconds,token);std::fflush(stdout);return t;
}
int main(int argc,char**argv){try{
 unit_tests();if(argc==2&&std::string(argv[1])=="--unit")return 0;
 require(argc<=2,"usage: benchmark_greedy_pipeline_grouped [fixed360_bundle | --unit]");
 const char*path=argc==2?argv[1]:"build/yaoyao_graph_step_360.dsb";
 const char*hash="34463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333";
 PipelineGroupedModel ref(read_compact_bundle(path,hash));GreedyPipelineGroupedModel candidate(read_compact_bundle(path,hash));
 AdvanceStageGroupedByteCpuModel stage(read_compact_bundle(path,hash));
 std::printf("IDENTITY bundle=%s tokenizer_hash=%s threads_per_model=2 GPU=unused\n",path,hash);
 auto a=ref.initial(),b=candidate.initial(),c=stage.initial();unsigned token=256;
 for(unsigned i=0;i<32;++i){auto v=ref.step(token,a),s=stage.step(token,c);equal(v,s);auto t=candidate.greedy_step(token,b);require(t==scan(v),"checkpoint token mismatch");states(a,b);states(a,c);token=t;}
 std::puts("CHECK checkpoint_32_states_token_bits=PASS stage_pipeline_full_logits_bits=PASS greedy_vector_omitted=1");
 auto rstep=[&](unsigned t,std::vector<LayerState>&s){return scan(ref.step(t,s));};
 auto gstep=[&](unsigned t,std::vector<LayerState>&s){return candidate.greedy_step(t,s);};
 auto sstep=[&](unsigned t,std::vector<LayerState>&s){return scan(stage.step(t,s));};
 Trial r[2],g[2],s[2];
 // Alternating order, fresh states, forced positions (EOS not an early exit).
 s[0]=bench(stage,"stage_full_logits_plus_scan",sstep);r[0]=bench(ref,"pipeline_full_logits_plus_scan",rstep);g[0]=bench(candidate,"pipeline_greedy",gstep);
 g[1]=bench(candidate,"pipeline_greedy",gstep);r[1]=bench(ref,"pipeline_full_logits_plus_scan",rstep);s[1]=bench(stage,"stage_full_logits_plus_scan",sstep);
 for(unsigned i=0;i<2;++i){require(r[i].tokens==g[i].tokens&&r[i].tokens==s[i].tokens&&r[i].tokens==r[0].tokens,"128 trace mismatch");states(r[i].state,g[i].state);states(r[i].state,s[i].state);}
 double rt=r[0].seconds+r[1].seconds,gt=g[0].seconds+g[1].seconds,st=s[0].seconds+s[1].seconds;
 std::printf("SUMMARY pairs=2 positions_per_trial=128 trace_and_final_states_bits=PASS pipeline_scan_tps=%.3f greedy_tps=%.3f stage_scan_tps=%.3f speedup_vs_pipeline=%.6f speedup_vs_stage=%.6f deployed=0\n",256/rt,256/gt,256/st,rt/gt,st/gt);
 return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
