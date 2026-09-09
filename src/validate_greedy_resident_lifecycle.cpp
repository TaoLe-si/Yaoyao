#define NOMINMAX
#define TAO_INPUT_SCALE
#define TAO_CPU_AVX2
#include "pipeline_grouped_model.hpp"
#include "greedy_resident_lifecycle.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
using namespace tao::dual;
static void need(bool ok,const char*message){if(!ok)throw std::runtime_error(message);}
static void vec_equal(const Vec&a,const Vec&b){need(a.size()==b.size(),"shape");for(size_t i=0;i<a.size();++i)need(std::isfinite(a[i])&&std::isfinite(b[i])&&!std::memcmp(&a[i],&b[i],sizeof(float)),"state bits");}
static void state_equal(const std::vector<LayerState>&a,const std::vector<LayerState>&b){need(a.size()==b.size(),"layer count");for(size_t i=0;i<a.size();++i){vec_equal(a[i].s,b[i].s);vec_equal(a[i].m,b[i].m);}}
static unsigned scan(Vec v){for(float z:v)need(std::isfinite(z),"nonfinite");for(auto t:{256u,257u,258u})v.at(t)=-std::numeric_limits<float>::infinity();return unsigned(std::max_element(v.begin(),v.end())-v.begin());}
// Independent full-logit resident reference. Never calls candidate lifecycle.
struct Reference {
 const PipelineGroupedModel&m;std::vector<LayerState> state;bool fresh=true;
 explicit Reference(const PipelineGroupedModel&model):m(model),state(m.initial()){}
 void reset(){state=m.initial();fresh=true;}
 template<class Select> GreedyResidentReply reply(const std::vector<uint32_t>&prompt,Select select){
  if(fresh){m.advance(256,state);fresh=false;}
  m.advance(257,state);for(auto t:prompt)m.advance(t,state);m.advance(259,state);
  Vec logits=m.step(258,state);GreedyResidentReply result;
  for(unsigned i=0;i<64;++i){
   auto t=select(i,scan(std::move(logits)));
   if(t==259||t==260){m.advance(t,state);result.end=int(t);if(t==260)reset();break;}
   result.ids.push_back(t);
   if(i==63)m.advance(t,state);else logits=m.step(t,state);
  }
  if(result.end<0)m.advance(259,state);
  return result;
 }
};
int main(int argc,char**argv){try{
 need(argc<=2,"usage: validate_greedy_resident_lifecycle [fixed360_bundle]");
 const char*path=argc==2?argv[1]:"build/yaoyao_graph_step_360.dsb";
 const char*hash="34463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333";
 PipelineGroupedModel baseline(read_compact_bundle(path,hash));GreedyPipelineGroupedModel model(read_compact_bundle(path,hash));
 Reference reference(baseline);GreedyResidentSession candidate(model);
 auto check=[&](const char*name,const std::vector<uint32_t>&prompt,const std::vector<uint32_t>&forced,bool reset){
  if(reset){reference.reset();candidate.reset();}
  std::vector<uint32_t> predictions;
  std::vector<std::vector<LayerState>> prediction_states;
  auto a=reference.reply(prompt,[&](unsigned i,uint32_t t){predictions.push_back(t);prediction_states.push_back(reference.state);return forced.empty()?t:forced.at(i);});
  size_t visits=0;
  auto b=candidate.reply_select(prompt,[&](unsigned i,uint32_t t){need(i<predictions.size()&&t==predictions[i],"prediction mismatch");state_equal(prediction_states[i],candidate.state);++visits;return forced.empty()?t:forced.at(i);});
  need(visits==predictions.size(),"prediction count");need(a.ids==b.ids&&a.end==b.end,"reply tokens/end");need(reference.fresh==candidate.fresh,"fresh flag");state_equal(reference.state,candidate.state);
  std::printf("CASE %s PASS predicted=%zu emitted=%zu end=%d fresh=%d states_bits=exact\n",name,visits,a.ids.size(),a.end,int(candidate.fresh));std::fflush(stdout);
 };
 check("1_fresh_empty_prompt_end259",{}, {259},false);
 check("2_multiturn_emit_then_end259",{72,105},{65,66,259},false);
 check("3_explicit_reset",{82},{259},true);
 check("4_end260_resets",{81},{67,260},false);
 check("5_fresh_after260",{88},{259},false);
 std::vector<uint32_t> tail(64,65);for(size_t i=0;i<tail.size();++i)tail[i]=uint32_t(65+i%20);
 check("6_forced64_tail_once",{84,97,105,108},tail,false);
 tail.back()=259;check("7_end259_at64",{69},tail,false);
 check("8_actual_greedy_multiturn",{72,101,108,108,111},{},false);
 std::puts("SUMMARY lifecycle_cases=8 PASS exact_predictions_emitted_tokens_end_flags_states=1 full_logits_reference_preserved=1 GPU=unused");
 return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
