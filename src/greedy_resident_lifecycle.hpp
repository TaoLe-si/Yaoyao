#pragma once
#include "greedy_pipeline_grouped_model.hpp"
#include <chrono>
namespace tao::dual {
struct GreedyResidentReply {
 std::vector<uint32_t> ids;int end=-1;double ttft=0,decode_seconds=0;
};
// Own session state, borrow an immutable model which must outlive the session.
// A prediction consumes its input exactly once; returned token is NOT consumed.
class GreedyResidentSession {
 const GreedyPipelineGroupedModel& model_;
public:
 std::vector<LayerState> state;bool fresh=true;
 explicit GreedyResidentSession(const GreedyPipelineGroupedModel&model):model_(model),state(model.initial()){}
 void reset(){state=model_.initial();fresh=true;}
 template<class Select> GreedyResidentReply reply_select(const std::vector<uint32_t>&prompt,Select select){
  using Clock=std::chrono::steady_clock;const auto begin=Clock::now();auto first=begin,last=begin;
  GreedyResidentReply result;result.ids.reserve(64);
  if(fresh){model_.advance(256,state);fresh=false;}
  model_.advance(257,state);
  for(auto t:prompt)model_.advance(t,state);
  model_.advance(259,state);
  uint32_t next=model_.greedy_step(258,state);
  for(unsigned i=0;i<64;++i){
   uint32_t t=select(i,next);auto now=Clock::now();
   if(t==259||t==260){model_.advance(t,state);result.end=int(t);if(t==260)reset();break;}
   if(result.ids.empty())first=now;last=now;result.ids.push_back(t);
   if(i==63)model_.advance(t,state);else next=model_.greedy_step(t,state);
  }
  if(result.end<0)model_.advance(259,state);
  if(!result.ids.empty())result.ttft=std::chrono::duration<double>(first-begin).count();
  if(result.ids.size()>1)result.decode_seconds=std::chrono::duration<double>(last-first).count();
  return result;
 }
 GreedyResidentReply reply(const std::vector<uint32_t>&prompt){return reply_select(prompt,[](unsigned,uint32_t t){return t;});}
};
}
