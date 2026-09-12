#pragma once
#include "greedy_pipeline_grouped_model.hpp"
#include <chrono>
#include <cstdlib>
#include <vector>
#include <cstdint>
namespace tao::dual {
// 推理端重复惩罚 + 最大输出。默认 pen=1.0 / max_out=64。
// 惩罚覆盖本回复已生成的每一个 token（无窗口）；每出现一次累加 pen（频率，不是只打一次的 presence）。
// pen<=0 关闭，与无惩罚逐位一致。另：3-gram 闭环硬阻断；连续同一 token 满 4 次则收束。
// CLI --rep-pen/--max-out，环境变量 TAO_REP_PEN/TAO_MAX_OUT，运行时 /rep-pen /max-out。
struct RepPenConfig { float pen=1.0f; unsigned max_out=64; };
inline RepPenConfig g_rep{};
struct GreedyResidentReply {
 std::vector<uint32_t> ids;int end=-1;double ttft=0,decode_seconds=0;
 unsigned generated=0, max_out=0; float pen=0;
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
  GreedyResidentReply result;
  const float pen=g_rep.pen;
  unsigned cap=g_rep.max_out; if(cap<1u)cap=1u; if(cap>4096u)cap=4096u;
  result.pen=pen; result.max_out=cap; result.ids.reserve(cap);
  if(fresh){model_.advance(256,state);fresh=false;}
  model_.advance(257,state);
  for(auto t:prompt)model_.advance(t,state);
  model_.advance(259,state);
  // 只惩罚本回复已生成的 token（全集），prompt 不参与。
  std::vector<float> adj;
  if(pen>0) adj.assign(size_t(model_.vocab()),0.0f);
  constexpr float kNgramBlock=1.0e6f;
  constexpr int kNgram=3;
  auto ngram_extra=[&]()->std::vector<uint32_t>{
   std::vector<uint32_t> extra;
   if(adj.empty()||int(result.ids.size())<kNgram)return extra;
   const size_t pre=size_t(kNgram-1);
   const auto& gen=result.ids;
   for(size_t i=0;i+pre<gen.size();++i){
    bool match=true;
    for(size_t j=0;j<pre;++j)if(gen[gen.size()-pre+j]!=gen[i+j]){match=false;break;}
    if(!match)continue;
    const size_t nxt=i+pre;
    if(nxt>=gen.size())continue;
    const uint32_t b=gen[nxt];
    if(b==259||b==260||b>=adj.size())continue;
    adj[b]+=kNgramBlock;extra.push_back(b);
   }
   if(!extra.empty())model_.set_head_adjustment(adj.data());
   return extra;
  };
  auto undo_ngram=[&](const std::vector<uint32_t>& extra){
   for(uint32_t b:extra)if(b<adj.size())adj[b]-=kNgramBlock;
   if(!adj.empty())model_.set_head_adjustment(adj.data());
  };
  uint32_t next=model_.greedy_step(258,state);
  for(unsigned i=0;i<cap;++i){
   uint32_t t=select(i,next);auto now=Clock::now();
   if(t==259||t==260){model_.advance(t,state);result.end=int(t);if(t==260)reset();break;}
   if(result.ids.empty())first=now;last=now;result.ids.push_back(t);
   if(i+1u==cap){model_.advance(t,state);break;}
   if(!adj.empty()&&t<adj.size()){
    adj[t]+=pen;
    model_.set_head_adjustment(adj.data());
    unsigned run=1;
    for(size_t k=result.ids.size();k>1&&result.ids[k-1]==result.ids[k-2];--k){++run;if(run>=4)break;}
    if(run>=4){model_.advance(259,state);result.end=259;break;}
   }
   auto extra=ngram_extra();
   next=model_.greedy_step(t,state);
   undo_ngram(extra);
  }
  result.generated=unsigned(result.ids.size());
  if(!adj.empty())model_.set_head_adjustment(nullptr);
  if(result.end<0)model_.advance(259,state);
  if(!result.ids.empty())result.ttft=std::chrono::duration<double>(first-begin).count();
  if(result.ids.size()>1)result.decode_seconds=std::chrono::duration<double>(last-first).count();
  return result;
 }
 GreedyResidentReply reply(const std::vector<uint32_t>&prompt){return reply_select(prompt,[](unsigned,uint32_t t){return t;});}
};
}
