#pragma once
#include "greedy_pipeline_grouped_model.hpp"
#include <chrono>
#include <cstdlib>
#include <vector>
#include <cstdint>
namespace tao::dual {
// 推理端重复惩罚配置。默认 pen=1.0 / win=32 为实测推荐值：
//   step_2200: loop 68.3%->45.8%（win=8）；step_4850: win 8->32 再砍半
//   loop 44.2%->25.8% 且 F1 0.0516->0.0522（短语级循环会跨 8-token 窗口逃逸，
//   必须覆盖更长的已生成序列）。pen<=0 = 关闭（与无惩罚实现逐位一致）。
// CLI --rep-pen/--rep-win 或环境变量 TAO_REP_PEN/TAO_REP_WIN 可覆盖。
struct RepPenConfig { float pen=1.0f; unsigned win=32; };
inline RepPenConfig g_rep{};
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
  // 推理端重复惩罚（exposure bias 缓解）：只惩罚本回复内已生成的 token，
  // prompt 不参与，避免破坏"原样复制"类任务。
  const float pen=g_rep.pen;
  const unsigned win=g_rep.win;
  std::vector<float> adj;std::vector<uint32_t> winTok,touched;
  if(pen>0&&win>0){adj.assign(size_t(model_.vocab()),0.0f);adj.shrink_to_fit();}
  uint32_t next=model_.greedy_step(258,state);
  for(unsigned i=0;i<64;++i){
   uint32_t t=select(i,next);auto now=Clock::now();
   if(t==259||t==260){model_.advance(t,state);result.end=int(t);if(t==260)reset();break;}
   if(result.ids.empty())first=now;last=now;result.ids.push_back(t);
   if(i==63){model_.advance(t,state);break;}
   if(!adj.empty()){
    for(auto x:touched)adj[x]=0;
    touched.clear();
    winTok.push_back(t);
    if(winTok.size()>size_t(win))winTok.erase(winTok.begin());
    for(auto x:winTok)if(adj[x]==0){adj[x]=pen;touched.push_back(x);}
    model_.set_head_adjustment(adj.data());
   }
   next=model_.greedy_step(t,state);
  }
  if(!adj.empty())model_.set_head_adjustment(nullptr);
  if(result.end<0)model_.advance(259,state);
  if(!result.ids.empty())result.ttft=std::chrono::duration<double>(first-begin).count();
  if(result.ids.size()>1)result.decode_seconds=std::chrono::duration<double>(last-first).count();
  return result;
 }
 GreedyResidentReply reply(const std::vector<uint32_t>&prompt){return reply_select(prompt,[](unsigned,uint32_t t){return t;});}
};
}
