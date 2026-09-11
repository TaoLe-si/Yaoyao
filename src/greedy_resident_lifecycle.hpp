#pragma once
#include "greedy_pipeline_grouped_model.hpp"
#include <chrono>
#include <cstdlib>
#include <vector>
#include <cstdint>
namespace tao::dual {
// 推理端重复惩罚（乘法，HF 式）：命中 token 的正 logit 除以 pen、负 logit 乘以 pen。
//   pen=1.0 等价关闭。win=0 = 惩罚覆盖本回合全部已生成 token（默认）：本回合出现过
//   一次即进入惩罚集，重放时会再次命中，因此对每个 token 都有效，不是固定窗口。
//   win=W>0 = 仅惩罚最近 W 个（旧滑窗行为，仅作对照）。
// 实测（step_7573，held-out n=200，乱码/空回复/长度比 一并记录）：
//   无惩罚            loop 104/200  F1 0.0414  乱码 9
//   减法 λ=1.3 win=32  loop  22/200  F1 0.0515  乱码 6   ← 旧实现（全窗口时恶化到 104）
//   乘法 1.2  win=0    loop   7/200  F1 0.0556  乱码 4
//   乘法 1.3  win=0    loop   0/200  F1 0.0547  乱码 2  ← 默认
//   乘法 1.5  win=0    loop   0/200  F1 0.0534  乱码 1  （开始以 F1 换安全）
// CLI --rep-pen/--rep-win 或环境变量 TAO_REP_PEN/TAO_REP_WIN 可覆盖。
struct RepPenConfig { float pen=1.3f; unsigned win=0; };
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
  if(pen>0){adj.assign(size_t(model_.vocab()),0.0f);adj.shrink_to_fit();}
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
    if(win&&winTok.size()>size_t(win))winTok.erase(winTok.begin());
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
