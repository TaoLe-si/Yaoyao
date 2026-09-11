#define NOMINMAX
#define TAO_INPUT_SCALE
#define TAO_CPU_AVX2
#include "tokenizer_file.hpp"
#include "greedy_resident_lifecycle.hpp"
#include <iostream>
#include <thread>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>
int main(int argc,char**argv){try{
 // CLI: h2r_cpu MODEL [--rep-pen F] [--rep-win N]
 // 重复惩罚默认 1.0/8（实测推荐值）；CLI 未给时回退环境变量。
 std::vector<std::string> args(argv+1,argv+argc);
 std::string modelpath; bool cliPen=false,cliWin=false;
 for(size_t i=0;i<args.size();++i){
  if(args[i]=="--rep-pen"){ if(i+1>=args.size())throw std::runtime_error("--rep-pen needs a value");
   tao::dual::g_rep.pen=strtof(args[++i].c_str(),nullptr); cliPen=true; }
  else if(args[i]=="--rep-win"){ if(i+1>=args.size())throw std::runtime_error("--rep-win needs a value");
   unsigned long w=std::strtoul(args[++i].c_str(),nullptr,10);
   if(w==0||w>64)throw std::runtime_error("--rep-win range 1..64");
   tao::dual::g_rep.win=unsigned(w); cliWin=true; }
  else if(!args[i].empty()&&args[i][0]=='-')throw std::runtime_error("unknown option "+args[i]);
  else modelpath=args[i];
 }
 if(modelpath.empty())throw std::runtime_error("model required");
 if(!cliPen){ if(const char* e=std::getenv("TAO_REP_PEN"))tao::dual::g_rep.pen=strtof(e,nullptr); }
 if(!cliWin){ if(const char* e=std::getenv("TAO_REP_WIN")){unsigned long w=std::strtoul(e,nullptr,10);
  if(w>0&&w<=64)tao::dual::g_rep.win=unsigned(w);} }
 auto start=std::chrono::steady_clock::now();std::string hash;
 // 词表路径可用 TAO_TOKENIZER 覆盖：换词表后必须与训练所用词表一致，
 // 否则同一 id 序列会被解码成完全不同的文本。
 const char* tokpath=std::getenv("TAO_TOKENIZER"); if(!tokpath)tokpath="build/formal_tokenizer.bbp";
 auto tokenizer=tao::text::load_tokenizer(tokpath,hash);
 tao::dual::GreedyPipelineGroupedModel model(tao::dual::read_compact_bundle(argv[1],hash));
 // CPU 行并行线程数：生产解码器此前从不调用 set_cpu_threads，而 pool 默认构造为
 // {1u}（thread_count()=2），在 8 核/16 逻辑核机器上只用了 2 个核。此处显式使用
 // 机器可用核数，上限 8（记录：16 线程会 SMT 饥饿）。行并行按行切分，每行点积在
 // 单线程内完成、不跨线程归约，故不改变数值（下方 THREADS 与逐字一致性已验证）。
 {
  unsigned hw=std::thread::hardware_concurrency(); if(hw==0u)hw=4u;
  unsigned nt=hw<8u?hw:8u;
  if(const char* e=std::getenv("TAO_CPU_THREADS")){int v=std::atoi(e); if(v>0)nt=unsigned(v);}
  if(nt>1u)model.set_cpu_threads(nt);
  std::cerr<<"THREADS="<<nt<<" hw="<<hw<<std::endl;
  std::cerr<<"REP_PEN="<<tao::dual::g_rep.pen<<" REP_WIN="<<tao::dual::g_rep.win<<std::endl;
 }
 tao::dual::GreedyResidentSession session(model);
 std::cerr<<"READY loads=1 load_seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<std::endl;
 std::string prompt;size_t request=0;
 while(std::getline(std::cin,prompt)){
  if(prompt=="/quit")break;
  if(prompt=="/reset"){session.reset();std::cout<<"RESET"<<std::endl;continue;}
  if(prompt.size()>8192){std::cout<<"ERROR prompt exceeds8192bytes"<<std::endl;continue;}
  auto begin=std::chrono::steady_clock::now();auto encoded=tokenizer.encode(prompt);
  auto compute_begin=std::chrono::steady_clock::now();auto result=session.reply(encoded);
  result.ttft+=std::chrono::duration<double>(compute_begin-begin).count();
  auto output=tokenizer.decode(result.ids);
  std::cout<<"BEGIN_REPLY "<<++request<<"\n"<<(output.empty()?"[empty reply]":output)<<"\nEND_REPLY tokens="<<result.ids.size()<<" end="<<result.end<<" truncated="<<(result.end<0)<<" loads=1";
  if(!result.ids.empty())std::cout<<" ttft_compute="<<result.ttft;
  if(result.ids.size()>1)std::cout<<" decode_tps="<<(result.ids.size()-1)/result.decode_seconds;
  else std::cout<<" decode_tps=NA";
  std::cout<<std::endl;
 }
 return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}}
