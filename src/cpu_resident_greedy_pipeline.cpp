#define NOMINMAX
#define TAO_INPUT_SCALE
#define TAO_CPU_AVX2
#include "tokenizer_file.hpp"
#include "greedy_resident_lifecycle.hpp"
#include <iostream>
#include <thread>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
namespace {
void print_cfg(std::ostream& o){
 o<<"REP_PEN="<<tao::dual::g_rep.pen
  <<" MAX_OUT="<<tao::dual::g_rep.max_out<<std::endl;
}
bool set_pen(const char* s){
 char* e=nullptr; const float v=strtof(s,&e);
 if(!s||e==s||!std::isfinite(v)||v<0.f) return false;
 tao::dual::g_rep.pen=v; return true;
}
bool set_max(const char* s){
 char* e=nullptr; const unsigned long m=strtoul(s,&e,10);
 if(!s||e==s||m<1||m>4096) return false;
 tao::dual::g_rep.max_out=unsigned(m); return true;
}
const char* after_key(const std::string& line, const char* key){
 const size_t n=std::strlen(key);
 if(line.size()<n||line.compare(0,n,key)!=0) return nullptr;
 if(line.size()==n) return "";
 if(line[n]!=' '&&line[n]!='=') return nullptr;
 size_t i=n+1; while(i<line.size()&&line[i]==' ') ++i;
 return line.c_str()+i;
}
bool handle_assign(const std::string& line){
 const char* v=nullptr;
 if((v=after_key(line,"/rep-pen"))||(v=after_key(line,"/rep_pen"))){
  if(!v||!*v||!set_pen(v)){ std::cout<<"ERROR --rep-pen needs a finite value >=0"<<std::endl; return true; }
  std::cout<<"CONFIG "; print_cfg(std::cout); return true;
 }
 if((v=after_key(line,"/rep-win"))||(v=after_key(line,"/rep_win"))||line=="/rep-win"||line=="/rep_win"){
  std::cout<<"ERROR repetition penalty applies to all generated tokens; window is not configurable"<<std::endl; return true;
 }
 if((v=after_key(line,"/max-out"))||(v=after_key(line,"/max_out"))||(v=after_key(line,"/max"))){
  if(!v||!*v||!set_max(v)){ std::cout<<"ERROR --max-out range 1..4096"<<std::endl; return true; }
  std::cout<<"CONFIG "; print_cfg(std::cout); return true;
 }
 if(line=="/config"){ std::cout<<"CONFIG "; print_cfg(std::cout); return true; }
 return false;
}
}

int main(int argc,char**argv){try{
 // CLI: h2r_cpu MODEL [--rep-pen F] [--max-out N]
 // 默认 pen=1.0 / max_out=64。惩罚覆盖本回复全部已生成 token。
 std::vector<std::string> args(argv+1,argv+argc);
 std::string modelpath; bool cliPen=false,cliMax=false;
 for(size_t i=0;i<args.size();++i){
  if(args[i]=="--rep-pen"){
   if(i+1>=args.size()||!set_pen(args[++i].c_str()))throw std::runtime_error("--rep-pen needs a finite value >=0");
   cliPen=true;
  }else if(args[i]=="--rep-win"){
   throw std::runtime_error("repetition penalty applies to all generated tokens; --rep-win removed");
  }else if(args[i]=="--max-out"||args[i]=="--max"){
   if(i+1>=args.size()||!set_max(args[++i].c_str()))throw std::runtime_error("--max-out range 1..4096");
   cliMax=true;
  }else if(!args[i].empty()&&args[i][0]=='-')throw std::runtime_error("unknown option "+args[i]);
  else modelpath=args[i];
 }
 if(modelpath.empty())throw std::runtime_error("model required");
 if(!cliPen){ if(const char* e=std::getenv("TAO_REP_PEN")) if(!set_pen(e)) throw std::runtime_error("TAO_REP_PEN invalid"); }
 if(std::getenv("TAO_REP_WIN")) throw std::runtime_error("TAO_REP_WIN removed; penalty applies to all generated tokens");
 if(!cliMax){ if(const char* e=std::getenv("TAO_MAX_OUT")) if(!set_max(e)) throw std::runtime_error("TAO_MAX_OUT range 1..4096"); }
 auto start=std::chrono::steady_clock::now();std::string hash;
 const char* tokpath=std::getenv("TAO_TOKENIZER"); if(!tokpath)tokpath="build/formal_tokenizer.bbp";
 auto tokenizer=tao::text::load_tokenizer(tokpath,hash);
 tao::dual::GreedyPipelineGroupedModel model(tao::dual::read_compact_bundle(modelpath,hash));
 {
  unsigned hw=std::thread::hardware_concurrency(); if(hw==0u)hw=4u;
  unsigned nt=hw<8u?hw:8u;
  if(const char* e=std::getenv("TAO_CPU_THREADS")){int v=std::atoi(e); if(v>0)nt=unsigned(v);}
  if(nt>1u)model.set_cpu_threads(nt);
  std::cerr<<"THREADS="<<nt<<" hw="<<hw<<std::endl;
  print_cfg(std::cerr);
 }
 tao::dual::GreedyResidentSession session(model);
 std::cerr<<"READY loads=1 load_seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<std::endl;
 std::string prompt;size_t request=0;
 while(std::getline(std::cin,prompt)){
  if(prompt=="/quit")break;
  if(prompt=="/reset"){session.reset();std::cout<<"RESET"<<std::endl;continue;}
  if(handle_assign(prompt)) continue;
  if(prompt.size()>8192){std::cout<<"ERROR prompt exceeds8192bytes"<<std::endl;continue;}
  auto begin=std::chrono::steady_clock::now();auto encoded=tokenizer.encode(prompt);
  auto compute_begin=std::chrono::steady_clock::now();auto result=session.reply(encoded);
  result.ttft+=std::chrono::duration<double>(compute_begin-begin).count();
  auto output=tokenizer.decode(result.ids);
  std::cout<<"BEGIN_REPLY "<<++request<<"\n"<<(output.empty()?"[empty reply]":output)
   <<"\nEND_REPLY tokens="<<result.ids.size()<<" end="<<result.end
   <<" truncated="<<(result.end<0)<<" loads=1"
   <<" rep_pen="<<result.pen<<" max_out="<<result.max_out;
  if(!result.ids.empty())std::cout<<" ttft_compute="<<result.ttft;
  if(result.ids.size()>1)std::cout<<" decode_tps="<<(result.ids.size()-1)/result.decode_seconds;
  else std::cout<<" decode_tps=NA";
  std::cout<<std::endl;
 }
 return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}}
