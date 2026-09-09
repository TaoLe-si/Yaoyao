#define NOMINMAX
#define TAO_INPUT_SCALE
#define TAO_CPU_AVX2
#include "tokenizer_file.hpp"
#include "greedy_resident_lifecycle.hpp"
#include <iostream>
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("model required");
 auto start=std::chrono::steady_clock::now();std::string hash;
 auto tokenizer=tao::text::load_tokenizer("build/formal_tokenizer.bbp",hash);
 tao::dual::GreedyPipelineGroupedModel model(tao::dual::read_compact_bundle(argv[1],hash));
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
