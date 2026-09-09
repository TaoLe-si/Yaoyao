#define NOMINMAX
#define TAO_INPUT_SCALE
#include "tokenizer_file.hpp"
#include "dual_model_bundle.hpp"
#include <chrono>
#include <cstdio>
int main(int argc,char**argv){try{
if(argc!=2)throw std::runtime_error("model required");std::string hash;auto b=tao::text::load_tokenizer("build/formal_tokenizer.bbp",hash);auto m=tao::dual::load_bundle(argv[1],hash);
std::vector<std::vector<std::string>> sessions={{u8"你好，请介绍一下自己。",u8"我叫小林，我喜欢蓝色。",u8"我叫什么名字？我喜欢什么颜色？"},{"What is 2 + 3?","Explain your previous answer in one sentence.","Now multiply that result by 4."},{u8"请把这句话翻译成英文：今天阳光很好。",u8"请只输出三个水果的名字，用逗号分隔。",u8"把你刚才列出的第二个水果再说一遍。"}};
printf("MODEL %s\nTOKENIZER %s\nPOLICY greedy max64/session_state_preserved no_min_length role_tokens_excluded\n",argv[1],hash.c_str());
for(size_t sid=0;sid<sessions.size();++sid){auto state=m.initial();tao::dual::Vec logits=m.step(256,state);bool closed=false;printf("SESSION %zu\n",sid+1);
for(size_t turn=0;turn<sessions[sid].size();++turn){if(closed){printf("SKIP session ended by EOS\n");break;}auto prompt=sessions[sid][turn];printf("USER %s\n",prompt.c_str());auto start=std::chrono::steady_clock::now();logits=m.step(257,state);for(auto id:b.encode(prompt))logits=m.step(id,state);logits=m.step(259,state);logits=m.step(258,state);std::vector<uint32_t> ids;int end=-1;
for(int k=0;k<64;++k){for(float v:logits)if(!std::isfinite(v))throw std::runtime_error("nonfinite logits");for(unsigned id:{256u,257u,258u})logits[id]=-std::numeric_limits<float>::infinity();unsigned id=std::max_element(logits.begin(),logits.end())-logits.begin();logits=m.step(id,state);if(id==259||id==260){end=id;closed=id==260;break;}ids.push_back(id);}
auto text=b.decode(ids);printf("ASSISTANT_HEX ");for(unsigned char c:text)printf("%02x",c);printf("\nMETA turn=%zu tokens=%zu end=%d truncated=%d seconds_including_prefill=%.6f\n",turn+1,ids.size(),end,end<0,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());if(end<0){logits=m.step(259,state);printf("CONTROL injected TURN_END after length limit, not generated\n");}fflush(stdout);
}}
return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
