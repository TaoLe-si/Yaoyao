#define NOMINMAX
#define TAO_INPUT_SCALE
#include "tokenizer_file.hpp"
#include "dual_model_bundle.hpp"
#include <chrono>
#include <cstdio>
int main(int argc,char**argv){try{
if(argc!=2)throw std::runtime_error("model required");auto load_start=std::chrono::steady_clock::now();std::string hash;auto b=tao::text::load_tokenizer("build/formal_tokenizer.bbp",hash);auto m=tao::dual::load_bundle(argv[1],hash);
printf("LOAD tokenizer_and_model_seconds=%.6f\n",std::chrono::duration<double>(std::chrono::steady_clock::now()-load_start).count());
std::vector<std::vector<std::string>> sessions={{u8"你好，请介绍一下自己。",u8"我叫小林，我喜欢蓝色。",u8"我叫什么名字？我喜欢什么颜色？"},{"What is 2 + 3?","Explain your previous answer in one sentence.","Now multiply that result by 4."},{u8"请把这句话翻译成英文：今天阳光很好。",u8"请只输出三个水果的名字，用逗号分隔。",u8"把你刚才列出的第二个水果再说一遍。"}};
printf("MODEL %s\nTOKENIZER %s\nPOLICY greedy max64/session_state_preserved no_min_length role_tokens_excluded\n",argv[1],hash.c_str());
for(size_t sid=0;sid<sessions.size();++sid){auto state=m.initial();tao::dual::Vec logits=m.step(256,state);bool closed=false;printf(u8"\n===== 对话组 %zu =====\n",sid+1);
for(size_t turn=0;turn<sessions[sid].size();++turn){if(closed){printf("SKIP session ended by EOS\n");break;}auto prompt=sessions[sid][turn];printf(u8"用户：%s\n",prompt.c_str());auto start=std::chrono::steady_clock::now();logits=m.step(257,state);for(auto id:b.encode(prompt))logits=m.step(id,state);logits=m.step(259,state);logits=m.step(258,state);auto decode_start=std::chrono::steady_clock::now();auto first=decode_start,last=decode_start;std::vector<uint32_t> ids;int end=-1;
for(int k=0;k<64;++k){for(float v:logits)if(!std::isfinite(v))throw std::runtime_error("nonfinite logits");for(unsigned id:{256u,257u,258u})logits[id]=-std::numeric_limits<float>::infinity();unsigned id=std::max_element(logits.begin(),logits.end())-logits.begin();auto emitted=std::chrono::steady_clock::now();if(id==259||id==260){logits=m.step(id,state);end=id;closed=id==260;break;}if(ids.empty())first=emitted;last=emitted;ids.push_back(id);logits=m.step(id,state);}
auto decode_end=std::chrono::steady_clock::now();double ds=std::chrono::duration<double>(decode_end-decode_start).count(),ps=std::chrono::duration<double>(decode_start-start).count();auto text=b.decode(ids);printf(u8"夭夭：");if(text.empty())printf(u8"〔空回复〕");else fwrite(text.data(),1,text.size(),stdout);printf("\nMETA turn=%zu tokens=%zu end=%d truncated=%d seconds_including_prefill=%.6f\n",turn+1,ids.size(),end,end<0,ps+ds);printf(u8"提示词处理：%.6f秒；解码循环：%.6f秒（含正文采样和前向、结束token处理；不含打印和文本转换）\n",ps,ds);if(ids.empty())printf(u8"TTFT与解码速度：N/A（无正文token）\n");else{printf("TTFT_compute_seconds=%.6f\n",std::chrono::duration<double>(first-start).count());if(ids.size()>1){double interval=std::chrono::duration<double>(last-first).count();printf("DECODE inter_token_count=%zu first_to_last_seconds=%.6f tokens_per_second=%.3f\n",ids.size()-1,interval,(ids.size()-1)/interval);}else printf("DECODE tokens_per_second=N/A only_one_token\n");}printf("TIMING_SCOPE internal_token_ready_not_streaming_UI; state_finalization_in_loop_not_in_first_to_last\n");if(end<0){logits=m.step(259,state);printf("CONTROL injected TURN_END after length limit, not generated\n");}fflush(stdout);
}}
return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
