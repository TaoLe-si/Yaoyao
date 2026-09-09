#define NOMINMAX
#define TAO_INPUT_SCALE
#include "tokenizer_file.hpp"
#include "dual_model_bundle.hpp"
#include "bpe_pilot_reader.hpp"
#include <cstdio>
int main(int argc,char**argv){try{if(argc!=2)throw std::runtime_error("model required");std::string hash;auto b=tao::text::load_tokenizer("build/formal_tokenizer.bbp",hash);auto m=tao::dual::load_bundle(argv[1],hash);for(auto split:{"train","validation"}){std::ifstream f(std::string("build/bpe_pilot_")+split+".bin",std::ios::binary);auto docs=tao::data::read_bpe_pilot(f,hash);for(size_t d=0;d<8&&d<docs.size();++d){auto state=m.initial();for(size_t i=1;i<docs[d].size();++i){auto logits=m.step(docs[d][i-1].id,state);if(docs[d][i-1].id!=tao::data::ASSISTANT)continue;int target=docs[d][i].id;float mx=*std::max_element(logits.begin(),logits.end());double sum=0;for(float v:logits)sum+=exp(double(v)-mx);unsigned top=std::max_element(logits.begin(),logits.end())-logits.begin();printf("%s doc=%zu first_target=%d supervised=%d top=%u p_end=%.8f p_top=%.8f p_target=%.8f\n",split,d,target,docs[d][i].loss,top,exp(double(logits[259])-mx)/sum,exp(double(logits[top])-mx)/sum,exp(double(logits[target])-mx)/sum);break;}}}return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
