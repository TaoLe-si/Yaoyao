#define NOMINMAX
#define TAO_INPUT_SCALE
#define TAO_CPU_AVX2
#include "tokenizer_file.hpp"
#include "dual_model_bundle.hpp"
#include <cstdio>
int main(){try{std::string h;auto tok=tao::text::load_tokenizer("build/formal_tokenizer.bbp",h);auto a=tao::dual::load_bundle("build/full_reference_step_61.dsb",h),b=tao::dual::load_bundle("build/full_candidate_step_61.dsb",h);double ss=0,rr=0;float max=0;size_t positions=0,topdiff=0;for(auto prompt:{"Hello, introduce yourself.","What is 2 + 3?",u8"请只输出三个水果的名字，用逗号分隔。"}){auto sa=a.initial(),sb=b.initial();std::vector<uint32_t>ids={256,257};auto text=tok.encode(prompt);ids.insert(ids.end(),text.begin(),text.end());ids.push_back(259);ids.push_back(258);for(auto id:ids){auto x=a.step(id,sa),y=b.step(id,sb);++positions;topdiff+=(std::max_element(x.begin(),x.end())-x.begin())!=(std::max_element(y.begin(),y.end())-y.begin());for(size_t i=0;i<x.size();++i){if(!std::isfinite(x[i])||!std::isfinite(y[i]))throw std::runtime_error("nonfinite");double d=double(x[i])-y[i];ss+=d*d;rr+=double(x[i])*x[i];max=std::max(max,float(std::abs(d)));}}}printf("positions=%zu logits_maxabs=%.9g logits_relative_l2=%.9g unmasked_top1_differences=%zu\n",positions,max,std::sqrt(ss/rr),topdiff);return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
