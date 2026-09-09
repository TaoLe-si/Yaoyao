#define NOMINMAX
#include "dual_model_bundle.hpp"
#include "tokenizer_file.hpp"
#include <cstdio>
int main(){try{std::string h;tao::text::load_tokenizer("build/formal_tokenizer.bbp",h);auto m=tao::dual::load_bundle("build/yaoyao_parallel_step_60.dsb",h);for(const auto&kv:m.w){if(kv.first!="embedding"&&kv.first.find("norm")==std::string::npos&&kv.first.find("gate.bias")==std::string::npos)continue;double ss=0,sum=0;float mx=0;for(float x:kv.second){ss+=double(x)*x;sum+=x;mx=std::max(mx,std::abs(x));}printf("%s mean=%.6f rms=%.6f maxabs=%.6f\n",kv.first.c_str(),sum/kv.second.size(),sqrt(ss/kv.second.size()),mx);}return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
