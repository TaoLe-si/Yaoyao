#define NOMINMAX
#include "dual_state_cpu.hpp"
#include "dual_model_bundle.hpp"
#include "tokenizer_file.hpp"
#include "bpe_pilot_reader.hpp"
#include <cstdio>
int main(int argc,char**argv){try{using namespace tao::dual;if(argc!=2)throw std::runtime_error("model path required");std::string hash;tao::text::load_tokenizer("build/formal_tokenizer.bbp",hash);auto cpu=load_bundle(argv[1],hash);std::ifstream f("build/bpe_pilot_validation.bin",std::ios::binary);auto docs=tao::data::read_bpe_pilot(f,hash);double loss=0;size_t n=0;for(auto&t:docs){auto state=cpu.initial();for(size_t i=1;i<t.size();++i){auto logits=cpu.step(t[i-1].id,state);if(t[i].loss){double mx=*std::max_element(logits.begin(),logits.end()),sum=0;for(float v:logits)sum+=std::exp(double(v)-mx);double l=std::log(sum)+mx-logits[t[i].id];if(!std::isfinite(l))throw std::runtime_error("nonfinite");loss+=l;++n;}}}printf("VALIDATION backend=CPU docs=%zu supervised=%zu NLL=%.9f model=%s\n",docs.size(),n,loss/n,argv[1]);return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
