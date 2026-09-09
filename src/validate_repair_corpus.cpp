#define NOMINMAX
#include "tokenizer_file.hpp"
#include "bpe_pilot_reader.hpp"
#include "batched_slot_plan.hpp"
#include <cstdio>
#include <sstream>
int main(int argc,char**argv){try{if(argc!=2)throw std::runtime_error("dataset path");std::string h;tao::text::load_tokenizer("build/formal_tokenizer.bbp",h);std::ifstream f(argv[1],std::ios::binary);if(!f)throw std::runtime_error("open");std::string raw((std::istreambuf_iterator<char>(f)),{});std::istringstream in(raw);auto docs=tao::data::read_bpe_pilot(in,h);if(docs.empty())throw std::runtime_error("empty");size_t tokens=0,targets=0,ends=0;for(auto&d:docs)for(auto&t:d){++tokens;targets+=t.loss;ends+=t.loss&&t.id==259;if(t.id==260)throw std::runtime_error("EOS");}size_t expectedP=tokens-docs.size(),expectedN=targets;tao::data::PilotCursor c(std::move(docs),4,false);size_t p=0,n=0,b=0;for(;;){auto x=tao::data::take_batch(c,256);if(!x.timesteps)break;p+=x.positions;n+=x.supervised;++b;}if(p!=expectedP||n!=expectedN)throw std::runtime_error("coverage");printf("PASS docs=%zu tokens=%zu supervised=%zu end_targets=%zu positions=%zu batch_rounds=%zu SHA256=%s tokenizer=%s\n",c.docs.size(),tokens,targets,ends,p,b,tao::text::sha256(raw).c_str(),h.c_str());return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
