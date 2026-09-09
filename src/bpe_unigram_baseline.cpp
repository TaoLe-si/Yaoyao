#define NOMINMAX
#include "tokenizer_file.hpp"
#include "bpe_pilot_reader.hpp"
#include <cmath>
#include <cstdio>
int main(){try{std::string hash;tao::text::load_tokenizer("build/formal_tokenizer.bbp",hash);std::ifstream train("build/bpe_pilot_train.bin",std::ios::binary);auto docs=tao::data::read_bpe_pilot(train,hash);std::vector<size_t>count(16384,0);size_t total=0;for(const auto&d:docs)for(size_t i=1;i<d.size();++i)if(d[i].loss){++count[d[i].id];++total;}
// Fixed add-one smoothing, no validation selection or special-token exclusion.
std::ifstream val("build/bpe_pilot_validation.bin",std::ios::binary);auto held=tao::data::read_bpe_pilot(val,hash);double loss=0;size_t n=0,unseen=0;for(const auto&d:held)for(size_t i=1;i<d.size();++i)if(d[i].loss){loss-=std::log(double(count[d[i].id]+1)/double(total+count.size()));++n;unseen+=count[d[i].id]==0;}
printf("BASELINE train_targets=%zu validation_docs=%zu validation_targets=%zu unseen_targets=%zu smoothing=add1 vocab=16384 validation_NLL=%.9f uniform_NLL=%.9f tokenizer=%s\n",total,held.size(),n,unseen,loss/n,std::log(16384.),hash.c_str());return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
