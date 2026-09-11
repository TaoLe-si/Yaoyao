#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
namespace tao::dual {
struct Config { uint32_t layers=2,d=512,s=128,m=512,e=1024,vocab=16384; uint32_t dk=64; void validate()const{if(!layers||!d||!s||!m||!e||vocab<261)throw std::invalid_argument("model dimensions");
#ifdef TAO_DELTA_MEM
if(!dk||dk>d)throw std::invalid_argument("key dimension");
#endif
} uint64_t memory_size()const{
#ifdef TAO_DELTA_MEM
return uint64_t(m)*dk;
#else
return m;
#endif
} };
struct TensorSpec { std::string name; uint32_t rows,cols; bool ternary; uint64_t elements()const{return uint64_t(rows)*cols;} };
inline std::vector<TensorSpec> schema(const Config&c){c.validate();std::vector<TensorSpec> t;auto add=[&](std::string name,uint32_t r,uint32_t k,bool q=true){t.push_back({name,r,k,q});};add("embedding",c.vocab,c.d);add("vocab.bias",c.vocab,1,false);add("final.norm",c.d,1,false);
for(uint32_t l=0;l<c.layers;++l){auto p="layer."+std::to_string(l)+".";for(auto branch:{"s.candidate","s.gate"}){add(p+branch+".x",c.s,c.d);add(p+branch+".s",c.s,c.s);add(p+branch+".bias",c.s,1,false);}
#ifdef TAO_DELTA_MEM
add(p+"mem.key",c.dk,c.d);add(p+"mem.query",c.dk,c.d);add(p+"mem.value",c.m,c.d);add(p+"mem.beta",1,c.d,false);add(p+"mem.beta.bias",1,1,false);
#else
for(auto branch:{"m.candidate","m.gate"}){add(p+branch+".x",c.m,c.d);add(p+branch+".s",c.m,c.s);add(p+branch+".m",c.m,c.m);add(p+branch+".bias",c.m,1,false);}
#endif
add(p+"read.s",c.d,c.s);
#ifndef TAO_DELTA_MEM
add(p+"read.m",c.d,c.m);
#endif
#ifdef TAO_NO_FFN
for(auto norm:{"input.norm","read.norm"})add(p+norm,c.d,1,false);
#else
add(p+"ff.up",c.e,c.d);add(p+"ff.down",c.d,c.e);for(auto norm:{"input.norm","read.norm","ff.norm"})add(p+norm,c.d,1,false);
#endif
}
return t;}
}
