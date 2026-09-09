#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
namespace tao::dual {
struct Config { uint32_t layers=8,d=512,s=128,m=512,e=1024,vocab=16384; void validate()const{if(!layers||!d||!s||!m||!e||vocab<261)throw std::invalid_argument("model dimensions");} };
struct TensorSpec { std::string name; uint32_t rows,cols; bool ternary; uint64_t elements()const{return uint64_t(rows)*cols;} };
inline std::vector<TensorSpec> schema(const Config&c){c.validate();std::vector<TensorSpec> t;auto add=[&](std::string name,uint32_t r,uint32_t k,bool q=true){t.push_back({name,r,k,q});};add("embedding",c.vocab,c.d);add("vocab.bias",c.vocab,1,false);add("final.norm",c.d,1,false);
for(uint32_t l=0;l<c.layers;++l){auto p="layer."+std::to_string(l)+".";for(auto branch:{"s.candidate","s.gate"}){add(p+branch+".x",c.s,c.d);add(p+branch+".s",c.s,c.s);add(p+branch+".bias",c.s,1,false);}for(auto branch:{"m.candidate","m.gate"}){add(p+branch+".x",c.m,c.d);add(p+branch+".s",c.m,c.s);add(p+branch+".m",c.m,c.m);add(p+branch+".bias",c.m,1,false);}add(p+"read.s",c.d,c.s);add(p+"read.m",c.d,c.m);add(p+"ff.up",c.e,c.d);add(p+"ff.down",c.d,c.e);for(auto norm:{"input.norm","read.norm","ff.norm"})add(p+norm,c.d,1,false);}
return t;}
}
