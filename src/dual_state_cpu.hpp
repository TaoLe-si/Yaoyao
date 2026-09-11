#pragma once
#include "dual_state_config.hpp"
#include <map>
#include <cmath>
#ifdef TAO_CPU_AVX2
#include "cpu_dot_avx2.hpp"
#endif
namespace tao::dual {
using Vec=std::vector<float>;
struct LayerState{Vec s,m;};
struct CpuModel{
Config c;std::map<std::string,Vec>w;
#ifdef TAO_CPU_AVX2
bool fast=true;
#endif
explicit CpuModel(Config cfg):c(cfg){for(auto&t:schema(c))w.emplace(t.name,Vec(t.elements(),0));}
size_t memory_size()const{
#ifdef TAO_DELTA_MEM
return size_t(c.m)*c.dk;
#else
return c.m;
#endif
}
std::vector<LayerState> initial()const{return std::vector<LayerState>(c.layers,LayerState{Vec(c.s),Vec(memory_size())});}
Vec linear(const std::string&name,const Vec&x,uint32_t rows)const{const auto&a=w.at(name);if(a.size()!=uint64_t(rows)*x.size())throw std::runtime_error("matrix shape");Vec y(rows);
#ifdef TAO_CPU_AVX2
if(fast){for(uint32_t r=0;r<rows;++r)y[r]=tao_dot_avx2(a.data()+r*x.size(),x.data(),x.size());return y;}
#endif
for(uint32_t r=0;r<rows;++r)for(size_t j=0;j<x.size();++j)y[r]+=a[r*x.size()+j]*x[j];return y;}
static void add(Vec&a,const Vec&b){if(a.size()!=b.size())throw std::runtime_error("vector shape");for(size_t i=0;i<a.size();++i)a[i]+=b[i];}
Vec norm(const Vec&x,const std::string&name)const{const auto&g=w.at(name);if(g.size()!=x.size())throw std::runtime_error("norm shape");float sum=0;for(float z:x)sum+=z*z;float inv=1/std::sqrt(sum/x.size()+1e-5f);Vec y(x.size());for(size_t j=0;j<x.size();++j)y[j]=x[j]*inv*g[j];return y;}
static float sigmoid(float x){if(x>=0)return 1/(1+std::exp(-x));float e=std::exp(x);return e/(1+e);}
Vec step(uint32_t token,std::vector<LayerState>&state)const{if(token>=c.vocab||state.size()!=c.layers)throw std::invalid_argument("token/state");for(const auto&z:state)if(z.s.size()!=c.s||z.m.size()!=memory_size())throw std::invalid_argument("state shape");const auto&emb=w.at("embedding");Vec x(emb.begin()+uint64_t(token)*c.d,emb.begin()+uint64_t(token+1)*c.d);
#ifdef TAO_INPUT_SCALE
for(float&v:x)v*=std::sqrt(float(c.d));
#endif
for(uint32_t l=0;l<c.layers;++l){auto p="layer."+std::to_string(l)+".";auto xn=norm(x,p+"input.norm");auto&s=state[l].s;auto&m=state[l].m;
auto temp=[&](const std::string&branch){Vec z=linear(p+branch+".x",xn,c.s);add(z,linear(p+branch+".s",s,c.s));add(z,w.at(p+branch+".bias"));return z;};auto u=temp("s.candidate"),a=temp("s.gate");for(size_t j=0;j<s.size();++j)s[j]+=sigmoid(a[j])*(std::tanh(u[j])-s[j]);
#ifdef TAO_DELTA_MEM
// H2R 增量规则矩阵记忆：M 为 dv x dk，行主序。写=m - beta(m k)k^T + beta v k^T，读=o=M q（先写后读）
const uint32_t dk=c.dk,dv=c.m;
auto k=linear(p+"mem.key",xn,dk);auto q=linear(p+"mem.query",xn,dk);auto v=linear(p+"mem.value",xn,dv);
float kn=0;for(float z:k)kn+=z*z;kn=1.0f/(std::sqrt(kn)+1e-6f);for(float&z:k)z*=kn;
float beta=sigmoid(linear(p+"mem.beta",xn,1)[0]+w.at(p+"mem.beta.bias")[0]);
Vec mk(dv,0);for(uint32_t i=0;i<dv;++i){float acc=0;const float*row=m.data()+size_t(i)*dk;for(uint32_t j=0;j<dk;++j)acc+=row[j]*k[j];mk[i]=acc;}
for(uint32_t i=0;i<dv;++i){float g=beta*(v[i]-mk[i]);float*row=m.data()+size_t(i)*dk;for(uint32_t j=0;j<dk;++j)row[j]+=g*k[j];}
Vec o(dv,0);for(uint32_t i=0;i<dv;++i){float acc=0;const float*row=m.data()+size_t(i)*dk;for(uint32_t j=0;j<dk;++j)acc+=row[j]*q[j];o[i]=acc;}
auto r=linear(p+"read.s",s,c.d);add(r,o);add(x,norm(r,p+"read.norm"));
#else
auto memory=[&](const std::string&branch){Vec z=linear(p+branch+".x",xn,c.m);add(z,linear(p+branch+".s",s,c.m));add(z,linear(p+branch+".m",m,c.m));add(z,w.at(p+branch+".bias"));return z;};auto v=memory("m.candidate"),g=memory("m.gate");for(size_t j=0;j<m.size();++j)m[j]+=sigmoid(g[j])*(std::tanh(v[j])-m[j]);
auto r=linear(p+"read.s",s,c.d);add(r,linear(p+"read.m",m,c.d));add(x,norm(r,p+"read.norm"));
#endif
#ifndef TAO_NO_FFN
auto f=linear(p+"ff.up",norm(x,p+"ff.norm"),c.e);for(float&z:f)z*=sigmoid(z);add(x,linear(p+"ff.down",f,c.d));
#endif
}
auto logits=linear("embedding",norm(x,"final.norm"),c.vocab);add(logits,w.at("vocab.bias"));return logits;
}
};
}
