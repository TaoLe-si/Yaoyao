#pragma once
#include "dual_state_cpu.hpp"
#include "cpu_prefetch_candidate.hpp"
#include "cpu_compact_bundle.hpp"
#include <map>
#include <cmath>
#ifdef TAO_CPU_AVX2
#include "cpu_dot_avx2.hpp"
#endif
namespace tao::dual {


struct PrefetchCpuModel{
Config c;std::map<std::string,Vec>w;std::map<std::string,PrefetchRows>packed;
explicit PrefetchCpuModel(CompactBundleData&&src):c(src.c),w(std::move(src.vectors)){for(auto&t:schema(c))if(t.ternary){auto&m=src.matrices.at(t.name);PrefetchRows p(Vec(t.cols,0),1,t.cols);p.rows=t.rows;p.q=std::move(m.q);p.scale=std::move(m.scale);packed.emplace(t.name,std::move(p));}}
explicit PrefetchCpuModel(const CpuModel&src):c(src.c){for(auto&t:schema(c)){if(t.ternary)packed.emplace(t.name,PrefetchRows(src.w.at(t.name),t.rows,t.cols));else w.emplace(t.name,src.w.at(t.name));}}
#ifdef TAO_CPU_AVX2
bool fast=true;
// Opt-in experimental backend; existing callers retain the single-row kernel.
bool four_rows=false;
#endif
size_t weight_bytes()const{size_t n=0;for(auto&kv:w)n+=kv.second.size()*sizeof(float);for(auto&kv:packed)n+=kv.second.q.size()+kv.second.scale.size()*sizeof(float);return n;}
std::vector<LayerState> initial()const{return std::vector<LayerState>(c.layers,LayerState{Vec(c.s),Vec(c.m)});}
Vec linear(const std::string&name,const Vec&x,uint32_t rows)const{const auto&p=packed.at(name);if(p.rows!=rows||p.cols!=x.size())throw std::runtime_error("matrix shape");Vec y(rows);
#ifdef TAO_CPU_AVX2
if(fast){if(four_rows)p.matvec_four_rows(x.data(),y.data());else for(uint32_t r=0;r<rows;++r)y[r]=p.dot(r,x.data());return y;}
#endif
for(uint32_t r=0;r<rows;++r)for(size_t j=0;j<x.size();++j)y[r]+=(float(p.q[r*x.size()+j])*p.scale[r])*x[j];return y;}
static void add(Vec&a,const Vec&b){if(a.size()!=b.size())throw std::runtime_error("vector shape");for(size_t i=0;i<a.size();++i)a[i]+=b[i];}
Vec norm(const Vec&x,const std::string&name)const{const auto&g=w.at(name);if(g.size()!=x.size())throw std::runtime_error("norm shape");float sum=0;for(float z:x)sum+=z*z;float inv=1/std::sqrt(sum/x.size()+1e-5f);Vec y(x.size());for(size_t j=0;j<x.size();++j)y[j]=x[j]*inv*g[j];return y;}
static float sigmoid(float x){if(x>=0)return 1/(1+std::exp(-x));float e=std::exp(x);return e/(1+e);}
Vec step(uint32_t token,std::vector<LayerState>&state)const{if(token>=c.vocab||state.size()!=c.layers)throw std::invalid_argument("token/state");for(const auto&z:state)if(z.s.size()!=c.s||z.m.size()!=c.m)throw std::invalid_argument("state shape");const auto&emb=packed.at("embedding");Vec x(c.d);for(size_t j=0;j<c.d;++j)x[j]=float(emb.q[size_t(token)*c.d+j])*emb.scale[token];
#ifdef TAO_INPUT_SCALE
for(float&v:x)v*=std::sqrt(float(c.d));
#endif
for(uint32_t l=0;l<c.layers;++l){auto p="layer."+std::to_string(l)+".";auto xn=norm(x,p+"input.norm");auto&s=state[l].s;auto&m=state[l].m;
auto temp=[&](const std::string&branch){Vec z=linear(p+branch+".x",xn,c.s);add(z,linear(p+branch+".s",s,c.s));add(z,w.at(p+branch+".bias"));return z;};auto u=temp("s.candidate"),a=temp("s.gate");for(size_t j=0;j<s.size();++j)s[j]+=sigmoid(a[j])*(std::tanh(u[j])-s[j]);
auto memory=[&](const std::string&branch){Vec z=linear(p+branch+".x",xn,c.m);add(z,linear(p+branch+".s",s,c.m));add(z,linear(p+branch+".m",m,c.m));add(z,w.at(p+branch+".bias"));return z;};auto v=memory("m.candidate"),g=memory("m.gate");for(size_t j=0;j<m.size();++j)m[j]+=sigmoid(g[j])*(std::tanh(v[j])-m[j]);
auto r=linear(p+"read.s",s,c.d);add(r,linear(p+"read.m",m,c.d));add(x,norm(r,p+"read.norm"));auto f=linear(p+"ff.up",norm(x,p+"ff.norm"),c.e);for(float&z:f)z*=sigmoid(z);add(x,linear(p+"ff.down",f,c.d));}
auto logits=linear("embedding",norm(x,"final.norm"),c.vocab);add(logits,w.at("vocab.bias"));return logits;
}
};
}
