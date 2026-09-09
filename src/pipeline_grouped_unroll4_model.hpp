#pragma once
#include "cpu_row_parallel_executor.hpp"
#include "dual_state_cpu.hpp"
#include "cpu_pipeline_rows_unroll4.hpp"
#include "cpu_compact_bundle.hpp"
#include <map>
#include <array>
#include <utility>
#include <cmath>
#ifdef TAO_CPU_AVX2
#include "cpu_dot_avx2.hpp"
#endif
namespace tao::dual {


// Standalone owning CPU candidate; no GPU/training or baseline changes.
// Exactly one caller plus one persistent sleeping worker for all projections.
struct PipelineGroupedUnroll4Model{
Config c;std::map<std::string,Vec>w;std::map<std::string,PipelineRowsUnroll4>packed;
explicit PipelineGroupedUnroll4Model(CompactBundleData&&src):c(src.c),w(std::move(src.vectors)){for(auto&t:schema(c))if(t.ternary){auto&m=src.matrices.at(t.name);PipelineRowsUnroll4 p(Vec(t.cols,0),1,t.cols);p.rows=t.rows;p.q=std::move(m.q);p.scale=std::move(m.scale);packed.emplace(t.name,std::move(p));}}
explicit PipelineGroupedUnroll4Model(const CpuModel&src):c(src.c){for(auto&t:schema(c)){if(t.ternary)packed.emplace(t.name,PipelineRowsUnroll4(src.w.at(t.name),t.rows,t.cols));else w.emplace(t.name,src.w.at(t.name));}}
#ifdef TAO_CPU_AVX2
bool fast=true;
// Opt-in experimental backend; existing callers retain the single-row kernel.
// Only the original single-row dot arithmetic is used.

#endif
size_t weight_bytes()const{size_t n=0;for(auto&kv:w)n+=kv.second.size()*sizeof(float);for(auto&kv:packed)n+=kv.second.q.size()+kv.second.scale.size()*sizeof(float);return n;}
std::vector<LayerState> initial()const{return std::vector<LayerState>(c.layers,LayerState{Vec(c.s),Vec(c.m)});}
Vec linear(const std::string&name,const Vec&x,uint32_t rows)const{const auto&p=packed.at(name);if(p.rows!=rows||p.cols!=x.size())throw std::runtime_error("matrix shape");Vec y(rows);
#ifdef TAO_CPU_AVX2
if(fast){executor.run(rows,p.cols,[&](size_t begin,size_t end){for(size_t r=begin;r<end;++r)y[r]=p.dot(r,x.data());});return y;}
#endif
for(uint32_t r=0;r<rows;++r)for(size_t j=0;j<x.size();++j)y[r]+=(float(p.q[r*x.size()+j])*p.scale[r])*x[j];return y;}
static void add(Vec&a,const Vec&b){if(a.size()!=b.size())throw std::runtime_error("vector shape");for(size_t i=0;i<a.size();++i)a[i]+=b[i];}
Vec norm(const Vec&x,const std::string&name)const{const auto&g=w.at(name);if(g.size()!=x.size())throw std::runtime_error("norm shape");float sum=0;for(float z:x)sum+=z*z;float inv=1/std::sqrt(sum/x.size()+1e-5f);Vec y(x.size());for(size_t j=0;j<x.size();++j)y[j]=x[j]*inv*g[j];return y;}
static float sigmoid(float x){if(x>=0)return 1/(1+std::exp(-x));float e=std::exp(x);return e/(1+e);}
private:

// A group is one synchronous dispatch, not one dispatch per projection.
// Cost coordinates concatenate rows*cols, then snap the half-cost boundary
// upward to a whole row. No dot is split: each side differs from half-cost
// by less than one row cost (candidate/gate groups split exactly in half).
// Inputs/outputs are borrowed until run returns, including exception paths.
template<size_t N> std::array<Vec,N> group(const std::array<std::string,N>& names,
                                         const std::array<const Vec*,N>& inputs,
                                         uint32_t rows)const {
    std::array<Vec,N> out;
    std::array<const PipelineRowsUnroll4*,N> matrices{};
    std::array<size_t,N+1> offsets{};
    for(size_t i=0;i<N;++i){
        const auto& p=packed.at(names[i]);
        if(p.rows!=rows||p.cols!=inputs[i]->size()||p.cols==0)
            throw std::runtime_error("matrix shape");
        if(p.rows>(std::numeric_limits<size_t>::max()-offsets[i])/p.cols)
            throw std::overflow_error("group cost overflow");
        matrices[i]=&p;out[i].resize(rows);
        offsets[i+1]=offsets[i]+p.rows*p.cols;
    }
#ifdef TAO_CPU_AVX2
    if(fast){
        executor.run(offsets[N],1,[&](size_t begin,size_t end){
            for(size_t i=0;i<N;++i){
                if(end<=offsets[i]||begin>=offsets[i+1])continue;
                const auto& p=*matrices[i];
                const size_t lo=begin>offsets[i]?begin-offsets[i]:0;
                const size_t hi=end<offsets[i+1]?end-offsets[i]:offsets[i+1]-offsets[i];
                // Adjacent cost intervals share the same ceil(boundary/cols):
                // caller [0,k), worker [k,rows). Thus no missing/duplicate rows.
                const size_t first=lo/p.cols+(lo%p.cols!=0);
                const size_t last=hi/p.cols+(hi%p.cols!=0);
                for(size_t r=first;r<last;++r)out[i][r]=p.dot(r,inputs[i]->data());
            }
        });
        return out;
    }
#endif
    for(size_t i=0;i<N;++i){
        const auto& p=*matrices[i];const auto& x=*inputs[i];
        for(uint32_t r=0;r<rows;++r)for(size_t j=0;j<x.size();++j)
            out[i][r]+=(float(p.q[r*x.size()+j])*p.scale[r])*x[j];
    }
    return out;
}

Vec recurrent(uint32_t token,std::vector<LayerState>&state)const{if(token>=c.vocab||state.size()!=c.layers)throw std::invalid_argument("token/state");for(const auto&z:state)if(z.s.size()!=c.s||z.m.size()!=c.m)throw std::invalid_argument("state shape");const auto&emb=packed.at("embedding");Vec x(c.d);for(size_t j=0;j<c.d;++j)x[j]=float(emb.q[size_t(token)*c.d+j])*emb.scale[token];
#ifdef TAO_INPUT_SCALE
for(float&v:x)v*=std::sqrt(float(c.d));
#endif
for(uint32_t l=0;l<c.layers;++l){auto p="layer."+std::to_string(l)+".";auto xn=norm(x,p+"input.norm");auto&s=state[l].s;auto&m=state[l].m;
auto st=group<4>({p+"s.candidate.x",p+"s.candidate.s",p+"s.gate.x",p+"s.gate.s"},{&xn,&s,&xn,&s},c.s);
auto u=std::move(st[0]),a=std::move(st[2]);add(u,st[1]);add(u,w.at(p+"s.candidate.bias"));add(a,st[3]);add(a,w.at(p+"s.gate.bias"));for(size_t j=0;j<s.size();++j)s[j]+=sigmoid(a[j])*(std::tanh(u[j])-s[j]);
auto mt=group<6>({p+"m.candidate.x",p+"m.candidate.s",p+"m.candidate.m",p+"m.gate.x",p+"m.gate.s",p+"m.gate.m"},{&xn,&s,&m,&xn,&s,&m},c.m);
auto v=std::move(mt[0]),g=std::move(mt[3]);add(v,mt[1]);add(v,mt[2]);add(v,w.at(p+"m.candidate.bias"));add(g,mt[4]);add(g,mt[5]);add(g,w.at(p+"m.gate.bias"));for(size_t j=0;j<m.size();++j)m[j]+=sigmoid(g[j])*(std::tanh(v[j])-m[j]);
auto rt=group<2>({p+"read.s",p+"read.m"},{&s,&m},c.d);auto r=std::move(rt[0]);add(r,rt[1]);add(x,norm(r,p+"read.norm"));auto f=linear(p+"ff.up",norm(x,p+"ff.norm"),c.e);for(float&z:f)z*=sigmoid(z);add(x,linear(p+"ff.down",f,c.d));}
return x;
}

public:
    void advance(uint32_t token,std::vector<LayerState>& state)const {
        (void)recurrent(token,state);
    }
    Vec step(uint32_t token,std::vector<LayerState>& state)const {
        auto x=recurrent(token,state);
        auto logits=linear("embedding",norm(x,"final.norm"),c.vocab);
        add(logits,w.at("vocab.bias"));return logits;
    }
private:
// Last member: worker shuts down before owned weights are destroyed.
mutable CpuRowParallelExecutor executor;
};
}
