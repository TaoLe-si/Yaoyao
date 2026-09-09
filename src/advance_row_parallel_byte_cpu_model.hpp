#pragma once
#include "row_parallel_byte_cpu_model.hpp"
namespace tao::dual {
// Independent compact-byte candidate. Only unconsumed output-head work is skipped.
// All recurrent layers, including final read/feedforward operations, still run.
struct AdvanceRowParallelByteCpuModel : RowParallelByteCpuModel {
    using RowParallelByteCpuModel::RowParallelByteCpuModel;
private:
Vec recurrent(uint32_t token,std::vector<LayerState>&state)const{if(token>=c.vocab||state.size()!=c.layers)throw std::invalid_argument("token/state");for(const auto&z:state)if(z.s.size()!=c.s||z.m.size()!=c.m)throw std::invalid_argument("state shape");const auto&emb=packed.at("embedding");Vec x(c.d);for(size_t j=0;j<c.d;++j)x[j]=float(emb.q[size_t(token)*c.d+j])*emb.scale[token];
#ifdef TAO_INPUT_SCALE
for(float&v:x)v*=std::sqrt(float(c.d));
#endif
for(uint32_t l=0;l<c.layers;++l){auto p="layer."+std::to_string(l)+".";auto xn=norm(x,p+"input.norm");auto&s=state[l].s;auto&m=state[l].m;
auto temp=[&](const std::string&branch){Vec z=linear(p+branch+".x",xn,c.s);add(z,linear(p+branch+".s",s,c.s));add(z,w.at(p+branch+".bias"));return z;};auto u=temp("s.candidate"),a=temp("s.gate");for(size_t j=0;j<s.size();++j)s[j]+=sigmoid(a[j])*(std::tanh(u[j])-s[j]);
auto memory=[&](const std::string&branch){Vec z=linear(p+branch+".x",xn,c.m);add(z,linear(p+branch+".s",s,c.m));add(z,linear(p+branch+".m",m,c.m));add(z,w.at(p+branch+".bias"));return z;};auto v=memory("m.candidate"),g=memory("m.gate");for(size_t j=0;j<m.size();++j)m[j]+=sigmoid(g[j])*(std::tanh(v[j])-m[j]);
auto r=linear(p+"read.s",s,c.d);add(r,linear(p+"read.m",m,c.d));add(x,norm(r,p+"read.norm"));auto f=linear(p+"ff.up",norm(x,p+"ff.norm"),c.e);for(float&z:f)z*=sigmoid(z);add(x,linear(p+"ff.down",f,c.d));}
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
};
}
