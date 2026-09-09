#pragma once
#include "byte_cpu_model.hpp"

namespace tao::dual {
// Non-owning immutable-weight view. Source must outlive this object and must not
// be mutated/moved. Single-thread, non-reentrant scratch; step result is borrowed
// until next step. Uses the original single-row AVX2 dot, never four-row kernels.
class PreboundByteCpuModel {
    struct Branch { const CpuTernaryRows *x,*s,*m; const Vec* bias; };
    struct Layer {
        Branch sc,sg,mc,mg;
        const CpuTernaryRows *rs,*rm,*up,*down;
        const Vec *input_norm,*read_norm,*ff_norm;
    };
    const ByteCpuModel& source;
    const Config c;
    std::vector<Layer> layers;
    const CpuTernaryRows* embedding;
    const Vec *final_norm,*vocab_bias;
    Vec x,xn,u,a,v,g,st,mt,r,dt,dn,f,logits;
    const CpuTernaryRows* matrix(const std::string& name,size_t rows,size_t cols) {
        const auto& p=source.packed.at(name);
        if(p.rows!=rows||p.cols!=cols||p.q.size()!=rows*cols||p.scale.size()!=rows)
            throw std::runtime_error("prebound matrix shape");
        return &p;
    }
    const Vec* vector(const std::string& name,size_t n) {
        const auto& z=source.w.at(name);
        if(z.size()!=n) throw std::runtime_error("prebound vector shape");
        return &z;
    }
    Branch branch(const std::string& p,size_t n,bool memory) {
        return {matrix(p+".x",n,c.d),matrix(p+".s",n,c.s),
                memory?matrix(p+".m",n,c.m):nullptr,vector(p+".bias",n)};
    }
    static void linear(const CpuTernaryRows* p,const Vec& in,Vec& out) {
        for(size_t row=0;row<p->rows;++row) out[row]=p->dot(row,in.data());
    }
    static void add(Vec& out,const Vec& in) {
        for(size_t j=0;j<out.size();++j) out[j]+=in[j];
    }
    static void norm(const Vec& in,const Vec* gain,Vec& out) {
        float sum=0;for(float z:in)sum+=z*z;
        float inv=1/std::sqrt(sum/in.size()+1e-5f);
        for(size_t j=0;j<in.size();++j)out[j]=in[j]*inv*(*gain)[j];
    }
    void temporal(const Branch& b,const Vec& s,Vec& out) {
        linear(b.x,xn,out);linear(b.s,s,st);add(out,st);add(out,*b.bias);
    }
    void memory(const Branch& b,const Vec& s,const Vec& m,Vec& out) {
        linear(b.x,xn,out);linear(b.s,s,mt);add(out,mt);
        linear(b.m,m,mt);add(out,mt);add(out,*b.bias);
    }
public:
    explicit PreboundByteCpuModel(const ByteCpuModel& src)
      :source(src),c(src.c),x(c.d),xn(c.d),u(c.s),a(c.s),v(c.m),g(c.m),
       st(c.s),mt(c.m),r(c.d),dt(c.d),dn(c.d),f(c.e),logits(c.vocab) {
        embedding=matrix("embedding",c.vocab,c.d);
        final_norm=vector("final.norm",c.d);vocab_bias=vector("vocab.bias",c.vocab);
        layers.reserve(c.layers);
        for(uint32_t l=0;l<c.layers;++l) {
            auto p="layer."+std::to_string(l)+".";
            layers.push_back({branch(p+"s.candidate",c.s,false),branch(p+"s.gate",c.s,false),
                branch(p+"m.candidate",c.m,true),branch(p+"m.gate",c.m,true),
                matrix(p+"read.s",c.d,c.s),matrix(p+"read.m",c.d,c.m),
                matrix(p+"ff.up",c.e,c.d),matrix(p+"ff.down",c.d,c.e),
                vector(p+"input.norm",c.d),vector(p+"read.norm",c.d),vector(p+"ff.norm",c.d)});
        }
    }
    PreboundByteCpuModel(ByteCpuModel&&)=delete;
    PreboundByteCpuModel(const PreboundByteCpuModel&)=delete;
    PreboundByteCpuModel& operator=(const PreboundByteCpuModel&)=delete;
    std::vector<LayerState> initial()const{return source.initial();}
    size_t weight_bytes()const{return source.weight_bytes();}
    size_t scratch_bytes()const{return (x.size()+xn.size()+u.size()+a.size()+v.size()+g.size()+st.size()+mt.size()+r.size()+dt.size()+dn.size()+f.size()+logits.size())*sizeof(float);}
    const Vec& step(uint32_t token,std::vector<LayerState>& state) {
        if(token>=c.vocab||state.size()!=c.layers)throw std::invalid_argument("token/state");
        for(const auto& z:state)if(z.s.size()!=c.s||z.m.size()!=c.m)throw std::invalid_argument("state shape");
        for(size_t j=0;j<c.d;++j)x[j]=float(embedding->q[size_t(token)*c.d+j])*embedding->scale[token];
#ifdef TAO_INPUT_SCALE
        for(float& z:x)z*=std::sqrt(float(c.d));
#endif
        for(size_t l=0;l<layers.size();++l) {
            const auto& b=layers[l];auto& s=state[l].s;auto& m=state[l].m;
            norm(x,b.input_norm,xn);
            temporal(b.sc,s,u);temporal(b.sg,s,a);
            for(size_t j=0;j<s.size();++j)s[j]+=ByteCpuModel::sigmoid(a[j])*(std::tanh(u[j])-s[j]);
            memory(b.mc,s,m,v);memory(b.mg,s,m,g);
            for(size_t j=0;j<m.size();++j)m[j]+=ByteCpuModel::sigmoid(g[j])*(std::tanh(v[j])-m[j]);
            linear(b.rs,s,r);linear(b.rm,m,dt);add(r,dt);
            norm(r,b.read_norm,dn);add(x,dn);
            norm(x,b.ff_norm,dn);linear(b.up,dn,f);
            for(float& z:f)z*=ByteCpuModel::sigmoid(z);
            linear(b.down,f,dt);add(x,dt);
        }
        norm(x,final_norm,dn);linear(embedding,dn,logits);add(logits,*vocab_bias);
        return logits;
    }
};
}
