#pragma once
#include "dual_state_cuda_probe.cuh"
#include <memory>
namespace tao::dual {
__device__ float ds_sigmoid(float z){if(z>=0)return 1/(1+expf(-z));float e=expf(z);return e/(1+e);}
__global__ void ds_add(float*a,const float*b,int n){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<n)a[j]+=b[j];}
__global__ void ds_update(float*s,const float*u,const float*g,int n){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<n)s[j]+=ds_sigmoid(g[j])*(tanhf(u[j])-s[j]);}
__global__ void ds_silu(float*x,int n){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<n)x[j]*=ds_sigmoid(x[j]);}
struct CudaResident {
using B=std::shared_ptr<Device>;Config c;std::map<std::string,B>w;std::vector<B>s,m;
explicit CudaResident(const CpuModel&cpu):c(cpu.c){for(const auto&kv:cpu.w)w[kv.first]=std::make_shared<Device>(kv.second);for(unsigned l=0;l<c.layers;++l){s.push_back(std::make_shared<Device>(Vec(c.s)));m.push_back(std::make_shared<Device>(Vec(c.m)));}}
void reset(){for(unsigned l=0;l<c.layers;++l){check(cudaMemset(s[l]->p,0,c.s*sizeof(float)));check(cudaMemset(m[l]->p,0,c.m*sizeof(float)));}}
B linear(const std::string&name,B x,unsigned rows){auto a=w.at(name);if(a->n!=size_t(rows)*x->n)throw std::runtime_error("shape");auto y=std::make_shared<Device>(rows);matvec<<<(rows+127)/128,128>>>(a->p,x->p,y->p,rows,int(x->n));check(cudaGetLastError());return y;}
B norm(B x,const std::string&name){auto y=std::make_shared<Device>(x->n);rms<<<1,1>>>(x->p,w.at(name)->p,y->p,int(x->n));check(cudaGetLastError());return y;}
void add(B a,B b){if(a->n!=b->n)throw std::runtime_error("add shape");ds_add<<<(a->n+127)/128,128>>>(a->p,b->p,int(a->n));check(cudaGetLastError());}
Vec step(unsigned token){if(token>=c.vocab)throw std::invalid_argument("token");auto x=std::make_shared<Device>(c.d);check(cudaMemcpy(x->p,w.at("embedding")->p+size_t(token)*c.d,c.d*sizeof(float),cudaMemcpyDeviceToDevice));
for(unsigned l=0;l<c.layers;++l){auto p="layer."+std::to_string(l)+".";auto xn=norm(x,p+"input.norm");auto branch=[&](std::string name,bool memory){unsigned n=memory?c.m:c.s;auto z=linear(p+name+".x",xn,n);add(z,linear(p+name+".s",s[l],n));if(memory)add(z,linear(p+name+".m",m[l],n));add(z,w.at(p+name+".bias"));return z;};auto u=branch("s.candidate",false),a=branch("s.gate",false);ds_update<<<(c.s+127)/128,128>>>(s[l]->p,u->p,a->p,c.s);check(cudaGetLastError());auto v=branch("m.candidate",true),g=branch("m.gate",true);ds_update<<<(c.m+127)/128,128>>>(m[l]->p,v->p,g->p,c.m);check(cudaGetLastError());auto r=linear(p+"read.s",s[l],c.d);add(r,linear(p+"read.m",m[l],c.d));add(x,norm(r,p+"read.norm"));auto f=linear(p+"ff.up",norm(x,p+"ff.norm"),c.e);ds_silu<<<(c.e+127)/128,128>>>(f->p,c.e);check(cudaGetLastError());add(x,linear(p+"ff.down",f,c.d));}
auto logits=linear("embedding",norm(x,"final.norm"),c.vocab);add(logits,w.at("vocab.bias"));return logits->host();}
};
}
