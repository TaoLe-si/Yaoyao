#pragma once
#include "dual_state_cpu.hpp"
#include <cuda_runtime.h>
#include <map>
#include <cmath>
#include <chrono>
namespace tao::dual {


inline void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
#ifdef TAO_ALLOC_PROFILE
inline double allocation_seconds=0,free_seconds=0;inline size_t allocation_calls=0,free_calls=0;
#endif
#ifdef TAO_DEVICE_POOL
struct DevicePool {std::map<size_t,std::vector<float*>> bins;size_t cached=0,hits=0,misses=0;static constexpr size_t cap=size_t(512)*1024*1024;
float* acquire(size_t n){auto&b=bins[n];if(!b.empty()){float*p=b.back();b.pop_back();cached-=n*4;++hits;return p;}float*p=nullptr;check(cudaMalloc(&p,n*4));++misses;return p;}
void release(float*p,size_t n){if(n*4<=cap-cached){bins[n].push_back(p);cached+=n*4;}else cudaFree(p);}
~DevicePool(){for(auto&kv:bins)for(float*p:kv.second)cudaFree(p);}};
inline DevicePool& device_pool(){static DevicePool pool;return pool;}
#endif
struct Device{float*p=nullptr;size_t n;explicit Device(size_t count):n(count){
#ifdef TAO_ALLOC_PROFILE
auto started=std::chrono::steady_clock::now();
#endif
#ifdef TAO_DEVICE_POOL
p=device_pool().acquire(n);
#elif defined(TAO_ASYNC_ALLOC)
check(cudaMallocAsync(&p,n*sizeof(float),0));
#else
check(cudaMalloc(&p,n*sizeof(float)));
#endif
#ifdef TAO_ALLOC_PROFILE
allocation_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();++allocation_calls;
#endif
}explicit Device(const Vec&v):Device(v.size()){check(cudaMemcpy(p,v.data(),n*sizeof(float),cudaMemcpyHostToDevice));}~Device(){
#ifdef TAO_ALLOC_PROFILE
auto started=std::chrono::steady_clock::now();
#endif
#ifdef TAO_DEVICE_POOL
device_pool().release(p,n);
#elif defined(TAO_ASYNC_ALLOC)
cudaFreeAsync(p,0);
#else
cudaFree(p);
#endif
#ifdef TAO_ALLOC_PROFILE
free_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();++free_calls;
#endif
}Device(const Device&)=delete;Vec host(){Vec v(n);check(cudaMemcpy(v.data(),p,n*sizeof(float),cudaMemcpyDeviceToHost));return v;}};
__global__ void matvec(const float*a,const float*x,float*y,int rows,int cols){int r=blockIdx.x*blockDim.x+threadIdx.x;if(r<rows){float z=0;for(int j=0;j<cols;++j)z+=a[r*cols+j]*x[j];y[r]=z;}}
__global__ void rms(const float*x,const float*g,float*y,int n){float sum=0;for(int j=0;j<n;++j)sum+=x[j]*x[j];float inv=1/sqrtf(sum/n+1e-5f);for(int j=0;j<n;++j)y[j]=x[j]*inv*g[j];}
struct CudaProbeModel{
Config c;std::map<std::string,Vec>w;
explicit CudaProbeModel(Config cfg):c(cfg){for(auto&t:schema(c))w.emplace(t.name,Vec(t.elements(),0));}
std::vector<LayerState> initial()const{
    std::vector<LayerState> st;st.reserve(c.layers);
    for(uint32_t i=0;i<c.layers;++i)st.emplace_back(c.s,c.m);
    return st;}
Vec linear(const std::string&name,const Vec&x,uint32_t rows)const{const auto&a=w.at(name);if(a.size()!=uint64_t(rows)*x.size())throw std::runtime_error("shape");Device da(a),dx(x);Device dy(rows);matvec<<<(rows+127)/128,128>>>(da.p,dx.p,dy.p,rows,x.size());check(cudaGetLastError());return dy.host();}
Vec linear(const std::string&name,const FSpan&x,uint32_t rows)const{return linear(name,Vec(x.data(),x.data()+x.size()),rows);}

static void add(Vec&a,const Vec&b){if(a.size()!=b.size())throw std::runtime_error("vector shape");for(size_t i=0;i<a.size();++i)a[i]+=b[i];}
Vec norm(const Vec&x,const std::string&name)const{Device dx(x),dg(w.at(name)),dy(x.size());rms<<<1,1>>>(dx.p,dg.p,dy.p,x.size());check(cudaGetLastError());return dy.host();}

static float sigmoid(float x){if(x>=0)return 1/(1+std::exp(-x));float e=std::exp(x);return e/(1+e);}
Vec step(uint32_t token,std::vector<LayerState>&state)const{if(token>=c.vocab||state.size()!=c.layers)throw std::invalid_argument("token/state");for(const auto&z:state)if(z.s.size()!=c.s||z.m.size()!=c.m)throw std::invalid_argument("state shape");const auto&emb=w.at("embedding");Vec x(emb.begin()+uint64_t(token)*c.d,emb.begin()+uint64_t(token+1)*c.d);
for(uint32_t l=0;l<c.layers;++l){auto p="layer."+std::to_string(l)+".";auto xn=norm(x,p+"input.norm");auto&s=state[l].s;auto&m=state[l].m;
auto temp=[&](const std::string&branch){Vec z=linear(p+branch+".x",xn,c.s);add(z,linear(p+branch+".s",s,c.s));add(z,w.at(p+branch+".bias"));return z;};auto u=temp("s.candidate"),a=temp("s.gate");for(size_t j=0;j<s.size();++j)s[j]+=sigmoid(a[j])*(std::tanh(u[j])-s[j]);
auto memory=[&](const std::string&branch){Vec z=linear(p+branch+".x",xn,c.m);add(z,linear(p+branch+".s",s,c.m));add(z,linear(p+branch+".m",m,c.m));add(z,w.at(p+branch+".bias"));return z;};auto v=memory("m.candidate"),g=memory("m.gate");for(size_t j=0;j<m.size();++j)m[j]+=sigmoid(g[j])*(std::tanh(v[j])-m[j]);
auto r=linear(p+"read.s",s,c.d);add(r,linear(p+"read.m",m,c.d));add(x,norm(r,p+"read.norm"));auto f=linear(p+"ff.up",norm(x,p+"ff.norm"),c.e);for(float&z:f)z*=sigmoid(z);add(x,linear(p+"ff.down",f,c.d));}
auto logits=linear("embedding",norm(x,"final.norm"),c.vocab);add(logits,w.at("vocab.bias"));return logits;
}
};
}
