#pragma once
#include "dual_state_cuda_probe.cuh"
#include <memory>
namespace tao::dual {
__device__ float ds_sigmoid(float z){if(z>=0)return 1/(1+expf(-z));float e=expf(z);return e/(1+e);}

// ============================================================================
// 【R4 统一 · doc 24】Padé 快速激活 —— 与 cpu_fast_activation.hpp **逐位同式**
//
//     tanh(x)    ≈ x(27 + x²) / (27 + 9x²)，|x| > 3 时夹紧到 ±1
//     sigmoid(x) = 0.5 * (1 + tanh(x))
//
// 动机：解码器（greedy_pipeline_grouped_model.hpp）默认 fast_act_=true，用上面的近似；
// 而训练器此前用精确 expf/tanhf。**训练/推理算子不一致**，违反 R4 铁律。
// 用户决策：统一到解码器一侧（该近似让 CPU 解码快 49.5×，是既定性能选择）。
// 因此训练器也使用同一近似，两侧共用同一式子。
//
// 回退开关：-DTAO_TRAIN_EXACT_ACT 恢复精确激活（仅用于对照实验，
// 此时解码器必须同时 TAO_FAST_ACT=0，否则重新引入不一致）。
//
// 导数（解析，非数值差分）：
//     f(x)  = (27x + x³) / (27 + 9x²)
//     f'(x) = (729 - 162x² + 9x⁴) / (27 + 9x²)²
//   夹紧区 f' = 0。校验：f'(0)=1（与 d/dx tanh 一致）；f'(3)=0（与夹紧平滑衔接）。
// ============================================================================
#ifndef TAO_TRAIN_EXACT_ACT
#define TAO_TRAIN_FAST_ACT 1
#endif
__device__ float ds_fast_tanh(float x){
    if(x> 3.0f)return  1.0f;
    if(x<-3.0f)return -1.0f;
    const float q=x*x;
    return x*(27.0f+q)/(27.0f+9.0f*q);
}
__device__ float ds_fast_tanh_grad(float x){
    if(x>3.0f||x<-3.0f)return 0.0f;
    const float q=x*x,den=27.0f+9.0f*q;
    return (729.0f-162.0f*q+9.0f*q*q)/(den*den);
}
__device__ float ds_fast_sigmoid(float x){return 0.5f*(1.0f+ds_fast_tanh(x));}
__device__ float ds_fast_sigmoid_grad(float x){return 0.5f*ds_fast_tanh_grad(x);}

// 状态更新专用激活（**不**改动 ds_sigmoid 本身：mem.beta 与 silu 必须保持精确，
// 因为 CPU 解码器的 beta 用精确 sigmoid，改了就再次引入不一致）。
__device__ float ds_act_tanh(float x){
#ifdef TAO_TRAIN_FAST_ACT
    return ds_fast_tanh(x);
#else
    return tanhf(x);
#endif
}
__device__ float ds_act_sigmoid(float x){
#ifdef TAO_TRAIN_FAST_ACT
    return ds_fast_sigmoid(x);
#else
    return ds_sigmoid(x);
#endif
}
__device__ float ds_act_tanh_grad(float x){
#ifdef TAO_TRAIN_FAST_ACT
    return ds_fast_tanh_grad(x);
#else
    const float t=tanhf(x);return 1.0f-t*t;
#endif
}
__device__ float ds_act_sigmoid_grad(float x){
#ifdef TAO_TRAIN_FAST_ACT
    return ds_fast_sigmoid_grad(x);
#else
    const float s=ds_sigmoid(x);return s*(1.0f-s);
#endif
}

__global__ void ds_add(float*a,const float*b,int n){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<n)a[j]+=b[j];}
__global__ void ds_update(float*s,const float*u,const float*g,int n){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<n)s[j]+=ds_act_sigmoid(g[j])*(ds_act_tanh(u[j])-s[j]);}
__global__ void ds_silu(float*x,int n){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<n)x[j]*=ds_sigmoid(x[j]);}
struct CudaResident {
using B=std::shared_ptr<Device>;Config c;std::map<std::string,B>w;std::vector<B>s,m;
explicit CudaResident(const CpuModel&cpu):c(cpu.c){for(const auto&kv:cpu.w)w[kv.first]=std::make_shared<Device>(kv.second);for(unsigned l=0;l<c.layers;++l){s.push_back(std::make_shared<Device>(Vec(c.s)));m.push_back(std::make_shared<Device>(Vec(size_t(c.memory_size()))));}}
void reset(){for(unsigned l=0;l<c.layers;++l){check(cudaMemset(s[l]->p,0,c.s*sizeof(float)));check(cudaMemset(m[l]->p,0,c.m*sizeof(float)));}}
B linear(const std::string&name,B x,unsigned rows){auto a=w.at(name);if(a->n!=size_t(rows)*x->n)throw std::runtime_error("shape");auto y=std::make_shared<Device>(rows);matvec<<<(rows+127)/128,128>>>(a->p,x->p,y->p,rows,int(x->n));check(cudaGetLastError());return y;}
B norm(B x,const std::string&name){auto y=std::make_shared<Device>(x->n);rms<<<1,1>>>(x->p,w.at(name)->p,y->p,int(x->n));check(cudaGetLastError());return y;}
void add(B a,B b){if(a->n!=b->n)throw std::runtime_error("add shape");ds_add<<<(a->n+127)/128,128>>>(a->p,b->p,int(a->n));check(cudaGetLastError());}
Vec step(unsigned token){if(token>=c.vocab)throw std::invalid_argument("token");auto x=std::make_shared<Device>(c.d);check(cudaMemcpy(x->p,w.at("embedding")->p+size_t(token)*c.d,c.d*sizeof(float),cudaMemcpyDeviceToDevice));
for(unsigned l=0;l<c.layers;++l){auto p="layer."+std::to_string(l)+".";auto xn=norm(x,p+"input.norm");auto branch=[&](std::string name,bool memory){unsigned n=memory?c.m:c.s;auto z=linear(p+name+".x",xn,n);add(z,linear(p+name+".s",s[l],n));if(memory)add(z,linear(p+name+".m",m[l],n));add(z,w.at(p+name+".bias"));return z;};auto u=branch("s.candidate",false),a=branch("s.gate",false);ds_update<<<(c.s+127)/128,128>>>(s[l]->p,u->p,a->p,c.s);check(cudaGetLastError());auto v=branch("m.candidate",true),g=branch("m.gate",true);ds_update<<<(c.m+127)/128,128>>>(m[l]->p,v->p,g->p,c.m);check(cudaGetLastError());auto r=linear(p+"read.s",s[l],c.d);add(r,linear(p+"read.m",m[l],c.d));add(x,norm(r,p+"read.norm"));auto f=linear(p+"ff.up",norm(x,p+"ff.norm"),c.e);ds_silu<<<(c.e+127)/128,128>>>(f->p,c.e);check(cudaGetLastError());add(x,linear(p+"ff.down",f,c.d));}
auto logits=linear("embedding",norm(x,"final.norm"),c.vocab);add(logits,w.at("vocab.bias"));return logits->host();}
};
}
