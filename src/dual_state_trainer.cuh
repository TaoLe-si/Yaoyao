#pragma once
#include "dual_state_cuda_optimizer.cuh"
namespace tao::dual {
struct GpuTrainer {
TrainGraph graph;std::map<std::string,std::shared_ptr<Device>>master,moment,variance,scales;std::vector<TensorSpec>spec;unsigned steps=0;
explicit GpuTrainer(const CpuModel&cpu):graph(cpu),spec(schema(cpu.c)){for(auto&t:spec){master[t.name]=std::make_shared<Device>(cpu.w.at(t.name));moment[t.name]=std::make_shared<Device>(Vec(t.elements()));variance[t.name]=std::make_shared<Device>(Vec(t.elements()));if(t.ternary)scales[t.name]=std::make_shared<Device>(t.rows);}project();}
void project(){for(auto&t:spec){auto a=master.at(t.name);for(float v:a->host())if(!std::isfinite(v))throw std::runtime_error("nonfinite master");auto q=graph.w.at(t.name);if(t.ternary){ds_project<<<(t.rows+31)/32,32>>>(a->p,q->value.p,scales.at(t.name)->p,t.rows,t.cols);check(cudaGetLastError());for(float v:scales.at(t.name)->host())if(!std::isfinite(v)||v<=0)throw std::runtime_error("scale");}else check(cudaMemcpy(q->value.p,a->p,a->n*4,cudaMemcpyDeviceToDevice));}}
// Discard temporal graph while preserving state values and accumulated parameter gradients.
void detach(){auto copy=[](Node old){auto n=std::make_shared<GradNode>(old->value.n);check(cudaMemcpy(n->value.p,old->value.p,old->value.n*4,cudaMemcpyDeviceToDevice));return n;};for(auto&n:graph.s)n=copy(n);for(auto&n:graph.m)n=copy(n);graph.tape.reverse.clear();}
void zero_grad(){for(auto&kv:graph.w)check(cudaMemset(kv.second->grad.p,0,kv.second->grad.n*4));}
float update(size_t supervised,float lr=.0003f){if(!supervised)return 0;double norm=0;for(auto&t:spec)for(float g:graph.w.at(t.name)->grad.host()){if(!std::isfinite(g))throw std::runtime_error("gradient");double z=double(g)/supervised;norm+=z*z;}norm=std::sqrt(norm);float factor=float(1./supervised/std::max(1.,norm));++steps;for(auto&t:spec){auto a=master.at(t.name);ds_adamw<<<(a->n+127)/128,128>>>(a->p,graph.w.at(t.name)->grad.p,moment.at(t.name)->p,variance.at(t.name)->p,a->n,factor,lr,t.ternary?.01f:0.f,1-std::pow(.9f,float(steps)),1-std::pow(.999f,float(steps)));check(cudaGetLastError());}check(cudaDeviceSynchronize());project();zero_grad();return float(norm);}
};
}
