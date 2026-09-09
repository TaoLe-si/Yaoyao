#pragma once
#include "dual_projection_sorted.cuh"
#ifdef TAO_GPU_HEALTH
#include "gpu_health_reduce.cuh"
#include "gpu_adam_device.cuh"
#endif
namespace tao::dual {
struct SortedGpuTrainer {
TrainGraph graph;std::map<std::string,std::shared_ptr<Device>>master,moment,variance,scales;std::vector<TensorSpec>spec;unsigned steps=0;
explicit SortedGpuTrainer(const CpuModel&cpu):graph(cpu),spec(schema(cpu.c)){for(auto&t:spec){master[t.name]=std::make_shared<Device>(cpu.w.at(t.name));moment[t.name]=std::make_shared<Device>(Vec(t.elements()));variance[t.name]=std::make_shared<Device>(Vec(t.elements()));if(t.ternary)scales[t.name]=std::make_shared<Device>(t.rows);}project();}
void project(){
#ifdef TAO_GPU_HEALTH
GpuHealth health;
#endif
for(auto&t:spec){auto a=master.at(t.name);
#ifdef TAO_GPU_HEALTH
health.add(*a);
#else
for(float v:a->host())if(!std::isfinite(v))throw std::runtime_error("nonfinite master");
#endif
auto q=graph.w.at(t.name);if(t.ternary){project_sorted(*a,q->value,*scales.at(t.name),t.rows,t.cols);check(cudaGetLastError());
#ifdef TAO_GPU_HEALTH
health.add(*scales.at(t.name),true);
#else
for(float v:scales.at(t.name)->host())if(!std::isfinite(v)||v<=0)throw std::runtime_error("scale");
#endif
}else check(cudaMemcpy(q->value.p,a->p,a->n*4,cudaMemcpyDeviceToDevice));}
#ifdef TAO_GPU_HEALTH
health.finish();
#endif
}
// Discard temporal graph while preserving state values and accumulated parameter gradients.
void detach(){auto copy=[](Node old){auto n=std::make_shared<GradNode>(old->value.n);check(cudaMemcpy(n->value.p,old->value.p,old->value.n*4,cudaMemcpyDeviceToDevice));return n;};for(auto&n:graph.s)n=copy(n);for(auto&n:graph.m)n=copy(n);graph.tape.reverse.clear();}
void zero_grad(){for(auto&kv:graph.w)check(cudaMemset(kv.second->grad.p,0,kv.second->grad.n*4));}
float update(size_t supervised,float lr=.0003f){if(!supervised)return 0;double norm=0;
#ifdef TAO_GPU_HEALTH
GpuHealth health;for(auto&t:spec)health.add(graph.w.at(t.name)->grad,false,double(supervised));
#else
for(auto&t:spec)for(float g:graph.w.at(t.name)->grad.host()){if(!std::isfinite(g))throw std::runtime_error("gradient");double z=double(g)/supervised;norm+=z*z;}
#endif
norm=std::sqrt(norm);float factor=float(1./supervised/std::max(1.,norm));++steps;for(auto&t:spec){auto a=master.at(t.name);
#ifdef TAO_GPU_HEALTH
ds_adamw_device<<<(a->n+127)/128,128>>>(a->p,graph.w.at(t.name)->grad.p,moment.at(t.name)->p,variance.at(t.name)->p,a->n,health.sum,health.bad,double(supervised),lr,t.ternary?.01f:0.f,1-std::pow(.9f,float(steps)),1-std::pow(.999f,float(steps)));
#else
ds_adamw<<<(a->n+127)/128,128>>>(a->p,graph.w.at(t.name)->grad.p,moment.at(t.name)->p,variance.at(t.name)->p,a->n,factor,lr,t.ternary?.01f:0.f,1-std::pow(.9f,float(steps)),1-std::pow(.999f,float(steps)));
#endif
check(cudaGetLastError());}
#ifdef TAO_GPU_HEALTH
norm=std::sqrt(health.finish());
#endif
check(cudaDeviceSynchronize());project();zero_grad();return float(norm);}
};
}
