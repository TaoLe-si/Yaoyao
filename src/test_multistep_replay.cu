#define TAO_ASYNC_ALLOC
#define TAO_ASYNC_D2D
#define TAO_DEVICE_ZERO_GRAD
#define TAO_DEFER_BACKWARD_SYNC
#include "batch_train_graph.cuh"
#include "gpu_batch_plan.cuh"
#include "dual_state_initialization.hpp"
#include <cstdio>
int main(){using namespace tao::dual;try{
auto cpu=initialize(Config{2,16,8,16,32,261},713);BatchTrainGraph b(cpu,4);
tao::data::BatchPlan p;p.slots=4;p.timesteps=3;p.positions=12;p.supervised=12;for(unsigned i=0;i<12;++i)p.items.push_back({65+i,81+i,true,i<4,true});GpuBatchPlan data(p,261);
auto run=[&](BatchTrainGraph&g){for(size_t t=0;t<3;++t){auto y=g.step_device(data.inputs,data.active,data.reset,t*4);data.seed(y,t);}g.tape.backward();};
check(cudaDeviceSynchronize());cudaGraph_t graph;cudaGraphExec_t executable;check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));run(b);check(cudaStreamEndCapture(cudaStreamPerThread,&graph));check(cudaGraphInstantiate(&executable,graph,cudaGraphInstantiateFlagAutoFreeOnLaunch));check(cudaGraphLaunch(executable,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
Vec ids(12),active(12,1),reset(12,0);std::vector<unsigned>target(12),supervised(12,1);for(unsigned i=0;i<12;++i){ids[i]=100+i;target[i]=120+i;if(i<4)reset[i]=1;}reset[6]=1;active[7]=0;active[11]=0;supervised[7]=0;supervised[11]=0;supervised[4]=0;
check(cudaMemcpy(data.inputs->p,ids.data(),48,cudaMemcpyHostToDevice));check(cudaMemcpy(data.active->p,active.data(),48,cudaMemcpyHostToDevice));check(cudaMemcpy(data.reset->p,reset.data(),48,cudaMemcpyHostToDevice));check(cudaMemcpy(data.targets.p,target.data(),48,cudaMemcpyHostToDevice));check(cudaMemcpy(data.masks.p,supervised.data(),48,cudaMemcpyHostToDevice));
for(auto&kv:b.w)check(cudaMemsetAsync(kv.second->grad.p,0,kv.second->grad.n*4));check(cudaGraphLaunch(executable,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));auto captured_loss=data.loss.host();
BatchTrainGraph fresh(cpu,4);run(fresh);check(cudaDeviceSynchronize());float max=0;auto compare=[&](Vec x,Vec y){for(size_t i=0;i<x.size();++i){if(!std::isfinite(y[i]))throw std::runtime_error("nonfinite");max=std::max(max,std::abs(x[i]-y[i]));}};compare(captured_loss,data.loss.host());for(auto&kv:fresh.w)compare(kv.second->grad.host(),b.w.at(kv.first)->grad.host());for(size_t l=0;l<b.s.size();++l){compare(fresh.s[l]->value.host(),b.s[l]->value.host());compare(fresh.m[l]->value.host(),b.m[l]->value.host());}
size_t nodes=0;check(cudaGraphGetNodes(graph,nullptr,&nodes));printf("MULTISTEP_REPLAY nodes=%zu maxabs=%.9g changed_masks_and_targets=1\n",nodes,max);check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));return max==0?0:1;
}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 2;}}
