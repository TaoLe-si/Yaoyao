#define TAO_ASYNC_ALLOC
#define TAO_ASYNC_D2D
#define TAO_DEVICE_ZERO_GRAD
#define TAO_DEFER_BACKWARD_SYNC
#include "batch_train_graph.cuh"
#include "dual_state_initialization.hpp"
#include <cstdio>
int main(){using namespace tao::dual;try{
auto cpu=initialize(Config{2,16,8,16,32,261},713);BatchTrainGraph a(cpu,4),b(cpu,4);
auto ids=std::make_shared<Device>(Vec{65,66,67,68});auto mask=std::make_shared<Device>(Vec(4,1));Device seed(Vec(1044,.01f));
auto direct=a.step_device(ids,mask,mask,0);tape_copy(direct->grad.p,seed.p,1044*4);a.tape.backward();check(cudaDeviceSynchronize());
auto initial_s=b.s,initial_m=b.m;auto restart=std::make_shared<Device>(Vec(4,0));
cudaGraph_t graph;cudaGraphExec_t executable;check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
auto captured=b.step_device(ids,mask,restart,0);tape_copy(captured->grad.p,seed.p,1044*4);b.tape.backward();
check(cudaStreamEndCapture(cudaStreamPerThread,&graph));check(cudaGraphInstantiate(&executable,graph,cudaGraphInstantiateFlagAutoFreeOnLaunch));check(cudaGraphLaunch(executable,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
for(auto&kv:cpu.w){for(float&v:kv.second)v*=.99f;check(cudaMemcpy(b.w.at(kv.first)->value.p,kv.second.data(),kv.second.size()*4,cudaMemcpyHostToDevice));}
auto set_state=[](std::vector<Node>&sv,std::vector<Node>&mv){for(size_t l=0;l<sv.size();++l){Vec s(sv[l]->value.n,.15f),m(mv[l]->value.n,-.07f);check(cudaMemcpy(sv[l]->value.p,s.data(),s.size()*4,cudaMemcpyHostToDevice));check(cudaMemcpy(mv[l]->value.p,m.data(),m.size()*4,cudaMemcpyHostToDevice));check(cudaMemsetAsync(sv[l]->grad.p,0,s.size()*4));check(cudaMemsetAsync(mv[l]->grad.p,0,m.size()*4));}};set_state(initial_s,initial_m);
Vec changed{81,82,83,84};check(cudaMemcpy(ids->p,changed.data(),16,cudaMemcpyHostToDevice));for(auto&kv:b.w)check(cudaMemsetAsync(kv.second->grad.p,0,kv.second->grad.n*4));check(cudaGraphLaunch(executable,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));BatchTrainGraph fresh(cpu,4);set_state(fresh.s,fresh.m);auto new_direct=fresh.step_device(ids,mask,restart,0);tape_copy(new_direct->grad.p,seed.p,1044*4);fresh.tape.backward();check(cudaDeviceSynchronize());
float max=0;auto compare=[&](Vec x,Vec y){for(size_t i=0;i<x.size();++i){if(!std::isfinite(y[i]))throw std::runtime_error("nonfinite");max=std::max(max,std::abs(x[i]-y[i]));}};compare(new_direct->value.host(),captured->value.host());for(auto&kv:fresh.w)compare(kv.second->grad.host(),b.w.at(kv.first)->grad.host());
for(size_t l=0;l<b.s.size();++l){compare(fresh.s[l]->value.host(),b.s[l]->value.host());compare(fresh.m[l]->value.host(),b.m[l]->value.host());}
size_t nodes=0;check(cudaGraphGetNodes(graph,nullptr,&nodes));printf("CHANGED_STATE_WEIGHT_REPLAY nodes=%zu maxabs=%.9g launches=2\n",nodes,max);check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));return max==0?0:1;
}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 2;}}
