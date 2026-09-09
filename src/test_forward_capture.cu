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
cudaGraph_t graph;cudaGraphExec_t executable;check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
auto captured=b.step_device(ids,mask,mask,0);tape_copy(captured->grad.p,seed.p,1044*4);b.tape.backward();
check(cudaStreamEndCapture(cudaStreamPerThread,&graph));check(cudaGraphInstantiate(&executable,graph,0));check(cudaGraphLaunch(executable,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
float max=0;auto compare=[&](Vec x,Vec y){for(size_t i=0;i<x.size();++i){if(!std::isfinite(y[i]))throw std::runtime_error("nonfinite");max=std::max(max,std::abs(x[i]-y[i]));}};compare(direct->value.host(),captured->value.host());for(auto&kv:a.w)compare(kv.second->grad.host(),b.w.at(kv.first)->grad.host());
size_t nodes=0;check(cudaGraphGetNodes(graph,nullptr,&nodes));printf("FULL_CAPTURE nodes=%zu maxabs=%.9g single_launch=1\n",nodes,max);check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));return max==0?0:1;
}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 2;}}
