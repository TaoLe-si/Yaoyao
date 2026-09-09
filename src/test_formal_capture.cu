#define TAO_INPUT_SCALE
#define TAO_DEVICE_ZERO_GRAD
#define TAO_DEFER_BACKWARD_SYNC
#include "batch_train_graph.cuh"
#include "dual_state_initialization.hpp"
#include <cstdio>
#include <chrono>
int main(){using namespace tao::dual;try{
auto cpu=initialize(Config{},713);BatchTrainGraph a(cpu,4),b(cpu,4);
for(int t=0;t<3;++t){std::vector<unsigned>ids{unsigned(65+t),unsigned(70+t),unsigned(75+t),unsigned(80+t)};std::vector<bool>active{true,t<2,true,t>0},reset{t==0,t==0,t==1,t==1};
auto x=a.step(ids,active,reset),y=b.step(ids,active,reset);Vec grad(size_t(cpu.c.vocab)*4,0);for(int k=0;k<4;++k)if(active[k])grad[size_t(k)*cpu.c.vocab+10+t]=.1f;check(cudaMemcpy(x->grad.p,grad.data(),grad.size()*4,cudaMemcpyHostToDevice));check(cudaMemcpy(y->grad.p,grad.data(),grad.size()*4,cudaMemcpyHostToDevice));}
check(cudaDeviceSynchronize());auto clock=[](){return std::chrono::steady_clock::now();};auto direct_start=clock();a.tape.backward();check(cudaDeviceSynchronize());auto direct_end=clock();cudaGraph_t graph;cudaGraphExec_t executable;
auto capture_start=clock();check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));b.tape.backward();check(cudaStreamEndCapture(cudaStreamPerThread,&graph));auto capture_end=clock();check(cudaGraphInstantiate(&executable,graph,0));auto instantiate_end=clock();check(cudaGraphLaunch(executable,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));auto launch_end=clock();printf("TIMING direct_ms=%.3f capture_ms=%.3f instantiate_ms=%.3f launch_sync_ms=%.3f\n",std::chrono::duration<double,std::milli>(direct_end-direct_start).count(),std::chrono::duration<double,std::milli>(capture_end-capture_start).count(),std::chrono::duration<double,std::milli>(instantiate_end-capture_end).count(),std::chrono::duration<double,std::milli>(launch_end-instantiate_end).count());
size_t nodes=0;check(cudaGraphGetNodes(graph,nullptr,&nodes));float max=0;for(auto&kv:a.w){auto x=kv.second->grad.host(),y=b.w.at(kv.first)->grad.host();for(size_t i=0;i<x.size();++i){if(!std::isfinite(y[i]))throw std::runtime_error("nonfinite");max=std::max(max,std::abs(x[i]-y[i]));}}
check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));printf("RECURRENT_CAPTURE nodes=%zu tensors=%zu max_gradient_difference=%.9g\n",nodes,a.w.size(),max);return max==0?0:1;
}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 2;}}
