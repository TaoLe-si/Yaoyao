#define TAO_INPUT_SCALE
#define TAO_DEVICE_ZERO_GRAD
#define TAO_DEFER_BACKWARD_SYNC
#include "batch_train_graph.cuh"
#include "dual_state_initialization.hpp"
#include <cstdio>
int main(){using namespace tao::dual;try{
auto cpu=initialize(Config{2,16,8,16,32,261},713);BatchTrainGraph a(cpu,4),b(cpu,4);
for(int t=0;t<3;++t){std::vector<unsigned>ids{unsigned(65+t),unsigned(70+t),unsigned(75+t),unsigned(80+t)};std::vector<bool>active{true,t<2,true,t>0},reset{t==0,t==0,t==1,t==1};
auto x=a.step(ids,active,reset),y=b.step(ids,active,reset);Vec grad(1044,0);for(int k=0;k<4;++k)if(active[k])grad[k*261+10+t]=.1f;check(cudaMemcpy(x->grad.p,grad.data(),grad.size()*4,cudaMemcpyHostToDevice));check(cudaMemcpy(y->grad.p,grad.data(),grad.size()*4,cudaMemcpyHostToDevice));}
check(cudaDeviceSynchronize());a.tape.backward();check(cudaDeviceSynchronize());cudaGraph_t graph;cudaGraphExec_t executable;
check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));b.tape.backward();check(cudaStreamEndCapture(cudaStreamPerThread,&graph));check(cudaGraphInstantiate(&executable,graph,0));check(cudaGraphLaunch(executable,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
size_t nodes=0;check(cudaGraphGetNodes(graph,nullptr,&nodes));float max=0;for(auto&kv:a.w){auto x=kv.second->grad.host(),y=b.w.at(kv.first)->grad.host();for(size_t i=0;i<x.size();++i){if(!std::isfinite(y[i]))throw std::runtime_error("nonfinite");max=std::max(max,std::abs(x[i]-y[i]));}}
check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));printf("RECURRENT_CAPTURE nodes=%zu tensors=%zu max_gradient_difference=%.9g\n",nodes,a.w.size(),max);return max==0?0:1;
}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 2;}}
