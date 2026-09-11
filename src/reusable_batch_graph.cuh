#pragma once
#include "batch_slot_bridge.cuh"
#include "gpu_batch_plan.cuh"
namespace tao::dual {
// Requires TAO_ASYNC_ALLOC, TAO_ASYNC_D2D, deferred backward sync, per-thread default stream.
struct ReusableBatchGraph {
BatchTrainGraph batch;GpuBatchPlan data;std::vector<Node>initial_s,initial_m;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
static tao::data::BatchPlan blank(unsigned slots,unsigned width){tao::data::BatchPlan p;p.slots=slots;p.timesteps=width;p.items.resize(size_t(slots)*width);return p;}
ReusableBatchGraph(SortedGpuTrainer&tr,unsigned slots,unsigned width):batch(tr.graph.c,slots,tr.graph.w),data(blank(slots,width),tr.graph.c.vocab){
initial_s=batch.s;initial_m=batch.m;check(cudaDeviceSynchronize());
check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
for(size_t t=0;t<width;++t){auto y=batch.step_device(data.inputs,data.active,data.reset,t*slots);data.seed(y,t);}batch.tape.backward();
check(cudaStreamEndCapture(cudaStreamPerThread,&graph));check(cudaGraphInstantiate(&executable,graph,cudaGraphInstantiateFlagAutoFreeOnLaunch));
}
~ReusableBatchGraph(){cudaStreamSynchronize(cudaStreamPerThread);if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);}
void run(const tao::data::BatchPlan&p,SequenceSlots&states){
if(p.slots!=data.slots||p.timesteps>data.steps||states.active!=-1)throw std::runtime_error("graph batch boundary");
auto padded=p;padded.timesteps=data.steps;padded.items.resize(data.steps*data.slots);GpuBatchPlan upload(padded,data.vocab);
auto copy=[](Device&a,Device&b){check(cudaMemcpyAsync(a.p,b.p,a.n*4,cudaMemcpyDeviceToDevice,0));};
copy(*data.inputs,*upload.inputs);copy(*data.active,*upload.active);copy(*data.reset,*upload.reset);copy(data.targets,upload.targets);copy(data.masks,upload.masks);
for(unsigned l=0;l<batch.c.layers;++l){check(cudaMemsetAsync(initial_s[l]->grad.p,0,initial_s[l]->grad.n*4));check(cudaMemsetAsync(initial_m[l]->grad.p,0,initial_m[l]->grad.n*4));for(unsigned k=0;k<batch.slots;++k){check(cudaMemcpyAsync(initial_s[l]->value.p+k*batch.c.s,states.s[k][l]->value.p,batch.c.s*4,cudaMemcpyDeviceToDevice,0));check(cudaMemcpyAsync(initial_m[l]->value.p+k*batch.c.memory_size(),states.m[k][l]->value.p,batch.c.memory_size()*4,cudaMemcpyDeviceToDevice,0));}}
check(cudaGraphLaunch(executable,cudaStreamPerThread));
for(unsigned l=0;l<batch.c.layers;++l)for(unsigned k=0;k<batch.slots;++k){check(cudaMemcpyAsync(states.s[k][l]->value.p,batch.s[l]->value.p+k*batch.c.s,batch.c.s*4,cudaMemcpyDeviceToDevice,0));check(cudaMemcpyAsync(states.m[k][l]->value.p,batch.m[l]->value.p+k*batch.c.memory_size(),batch.c.memory_size()*4,cudaMemcpyDeviceToDevice,0));}
}
};
}
