#pragma once
#include "batch_train_graph.cuh"
#include "dual_sequence_slots.cuh"
namespace tao::dual {
inline void bridge_slots(BatchTrainGraph&b,SequenceSlots&serial,bool to_batch){if(serial.active!=-1||!serial.trainer.graph.tape.reverse.empty()||!b.tape.reverse.empty()||b.slots!=serial.s.size())throw std::runtime_error("bridge requires detached boundaries");auto&c=serial.trainer.graph.c;if(c.layers!=b.c.layers||c.s!=b.c.s||c.m!=b.c.m)throw std::runtime_error("bridge state shape");for(unsigned l=0;l<b.c.layers;++l)for(unsigned k=0;k<b.slots;++k){auto copy=[&](Node batch,Node single,unsigned n){float*bp=batch->value.p+size_t(k)*n;check(cudaMemcpyAsync(to_batch?bp:single->value.p,to_batch?single->value.p:bp,n*4,cudaMemcpyDeviceToDevice,0));};copy(b.s[l],serial.s[k][l],b.c.s);copy(b.m[l],serial.m[k][l],b.c.m);}}
inline void detach_batch(BatchTrainGraph&b){auto copy=[](Node old){auto n=std::make_shared<GradNode>(old->value.n);check(cudaMemcpyAsync(n->value.p,old->value.p,old->value.n*4,cudaMemcpyDeviceToDevice,0));return n;};for(auto&v:b.s)v=copy(v);for(auto&v:b.m)v=copy(v);b.tape.reverse.clear();}
}
