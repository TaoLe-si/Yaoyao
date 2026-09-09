#pragma once
#include "dual_state_sorted_trainer.cuh"
namespace tao::dual {
// Serial slot execution shares parameter nodes/gradients, independent detached states.
struct SequenceSlots {
SortedGpuTrainer&trainer;std::vector<std::vector<Node>>s,m;int active=-1;
SequenceSlots(SortedGpuTrainer&t,unsigned count):trainer(t){if(!count)throw std::invalid_argument("slots");for(unsigned i=0;i<count;++i){std::vector<Node>a,b;for(unsigned l=0;l<t.graph.c.layers;++l){a.push_back(t.graph.tape.leaf(Vec(t.graph.c.s)));b.push_back(t.graph.tape.leaf(Vec(t.graph.c.m)));}s.push_back(a);m.push_back(b);}}
void begin(unsigned slot,bool reset){if(active!=-1||slot>=s.size()||!trainer.graph.tape.reverse.empty())throw std::runtime_error("slot lifecycle");active=int(slot);if(reset){for(unsigned l=0;l<trainer.graph.c.layers;++l){check(cudaMemset(s[slot][l]->value.p,0,s[slot][l]->value.n*4));check(cudaMemset(m[slot][l]->value.p,0,m[slot][l]->value.n*4));}}trainer.graph.s=s[slot];trainer.graph.m=m[slot];}
void finish(bool has_loss){if(active<0)throw std::runtime_error("slot lifecycle");if(has_loss)trainer.graph.tape.backward();trainer.detach();s[active]=trainer.graph.s;m[active]=trainer.graph.m;active=-1;}
};
}
