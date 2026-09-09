#define NOMINMAX
#include "dual_sequence_slots.cuh"
#include "dual_state_initialization.hpp"
#include "dual_model_bundle.hpp"
#include "bpe_pilot_reader.hpp"
#include <cstdio>
int main(){try{using namespace tao::dual;std::string hash="34463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333";std::ifstream f("build/bpe_pilot_train.bin",std::ios::binary);auto docs=tao::data::read_bpe_pilot(f,hash);SortedGpuTrainer tr(initialize(Config{},713));double loss=0;size_t n=0;auto&t=docs.at(0);for(size_t i=1;i<t.size()&&i<=256;++i){auto y=tr.graph.step(t[i-1].id);loss+=seed_loss(y,t[i].id,t[i].loss);n+=t[i].loss;}if(!n)throw std::runtime_error("no supervision");tr.graph.tape.backward();tr.detach();float norm=tr.update(n,3e-6f);CpuModel cpu(tr.graph.c);for(auto&kv:cpu.w)kv.second=tr.graph.w.at(kv.first)->value.host();save_bundle(cpu,"build/yaoyao_formal_step1.dsb",hash);printf("PASS formal_GPU_step=%u supervised=%zu preupdate_NLL=%.6f norm=%.6f exported=yaoyao_formal_step1.dsb\n",tr.steps,n,loss/n,norm);return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
