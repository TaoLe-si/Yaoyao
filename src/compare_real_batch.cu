#define TAO_INPUT_SCALE
#define TAO_DEVICE_ZERO_GRAD
#define TAO_DEVICE_POOL
#define TAO_ASYNC_D2D
#include "batch_train_graph.cuh"
#include "gpu_batch_plan.cuh"
#include "bpe_pilot_reader.hpp"
#include "tokenizer_file.hpp"
#include "dual_model_bundle.hpp"
#include <chrono>
#include <cstdio>
int main(){try{using namespace tao::dual;std::string h;tao::text::load_tokenizer("build/formal_tokenizer.bbp",h);std::ifstream f("build/bpe_pilot_train.bin",std::ios::binary);auto docs=tao::data::read_bpe_pilot(f,h);tao::data::PilotCursor cursor(std::move(docs),4,false);auto plan=tao::data::take_batch(cursor,64);auto cpu=load_bundle("build/yaoyao_controlled_step_60.dsb",h);BatchTrainGraph graph(cpu,4);GpuBatchPlan device(plan,cpu.c.vocab);check(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();for(size_t t=0;t<plan.timesteps;++t){std::vector<unsigned>ids(4);std::vector<bool>active(4),reset(4);for(size_t k=0;k<4;++k){auto&p=plan.items[t*4+k];ids[k]=p.input;active[k]=p.active;reset[k]=p.reset;}auto y=graph.step(ids,active,reset);device.seed(y,t);}if(plan.supervised)graph.tape.backward();check(cudaDeviceSynchronize());double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();double loss=0;for(float v:device.loss.host()){if(!std::isfinite(v))throw std::runtime_error("loss");loss+=v;}std::ofstream dump("build/real_batch_grads.bin",std::ios::binary);for(auto&kv:graph.w){auto v=kv.second->grad.host();dump.write(reinterpret_cast<const char*>(v.data()),v.size()*4);}dump.close();if(!dump)throw std::runtime_error("dump");double norm2=0;for(auto&kv:graph.w)for(float v:kv.second->grad.host()){if(!std::isfinite(v))throw std::runtime_error("gradient");norm2+=double(v)*v;}printf("REAL_BATCH start_of_dataset=1 shape=formal slots=4 maxwidth=64 positions=%zu supervised=%zu seconds=%.6f loss_sum=%.9f gradient_norm=%.9f no_update_no_checkpoint\n",plan.positions,plan.supervised,seconds,loss,std::sqrt(norm2));return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
