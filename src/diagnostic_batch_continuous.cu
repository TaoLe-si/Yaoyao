#define TAO_ASYNC_D2D
#define TAO_DEFER_BACKWARD_SYNC
#define TAO_GPU_HEALTH
#define TAO_DEVICE_POOL
#define TAO_WARP_MATVEC
#define TAO_PARALLEL_RMS
#define TAO_TILED_DX
#ifndef DIAGNOSTIC_PREFIX
#define DIAGNOSTIC_PREFIX "batch_continuous_step_"
#endif
#define TAO_INPUT_SCALE
#define NOMINMAX
#define TAO_DEVICE_ZERO_GRAD
#include "slot_checkpoint_file.cuh"
#include "dual_state_initialization.hpp"
#include "bpe_pilot_reader.hpp"
#include "tokenizer_file.hpp"
#include <cstdio>
#include "training_control.hpp"
#include <chrono>
#include "gpu_deferred_loss.cuh"
#include "batch_slot_bridge.cuh"
#include "gpu_batch_plan.cuh"
int main(int argc,char**argv){try{
using namespace tao::dual;
constexpr unsigned budget=100;
bool profile=argc==3&&std::string(argv[2])=="--profile-block";
bool preflight=argc==3&&std::string(argv[2])=="--preflight";
if(argc!=2&&!preflight&&!profile)throw std::runtime_error("usage: checkpoint.scp [--preflight]");
std::string th;tao::text::load_tokenizer("build/formal_tokenizer.bbp",th);
std::ifstream f("build/bpe_pilot_train.bin",std::ios::binary);
std::string raw((std::istreambuf_iterator<char>(f)),{});
auto dh=tao::text::sha256(raw);std::istringstream input(raw);
auto docs=tao::data::read_bpe_pilot(input,th);
SortedGpuTrainer tr(initialize(Config{},713));SequenceSlots slots(tr,4);
tao::data::PilotCursor cursor(std::move(docs),4,false);
auto legacy_identity=checkpoint_identity(tr.graph.c,dh,th,200)+"arch2-warmstart63-reset-adam-state-data-warmup4-cosine196\n";
auto identity=checkpoint_identity(tr.graph.c,dh,th,200)+"arch2-controlled-v1-external-lr\n";
{auto expected=capture(slots,cursor);SlotSnapshot snap;try{snap=load_slot_file(argv[1],identity,expected,cursor);}catch(const std::runtime_error&e){if(std::string(e.what())!="length"&&std::string(e.what())!="identity")throw;snap=load_slot_file(argv[1],legacy_identity,expected,cursor);}restore(snap,slots,cursor);}
if(tr.steps>UINT32_MAX-200)throw std::runtime_error("step overflow");
unsigned target_step=tr.steps+200;
TrainingControl control;control.target=target_step;
std::string control_path=std::string(argv[1])+".control";
control.target=tr.steps+2;control.stop=false;target_step=control.target;
printf("RUN start=%u target=%u lr=%.9g stop=%d\n",tr.steps,target_step,control.lr,control.stop);
if(preflight){printf("PREFLIGHT_OK restored=%u next_doc=%zu slots=%zu no_updates_no_writes\n",tr.steps,cursor.next,cursor.slots.size());return 0;}
BatchTrainGraph batch(tr.graph.c,4,tr.graph.w);bridge_slots(batch,slots,true);DeferredLoss block_loss(1024);
unsigned saved=std::filesystem::equivalent(argv[1],argv[1])&&std::filesystem::path(argv[1]).filename()==("yaoyao_controlled_step_"+std::to_string(tr.steps)+".scp")?tr.steps:UINT32_MAX;
auto save=[&](){if(saved==tr.steps)return;auto stem=std::string("build/")+DIAGNOSTIC_PREFIX+std::to_string(tr.steps);save_slot_file(capture(slots,cursor),stem+".scp",identity);CpuModel cpu(tr.graph.c);for(auto&kv:cpu.w)kv.second=tr.graph.w.at(kv.first)->value.host();save_bundle(cpu,stem+".dsb",th);std::ofstream meta(stem+".scp.control");meta.precision(9);meta<<"TC1 "<<control.lr<<" "<<control.target<<" 0\n";meta.close();if(!meta)throw std::runtime_error("control sidecar write");saved=tr.steps;printf("SAVED step=%u stem=%s\n",saved,stem.c_str());fflush(stdout);};
size_t supervised_dataset=0;for(auto&doc:cursor.docs)for(size_t i=1;i<doc.size();++i)supervised_dataset+=doc[i].loss;if(!supervised_dataset)throw std::runtime_error("no supervised data");
while(tr.steps<target_step){
// Isolated single-update diagnostic; no external training controller.
target_step=control.target;
if(control.stop||tr.steps>=target_step){save();printf("CONTROL_STOP saved boundary\n");break;}
bool exhausted=cursor.next==cursor.docs.size();for(auto&c:cursor.slots)if(c.doc!=std::numeric_limits<size_t>::max()&&c.target<cursor.docs[c.doc].size())exhausted=false;
if(exhausted){cursor.next=0;for(auto&c:cursor.slots)c=tao::data::Cursor{};printf("DATASET_RESTART deterministic order\n");}
if(false){save();printf("STOP optimizer boundary\n");break;}
auto update_start=std::chrono::steady_clock::now();
size_t n=0,positions=0;double loss=0;
for(int r=0;r<8;++r){auto plan=tao::data::take_batch(cursor,256);if(!plan.timesteps)continue;GpuBatchPlan data(plan,tr.graph.c.vocab);for(size_t t=0;t<plan.timesteps;++t){auto y=batch.step_device(data.inputs,data.active,data.reset,t*plan.slots);data.seed(y,t);}if(plan.supervised)batch.tape.backward();accumulate_loss<<<1,1>>>(data.loss.p,plan.items.size(),block_loss.total,block_loss.bad);detach_batch(batch);n+=plan.supervised;positions+=plan.positions;}bridge_slots(batch,slots,false);
if(!n){printf("NO_UPDATE no supervised targets\n");continue;}
loss=block_loss.collect();
unsigned step=tr.steps+1;
float lr=control.lr;
float norm=tr.update(n,lr);
double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-update_start).count();printf("TIMING step=%u seconds=%.6f positions_per_second=%.3f excludes_checkpoint_io=1\n",tr.steps,elapsed,positions/elapsed);
printf("UPDATE step=%u positions=%zu supervised=%zu train_preupdate_NLL=%.7f lr=%.9g norm=%.6f\n",tr.steps,positions,n,loss/n,lr,norm);fflush(stdout);
if(tr.steps%10==0||tr.steps==target_step)save();
}
save();return 0;
}catch(const std::exception&e){printf("FAIL %s; restore last complete checkpoint, no automatic retry\n",e.what());return 1;}}
