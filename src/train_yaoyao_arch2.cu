#define TAO_INPUT_SCALE
#define NOMINMAX
#define TAO_DEVICE_ZERO_GRAD
#include "slot_checkpoint_file.cuh"
#include "dual_state_initialization.hpp"
#include "bpe_pilot_reader.hpp"
#include "tokenizer_file.hpp"
#include <cstdio>
#include <chrono>
#include "dual_loss_parallel.cuh"
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
auto identity=checkpoint_identity(tr.graph.c,dh,th,200)+"arch2-warmstart63-reset-adam-state-data-warmup4-cosine196\n";
{
auto expected=capture(slots,cursor);
auto legacy=checkpoint_identity(tr.graph.c,dh,th,100);
legacy.replace(0,std::string(TAO_OPERATOR_ID).size(),"dual-state-1");
legacy+="block256-slots4-accum8-linear2-cosine98-peak0.0003-min0.00003\ncontinuous-v2-repeat-floor100\nce-parallel256-block-loss-v1\n";
auto snap=load_slot_file(argv[1],legacy,expected,cursor);
if(snap.steps!=63)throw std::runtime_error("expected source step63");
expected.w=std::move(snap.w); // Warm-start only master weights; fresh Adam/state/cursors.
restore(expected,slots,cursor);
printf("WARMSTART source_step=63 reset_optimizer=1 reset_states=1 reset_data=1\n");
}
if(tr.steps>UINT32_MAX-200)throw std::runtime_error("step overflow");
unsigned target_step=tr.steps+200;printf("RUN start=%u target=%u updates=200\n",tr.steps,target_step);
if(preflight){printf("PREFLIGHT_OK restored=%u next_doc=%zu slots=%zu no_updates_no_writes\n",tr.steps,cursor.next,cursor.slots.size());return 0;}
BlockLoss block_loss(256);
unsigned saved=tr.steps;
auto save=[&](){if(saved==tr.steps)return;auto stem="build/yaoyao_arch2_step_"+std::to_string(tr.steps);save_slot_file(capture(slots,cursor),stem+".scp",identity);CpuModel cpu(tr.graph.c);for(auto&kv:cpu.w)kv.second=tr.graph.w.at(kv.first)->value.host();save_bundle(cpu,stem+".dsb",th);saved=tr.steps;printf("SAVED step=%u stem=%s\n",saved,stem.c_str());fflush(stdout);};
size_t supervised_dataset=0;for(auto&doc:cursor.docs)for(size_t i=1;i<doc.size();++i)supervised_dataset+=doc[i].loss;if(!supervised_dataset)throw std::runtime_error("no supervised data");
while(tr.steps<target_step){
bool exhausted=cursor.next==cursor.docs.size();for(auto&c:cursor.slots)if(c.doc!=std::numeric_limits<size_t>::max()&&c.target<cursor.docs[c.doc].size())exhausted=false;
if(exhausted){cursor.next=0;for(auto&c:cursor.slots)c=tao::data::Cursor{};printf("DATASET_RESTART deterministic order\n");}
if(!profile&&std::filesystem::exists("build/STOP_TRAINING")){save();printf("STOP optimizer boundary\n");break;}
auto update_start=std::chrono::steady_clock::now();
size_t n=0,positions=0;double loss=0;
for(int r=0;r<8;++r)for(unsigned k=0;k<4;++k){
tao::data::Work work;if(!cursor.take(k,256,work))continue;
auto block_start=std::chrono::steady_clock::now();
slots.begin(k,work.reset);size_t local=0;auto&t=cursor.docs[work.doc];
for(size_t i=work.begin;i<work.end;++i){auto y=tr.graph.step(t[i-1].id);block_loss.seed(y,t[i].id,t[i].loss);local+=t[i].loss;++positions;}
loss+=block_loss.collect();slots.finish(local!=0);n+=local;
if(profile&&local){check(cudaDeviceSynchronize());double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-block_start).count();printf("PROFILE_BLOCK positions=%zu supervised=%zu seconds=%.6f positions_per_second=%.3f no_update_no_save\n",work.end-work.begin,local,seconds,(work.end-work.begin)/seconds);return 0;}
}
if(!n){printf("NO_UPDATE no supervised targets\n");continue;}
unsigned step=tr.steps+1;
float lr=step<=4?.0003f*step/4:float(.00003+.00027*.5*(1+std::cos(3.141592653589793*(step-4)/196)));
float norm=tr.update(n,lr);
double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-update_start).count();printf("TIMING step=%u seconds=%.6f positions_per_second=%.3f excludes_checkpoint_io=1\n",tr.steps,elapsed,positions/elapsed);
printf("UPDATE step=%u positions=%zu supervised=%zu train_preupdate_NLL=%.7f lr=%.9g norm=%.6f\n",tr.steps,positions,n,loss/n,lr,norm);fflush(stdout);
if(tr.steps%10==0||tr.steps==target_step)save();
}
save();return 0;
}catch(const std::exception&e){printf("FAIL %s; restore last complete checkpoint, no automatic retry\n",e.what());return 1;}}
