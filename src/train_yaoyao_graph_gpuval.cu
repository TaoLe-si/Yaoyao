#define TAO_ASYNC_D2D
#define TAO_DEFER_BACKWARD_SYNC
#define TAO_GPU_HEALTH
#define TAO_ASYNC_ALLOC
#define TAO_WARP_MATVEC
#define TAO_PARALLEL_RMS
#define TAO_TILED_DX
#ifndef DIAGNOSTIC_PREFIX
#define DIAGNOSTIC_PREFIX "yaoyao_graph_step_"
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
#include "reusable_batch_graph.cuh"
#include "gpu_fixed_validation_arch2.cuh"
int main(int argc,char**argv){try{
using namespace tao::dual;


bool validate_only=argc==3&&std::string(argv[2])=="--validate-only";
bool preflight=argc==3&&std::string(argv[2])=="--preflight";
if(argc!=2&&!preflight&&!validate_only)throw std::runtime_error("usage: checkpoint.scp [--preflight|--validate-only]");
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
control=read_control(control_path,control);target_step=control.target;
printf("RUN start=%u target=%u lr=%.9g stop=%d\n",tr.steps,target_step,control.lr,control.stop);
if(!preflight&&!validate_only&&(control.stop||std::filesystem::exists("build/STOP_TRAINING"))){printf("STOP restored checkpoint retained; no updates\n");return 0;}
if(preflight){printf("PREFLIGHT_OK restored=%u next_doc=%zu slots=%zu no_updates_no_writes\n",tr.steps,cursor.next,cursor.slots.size());return 0;}
fixed_validation::Evaluator validation(tr,fixed_validation::Corpus("build/bpe_pilot_validation.bin","build/formal_tokenizer.bbp",fixed_validation::Corpus::fixed_dataset_sha256,th));
auto validate=[&](){
    if(slots.active!=-1)throw std::runtime_error("validation slot boundary");
    auto start=std::chrono::steady_clock::now();auto v=validation.run();
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    printf("VALIDATION_TIMING validation_step=%u seconds=%.6f tokens=%zu positions=%llu parameter_h2d_bytes=%zu hotpath_h2d_bytes=%zu d2h_bytes=%zu\n",
        v.step,seconds,v.tokens,v.totals.positions,v.parameter_h2d_bytes,v.hotpath_h2d_bytes,v.d2h_bytes);
    printf("FIXED_VALIDATION step=%u backend=%s docs=%zu supervised=%llu positions=%llu NLL=%.12f dataset_sha256=%s tokenizer_sha256=%s seconds=%.6f\n",v.step,v.backend,v.documents,v.totals.supervised,v.totals.positions,v.nll(),v.dataset_sha256.c_str(),v.tokenizer_sha256.c_str(),seconds);fflush(stdout);return v;
};
if(validate_only){
    // Diagnostic-only full readbacks. Routine validation never calls capture.
    auto snapshot_bytes=[&](){
        auto a=capture(slots,cursor);std::string bytes;
        auto integer=[&](uint64_t v){for(int i=0;i<8;++i)bytes.push_back(char(v>>(8*i)));};
        auto vec=[&](const Vec&v){integer(v.size());if(!v.empty())bytes.append(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(float));};
        integer(a.steps);integer(a.next);integer(a.cursor.size());
        for(const auto&v:a.cursor){integer(v.doc);integer(v.target);}
        for(const auto*map:{&a.w,&a.m,&a.v}){
            integer(map->size());for(const auto&kv:*map){integer(kv.first.size());bytes+=kv.first;vec(kv.second);}
        }
        for(const auto*states:{&a.s,&a.mem}){
            integer(states->size());for(const auto&slot:*states){integer(slot.size());for(const auto&v:slot)vec(v);}
        }
        for(const auto&kv:tr.graph.w){vec(kv.second->value.host());vec(kv.second->grad.host());}
        for(const auto&kv:tr.scales)vec(kv.second->host());
        for(const auto*states:{&tr.graph.s,&tr.graph.m})for(const auto&v:*states){vec(v->value.host());vec(v->grad.host());}
        for(const auto*states:{&slots.s,&slots.m})for(const auto&slot:*states)for(const auto&v:slot)vec(v->grad.host());
        integer(uint64_t(slots.active));integer(tr.graph.tape.reverse.size());return bytes;
    };
    auto before=snapshot_bytes();auto a=validate();
    if(before!=snapshot_bytes())throw std::runtime_error("validation mutated training snapshot");
    auto b=validate();
    if(before!=snapshot_bytes())throw std::runtime_error("validation repeat mutated training snapshot");
    if(a.nll()!=b.nll()||a.totals.loss!=b.totals.loss||a.totals.positions!=b.totals.positions||
       a.totals.supervised!=b.totals.supervised||a.step!=b.step||a.dataset_sha256!=b.dataset_sha256||
       a.tokenizer_sha256!=b.tokenizer_sha256)throw std::runtime_error("validation repeat differs");
    printf("VALIDATE_ONLY_OK validation_step=%u repeats=2 exact_NLL_equal=1 capture_equal=1 effective_optimizer_slots_equal=1 no_updates_no_writes=1 snapshot_sha256=%s\n",
        tr.steps,tao::text::sha256(before).c_str());fflush(stdout);return 0;
}
ReusableBatchGraph replay(tr,4,256);DeferredLoss block_loss(1024);
unsigned saved=tr.steps;
auto save=[&](){if(saved==tr.steps)return;auto stem=std::string("build/")+DIAGNOSTIC_PREFIX+std::to_string(tr.steps);if(std::filesystem::exists(stem+".scp")||std::filesystem::exists(stem+".dsb"))throw std::runtime_error("refuse checkpoint overwrite");save_slot_file(capture(slots,cursor),stem+".scp",identity);CpuModel cpu(tr.graph.c);for(auto&kv:cpu.w)kv.second=tr.graph.w.at(kv.first)->value.host();save_bundle(cpu,stem+".dsb",th);std::ofstream meta(stem+".scp.control");meta.precision(9);meta<<"TC1 "<<control.lr<<" "<<control.target<<" 0\n";meta.close();if(!meta)throw std::runtime_error("control sidecar write");saved=tr.steps;printf("SAVED step=%u stem=%s\n",saved,stem.c_str());fflush(stdout);};
size_t supervised_dataset=0;for(auto&doc:cursor.docs)for(size_t i=1;i<doc.size();++i)supervised_dataset+=doc[i].loss;if(!supervised_dataset)throw std::runtime_error("no supervised data");
while(tr.steps<target_step){
try{control=read_control(control_path,control);}catch(...){save();throw;}
target_step=control.target;
if(control.stop||tr.steps>=target_step){save();printf("CONTROL_STOP saved boundary\n");break;}
bool exhausted=cursor.next==cursor.docs.size();for(auto&c:cursor.slots)if(c.doc!=std::numeric_limits<size_t>::max()&&c.target<cursor.docs[c.doc].size())exhausted=false;
if(exhausted){cursor.next=0;for(auto&c:cursor.slots)c=tao::data::Cursor{};printf("DATASET_RESTART deterministic order\n");}
if(std::filesystem::exists("build/STOP_TRAINING")){save();printf("STOP optimizer boundary\n");break;}
auto update_start=std::chrono::steady_clock::now();
size_t n=0,positions=0;double loss=0;
for(int r=0;r<8;++r){auto plan=tao::data::take_batch(cursor,256);if(!plan.timesteps)continue;replay.run(plan,slots);accumulate_loss<<<1,1>>>(replay.data.loss.p,replay.data.loss.n,block_loss.total,block_loss.bad);n+=plan.supervised;positions+=plan.positions;}
if(!n){printf("NO_UPDATE no supervised targets\n");continue;}
loss=block_loss.collect();

float lr=control.lr;
float norm=tr.update(n,lr);
double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-update_start).count();printf("TIMING step=%u seconds=%.6f positions_per_second=%.3f excludes_checkpoint_io=1\n",tr.steps,elapsed,positions/elapsed);
printf("UPDATE step=%u positions=%zu supervised=%zu train_preupdate_NLL=%.7f lr=%.9g norm=%.6f\n",tr.steps,positions,n,loss/n,lr,norm);fflush(stdout);
validate(); // Every completed optimizer update; no validation gradients.
if(tr.steps%100==0||tr.steps==target_step)save();
}
save();return 0;
}catch(const std::exception&e){printf("FAIL %s; restore last complete checkpoint, no automatic retry\n",e.what());return 1;}}
