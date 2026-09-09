#define NOMINMAX
#include "slot_checkpoint_file.cuh"
#include "dual_state_initialization.hpp"
#include "bpe_pilot_reader.hpp"
#include "tokenizer_file.hpp"
#include <cstdio>
int main(int argc,char**argv){try{
using namespace tao::dual;
constexpr unsigned budget=100;
bool preflight=argc==3&&std::string(argv[2])=="--preflight";
if(argc!=2&&!preflight)throw std::runtime_error("usage: checkpoint.scp [--preflight]");
std::string th;tao::text::load_tokenizer("build/formal_tokenizer.bbp",th);
std::ifstream f("build/bpe_pilot_train.bin",std::ios::binary);
std::string raw((std::istreambuf_iterator<char>(f)),{});
auto dh=tao::text::sha256(raw);std::istringstream input(raw);
auto docs=tao::data::read_bpe_pilot(input,th);
SortedGpuTrainer tr(initialize(Config{},713));SequenceSlots slots(tr,4);
tao::data::PilotCursor cursor(std::move(docs),4,false);
auto identity=checkpoint_identity(tr.graph.c,dh,th,budget)+"block256-slots4-accum8-linear2-cosine98-peak0.0003-min0.00003\n";
auto old_identity=identity;identity+="continuous-v2-repeat-floor100\n";
{ // Release host snapshot copies immediately after GPU restoration.
auto expected=capture(slots,cursor);SlotSnapshot snap;
try{snap=load_slot_file(argv[1],identity,expected,cursor);}catch(const std::runtime_error&e){if(std::string(e.what())!="length"&&std::string(e.what())!="identity")throw;snap=load_slot_file(argv[1],old_identity,expected,cursor);}
restore(snap,slots,cursor);
}
if(tr.steps>UINT32_MAX-200)throw std::runtime_error("step overflow");
unsigned target_step=tr.steps+200;printf("RUN start=%u target=%u updates=200\n",tr.steps,target_step);
if(preflight){printf("PREFLIGHT_OK restored=%u next_doc=%zu slots=%zu no_updates_no_writes\n",tr.steps,cursor.next,cursor.slots.size());return 0;}
unsigned saved=tr.steps;
auto save=[&](){if(saved==tr.steps)return;auto stem="build/yaoyao_continuous_step_"+std::to_string(tr.steps);save_slot_file(capture(slots,cursor),stem+".scp",identity);CpuModel cpu(tr.graph.c);for(auto&kv:cpu.w)kv.second=tr.graph.w.at(kv.first)->value.host();save_bundle(cpu,stem+".dsb",th);saved=tr.steps;printf("SAVED step=%u stem=%s\n",saved,stem.c_str());fflush(stdout);};
size_t supervised_dataset=0;for(auto&doc:cursor.docs)for(size_t i=1;i<doc.size();++i)supervised_dataset+=doc[i].loss;if(!supervised_dataset)throw std::runtime_error("no supervised data");
while(tr.steps<target_step){
bool exhausted=cursor.next==cursor.docs.size();for(auto&c:cursor.slots)if(c.doc!=std::numeric_limits<size_t>::max()&&c.target<cursor.docs[c.doc].size())exhausted=false;
if(exhausted){cursor.next=0;for(auto&c:cursor.slots)c=tao::data::Cursor{};printf("DATASET_RESTART deterministic order\n");}
if(std::filesystem::exists("build/STOP_TRAINING")){save();printf("STOP optimizer boundary\n");break;}
size_t n=0,positions=0;double loss=0;
for(int r=0;r<8;++r)for(unsigned k=0;k<4;++k){
tao::data::Work work;if(!cursor.take(k,256,work))continue;
slots.begin(k,work.reset);size_t local=0;auto&t=cursor.docs[work.doc];
for(size_t i=work.begin;i<work.end;++i){auto y=tr.graph.step(t[i-1].id);float v=seed_loss(y,t[i].id,t[i].loss);if(!std::isfinite(v))throw std::runtime_error("loss");loss+=v;local+=t[i].loss;++positions;}
slots.finish(local!=0);n+=local;
}
if(!n){printf("NO_UPDATE no supervised targets\n");continue;}
unsigned step=tr.steps+1;
float lr=step<=2?.0003f*step/2:step>=100?.00003f:float(.00003+.00027*.5*(1+std::cos(3.141592653589793*(step-2)/98)));
float norm=tr.update(n,lr);
printf("UPDATE step=%u positions=%zu supervised=%zu train_preupdate_NLL=%.7f lr=%.9g norm=%.6f\n",tr.steps,positions,n,loss/n,lr,norm);fflush(stdout);
if(tr.steps%10==0||tr.steps==target_step)save();
}
save();return 0;
}catch(const std::exception&e){printf("FAIL %s; restore last complete checkpoint, no automatic retry\n",e.what());return 1;}}
