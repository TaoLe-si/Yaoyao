// Filtered-corpus repair: keep 1240 master weights, zero Adam+states, new shuffled epoch, lr=.0001.
#define main preserved_filtered_production_main
#include "train_yaoyao_graph_gpuval.cu"
#undef main
#include "shuffled_native_checkpoint.cuh"
#include "shuffled_epoch_batch_plan.hpp"

namespace filtered {
using namespace tao::dual;
namespace fs=std::filesystem;
namespace disk=tao::dual::shuffled_native::detail;
using DeterministicShuffledEpochCursor=tao::data::ShuffledEpochCursor;
constexpr uint64_t seed=20260909;
const std::string policy="filtered-v1;source1240-master-only;zero-adam-m-v;zero-slot-s-m;fresh-shuffled-epoch;lr=.0001;graph=4x8x256;official-gpuval-every-update;max200;min-turn-16;max-turn-512";
void require(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
std::string bytes(const fs::path& p,uint64_t cap=UINT64_C(2)*1024*1024*1024){return disk::read_regular(p,cap);}
void stopped(){
    require(fs::is_regular_file("build/STOP_TRAINING"),"STOP_TRAINING must remain present");
    for(const char* p:{"build/STOP_REPAIR","build/graph_run.lock","build/repair_order.lock","build/expanded_repair.lock"})
        require(!fs::exists(fs::symlink_status(p)),std::string("repair forbidden by ")+p);
}
std::map<std::string,std::string> parse_manifest(const std::string& text){
    std::map<std::string,std::string> m;std::istringstream in(text);std::string row;
    while(std::getline(in,row)){
        if(!row.empty()&&row.back()=='\r')row.pop_back();
        auto tab=row.find('\t');require(tab!=std::string::npos&&tab>0&&tab+1<row.size()&&row.find('\t',tab+1)==std::string::npos,"manifest tab fields");
        require(m.emplace(row.substr(0,tab),row.substr(tab+1)).second,"duplicate manifest key");
    }
    return m;
}
}

int main(int argc,char** argv){try{
    using namespace filtered;
    require(argc==8,"usage: train_repair_filtered NEW_TRAIN_BIN PARENT_RECEIPT TRUSTED_RECEIPT_SHA256 UNIQUE_OUT UPDATES_1_TO_200 SOURCE_CKPT TRUSTED_SOURCE_MANIFEST_SHA256");
    stopped();disk::hash_self_test();
    std::string count=argv[5];require(!count.empty()&&count.size()<=3&&count.find_first_not_of("0123456789")==std::string::npos,"updates syntax");
    unsigned updates=unsigned(std::stoul(count));require(updates>=1&&updates<=200,"updates must be 1..200");
    auto out=disk::directory(argv[4]);require(!fs::exists(fs::symlink_status(out)),"output directory exists; never overwrite/retry");
    auto source=disk::directory(argv[6]);require(fs::is_directory(fs::symlink_status(source)),"source checkpoint missing");
    require(fs::weakly_canonical(source)!=fs::weakly_canonical(out),"source and output must differ");
    require(disk::digest(argv[7]),"trusted source manifest digest");
    const fs::path newpath=argv[1];
    require(newpath.filename()=="train.bin","require finalized train.bin");
    require(!fs::exists(newpath.string()+".partial"),"export still partial");
    const auto raw=bytes(newpath),newsha=disk::sha256(raw);
    fs::path manpath=newpath;manpath+=".manifest.tsv";
    if(!fs::exists(manpath)) manpath=newpath.parent_path()/"train.manifest.tsv";
    const auto completion=bytes(manpath,UINT64_C(64)*1024*1024),completion_sha=disk::sha256(completion);
    std::string th;tao::text::load_tokenizer("build/formal_tokenizer.bbp",th);
    require(th==fixed_validation::Corpus::fixed_tokenizer_sha256,"frozen tokenizer mismatch");
    auto manifest=parse_manifest(completion);
    auto mf=[&](const std::string& key){auto it=manifest.find(key);require(it!=manifest.end(),"missing manifest "+key);return it->second;};
    require(mf("status")=="complete","export manifest not complete");
    require(disk::digest(mf("bin_sha256"))&&mf("bin_sha256")==newsha,"manifest bin hash");
    require(mf("tokenizer_sha256")==th,"manifest tokenizer");
    require(mf("validation_sha256")==disk::sha256(bytes("build/bpe_pilot_validation.bin")),"manifest validation");
    require(mf("test_sha256")==disk::sha256(bytes("build/bpe_pilot_test.bin")),"manifest test");
    auto number=[&](const std::string& key){auto v=mf(key);require(!v.empty()&&v.find_first_not_of("0123456789")==std::string::npos,"manifest integer "+key);size_t used=0;auto n=std::stoull(v,&used);require(used==v.size(),"range");return n;};
    const auto manifest_tokens=number("tokens"),manifest_supervised=number("supervised"),manifest_accepted=number("accepted");
    const auto receipt=bytes(argv[2],32768);
    require(disk::digest(argv[3])&&disk::sha256(receipt)==argv[3],"parent receipt digest mismatch");
    std::istringstream accepted(receipt);std::string line;
    require(bool(std::getline(accepted,line))&&line=="TAO_REPAIR_CORPUS_ACCEPT_V1","receipt magic");
    auto field=[&](const std::string& key){
        require(bool(std::getline(accepted,line)),"receipt missing "+key);
        const auto prefix=key+" ";
        require(line.compare(0,prefix.size(),prefix)==0,"receipt field "+key);
        auto value=line.substr(prefix.size());require(disk::digest(value),"receipt digest "+key);return value;
    };
    require(field("dataset_sha256")==newsha&&field("tokenizer_sha256")==th&&field("manifest_sha256")==completion_sha
        &&!receipt.empty()&&receipt.back()=='\n'&&accepted.peek()==std::char_traits<char>::eof(),"receipt corpus mismatch");
    const auto oldraw=bytes("build/repair_export_20260909_v1/train.bin"),oldsha=disk::sha256(oldraw);
    require(newsha!=oldsha,"filtered data equals unfiltered dataset");
    std::istringstream oldin(oldraw),newin(raw);
    tao::data::PilotCursor oldcanonical(tao::data::read_bpe_pilot(oldin,th),4,false);
    tao::data::PilotCursor fresh(tao::data::read_bpe_pilot(newin,th),4,false);
    size_t dataset_targets=0,dataset_positions=0,dataset_tokens=0;
    for(const auto& doc:fresh.docs){require(doc.size()>=2,"short document");dataset_positions+=doc.size()-1;dataset_tokens+=doc.size();
        for(size_t i=1;i<doc.size();++i)dataset_targets+=doc[i].loss;}
    require(dataset_targets>0,"empty supervised dataset");
    require(manifest_tokens==dataset_tokens&&manifest_supervised==dataset_targets&&manifest_accepted==fresh.docs.size(),"manifest counts");
    const auto identity=bytes(source/"transition.identity",1024*1024);
    const auto verified=bytes(source/"verified.identity",1024*1024);
    require(!identity.empty()&&verified.size()>identity.size()&&verified.compare(0,identity.size(),identity)==0,"saved identities");
    stopped();SortedGpuTrainer tr(initialize(Config{},713));SequenceSlots slots(tr,4);
    DeterministicShuffledEpochCursor oldepoch(std::move(oldcanonical),seed,verified);
    shuffled_native::load_and_restore(source,argv[7],slots,oldepoch,seed,verified,identity);
    require(tr.steps==1240 && oldepoch.epoch()==1,"source must be expanded 1240 epoch 1");
    auto initial_keep=tr.steps;
    for(auto& kv:tr.moment)check(cudaMemset(kv.second->p,0,kv.second->n*sizeof(float)));
    for(auto& kv:tr.variance)check(cudaMemset(kv.second->p,0,kv.second->n*sizeof(float)));
    for(auto* states:{&slots.s,&slots.m})for(auto& slot:*states)for(auto& v:slot)
        check(cudaMemset(v->value.p,0,v->value.n*sizeof(float)));
    tr.steps=0;tr.zero_grad();check(cudaDeviceSynchronize());
    require(initial_keep==1240,"source step mutated before adam reset");
    fresh.next=fresh.docs.size();for(auto& c:fresh.slots)c=tao::data::Cursor{};
    const auto newidentity=checkpoint_identity(tr.graph.c,newsha,th,200)+policy+"\nsource_manifest_sha256="+std::string(argv[7])+"\nparent_receipt_sha256="+argv[3]+"\nsource_step=1240\n";
    const auto newverified=newidentity+"canonical-filtered-dataset-sha256="+newsha+"\n";
    DeterministicShuffledEpochCursor epoch(std::move(fresh),seed,newverified);epoch.begin_next_epoch(true);
    require(epoch.epoch()==1&&epoch.cursor().next==0,"initial filtered epoch");
    fixed_validation::Evaluator eval(tr,fixed_validation::Corpus("build/bpe_pilot_validation.bin","build/formal_tokenizer.bbp",fixed_validation::Corpus::fixed_dataset_sha256,th));
    auto validate=[&](){
        require(slots.active==-1,"validation boundary");const auto v=eval.run();
        printf("FIXED_VALIDATION step=%u backend=%s docs=%zu supervised=%llu positions=%llu NLL=%.12f dataset_sha256=%s tokenizer_sha256=%s\n",
            v.step,v.backend,v.documents,v.totals.supervised,v.totals.positions,v.nll(),v.dataset_sha256.c_str(),v.tokenizer_sha256.c_str());
        fflush(stdout);return v;
    };
    std::ostringstream metrics;metrics.precision(17);
    auto initial=validate();
    require(std::abs(initial.nll()-6.3094786034531669)<1e-9,"filtered start must keep 1240 master weights");
    metrics<<"INITIAL source_step=1240 optimizer_step="<<tr.steps<<" val_NLL="<<initial.nll()<<" dataset_positions="<<dataset_positions<<" dataset_targets="<<dataset_targets<<" lr=.0001 adam_reset=1\n";
    printf("%s",metrics.str().c_str());fflush(stdout);
    ReusableBatchGraph replay(tr,4,256);DeferredLoss loss(1024);
    for(unsigned u=0;u<updates;++u){
        stopped();size_t positions=0,targets=0;const auto before=tr.steps;
        for(unsigned r=0;r<8;++r){
            stopped();if(epoch.exhausted()){check(cudaStreamSynchronize(0));epoch.begin_next_epoch(true);}
            auto p=tao::data::take_batch(epoch,256);require(p.timesteps>0,"no work after exhausted-only rollover");
            replay.run(p,slots);accumulate_loss<<<1,1>>>(replay.data.loss.p,replay.data.loss.n,loss.total,loss.bad);check(cudaGetLastError());
            positions+=p.positions;targets+=p.supervised;
        }
        require(targets>0,"no supervised update");const double train_loss=loss.collect();
        stopped();const float norm=tr.update(targets,.0001f);check(cudaDeviceSynchronize());
        require(tr.steps==before+1&&std::isfinite(norm),"invalid update");auto v=validate();
        std::ostringstream uline;uline.precision(17);uline<<"UPDATE step="<<tr.steps<<" epoch="<<epoch.epoch()<<" positions="<<positions<<" targets="<<targets<<" train_preupdate_NLL="<<train_loss/targets<<" val_NLL="<<v.nll()<<" lr=.0001 norm="<<norm<<"\n";
        metrics<<uline.str();printf("%s",uline.str().c_str());fflush(stdout);
    }
    stopped();require(tr.steps==updates,"bounded final step");
    const auto commit=shuffled_native::save(out,slots,epoch,seed,newverified,newidentity);
    stopped();const auto dsb=out/"final.dsb",tmp=out/"final.dsb.tmp";
    require(!fs::exists(fs::symlink_status(dsb))&&!fs::exists(fs::symlink_status(tmp)),"quality DSB overwrite");
    CpuModel cpu(tr.graph.c);for(auto& kv:cpu.w)kv.second=tr.graph.w.at(kv.first)->value.host();
    save_bundle(cpu,tmp.string(),th);
    require(!fs::exists(fs::symlink_status(dsb)),"quality DSB appeared");fs::rename(tmp,dsb);
    printf("QUALITY_EXPORT path=%s sha256=%s inference_only=1\n",dsb.generic_string().c_str(),disk::sha256(bytes(dsb)).c_str());
    disk::publish_text(out/"pilot.metrics",metrics.str());
    disk::publish_text(out/"transition.identity",newidentity);
    disk::publish_text(out/"verified.identity",newverified);
    disk::publish_text(out/"accepted.receipt",receipt);disk::publish_text(out/"export.manifest",completion);
    printf("FINAL step=%u updates=%u resumable_wrapper=1 manifest_sha256=%s directory=%s new_dataset_sha256=%s policy_sha256=%s\n",tr.steps,updates,commit.manifest_sha256.c_str(),out.generic_string().c_str(),newsha.c_str(),disk::sha256(policy).c_str());fflush(stdout);
    return 0;
}catch(const std::exception& e){fprintf(stderr,"FILTERED_PILOT_FAIL %s; no automatic retry/resume; retain STOP_TRAINING\n",e.what());return 1;}}
