#define TAO_NO_FFN
// Continue filtered repair from any compatible shuffled-native checkpoint. lr=.00025 (max), up to 2000 updates total.
// Resume does not hardcode step count or val NLL. Skips warmup if start>=100; cosine decay 1700->2200 to 0.00005.
// Does not overwrite production, expanded_repair_v1, filtered_repair_v1, filtered_long_v2.
// Checkpoints every 100 optimizer steps under UNIQUE_OUT/step_N plus a staged inference DSB.
#define main preserved_long_production_main
#include "train_yaoyao_graph_gpuval.cu"
#undef main
#include "shuffled_native_checkpoint.cuh"
#include "shuffled_epoch_batch_plan.hpp"

namespace longrun {
using namespace tao::dual;
namespace fs=std::filesystem;
namespace disk=tao::dual::shuffled_native::detail;
using DeterministicShuffledEpochCursor=tao::data::ShuffledEpochCursor;
constexpr uint64_t seed=20260909;
const std::string policy="noffn-fresh-v1;zero-init;short100-full1900;lr-warm100-cos1500-2000;graph4x8x256;gpuval-each;save100";
void require(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
std::string bytes(const fs::path& p,uint64_t cap=UINT64_C(2)*1024*1024*1024){return disk::read_regular(p,cap);}
void stopped(){
    require(fs::is_regular_file("build/STOP_TRAINING"),"STOP_TRAINING must remain present");
    for(const char* p:{"build/STOP_REPAIR","build/graph_run.lock","build/repair_order.lock","build/expanded_repair.lock","build/filtered_long.lock"})
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
    if(!std::freopen("build/noffn_fresh.log","a",stdout)||!std::freopen("build/noffn_fresh.log","a",stderr))throw std::runtime_error("fixed log open");
    setvbuf(stdout,nullptr,_IONBF,0);setvbuf(stderr,nullptr,_IONBF,0);
    using namespace longrun;
    require(argc==8,"usage: train_repair_long SOURCE_CKPT_DIR TRUSTED_MANIFEST_SHA256 NEW_TRAIN_BIN PARENT_RECEIPT TRUSTED_RECEIPT_SHA256 UNIQUE_OUT UPDATES_1_TO_2000");
    stopped();disk::hash_self_test();
    std::string count=argv[7];require(!count.empty()&&count.size()<=4&&count.find_first_not_of("0123456789")==std::string::npos,"updates syntax");
    unsigned updates=unsigned(std::stoul(count));require(updates>=1&&updates<=2000,"updates must be 1..2000");
    auto source=disk::directory(argv[1]);


    auto out=disk::directory(argv[6]);require(!fs::exists(fs::symlink_status(out)),"output directory exists; never overwrite/retry");
    require(fs::weakly_canonical(source)!=fs::weakly_canonical(out),"source and output must differ");
    const fs::path newpath=argv[3];
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
    const auto receipt=bytes(argv[4],32768);
    require(disk::digest(argv[5])&&disk::sha256(receipt)==argv[5],"parent receipt digest mismatch");
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
    std::istringstream newin(raw);
    tao::data::PilotCursor canonical(tao::data::read_bpe_pilot(newin,th),4,false);
    size_t dataset_targets=0,dataset_positions=0;
    for(const auto& doc:canonical.docs){require(doc.size()>=2,"short document");dataset_positions+=doc.size()-1;
        for(size_t i=1;i<doc.size();++i)dataset_targets+=doc[i].loss;}
    require(dataset_targets>0,"empty supervised dataset");
    std::vector<std::vector<tao::data::Token>> full_docs=canonical.docs,short_docs;
    for(const auto& doc:full_docs)if(doc.size()<=256)short_docs.push_back(doc);
    require(!short_docs.empty(),"empty short curriculum");
    const std::string base_identity=checkpoint_identity(Config{},newsha,th,2000)+"noffn-fresh-v1;short100-full1900;graph4x8x256;save100;gpuval-each;seed20260911\n";
    std::string identity=base_identity+"stage=short\n",verified=identity+"corpus="+newsha+"\n";
    stopped();SortedGpuTrainer tr(initialize(Config{},20260911));SequenceSlots slots(tr,4);
    DeterministicShuffledEpochCursor epoch(tao::data::PilotCursor(short_docs,4,false),seed,verified);
    const auto continue_identity=identity,continue_verified=verified;
    fixed_validation::Evaluator eval(tr,fixed_validation::Corpus("build/bpe_pilot_validation.bin","build/formal_tokenizer.bbp",fixed_validation::Corpus::fixed_dataset_sha256,th));
    auto validate=[&](){
        require(slots.active==-1,"validation boundary");const auto v=eval.run();
        printf("FIXED_VALIDATION step=%u backend=%s docs=%zu supervised=%llu positions=%llu NLL=%.12f dataset_sha256=%s tokenizer_sha256=%s\n",
            v.step,v.backend,v.documents,v.totals.supervised,v.totals.positions,v.nll(),v.dataset_sha256.c_str(),v.tokenizer_sha256.c_str());
        fflush(stdout);return v;
    };
    std::ostringstream metrics;metrics.precision(17);
    auto initial=validate();
    // Accept any plausible resumed val NLL (loose tolerance). Exact match is not required.
    require(std::isfinite(initial.nll()), std::string("resumed val NLL out of plausible range: ")+std::to_string(initial.nll()));
    metrics<<"FRESH step="<<tr.steps<<" positions=0 targets=0 train_preupdate_NLL=NA val_NLL="<<initial.nll()<<" dataset_positions="<<dataset_positions<<" dataset_targets="<<dataset_targets<<" lr=.00025 adam_reset=0\n";
    printf("%s",metrics.str().c_str());fflush(stdout);
    ReusableBatchGraph replay(tr,4,256);DeferredLoss loss(1024);
    const unsigned start=tr.steps;
    const unsigned total_target=unsigned(start)+updates;
    constexpr float lr_max=0.00025f, lr_min=0.00005f;
    constexpr unsigned hard_target=2000u;
    const unsigned warmup_until=start<100u?start+100u:start;
    const unsigned decay_begin=1500u;
    auto schedule=[&](unsigned s)->float{
        if(s<100u){return lr_max*float(s+1u)/100.f;}
        if(s>=decay_begin){
            float t=float(s-decay_begin)/float(hard_target-decay_begin);
            if(t>1.f)t=1.f;
            float cos=0.5f*(1.f+std::cos(3.14159265f*t));
            return lr_min+(lr_max-lr_min)*cos;
        }
        return lr_max;
    };
    auto export_ckpt=[&](bool finish){
        stopped();
        if(!fs::exists(fs::symlink_status(out))) require(fs::create_directory(out),"output directory create");
        const unsigned step=tr.steps;
        auto ckpt=out/("step_"+std::to_string(step));
        require(!fs::exists(fs::symlink_status(ckpt)),"periodic checkpoint overwrite");
        const fs::path staged=fs::path(ckpt.string()+".quality.dsb.tmp");
        require(!fs::exists(fs::symlink_status(staged)),"quality DSB staging overwrite");
        CpuModel cpu(tr.graph.c);for(auto& kv:cpu.w)kv.second=tr.graph.w.at(kv.first)->value.host();
        save_bundle(cpu,staged.string(),th);
        printf("QUALITY_STAGED step=%u path=%s sha256=%s inference_only=1\n",step,staged.generic_string().c_str(),disk::sha256(bytes(staged)).c_str());fflush(stdout);
        const auto commit=shuffled_native::save(ckpt,slots,epoch,seed,verified,identity);
        stopped();
        const auto dsb=ckpt/"final.dsb";
        require(!fs::exists(fs::symlink_status(dsb)),"quality DSB overwrite");
        fs::rename(staged,dsb);
        disk::publish_text(ckpt/"transition.identity",identity);
        disk::publish_text(ckpt/"verified.identity",verified);
        disk::publish_text(ckpt/"continue.transition.identity",identity);
        disk::publish_text(ckpt/"continue.verified.identity",verified);
        disk::publish_text(ckpt/"accepted.receipt",receipt);
        disk::publish_text(ckpt/"export.manifest",completion);
        disk::publish_text(ckpt/"pilot.metrics",metrics.str());
        printf("CHECKPOINT step=%u path=%s manifest_sha256=%s dsb_sha256=%s\n",step,ckpt.generic_string().c_str(),commit.manifest_sha256.c_str(),disk::sha256(bytes(dsb)).c_str());fflush(stdout);
        if(finish){
            const auto root_dsb=out/"final.dsb";
            require(!fs::exists(fs::symlink_status(root_dsb)),"root quality DSB overwrite");
            fs::copy_file(dsb,root_dsb);
            disk::publish_text(out/"pilot.metrics",metrics.str());
            disk::publish_text(out/"transition.identity",identity);
            disk::publish_text(out/"verified.identity",verified);
            printf("QUALITY_EXPORT path=%s sha256=%s inference_only=1\n",root_dsb.generic_string().c_str(),disk::sha256(bytes(root_dsb)).c_str());
            printf("FINAL step=%u updates=%u resumable_wrapper=1 manifest_sha256=%s directory=%s new_dataset_sha256=%s policy_sha256=%s\n",tr.steps,updates,commit.manifest_sha256.c_str(),ckpt.generic_string().c_str(),newsha.c_str(),disk::sha256(policy).c_str());fflush(stdout);
        }
    };
    export_ckpt(false);
    for(unsigned u=0;u<updates;++u){
        if(tr.steps==100){identity=base_identity+"stage=full\n";verified=identity+"corpus="+newsha+"\n";epoch=DeterministicShuffledEpochCursor(tao::data::PilotCursor(full_docs,4,false),seed,verified);for(auto* states:{&slots.s,&slots.m})for(auto& slot:*states)for(auto& n:slot)check(cudaMemset(n->value.p,0,n->value.n*4));printf("CURRICULUM stage=full step=100 documents=%zu\n",full_docs.size());fflush(stdout);}
        if(fs::exists("build/STOP_NOFFN")){if(tr.steps%100)export_ckpt(false);printf("PAUSED saved step=%u\n",tr.steps);return 0;}
        stopped();size_t positions=0,targets=0;const auto before=tr.steps;
        for(unsigned r=0;r<8;++r){
            stopped();if(epoch.exhausted()){check(cudaStreamSynchronize(0));epoch.begin_next_epoch(true);}
            auto p=tao::data::take_batch(epoch,256);require(p.timesteps>0,"no work after exhausted-only rollover");
            replay.run(p,slots);accumulate_loss<<<1,1>>>(replay.data.loss.p,replay.data.loss.n,loss.total,loss.bad);check(cudaGetLastError());
            positions+=p.positions;targets+=p.supervised;
        }
        require(targets>0,"no supervised update; abort, no automatic retry");const double train_loss=loss.collect();
        stopped();const float lr=schedule(tr.steps);float norm=tr.update(targets,lr);check(cudaDeviceSynchronize());
        if(norm>1.5f){std::printf("NORM_WARNING trigger=norm_clip step=%u norm=%.6f lr=%.6f\n",tr.steps,norm,lr);std::fflush(stdout);}
        require(tr.steps==before+1&&std::isfinite(norm),"invalid update");auto v=validate();
        std::ostringstream uline;uline.precision(17);uline<<"UPDATE step="<<tr.steps<<" epoch="<<epoch.epoch()<<" positions="<<positions<<" targets="<<targets<<" train_preupdate_NLL="<<train_loss/targets<<" val_NLL="<<v.nll()<<" lr="<<lr<<" norm="<<norm<<"\n";
        metrics<<uline.str();printf("%s",uline.str().c_str());fflush(stdout);
        if(tr.steps%100u==0||u+1==updates) export_ckpt(u+1==updates);
    }
    stopped();require(tr.steps==start+updates,"bounded final step");
    require(fs::is_regular_file(out/"final.dsb"),"missing root quality DSB after final checkpoint");
    return 0;
}catch(const std::exception& e){fprintf(stderr,"LONG_PILOT_FAIL %s; no automatic retry/resume; retain STOP_TRAINING\n",e.what());return 1;}}