// SOURCE-ONLY bounded candidate; requires independent data completion + parent review.
// No auto-resume, production controls, tokenizer edits, or effective-only checkpoint.
#define main preserved_expanded_production_main
#include "train_yaoyao_graph_gpuval.cu"
#undef main
#include "shuffled_native_checkpoint.cuh"
#include "shuffled_epoch_batch_plan.hpp"

namespace expanded {
using namespace tao::dual;
namespace fs=std::filesystem;
namespace disk=tao::dual::shuffled_native::detail;
using DeterministicShuffledEpochCursor=tao::data::ShuffledEpochCursor;
constexpr uint64_t seed=20260909;
const std::string policy="expanded-v1;old1200-verified-oldcanonical;preserve-master-adam-step;reset-slot-s-m-newcursor;canonical-BPE-no-legacy-EOS-removal;splitmix64-fisher-yates-v1;seed=20260909;epoch1;rollover-only-exhausted;graph=4x8x256;lr=.00025;official-gpuval-every-update;max40";
void require(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
std::string bytes(const fs::path& p,uint64_t cap=UINT64_C(2)*1024*1024*1024){return disk::read_regular(p,cap);}
void stopped(){
    require(fs::is_regular_file("build/STOP_TRAINING"),"STOP_TRAINING must remain present");
    for(const char* p:{"build/STOP_REPAIR","build/graph_run.lock","build/repair_order.lock"})
        require(!fs::exists(fs::symlink_status(p)),std::string("repair forbidden by ")+p);
}
}
int main(int argc,char** argv){try{
    using namespace expanded;
    require(argc==6,"usage: train_repair_expanded NEW_TRAIN_BIN PARENT_RECEIPT TRUSTED_PARENT_RECEIPT_SHA256 UNIQUE_NEW_CHECKPOINT_DIR UPDATES_1_TO_40 (repository-root cwd)");
    stopped();disk::hash_self_test();
    std::string count=argv[5];require(!count.empty()&&count.size()<=2&&count.find_first_not_of("0123456789")==std::string::npos,"updates syntax");
    unsigned updates=unsigned(std::stoul(count));require(updates>=1&&updates<=40,"updates must be 1..40");
    auto out=disk::directory(argv[4]);require(!fs::exists(fs::symlink_status(out)),"output directory exists; never overwrite/retry");
    const fs::path newpath=argv[1];
    require(newpath.filename()=="train.bin","require finalized train.bin, not partial export");
    require(!fs::exists(newpath.string()+".partial")&&!fs::exists(newpath.parent_path()/"train.tsv.partial"),"export still partial");
    require(fs::weakly_canonical(newpath)!=fs::weakly_canonical("build/bpe_pilot_train.bin"),"new data must not alias old corpus");
    const auto raw=bytes(newpath),newsha=disk::sha256(raw);
    const auto completion=bytes(newpath.parent_path()/"train.manifest.tsv",UINT64_C(64)*1024*1024),completion_sha=disk::sha256(completion);
    require(!completion.empty(),"empty export manifest");
    std::string th;tao::text::load_tokenizer("build/formal_tokenizer.bbp",th);
    require(th==fixed_validation::Corpus::fixed_tokenizer_sha256,"frozen tokenizer mismatch");
    // Parse the real exporter key<TAB>value manifest, never invent a completion file.
    std::map<std::string,std::string> manifest;std::istringstream manifest_in(completion);std::string row;
    while(std::getline(manifest_in,row)){
        if(!row.empty()&&row.back()=='\r')row.pop_back();
        auto tab=row.find('\t');require(tab!=std::string::npos&&tab>0&&tab+1<row.size()&&row.find('\t',tab+1)==std::string::npos,"manifest tab fields");
        require(manifest.emplace(row.substr(0,tab),row.substr(tab+1)).second,"duplicate manifest key");
    }
    auto mf=[&](const std::string& key)->std::string{auto it=manifest.find(key);require(it!=manifest.end(),"missing manifest "+key);return it->second;};
    require(mf("status")=="complete","export manifest not complete");
    auto match_hash=[&](const std::string& key,const std::string& actual){auto h=mf(key);require(disk::digest(h)&&h==actual,"manifest hash mismatch "+key);};
    match_hash("bin_sha256",newsha);match_hash("tokenizer_sha256",th);
    match_hash("tsv_sha256",disk::sha256(bytes(newpath.parent_path()/"train.tsv")));
    match_hash("validation_sha256",disk::sha256(bytes("build/bpe_pilot_validation.bin")));
    match_hash("test_sha256",disk::sha256(bytes("build/bpe_pilot_test.bin")));
    auto number=[&](const std::string& key){auto v=mf(key);require(!v.empty()&&v.find_first_not_of("0123456789")==std::string::npos,"manifest integer "+key);size_t used=0;auto n=std::stoull(v,&used);require(used==v.size(),"manifest integer range");return n;};
    const auto manifest_tokens=number("tokens"),manifest_supervised=number("supervised"),manifest_accepted=number("accepted");
    const auto oldraw=bytes("build/bpe_pilot_train.bin"),oldsha=disk::sha256(oldraw);
    require(newsha!=oldsha,"expanded data equals old dataset");
    const auto source_sha=disk::sha256(bytes("build/yaoyao_graph_step_1200.scp"));
    const auto receipt=bytes(argv[2],32768);
    require(disk::digest(argv[3])&&disk::sha256(receipt)==argv[3],"parent receipt digest must be supplied independently by parent");
    // Independent strict parser: no duplicate, missing, reordered or extra fields.
    // Parent writes ONLY after completed export, independent native grammar/full
    // coverage validation, and composition-tool hash/count verification.
    std::istringstream accepted(receipt);std::string line;
    require(bool(std::getline(accepted,line))&&line=="TAO_REPAIR_CORPUS_ACCEPT_V1","parent receipt magic");
    auto field=[&](const std::string& key){
        require(bool(std::getline(accepted,line)),"parent receipt missing "+key);
        const auto prefix=key+" ";
        require(line.compare(0,prefix.size(),prefix)==0,"parent receipt field "+key);
        auto value=line.substr(prefix.size());require(disk::digest(value),"parent receipt digest "+key);return value;
    };
    const auto accepted_data=field("dataset_sha256");
    const auto accepted_tokenizer=field("tokenizer_sha256");
    const auto accepted_manifest=field("manifest_sha256");
    require(!receipt.empty()&&receipt.back()=='\n'&&accepted.peek()==std::char_traits<char>::eof(),"parent receipt trailing/missing LF");
    require(accepted_data==newsha&&accepted_tokenizer==th&&accepted_manifest==completion_sha,
            "parent receipt corpus/tokenizer/exact manifest bytes mismatch");
    std::istringstream oldin(oldraw),newin(raw);
    tao::data::PilotCursor oldcanonical(tao::data::read_bpe_pilot(oldin,th),4,false);
    tao::data::PilotCursor fresh(tao::data::read_bpe_pilot(newin,th),4,false);
    size_t dataset_targets=0,dataset_positions=0,dataset_tokens=0;
    for(const auto& doc:fresh.docs){require(doc.size()>=2,"short document");dataset_positions+=doc.size()-1;dataset_tokens+=doc.size();
        for(size_t i=0;i<doc.size();++i){require(unsigned(doc[i].id)<Config{}.vocab,"token out of range");if(i)dataset_targets+=doc[i].loss;}}
    require(dataset_targets>0,"empty supervised dataset");
    require(manifest_tokens==dataset_tokens&&manifest_supervised==dataset_targets&&manifest_accepted==fresh.docs.size(),"manifest token/supervised/accepted count mismatch");
    // All acceptance checks precede first GPU allocation. There is no bypass flag.
    stopped();SortedGpuTrainer tr(initialize(Config{},713));SequenceSlots slots(tr,4);
    const auto oldidentity=checkpoint_identity(tr.graph.c,oldsha,th,200)+"arch2-controlled-v1-external-lr\n";
    {auto source=load_slot_file("build/yaoyao_graph_step_1200.scp",oldidentity,capture(slots,oldcanonical),oldcanonical);
     require(source.steps==1200,"source must be real step1200");
     require(disk::sha256(bytes("build/yaoyao_graph_step_1200.scp"))==source_sha,"source changed during load");
     restore(source,slots,oldcanonical);}
    // Explicit domain transition, NOT resume of the old data cursor. Restore above
    // verifies OLD canonical docs and OLD identity before any NEW identity is used.
    // Do not reset master weights, Adam m/v, step, effective projection or scales.
    for(auto* states:{&slots.s,&slots.m})for(auto& slot:*states)for(auto& v:slot)
        check(cudaMemset(v->value.p,0,v->value.n*sizeof(float)));
    check(cudaDeviceSynchronize());require(tr.steps==1200,"step reset forbidden");
    // Tested exhausted-sentinel initialization; begin_next_epoch is never called
    // mid-epoch. Canonical docs are never physically reordered.
    fresh.next=fresh.docs.size();for(auto& c:fresh.slots)c=tao::data::Cursor{};
    const auto identity=checkpoint_identity(tr.graph.c,newsha,th,200)+policy+"\nsource_scp_sha256="+source_sha+"\nparent_receipt_sha256="+argv[3]+"\n";
    const auto verified=identity+"canonical-new-dataset-sha256="+newsha+"\n";
    DeterministicShuffledEpochCursor epoch(std::move(fresh),seed,verified);epoch.begin_next_epoch(true);
    require(epoch.epoch()==1&&epoch.cursor().next==0,"initial epoch reset");
    fixed_validation::Evaluator eval(tr,fixed_validation::Corpus("build/bpe_pilot_validation.bin","build/formal_tokenizer.bbp",fixed_validation::Corpus::fixed_dataset_sha256,th));
    auto validate=[&](){
        require(slots.active==-1,"validation boundary");const auto v=eval.run();
        // Use the Result fields/nll() directly, as the production trainer does.
        printf("FIXED_VALIDATION step=%u backend=%s docs=%zu supervised=%llu positions=%llu NLL=%.12f dataset_sha256=%s tokenizer_sha256=%s\n",
            v.step,v.backend,v.documents,v.totals.supervised,v.totals.positions,v.nll(),v.dataset_sha256.c_str(),v.tokenizer_sha256.c_str());
        fflush(stdout);return v;
    };
    std::ostringstream metrics;metrics.precision(17);
    auto initial=validate();
    metrics<<"INITIAL step="<<tr.steps<<" positions=0 targets=0 train_preupdate_NLL=NA val_NLL="<<initial.nll()<<" dataset_positions="<<dataset_positions<<" dataset_targets="<<dataset_targets<<"\n";
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
        require(targets>0,"no supervised update; abort, no automatic retry");const double train_loss=loss.collect();
        stopped();const float norm=tr.update(targets,.00025f);check(cudaDeviceSynchronize());
        require(tr.steps==before+1&&std::isfinite(norm),"invalid update");auto v=validate();
        std::ostringstream line;line.precision(17);line<<"UPDATE step="<<tr.steps<<" epoch="<<epoch.epoch()<<" positions="<<positions<<" targets="<<targets<<" train_preupdate_NLL="<<train_loss/targets<<" val_NLL="<<v.nll()<<" lr=.00025 norm="<<norm<<"\n";
        metrics<<line.str();printf("%s",line.str().c_str());fflush(stdout);
    }
    stopped();require(tr.steps==1200+updates,"bounded final step");
    const auto commit=shuffled_native::save(out,slots,epoch,seed,verified,identity);
    stopped();const auto dsb=out/"final.dsb",tmp=out/"final.dsb.tmp";
    require(!fs::exists(fs::symlink_status(dsb))&&!fs::exists(fs::symlink_status(tmp)),"quality DSB overwrite");
    CpuModel cpu(tr.graph.c);for(auto& kv:cpu.w)kv.second=tr.graph.w.at(kv.first)->value.host();
    save_bundle(cpu,tmp.string(),th);
    require(!fs::exists(fs::symlink_status(dsb)),"quality DSB appeared");fs::rename(tmp,dsb);
    const auto dsbsha=disk::sha256(bytes(dsb));
    printf("QUALITY_EXPORT path=%s sha256=%s inference_only=1\n",dsb.generic_string().c_str(),dsbsha.c_str());
    disk::publish_text(out/"pilot.metrics",metrics.str());
    disk::publish_text(out/"transition.identity",identity);
    disk::publish_text(out/"verified.identity",verified);
    disk::publish_text(out/"accepted.receipt",receipt);disk::publish_text(out/"export.manifest",completion);
    printf("FINAL step=%u updates=%u resumable_wrapper=1 manifest_sha256=%s directory=%s new_dataset_sha256=%s policy_sha256=%s\n",tr.steps,updates,commit.manifest_sha256.c_str(),out.generic_string().c_str(),newsha.c_str(),disk::sha256(policy).c_str());fflush(stdout);
    return 0;
}catch(const std::exception& e){fprintf(stderr,"EXPANDED_PILOT_FAIL %s; no automatic retry/resume; retain STOP_TRAINING\n",e.what());return 1;}}
