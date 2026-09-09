// Native CUDA correctness test only. Build like train_repair_order_pilot.cu
// (C++17, --default-stream per-thread). Run from the repository root:
//   test_shuffled_native_resume.exe build/UNUSED_PRIVATE_CHECKPOINT_DIRECTORY
// No validation, CPU training, production control changes, or effective-only export.
#define main preserved_shuffled_resume_production_main
#include "train_yaoyao_graph_gpuval.cu"
#undef main
#include "shuffled_native_checkpoint.cuh"
#include <cstring>

namespace resume_test {
using namespace tao::dual;
using tao::data::ShuffledEpochCursor;
void require(bool ok,const std::string& why) { if(!ok) throw std::runtime_error(why); }
void stopped() { require(std::filesystem::is_regular_file("build/STOP_TRAINING"),"production STOP_TRAINING must remain present"); }

// Same layout/semantics as take_batch(PilotCursor), but allocations go through
// ShuffledEpochCursor::take; documents are NEVER physically reordered.
tao::data::BatchPlan take_batch(ShuffledEpochCursor& epoch,size_t width) {
    tao::data::BatchPlan p;p.slots=epoch.cursor().slots.size();
    std::vector<tao::data::Work> work(p.slots);std::vector<bool> valid(p.slots);
    for(size_t k=0;k<p.slots;++k) {
        valid[k]=epoch.take(k,width,work[k]);
        if(valid[k])p.timesteps=std::max(p.timesteps,work[k].end-work[k].begin);
    }
    p.items.resize(p.slots*p.timesteps);
    for(size_t k=0;k<p.slots;++k)if(valid[k]) {
        const auto& w=work[k];const auto& doc=epoch.cursor().docs.at(w.doc);
        for(size_t t=0;t<w.end-w.begin;++t) {
            size_t i=w.begin+t;
            p.items[t*p.slots+k]={unsigned(doc[i-1].id),unsigned(doc[i].id),true,t==0&&w.reset,bool(doc[i].loss)};
            ++p.positions;p.supervised+=doc[i].loss;
        }
    }
    return p;
}
std::string update(SortedGpuTrainer& tr,SequenceSlots& slots,ReusableBatchGraph& replay,ShuffledEpochCursor& epoch) {
    stopped();const unsigned before=tr.steps;size_t supervised=0,positions=0;
    std::ostringstream trace;
    for(unsigned r=0;r<8;++r) {
        auto p=take_batch(epoch,256);
        require(p.timesteps>0,"test corpus exhausted inside eight-accumulation update");
        trace<<p.slots<<' '<<p.timesteps<<' '<<p.positions<<' '<<p.supervised<<'\n';
        for(const auto& x:p.items)trace<<x.input<<' '<<x.target<<' '<<x.active<<' '<<x.reset<<' '<<x.loss<<'\n';
        replay.run(p,slots);supervised+=p.supervised;positions+=p.positions;
    }
    require(supervised>0,"empty supervised update");
    float norm=tr.update(supervised,.00025f);
    check(cudaDeviceSynchronize());
    require(tr.steps==before+1 && std::isfinite(norm) && norm>0,"GPU update did not advance with finite nonzero gradient");
    printf("GPU_UPDATE step=%u slots=4 accumulation=8 width=256 lr=.00025 positions=%zu supervised=%zu norm=%.9g\n",tr.steps,positions,supervised,norm);
    fflush(stdout);stopped();return trace.str();
}
struct Observed {
    SlotSnapshot native;
    std::map<std::string,Vec> effective,scales;
    std::string epoch;
};
Observed observe(SequenceSlots& slots,const ShuffledEpochCursor& epoch) {
    check(cudaDeviceSynchronize());auto canonical=epoch.cursor();Observed out;
    out.native=capture(slots,canonical);
    for(const auto& kv:slots.trainer.graph.w)out.effective.emplace(kv.first,kv.second->value.host());
    for(const auto& kv:slots.trainer.scales)out.scales.emplace(kv.first,kv.second->host());
    out.epoch=epoch.save("exact-comparison-fixed-checkpoint-id");return out;
}
size_t compared_floats=0;
void equal_vec(const Vec& a,const Vec& b,const std::string& label) {
    require(a.size()==b.size(),label+" size");
    for(size_t i=0;i<a.size();++i) {
        if(!std::isfinite(a[i]) || !std::isfinite(b[i]))
            throw std::runtime_error(label+" nonfinite at "+std::to_string(i));
        if(std::memcmp(&a[i],&b[i],sizeof(float))!=0)
            throw std::runtime_error(label+" bit mismatch at "+std::to_string(i));
    }
    compared_floats+=a.size();
}
void equal_map(const std::map<std::string,Vec>& a,const std::map<std::string,Vec>& b,const std::string& label) {
    require(a.size()==b.size(),label+" map size");
    for(const auto& kv:a) {auto it=b.find(kv.first);require(it!=b.end(),label+" missing "+kv.first);equal_vec(kv.second,it->second,label+"/"+kv.first);}
}
void equal(const Observed& a,const Observed& b) {
    const auto& x=a.native;const auto& y=b.native;
    require(x.steps==y.steps && x.next==y.next,"step/next mismatch");
    require(x.cursor.size()==y.cursor.size(),"cursor size");
    for(size_t k=0;k<x.cursor.size();++k)
        require(x.cursor[k].doc==y.cursor[k].doc && x.cursor[k].target==y.cursor[k].target,"cursor slot "+std::to_string(k));
    equal_map(x.w,y.w,"master");equal_map(x.m,y.m,"Adam m");equal_map(x.v,y.v,"Adam v");
    equal_map(a.effective,b.effective,"effective");equal_map(a.scales,b.scales,"projection scales");
    for(unsigned kind=0;kind<2;++kind) {
        const auto& u=kind?x.mem:x.s;const auto& v=kind?y.mem:y.s;
        require(u.size()==v.size(),"state slot count");
        for(size_t k=0;k<u.size();++k) {
            require(u[k].size()==v[k].size(),"state layer count");
            for(size_t l=0;l<u[k].size();++l)
                equal_vec(u[k][l],v[k][l],std::string(kind?"mem":"s")+"/"+std::to_string(k)+"/"+std::to_string(l));
        }
    }
    require(a.epoch==b.epoch,"full canonical cursor/seed/epoch/permutation mismatch");
}
} // namespace resume_test

int main(int argc,char** argv) {try {
    using namespace tao::dual;using namespace resume_test;
    require(argc==2,"usage: test_shuffled_native_resume.exe UNIQUE_NEW_CHECKPOINT_DIRECTORY");stopped();
    auto directory=shuffled_native::detail::directory(argv[1]);
    require(!std::filesystem::exists(std::filesystem::symlink_status(directory)),"output directory already exists");
    std::string th;tao::text::load_tokenizer("build/formal_tokenizer.bbp",th);
    std::ifstream f("build/bpe_pilot_train.bin",std::ios::binary);require(bool(f),"training corpus open");
    std::string raw((std::istreambuf_iterator<char>(f)),{});require(!f.bad(),"training corpus read");
    auto dh=tao::text::sha256(raw);std::istringstream input(raw);
    auto docs=tao::data::read_bpe_pilot(input,th);
    // Exactly one full model on the device; restore never replaces graph nodes.
    SortedGpuTrainer tr(initialize(Config{},713));SequenceSlots slots(tr,4);
    tao::data::PilotCursor canonical(std::move(docs),4,false);
    const auto identity=checkpoint_identity(tr.graph.c,dh,th,200)+"arch2-controlled-v1-external-lr\n";
    {auto expected=capture(slots,canonical);
     auto source=load_slot_file("build/yaoyao_graph_step_1200.scp",identity,expected,canonical);
     require(source.steps==1200,"source checkpoint must be real step 1200");restore(source,slots,canonical);}
    // Deliberate matched fresh-epoch reset, NOT continuation of the step1200
    // dataset cursor. The API supports shuffle only at an exhausted boundary:
    // adopt an empty/exhausted scheduling sentinel, then begin epoch 1 normally.
    canonical.next=canonical.docs.size();for(auto& c:canonical.slots)c=tao::data::Cursor{};
    for(auto* states:{&slots.s,&slots.m})for(auto& slot:*states)for(auto& v:slot)
        check(cudaMemset(v->value.p,0,v->value.n*sizeof(float)));
    check(cudaDeviceSynchronize());
    constexpr uint64_t seed=20260909;
    const std::string verified=identity+"canonical-BPE-no-legacy-EOS-removal;matched-fresh-epoch-v1\n";
    tao::data::ShuffledEpochCursor epoch(std::move(canonical),seed,verified);
    epoch.begin_next_epoch(true);
    require(epoch.epoch()==1 && epoch.cursor().next==0,"initial shuffled epoch reset");
    bool nonidentity=false;for(size_t i=0;i<epoch.order().size();++i)nonidentity|=epoch.order()[i]!=i;
    require(nonidentity,"test needs nonidentity shuffle");
    ReusableBatchGraph replay(tr,4,256);
    update(tr,slots,replay,epoch);require(tr.steps==1201,"save step");
    const auto saved_epoch=epoch.save("exact-comparison-fixed-checkpoint-id");
    const auto commit=shuffled_native::save(directory,slots,epoch,seed,verified,identity);
    printf("SAVED step=1201 manifest_sha256=%s directory=%s\n",commit.manifest_sha256.c_str(),commit.directory.string().c_str());fflush(stdout);
    const auto trace_a=update(tr,slots,replay,epoch);
    const auto a=observe(slots,epoch);require(a.native.steps==1202,"uninterrupted step");
    // Trusted digest is retained from save(), never derived from candidate files.
    shuffled_native::load_and_restore(commit.directory,commit.manifest_sha256,slots,epoch,seed,verified,identity);
    require(tr.steps==1201 && epoch.save("exact-comparison-fixed-checkpoint-id")==saved_epoch,"saved boundary not restored");
    const auto trace_b=update(tr,slots,replay,epoch);
    require(trace_a==trace_b,"replayed token/input/target/reset/loss plans differ");
    const auto b=observe(slots,epoch);equal(a,b);stopped();
    require(compared_floats>0,"vacuous tensor comparison");
    printf("SHUFFLED_NATIVE_RESUME_PASS source=1200 saved=1201 compared_step=1202 GPU_updates_executed=3 exact_float_bits=%zu master_m_v_effective_scales_slot_states_cursor_epoch_order_step_equal=1 same_model_same_reusable_graph=1 validation=0\n",compared_floats);
    return 0;
} catch(const std::exception& e) {
    fprintf(stderr,"SHUFFLED_NATIVE_RESUME_FAIL %s\n",e.what());return 1;
}}
