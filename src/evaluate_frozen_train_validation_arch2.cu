#define TAO_ASYNC_D2D
#define TAO_DEFER_BACKWARD_SYNC
#define TAO_GPU_HEALTH
#define TAO_ASYNC_ALLOC
#define TAO_WARP_MATVEC
#define TAO_PARALLEL_RMS
#define TAO_TILED_DX
#define TAO_INPUT_SCALE
#define NOMINMAX
#define TAO_DEVICE_ZERO_GRAD
#include "gpu_frozen_diagnostic_arch2.cuh"
#include "dual_model_bundle.hpp"
#include <numeric>
#include <cstring>
#include <cstdio>
#include <chrono>
using namespace tao::dual;
namespace dv=tao::dual::diagnostic_validation;
static std::string bytes(const std::string&p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("open: "+p);std::string s((std::istreambuf_iterator<char>(f)),{});if(f.bad())throw std::runtime_error("read: "+p);return s;}
static void fresh(const std::string&p,const std::string&s){if(std::filesystem::exists(p))throw std::runtime_error("refuse output overwrite: "+p);std::ofstream f(p,std::ios::binary);f<<s;f.close();if(!f)throw std::runtime_error("write: "+p);}
int main(int argc,char**argv){try{
    if(argc!=6||std::string(argv[1])!="--diagnostic")throw std::runtime_error("usage: --diagnostic checkpoint1200.dsb EXPECTED_CHECKPOINT_SHA256 EXPECTED_TRAIN_SHA256 NEW_OUTPUT_PREFIX (run from D:/TaoVm)");
    if(!std::filesystem::exists("build/STOP_TRAINING"))throw std::runtime_error("STOP_TRAINING must remain present");
    const std::string model=argv[2],prefix=argv[5];
    for(auto suffix:{".manifest.tsv",".results.tsv"})if(std::filesystem::exists(prefix+suffix))throw std::runtime_error("output exists");
    auto checkpoint_sha=tao::text::sha256(bytes(model));if(checkpoint_sha!=argv[3])throw std::runtime_error("checkpoint SHA mismatch");
    std::string th;tao::text::load_tokenizer("build/formal_tokenizer.bbp",th);
    if(th!=dv::Corpus::fixed_tokenizer_sha256)throw std::runtime_error("tokenizer pin mismatch");
    auto train_raw=bytes("build/bpe_pilot_train.bin"),val_raw=bytes("build/bpe_pilot_validation.bin");
    auto train_sha=tao::text::sha256(train_raw),val_sha=tao::text::sha256(val_raw);
    if(train_sha!=argv[4]||val_sha!=dv::Corpus::fixed_dataset_sha256)throw std::runtime_error("dataset SHA mismatch");
    std::istringstream ti(train_raw),vi(val_raw);
    auto train=tao::data::read_bpe_pilot(ti,th),val=tao::data::read_bpe_pilot(vi,th);
    if(train.size()<23||val.size()!=23)throw std::runtime_error("corpus sizes");
    // Sort ALL training documents by complete token length, tie by original zero-based ID.
    // Divide ranks into 23 disjoint equal-population strata; select each stratum midpoint.
    // Selection is completely model/loss independent; no document truncation.
    std::vector<size_t> rank(train.size());std::iota(rank.begin(),rank.end(),0);
    std::sort(rank.begin(),rank.end(),[&](size_t a,size_t b){return train[a].size()!=train[b].size()?train[a].size()<train[b].size():a<b;});
    std::vector<size_t> ids,vids(23);std::iota(vids.begin(),vids.end(),0);
    std::vector<std::vector<tao::data::Token>> selected;
    std::ostringstream manifest;
    manifest<<"format\tfrozen-arch2-diagnostic-v1\ncheckpoint_sha256\t"<<checkpoint_sha<<"\ncheckpoint_step_label\t1200 (caller provenance; DSB does not encode step)\ntokenizer_sha256\t"<<th<<"\ntrain_dataset_sha256\t"<<train_sha<<"\nvalidation_dataset_sha256\t"<<val_sha<<"\nselection\t23_equal_population_length_rank_strata_midpoint_tie_doc_id\ntraining_corpus_docs\t"<<train.size()<<"\nset\tslot\tdoc_id_zero_based\ttokens\tstratum_rank_begin\tstratum_rank_end_exclusive\tselected_rank\tstratum_min_tokens\tstratum_max_tokens\n";
    for(size_t k=0;k<23;++k){size_t lo=k*rank.size()/23,hi=(k+1)*rank.size()/23,r=lo+(hi-lo-1)/2,id=rank[r];ids.push_back(id);selected.push_back(train[id]);
        manifest<<"train_sample\t"<<k<<'\t'<<id<<'\t'<<train[id].size()<<'\t'<<lo<<'\t'<<hi<<'\t'<<r<<'\t'<<train[rank[lo]].size()<<'\t'<<train[rank[hi-1]].size()<<'\n';}
    for(size_t k=0;k<23;++k)manifest<<"full_validation\t"<<k<<'\t'<<k<<'\t'<<val[k].size()<<"\tNA\tNA\tNA\tNA\tNA\n";
    dv::Corpus tc(std::move(selected),ids,train_sha,th),vc(std::move(val),vids,val_sha,th);
    if(vc.supervised!=4906)throw std::runtime_error("official validation targets");
    // Persist selection and input identities BEFORE any model inference.
    fresh(prefix+".manifest.tsv",manifest.str());
    auto cpu=load_bundle(model,th);
    SortedGpuTrainer tr(cpu); // Constructor projects; overwrite every graph value BELOW.
    // Restore decoded DSB effective float bytes after constructor projection. Never project again.
    for(const auto&kv:cpu.w){auto&v=tr.graph.w.at(kv.first)->value;if(v.n!=kv.second.size())throw std::runtime_error("shape");
        for(float f:kv.second)if(!std::isfinite(f))throw std::runtime_error("nonfinite DSB weight");
        check(cudaMemcpy(v.p,kv.second.data(),v.n*sizeof(float),cudaMemcpyHostToDevice));}
    tr.steps=1200; // Reporting label only, NOT restored optimizer state.
    auto verify=[&](){for(const auto&kv:cpu.w){auto h=tr.graph.w.at(kv.first)->value.host();if(h.size()!=kv.second.size()||std::memcmp(h.data(),kv.second.data(),h.size()*sizeof(float)))throw std::runtime_error("DSB effective bytes changed");}
        if(tr.steps!=1200||!tr.graph.tape.reverse.empty())throw std::runtime_error("frozen lifecycle");};
    verify();
    std::ostringstream result;result.precision(17);
    result<<"set\tdoc_id_zero_based\ttokens\tpositions\tsupervised\tloss_sum\tNLL\n";
    auto evaluate=[&](const char*label,const dv::Corpus&corpus){dv::Evaluator evaluator(tr,corpus);auto start=std::chrono::steady_clock::now();auto r=evaluator.run();
        for(size_t k=0;k<23;++k){const auto&t=r.per_doc[k];result<<label<<'\t'<<corpus.doc_ids[k]<<'\t'<<corpus.docs[k].size()<<'\t'<<t.positions<<'\t'<<t.supervised<<'\t'<<t.loss<<'\t';if(t.supervised)result<<t.loss/double(t.supervised);else result<<"NA";result<<'\n';}
        result<<label<<"\tALL\t"<<r.tokens<<'\t'<<r.totals.positions<<'\t'<<r.totals.supervised<<'\t'<<r.totals.loss<<'\t'<<r.nll()<<'\n';
        printf("DIAGNOSTIC set=%s step_label=1200 docs=23 supervised=%llu NLL=%.12f seconds=%.6f\n",label,r.totals.supervised,r.nll(),std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());fflush(stdout);verify();return r.nll();};
    double tn=evaluate("train_sample",tc),vn=evaluate("full_validation",vc);
    if(checkpoint_sha!=tao::text::sha256(bytes(model))||!std::filesystem::exists("build/STOP_TRAINING"))throw std::runtime_error("input checkpoint or STOP changed");
    fresh(prefix+".results.tsv",result.str());
    printf("DIAGNOSTIC_OK train_sample_NLL=%.12f full_validation_NLL=%.12f validation_minus_train=%.12f effective_bytes_equal=1 no_updates=1 no_checkpoint_writes=1\n",tn,vn,vn-tn);return 0;
}catch(const std::exception&e){fprintf(stderr,"DIAGNOSTIC_FAIL %s\n",e.what());return 1;}}
