#pragma once
#ifndef TAO_INPUT_SCALE
#error "Fixed validation requires arch2 TAO_INPUT_SCALE before all model includes"
#endif
#include "dual_state_sorted_trainer.cuh"
#include "gpu_batched_matvec.cuh"
#include "gpu_batch_rms.cuh"
#include "bpe_pilot_reader.hpp"
#include "tokenizer_file.hpp"
#include <sstream>
#include <algorithm>
#include <limits>
#include <cfloat>

// Forward-only: no Tape, GradNode allocations, CE seeds, backward, projection or Adam.
// Call only on the training host thread at an idle optimizer boundary. All launches
// explicitly use stream 0, matching trainer kernels (build --default-stream per-thread).
namespace tao::dual::fixed_validation {
template<class T> struct Buffer {
    T* p=nullptr; size_t n;
    explicit Buffer(size_t count):n(count) { check(cudaMalloc(&p,n*sizeof(T))); }
    ~Buffer(){ if(p) cudaFree(p); }
    Buffer(const Buffer&)=delete; Buffer& operator=(const Buffer&)=delete;
};
struct Totals { double loss; unsigned long long positions, supervised, bad; };
struct Item { unsigned input,target,active,loss; };
static __global__ void embed(const float*w,const Item*items,float*x,int d,int slots,float scale){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<d*slots)x[i]=w[size_t(items[i/d].input)*d+i%d]*scale;
}
static __global__ void plus(float*a,const float*b,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)a[i]+=b[i];}
static __global__ void bias(float*a,const float*b,int n,int slots){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n*slots)a[i]+=b[i%n];}
static __global__ void state_update(float*s,const float*u,const float*g,const Item*items,int n,int slots){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    // 【R4 统一 · doc 24】验证前向必须与训练前向（ds_update）用同一激活，
    // 否则验证 NLL 与训练 NLL 系统性偏离，会把"算子不一致"误读成"过拟合"。
    if(i<n*slots&&items[i/n].active)s[i]+=ds_act_sigmoid(g[i])*(ds_act_tanh(u[i])-s[i]);
}
// Stable double log-sum-exp, one block/document. Never creates or writes gradients.
static __global__ void ce(const float*z,const Item*items,Totals*totals,int vocab){
    int slot=blockIdx.x,t=threadIdx.x; Item a=items[slot];
    if(!a.active)return;
    if(!t)++totals[slot].positions;
    if(!a.loss)return;
    z+=size_t(slot)*vocab;
    __shared__ double red[256]; __shared__ unsigned bad[256];
    double mx=-DBL_MAX; unsigned invalid=0;
    for(int j=t;j<vocab;j+=256){double v=z[j];invalid+=!isfinite(v);mx=fmax(mx,v);}
    red[t]=mx;bad[t]=invalid;__syncthreads();
    for(int k=128;k;k/=2){if(t<k){red[t]=fmax(red[t],red[t+k]);bad[t]+=bad[t+k];}__syncthreads();}
    mx=red[0];double sum=0;
    for(int j=t;j<vocab;j+=256)sum+=exp(double(z[j])-mx);
    red[t]=sum;__syncthreads();
    for(int k=128;k;k/=2){if(t<k)red[t]+=red[t+k];__syncthreads();}
    if(!t){double loss=mx+log(red[0])-double(z[a.target]);
        ++totals[slot].supervised;totals[slot].bad+=bad[0]+!isfinite(loss);totals[slot].loss+=loss;}
}
static __global__ void reduce(const Totals*docs,Totals*out,int slots){
    if(threadIdx.x||blockIdx.x)return;Totals r{};
    for(int i=0;i<slots;++i){r.loss+=docs[i].loss;r.positions+=docs[i].positions;r.supervised+=docs[i].supervised;r.bad+=docs[i].bad;}
    *out=r;
}
struct Corpus {
    static constexpr const char* fixed_dataset_sha256 =
        "69900a8acbb28b09296b13a08c76657aa6261c129e929d59bc83ad9d2f1447b0";
    static constexpr const char* fixed_tokenizer_sha256 =
        "34463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333";
    std::string dataset_sha256,tokenizer_sha256;
    std::vector<std::vector<tao::data::Token>> docs;
    size_t steps=0,tokens=0,positions=0,supervised=0;
    // Dataset pin is mandatory here as well as in the caller. Tokenizer hash must
    // match the restored training identity; TLP2 header independently binds it.
    Corpus(const std::string&dataset,const std::string&tokenizer,
           const std::string&expected_dataset,const std::string&expected_tokenizer){
        tao::text::load_tokenizer(tokenizer,tokenizer_sha256);
        std::ifstream f(dataset,std::ios::binary);
        if(!f)throw std::runtime_error("fixed validation open");
        std::string raw((std::istreambuf_iterator<char>(f)),{});
        if(f.bad())throw std::runtime_error("fixed validation read");
        dataset_sha256=tao::text::sha256(raw);
        if(expected_dataset.size()!=64||expected_tokenizer.size()!=64||
           dataset_sha256!=fixed_dataset_sha256||dataset_sha256!=expected_dataset||
           tokenizer_sha256!=fixed_tokenizer_sha256||tokenizer_sha256!=expected_tokenizer)
            throw std::runtime_error("fixed validation identity mismatch");
        std::istringstream in(raw);docs=tao::data::read_bpe_pilot(in,tokenizer_sha256);
        for(const auto&d:docs){steps=(std::max)(steps,d.size()-1);tokens+=d.size();positions+=d.size()-1;
            for(size_t j=1;j<d.size();++j)supervised+=d[j].loss;}
        if(docs.size()!=23||supervised!=4906)throw std::runtime_error("fixed validation contract: 23 docs / 4906 targets");
    }
};
struct Result {
    Totals totals{}; unsigned step=0;
    std::string dataset_sha256,tokenizer_sha256;
    size_t documents=23,tokens=0,parameter_h2d_bytes=0,hotpath_h2d_bytes=0,d2h_bytes=sizeof(Totals);
    const char* backend="cuda-forward-arch2-effective-resident-v1";
    double nll()const{return totals.loss/double(totals.supervised);}
    static bool due(unsigned optimizer_step){return optimizer_step!=0;}
    static bool lr_decision_due(unsigned optimizer_step){return optimizer_step!=0 && optimizer_step%100==0;}
    // Stable machine-readable schema; caller emits this once after each due update.
    std::string metrics_json()const{
        std::ostringstream out;out.precision(17);
        out << "{\"validation_step\":" << step
            << ",\"docs\":" << documents
            << ",\"supervised\":" << totals.supervised
            << ",\"NLL\":" << nll()
            << ",\"datasetSHA\":\"" << dataset_sha256
            << "\",\"backend\":\"" << backend
            << "\",\"lr_decision_due\":" << (lr_decision_due(step)?"true":"false") << "}";
        return out.str();
    }
};
class Evaluator {
    SortedGpuTrainer&tr; Config c; Corpus corpus;
    static constexpr int slots=23;
    size_t width;
    Buffer<Item> items;
    Buffer<float> states_s,states_m,scratch;
    Buffer<Totals> doc_totals,total;
    std::map<std::string,const float*> weights;
    float* buf(int i){return scratch.p+size_t(i)*width*slots;}
    const float* w(const std::string&name)const{return weights.at(name);}
    void linear(const std::string&name,const float*x,float*y,unsigned rows,unsigned cols){
        ds_batched_matvec<<<dim3((rows+3)/4,slots),128,0,0>>>(w(name),x,y,rows,cols,slots);
    }
    void add(float*a,const float*b,unsigned n){plus<<<(n*slots+127)/128,128,0,0>>>(a,b,n*slots);}
    void norm(const float*x,const std::string&name,float*y,unsigned n){batch_rms_forward<<<slots,256,0,0>>>(x,w(name),y,n);}
    void add_bias(float*x,const std::string&name,unsigned n){bias<<<(n*slots+127)/128,128,0,0>>>(x,w(name),n,slots);}
    void forward(const Item*it){
        float*x=buf(0),*xn=buf(1),*u=buf(2),*g=buf(3),*tmp=buf(4),*r=buf(5),*f=buf(6),*out=buf(7);
        embed<<<(c.d*slots+127)/128,128,0,0>>>(w("embedding"),it,x,c.d,slots,std::sqrt(float(c.d)));
        for(unsigned l=0;l<c.layers;++l){
            std::string p="layer."+std::to_string(l)+".";
            float*s=states_s.p+size_t(l)*slots*c.s,*m=states_m.p+size_t(l)*slots*c.m;
            norm(x,p+"input.norm",xn,c.d);
            auto branch=[&](const std::string&name,bool mem,float*y){
                unsigned n=mem?c.m:c.s;
                linear(p+name+".x",xn,y,n,c.d);
                linear(p+name+".s",s,tmp,n,c.s);add(y,tmp,n);
                if(mem){linear(p+name+".m",m,tmp,n,c.m);add(y,tmp,n);}
                add_bias(y,p+name+".bias",n);
            };
            branch("s.candidate",false,u);branch("s.gate",false,g);
            state_update<<<(c.s*slots+127)/128,128,0,0>>>(s,u,g,it,c.s,slots);
            // Memory branches observe updated s, but both observe the same old m.
            branch("m.candidate",true,u);branch("m.gate",true,g);
            state_update<<<(c.m*slots+127)/128,128,0,0>>>(m,u,g,it,c.m,slots);
            linear(p+"read.s",s,r,c.d,c.s);linear(p+"read.m",m,tmp,c.d,c.m);add(r,tmp,c.d);
            norm(r,p+"read.norm",tmp,c.d);add(x,tmp,c.d);
#ifndef TAO_NO_FFN
            norm(x,p+"ff.norm",tmp,c.d);linear(p+"ff.up",tmp,f,c.e,c.d);
            ds_silu<<<(c.e*slots+127)/128,128,0,0>>>(f,c.e*slots);
            linear(p+"ff.down",f,tmp,c.d,c.e);add(x,tmp,c.d);
#endif
        }
        norm(x,"final.norm",tmp,c.d);linear("embedding",tmp,out,c.vocab,c.d);add_bias(out,"vocab.bias",c.vocab);
        ce<<<slots,256,0,0>>>(out,it,doc_totals.p,c.vocab);
    }
public:
    Evaluator(SortedGpuTrainer&trainer,Corpus fixed):tr(trainer),c(trainer.graph.c),corpus(std::move(fixed)),
        width((std::max)({c.d,c.s,c.m,c.e,c.vocab})),items(corpus.steps*slots),
        states_s(size_t(c.layers)*slots*c.s),states_m(size_t(c.layers)*slots*c.m),
        scratch(8*width*slots),doc_totals(slots),total(1){
        if(!width||width>size_t(std::numeric_limits<int>::max()/slots))throw std::runtime_error("validation dimensions");
        for(const auto&t:schema(c)){
            const auto&node=tr.graph.w.at(t.name);
            if(node->value.n!=t.elements())throw std::runtime_error("validation parameter shape");
            weights.emplace(t.name,node->value.p); // NEVER master or repacked/reprojected copies.
        }
        std::vector<Item> plan(items.n);
        for(size_t k=0;k<corpus.docs.size();++k){const auto&d=corpus.docs[k];
            for(size_t t=0;t+1<d.size();++t){
                if(unsigned(d[t].id)>=c.vocab||unsigned(d[t+1].id)>=c.vocab)throw std::runtime_error("validation token range");
                plan[t*slots+k]={unsigned(d[t].id),unsigned(d[t+1].id),1,unsigned(d[t+1].loss)};
            }
        }
        check(cudaMemcpy(items.p,plan.data(),items.n*sizeof(Item),cudaMemcpyHostToDevice));
    }
    Result run(){
        // External lifecycle contract: no replay/update on another thread/stream.
        if(!tr.graph.tape.reverse.empty())throw std::runtime_error("validation requires detached boundary");
        for(const auto&kv:weights)if(tr.graph.w.at(kv.first)->value.p!=kv.second)
            throw std::runtime_error("validation effective pointers replaced: rebuild evaluator");
        check(cudaStreamSynchronize(0));
        check(cudaMemsetAsync(states_s.p,0,states_s.n*sizeof(float),0));
        check(cudaMemsetAsync(states_m.p,0,states_m.n*sizeof(float),0));
        check(cudaMemsetAsync(doc_totals.p,0,doc_totals.n*sizeof(Totals),0));
        for(size_t t=0;t<corpus.steps;++t)forward(items.p+t*slots);
        reduce<<<1,1,0,0>>>(doc_totals.p,total.p,slots);check(cudaGetLastError());
        Result result;result.step=tr.steps;result.tokens=corpus.tokens;
        result.dataset_sha256=corpus.dataset_sha256;result.tokenizer_sha256=corpus.tokenizer_sha256;
        check(cudaMemcpyAsync(&result.totals,total.p,sizeof(Totals),cudaMemcpyDeviceToHost,0));
        check(cudaStreamSynchronize(0));
        if(result.totals.bad||!std::isfinite(result.totals.loss)||result.totals.supervised!=4906||
           result.totals.positions!=corpus.positions)throw std::runtime_error("invalid GPU fixed validation totals");
        return result;
    }
};
} // namespace tao::dual::fixed_validation
