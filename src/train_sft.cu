#define TAO_NO_FFN
// SFT（冷启动）训练器 —— 标准流程第 ⑤ 步，对应 R1 的第一阶段。
//
// 为什么必须有这一步：RL 无法让模型从零学会一种输出格式。必须先由 SFT
// 用示范教会它 "思考 <推理>答案：<X>" 的两段式，GRPO 才能在这个格式内搜索。
//
// 与 GRPO 训练器的唯一区别在数据侧：
//   - GRPO 读采样轨迹，按组内优势加权（可正可负）
//   - SFT  读示范语料，权重恒为 1，监督范围是 ASSISTANT..TURN_END 整段
// 训练器本体完全相同（同一个 SortedGpuTrainer 与加权 CE 通路）。
//
// 用法: train_sft REASON.txt TOKENIZER.bbp OUT_DIR UPDATES RESUME_DIR [SLOTS WIDTH] [LR]
#define main unused_probe_graph_main
#include "train_yaoyao_graph_gpuval.cu"
#undef main
#include "shuffled_epoch_batch_plan.hpp"
#include "dual_model_bundle.hpp"
#include "plain_lm_format.hpp"
#include "tokenizer_file.hpp"
#include <cmath>
#include <fstream>
#include <sstream>
#include <thread>

namespace sft {
using namespace tao::dual;
using tao::data::Token;
namespace fs = std::filesystem;
void require(bool ok,const std::string&why){ if(!ok) throw std::runtime_error(why); }

// 数据加载的并行度：默认取全部逻辑核，TAO_CPU_THREADS 可覆盖。
// 分词是纯 CPU 且占加载时间的绝大部分，必须多核，否则 20 万篇要串行很久。
inline unsigned load_threads(){
    unsigned hw=std::thread::hardware_concurrency(); if(hw==0u)hw=4u;
    if(const char*e=std::getenv("TAO_CPU_THREADS")){int v=std::atoi(e); if(v>0)hw=unsigned(v);}
    return hw;
}
template<class F>
inline void parallel_for(size_t n,unsigned threads,F&&f){
    if(n==0)return;
    if(threads<=1u||n<2u){for(size_t i=0;i<n;++i)f(i);return;}
    if(size_t(threads)>n)threads=unsigned(n);
    std::vector<std::thread> ts; ts.reserve(threads);
    std::vector<std::exception_ptr> errs(threads,nullptr);
    for(unsigned w=0;w<threads;++w) ts.emplace_back([&,w](){ try{ for(size_t i=w;i<n;i+=threads)f(i);}catch(...){errs[w]=std::current_exception();} });
    for(auto&t:ts)t.join();
    for(auto&e:errs) if(e)std::rethrow_exception(e);
}

// 读推理语料：DOC 分隔，每篇含 "U <问题>" 与 "A <回答>" 两行。
// 文档结构 [BOS, USER, q..., TURN_END, ASSISTANT, a..., TURN_END]
// 只监督 ASSISTANT 之后的回答段（问题段 loss=false）。
void load_reason(const fs::path&p,const tao::text::ByteBpe&tok,
                 std::vector<std::vector<Token>>&docs,std::vector<float>&w,size_t&sup,size_t&bad){
    std::ifstream f(p,std::ios::binary);
    require(bool(f),"reason open: "+p.string());
    std::stringstream ss; ss<<f.rdbuf();
    const std::string all=ss.str();

    // 第一步：切分 (问题, 回答) 对。I/O 与切分是单遍字符串扫描，很快。
    struct Pair{ std::string q,a; };
    std::vector<Pair> pairs;
    {
        size_t i=0; std::string cq,ca;
        auto flush=[&](){ if(!cq.empty()&&!ca.empty())pairs.push_back({std::move(cq),std::move(ca)}); cq.clear(); ca.clear(); };
        while(i<all.size()){
            size_t e=all.find('\n',i); if(e==std::string::npos)e=all.size();
            std::string line=all.substr(i,e-i);
            if(!line.empty()&&line.back()=='\r')line.pop_back();
            i=e+1;
            if(line=="DOC"){ flush(); continue; }
            if(line.size()>2&&line[0]=='U'&&line[1]==' ')cq=line.substr(2);
            else if(line.size()>2&&line[0]=='A'&&line[1]==' ')ca=line.substr(2);
        }
        flush();
    }

    // 第二步：并行分词（这是加载的瓶颈）。
    const unsigned nt=load_threads();
    std::vector<std::vector<Token>> built(pairs.size());
    std::vector<unsigned> suplen(pairs.size(),0u);
    std::vector<char> ok(pairs.size(),0);
    parallel_for(pairs.size(),nt,[&](size_t k){
        auto qi=tok.encode(pairs[k].q), ai=tok.encode(pairs[k].a);
        if(qi.empty()||ai.empty())return;
        std::vector<Token> d;
        d.reserve(qi.size()+ai.size()+4);
        d.push_back({tao::data::BOS,false});
        d.push_back({tao::data::USER,false});
        for(auto t:qi)d.push_back({int(t),false});
        d.push_back({tao::data::TURN_END,false});
        d.push_back({tao::data::ASSISTANT,false});
        for(auto t:ai)d.push_back({int(t),true});
        d.push_back({tao::data::TURN_END,true});
        suplen[k]=unsigned(ai.size()+1);
        built[k]=std::move(d);
        ok[k]=1;
    });

    // 第三步：按原顺序归并（顺序影响打乱前的文档序，必须稳定）。
    for(size_t k=0;k<pairs.size();++k){
        if(!ok[k]){ ++bad; continue; }
        sup+=suplen[k];
        docs.push_back(std::move(built[k]));
        w.push_back(1.0f);
    }
}
}

int main(int argc,char**argv){
    try{
        using namespace sft;
        require(argc>=6&&argc<=9,
            "usage: train_sft REASON.txt TOKENIZER.bbp OUT_DIR UPDATES RESUME_DIR [SLOTS WIDTH] [LR]");
        const fs::path reason=argv[1];
        const std::string tokpath=argv[2],out_s=argv[3];
        const unsigned updates=unsigned(std::stoul(argv[4]));
        const fs::path resume=argv[5];
        const unsigned slot_count=argc>=7?unsigned(std::stoul(argv[6])):16u;
        const unsigned graph_width=argc>=7?unsigned(std::stoul(argv[7])):256u;
        float lr=argc>=9?float(std::atof(argv[8])):5e-5f;
        require(slot_count>=1&&slot_count<=256,"slots 1..256");
        require(graph_width>=1&&graph_width<=4096,"width 1..4096");
        require(lr>0.f&&lr<=0.1f,"lr range");
        const fs::path out=out_s;
        bool resume_used=true;

        std::string th;
        auto tok=tao::text::load_tokenizer(tokpath,th);
        printf("SFT_TOKENIZER %s merges=%zu\n",tokpath.c_str(),tok.merges.size());
        fflush(stdout);

        std::vector<std::vector<Token>> docs; std::vector<float> weights;
        size_t supervised=0,bad=0;
        load_reason(reason,tok,docs,weights,supervised,bad);
        printf("SFT_DATA docs=%zu supervised=%zu skipped=%zu threads=%u\n",docs.size(),supervised,bad,load_threads());
        fflush(stdout);
        require(docs.size()>=size_t(slot_count),"need at least one doc per slot");
        if(std::getenv("TAO_SFT_DRYRUN")){ printf("SFT_DRYRUN_OK\n"); fflush(stdout); return 0; }

        Config cfg{};
        {
            auto ov=[&](const char*n,uint32_t&dst){ if(const char*v=std::getenv(n)){unsigned long x=std::strtoul(v,nullptr,10);
                require(x>=1&&x<=262144,"config override range");dst=uint32_t(x);} };
            ov("TAO_CFG_S",cfg.s); ov("TAO_CFG_D",cfg.d); ov("TAO_CFG_M",cfg.m);
            ov("TAO_CFG_E",cfg.e); ov("TAO_CFG_LAYERS",cfg.layers);
            ov("TAO_CFG_DK",cfg.dk); ov("TAO_CFG_VOCAB",cfg.vocab);
            cfg.validate();
            printf("CONFIG layers=%u d=%u s=%u m=%u e=%u vocab=%u dk=%u\n",cfg.layers,cfg.d,cfg.s,cfg.m,cfg.e,cfg.vocab,cfg.dk);
            fflush(stdout);
        }
        SortedGpuTrainer tr(initialize(cfg,20260912));
        // 从零训练：仅当 resume 目录确实有与本模型兼容的优化器状态时才载入。
        // 架构变更后旧 opt_state 尺寸不符，强行载入会失败或污染训练。
        if(const char* f=std::getenv("TAO_FRESH"); f && *f=='1'){resume_used=false;}
        if(resume_used){
            const fs::path op=resume/"opt_state.bin";
            if(fs::exists(op)){ tr.load_state(op.string()); }
            else { printf("SFT_NORESUME %s 无 opt_state.bin，从零开始\n",resume.generic_string().c_str()); }
        } else {
            printf("SFT_FRESH TAO_FRESH=1: DISCARD checkpoint (weights AND optimizer), train from random init seed 20260912\n");
        }
        printf("RESUMED steps=%u lr=%g slots=%u width=%u updates=%u\n",tr.steps,double(lr),slot_count,graph_width,updates);
        fflush(stdout);
        SequenceSlots slots(tr,slot_count);
        ReusableBatchGraph replay(tr,slot_count,graph_width);
        DeferredLoss loss(1024);

        tao::data::PilotCursor base(docs,slot_count,false);
        base.doc_weights=weights;
        const std::string identity=std::string("sft-v1;tok=")+th+"\n";
        tao::data::ShuffledEpochCursor epoch(std::move(base),20470101ull,identity);
        printf("SFT_START\n"); fflush(stdout);

        const bool save_opt=!std::getenv("TAO_OPT_STATE")||std::string(std::getenv("TAO_OPT_STATE"))!="0";
        auto export_ckpt=[&](bool finish){
            if(!fs::exists(fs::symlink_status(out)))require(fs::create_directory(out),"out create");
            const unsigned step=tr.steps;
            auto ckpt=out/("step_"+std::to_string(step));
            require(!fs::exists(fs::symlink_status(ckpt)),"checkpoint overwrite");
            require(fs::create_directory(ckpt),"checkpoint create");
            const fs::path staged=ckpt/"final.dsb.tmp";
            CpuModel cpu(tr.graph.c);
            for(auto&kv:cpu.w)kv.second=tr.graph.w.at(kv.first)->value.host();
            save_bundle(cpu,staged.string(),th);
            fs::rename(staged,ckpt/"final.dsb");
            if(save_opt)tr.save_state((out/"opt_state.bin").string());
            printf("CHECKPOINT step=%u path=%s\n",step,ckpt.generic_string().c_str()); fflush(stdout);
            if(finish){
                const auto root=out/"final.dsb";
                require(!fs::exists(fs::symlink_status(root)),"root DSB overwrite");
                fs::copy_file(ckpt/"final.dsb",root);
                printf("FINAL step=%u path=%s\n",tr.steps,root.generic_string().c_str()); fflush(stdout);
            }
        };

        for(unsigned u=0;u<updates;++u){
            size_t positions=0,targets=0;
            const auto before=tr.steps;
            for(unsigned r=0;r<8;++r){
                if(epoch.exhausted()){ check(cudaStreamSynchronize(0)); epoch.begin_next_epoch(true); }
                auto p=tao::data::take_batch(epoch,graph_width);
                require(p.timesteps>0,"no work after rollover");
                replay.run(p,slots);
                accumulate_loss<<<1,1>>>(replay.data.loss.p,replay.data.loss.n,loss.total,loss.bad);
                check(cudaGetLastError());
                positions+=p.positions; targets+=p.supervised;
            }
            require(targets>0,"no supervised update");
            const double wloss=loss.collect();
                    // LR 调度：常数 LR 会让末段在最优解附近震荡而无法收敛。
        // warmup 线性升到峰值，再余弦退火到 TAO_LR_MIN（默认峰值的 10%）。
        unsigned warmup=20u;
        if(const char* e=std::getenv("TAO_WARMUP")){int v=std::atoi(e); if(v>=1) warmup=(unsigned)v;}
        float lr_min=lr*0.1f;
        if(const char* e=std::getenv("TAO_LR_MIN")){float v=float(std::atof(e)); if(v>0.f) lr_min=v;}
        auto lr_sched=[&](unsigned s)->float{
            if(s<warmup) return lr*float(s+1u)/float(warmup);
            const unsigned total=(updates>warmup)?updates:warmup;
            double t=double(s-warmup)/double(total-warmup);
            if(t>1.0) t=1.0;
            return lr_min+(lr-lr_min)*float(0.5*(1.0+std::cos(3.14159265358979323846*t)));
        };
        const float lr_now=lr_sched(tr.steps);
        const float norm=tr.update(targets,lr_now);
            check(cudaDeviceSynchronize());
            require(tr.steps==before+1&&std::isfinite(norm),"invalid update");
            printf("SFT_UPDATE step=%u positions=%zu targets=%zu loss=%.6f lr=%g norm=%.6f\n",
                tr.steps,positions,targets,wloss/double(targets),double(lr_now),norm);
            fflush(stdout);
            if(tr.steps%10u==0)export_ckpt(false);
        }
        export_ckpt(true);
        require(fs::is_regular_file(out/"final.dsb"),"missing root DSB");
        return 0;
    }catch(const std::exception&e){ std::fprintf(stderr,"SFT_FAIL %s\n",e.what()); return 1; }
}
