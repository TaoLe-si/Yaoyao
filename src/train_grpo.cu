#define TAO_NO_FFN
// GRPO 训练器：实现论文 Eq.3 的完整目标（裁剪代理 + KL 惩罚 + 逐序列归一化）。
//
//   J = E[ (1/G) Σ_i (1/|o_i|) Σ_t min(r_t·A_i, clip(r_t,1-ε,1+ε)·A_i) - β·KL(π_θ‖π_ref) ]
//   r_t = π_θ(o_t|q,o_<t) / π_old(o_t|q,o_<t)
//
// 三项缺一不可：
//   ① 重要性比值 r 与裁剪 —— 限制单步更新幅度，缺了就退化为朴素加权 CE（易被
//      个别样本带偏，且不再是无偏的策略梯度）。
//   ② KL(π_θ‖π_ref) —— 防止策略偏离参考策略过远而崩溃。用 k3 无偏估计
//      (e^d - d - 1)，d = lp_ref - lp_θ；只保留其梯度项 (e^d - 1)。
//   ③ 逐序列 1/|o_i| —— 每个序列按自身 token 数取平均，长序列不因更长而主导。
//
// 整套目标在实现上归约为**逐 token 乘数** c_t（见 gpu_batch_loss.cuh 的推导），
// 故仍复用同一套前向/反向通路，无需为 RL 另写反向核。
// π_ref 取法：论文 Algorithm 1 每轮令 π_ref ← π_θ。本实现每批只做一次更新（μ=1），
// 采样策略即参考策略，故 d = lp_old - lp_θ，无需额外前向。
//
// 组内方差为 0 的题（所有采样奖励相同）→ 优势为 0 → 该组不产生任何梯度，
// 这正是 GRPO 的定义：没有相对好坏就没有学习信号。
//
// 用法: train_grpo TRAJ.tsv TOKENIZER.bbp OUT_DIR UPDATES RESUME_DIR [SLOTS WIDTH] [LR]
#define main unused_probe_graph_main
#include "train_yaoyao_graph_gpuval.cu"
#undef main
#include "shuffled_epoch_batch_plan.hpp"
#include "dual_model_bundle.hpp"
#include "plain_lm_format.hpp"
#include <cmath>
#include <fstream>
#include <sstream>

namespace grpo {
using namespace tao::dual;
using tao::data::Token;
namespace fs = std::filesystem;
void require(bool ok,const std::string&why){ if(!ok) throw std::runtime_error(why); }

struct Sample { size_t qid=0; double reward=0; bool hit_eos=false;
                std::vector<uint32_t> prompt, gen; std::vector<float> lp, lpr; double adv=0; };

std::vector<uint32_t> parse_ids(const std::string&s){
    std::vector<uint32_t> v; if(s.empty())return v;
    size_t i=0;
    while(i<s.size()){
        size_t j=s.find(',',i); if(j==std::string::npos)j=s.size();
        if(j>i)v.push_back((uint32_t)std::strtoul(s.substr(i,j-i).c_str(),nullptr,10));
        i=j+1;
    }
    return v;
}
// 逐 token 旧策略对数概率（rollout 侧导出）。第 8 个字段，逗号分隔。
std::vector<float> parse_floats(const std::string&s){
    std::vector<float> v; if(s.empty())return v;
    size_t i=0;
    while(i<s.size()){
        size_t j=s.find(',',i); if(j==std::string::npos)j=s.size();
        if(j>i)v.push_back(float(std::atof(s.substr(i,j-i).c_str())));
        i=j+1;
    }
    return v;
}
std::vector<std::string> split_tab(const std::string&s){
    std::vector<std::string> v; size_t i=0;
    while(i<=s.size()){ size_t j=s.find('\t',i); if(j==std::string::npos)j=s.size();
        v.push_back(s.substr(i,j-i)); i=j+1; if(j==s.size())break; }
    return v;
}
}

int main(int argc,char**argv){
    try{
        using namespace grpo;
        require(argc>=6 && argc<=9,
            "usage: train_grpo TRAJ.tsv TOKENIZER.bbp OUT_DIR UPDATES RESUME_DIR [SLOTS WIDTH] [LR]");
        const fs::path traj=argv[1];
        const std::string tokpath=argv[2], out_s=argv[3];
        const unsigned updates=unsigned(std::stoul(argv[4]));
        const fs::path resume=argv[5];
        const unsigned slot_count=argc>=7?unsigned(std::stoul(argv[6])):16u;
        const unsigned graph_width=argc>=7?unsigned(std::stoul(argv[7])):256u;
        float lr=argc>=9?float(std::atof(argv[8])):2e-5f;
        if(const char*e=std::getenv("TAO_LR")) lr=float(std::atof(e));
        require(lr>1e-8f&&lr<=0.1f,"lr range");
        require(slot_count>=1&&slot_count<=256,"slots 1..256");
        // width 下界放宽到 1（与 train_sft.cu 一致）：GEMM batch = slots，width 只是
        // 串行时间步；显存正比 slots*width，故「大 slots + 小 width」更划算。
        require(graph_width>=1&&graph_width<=4096,"width 1..4096");

        const fs::path out=out_s;
        require(!fs::exists(fs::symlink_status(out)),"output directory exists; never overwrite");
        require(fs::is_regular_file(resume/"opt_state.bin"),"resume: opt_state.bin missing");
        if(!std::freopen((out_s+".log").c_str(),"a",stdout))throw std::runtime_error("log open");
        if(!std::freopen((out_s+".log").c_str(),"a",stderr))throw std::runtime_error("log open");
        setvbuf(stdout,nullptr,_IONBF,0); setvbuf(stderr,nullptr,_IONBF,0);

        std::string th; tao::text::load_tokenizer(tokpath,th);
        if(!std::getenv("TAO_ALLOW_TOKENIZER"))
            require(th=="18b1c761bbc13d7f29ff99db86b3eb17502d7ba2afe45c549fc4381fffc037fa","frozen tokenizer mismatch");

        // ---- 读轨迹 ----
        std::ifstream tf(traj);
        require(bool(tf),"traj open: "+traj.string());
        std::vector<Sample> S; std::string line;
        while(std::getline(tf,line)){
            if(line.empty())continue;
            auto f=split_tab(line);
            require(f.size()>=7,"traj field count (need 7, got "+std::to_string(f.size())+")");
            Sample s;
            s.qid=(size_t)std::strtoull(f[0].c_str(),nullptr,10);
            s.reward=std::atof(f[2].c_str());
            s.hit_eos=(std::atoi(f[4].c_str())!=0);
            s.prompt=parse_ids(f[5]); s.gen=parse_ids(f[6]); if(f.size()>=8) s.lp=parse_floats(f[7]);
            // 第 9 字段：逐 token 参考策略对数概率（冻结参考模型）。KL 项需要它。
            if(f.size()>=9) s.lpr=parse_floats(f[8]);
            if(!s.prompt.empty()&&!s.gen.empty())S.push_back(std::move(s));
        }
        require(!S.empty(),"no usable trajectory");
        // ---- 组内归一化优势 ----
        size_t ngroups=0, nvar=0, ngrad=0;
        {
            std::vector<size_t> idx(S.size());
            for(size_t i=0;i<idx.size();++i)idx[i]=i;
            std::sort(idx.begin(),idx.end(),[&](size_t a,size_t b){return S[a].qid<S[b].qid;});
            size_t i=0;
            while(i<idx.size()){
                size_t j=i; while(j<idx.size()&&S[idx[j]].qid==S[idx[i]].qid)++j;
                const size_t n=j-i; ++ngroups;
                double mu=0; for(size_t k=i;k<j;++k)mu+=S[idx[k]].reward; mu/=double(n);
                double var=0; for(size_t k=i;k<j;++k)var+=(S[idx[k]].reward-mu)*(S[idx[k]].reward-mu); var/=double(n);
                const double sd=std::sqrt(var);
                if(sd>1e-9)++nvar;
                for(size_t k=i;k<j;++k){
                    S[idx[k]].adv = (sd>1e-9) ? (S[idx[k]].reward-mu)/sd : 0.0;
                    if(S[idx[k]].adv!=0.0)++ngrad;
                }
                i=j;
            }
        }
        // ---- 构造文档：[BOS, USER, prompt..., TURN_END, ASSISTANT, gen..., TURN_END] ----
        std::vector<std::vector<Token>> docs; docs.reserve(S.size());
        // doc_lp 与 docs 逐位置对齐：该位置 target token 在**旧策略**下的对数概率。
        // 论文 Eq.3 的重要性比值 r = exp(lp_θ - lp_old) 需要它，缺了它就无法算裁剪项。
        std::vector<std::vector<float>> doc_lp; doc_lp.reserve(S.size());
        // doc_lpref 同样逐位置对齐：该位置 target token 在**冻结参考策略**下的对数概率。
        // 论文 Eq.3 的 KL(π_θ‖π_ref) 需要它。若参考模型未提供（全 0），内核退化为
        // π_ref=π_old，KL 梯度恒为 0 —— 这是必须显式暴露出来的降级，不能静默。
        std::vector<std::vector<float>> doc_lpref; doc_lpref.reserve(S.size());
        std::vector<float> weights; weights.reserve(S.size());
        size_t supervised=0;
        for(auto& s:S){
            std::vector<Token> d; std::vector<float> lpv, lprv;
            d.push_back({tao::data::BOS,false});       lpv.push_back(0.f); lprv.push_back(0.f);
            d.push_back({tao::data::USER,false});      lpv.push_back(0.f); lprv.push_back(0.f);
            for(auto t:s.prompt){d.push_back({int(t),false}); lpv.push_back(0.f); lprv.push_back(0.f);}
            d.push_back({tao::data::TURN_END,false});  lpv.push_back(0.f); lprv.push_back(0.f);
            d.push_back({tao::data::ASSISTANT,false}); lpv.push_back(0.f); lprv.push_back(0.f);
            for(size_t g=0;g<s.gen.size();++g){
                d.push_back({int(s.gen[g]),true});
                lpv.push_back(g<s.lp.size()?s.lp[g]:0.f);
                lprv.push_back(g<s.lpr.size()?s.lpr[g]:0.f);
            }
            // 末尾 TURN_END：只有在拿到它的旧策略对数概率时才监督。
            // 拿不到（生成被 max 截断，未产生终止符）时标 loss=false —— 比编造一个
            // lp_old 更诚实：编造会让 r 失真，进而污染整个组的梯度。
            const bool haveTerm = s.lp.size() > s.gen.size();
            d.push_back({tao::data::TURN_END, haveTerm});
            lpv.push_back(haveTerm ? s.lp[s.gen.size()] : 0.f);
            lprv.push_back(haveTerm && s.lpr.size()>s.gen.size() ? s.lpr[s.gen.size()] : 0.f);
            if(d.size()<3||d.front().id!=tao::data::BOS||d.back().id!=tao::data::TURN_END)
                throw std::runtime_error("doc boundary");
            if(s.adv==0.0)continue;  // 组内无方差 → 无梯度，直接不入批
            supervised+=s.gen.size()+(haveTerm?1u:0u);
            // 论文 Eq.3 的逐序列归一化 1/|o_i|：每个序列的损失是它自身的 token 平均，
            // 使长序列不会仅因更长就主导梯度。取 w_i = A_i/|o_i| 后，配合 update() 内
            // 的 1/supervised 全局缩放，相对权重与论文一致（全局标量被优化器吸收）。
            const size_t olen=s.gen.size()+(haveTerm?1u:0u);
            weights.push_back(olen?float(s.adv)/float(olen):0.f);
            docs.push_back(std::move(d));
            doc_lp.push_back(std::move(lpv));
            doc_lpref.push_back(std::move(lprv));
        }
        size_t lp_positions=0,lp_have=0,lpr_have=0;
        for(auto&v:doc_lp)for(float x:v){++lp_positions;if(x!=0.f)++lp_have;}
        for(auto&v:doc_lpref)for(float x:v) if(x!=0.f)++lpr_have;
        printf("GRPO_LP positions=%zu with_logprob=%zu (%.1f%%)\n",
               lp_positions,lp_have,lp_positions?100.0*double(lp_have)/double(lp_positions):0.0);
        printf("GRPO_LPREF positions=%zu with_logprob=%zu (%.1f%%)\n",
               lp_positions,lpr_have,lp_positions?100.0*double(lpr_have)/double(lp_positions):0.0);
        if(!lpr_have) printf("GRPO_LPREF_NOTE no reference logprob -> KL term disabled (d=0, grad=0), clip only\n");
        fflush(stdout);
        printf("GRPO_DATA samples=%zu groups=%zu groups_with_variance=%zu with_gradient=%zu docs=%zu supervised=%zu\n",
            S.size(),ngroups,nvar,ngrad,docs.size(),supervised);
        fflush(stdout);
        require(docs.size()>=size_t(slot_count),"need at least one doc per slot");
        // 干跑：只校验数据通路（轨迹解析 / 组内优势 / 文档构造），不初始化 GPU。
        // 用于在主训练占用显存期间验证 GRPO 数据侧正确性。
        if(std::getenv("TAO_GRPO_DRYRUN")){ printf("GRPO_DRYRUN_OK\n"); fflush(stdout); return 0; }

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
        tr.load_state((resume/"opt_state.bin").string());
        printf("RESUMED steps=%u lr=%g slots=%u width=%u updates=%u\n",tr.steps,double(lr),slot_count,graph_width,updates);
        fflush(stdout);
        // 论文 Eq.3 的裁剪幅度 ε 与 KL 系数 β。默认 ε=0.2，β=0.04（R1 论文量级）。
        float grpo_eps=0.2f, grpo_beta=0.04f;
        if(const char*e=std::getenv("TAO_GRPO_CLIP")) grpo_eps=float(std::atof(e));
        if(const char*e=std::getenv("TAO_GRPO_BETA")) grpo_beta=float(std::atof(e));
        printf("GRPO_OBJECTIVE clip_eps=%g beta=%g ratio=exp(lp_theta-lp_old) kl=k3(pi_ref||pi_theta)\n",double(grpo_eps),double(grpo_beta));
        fflush(stdout);
        SequenceSlots slots(tr,slot_count);
        // 诊断缓冲必须在图捕获前传进去（指针会被烘焙进 graph 节点）。
        Device dbgbuf(size_t(slot_count)*size_t(graph_width)*3u);
        ReusableBatchGraph replay(tr,slot_count,graph_width,true,grpo_beta,grpo_eps,dbgbuf.p);
        DeferredLoss loss(1024);

        tao::data::PilotCursor base(docs,slot_count,false);
        base.doc_weights=weights; base.doc_lp=doc_lp; base.doc_lpref=doc_lpref;
        const std::string identity=std::string("grpo-v1;tok=")+th+"\n";
        tao::data::ShuffledEpochCursor epoch(std::move(base),20460101ull,identity);
        printf("GRPO_START\n"); fflush(stdout);

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
            // 诊断：读回逐 token (lp, r, c)，报告重要性比值与裁剪触发比例。
            // 这是"目标确实按论文公式在算"的直接证据：比值被显式求出并参与梯度。
            {
                std::vector<float> dv(size_t(slot_count)*graph_width*3u);
                check(cudaMemcpy(dv.data(),dbgbuf.p,dv.size()*4,cudaMemcpyDeviceToHost));
                double rs=0; size_t nn=0,clipn=0;
                for(size_t i=0;i<size_t(slot_count)*graph_width;++i){
                    const float r=dv[i*3+1];
                    if(r<=0.f)continue;                 // 未监督的位置
                    rs+=r;++nn;
                    if(r>1.f+grpo_eps||r<1.f-grpo_eps)++clipn;
                }
                printf("GRPO_DIAG step=%u supervised_tokens=%zu mean_ratio=%.6f clip_frac=%.4f\n",
                       tr.steps,nn,nn?rs/double(nn):0.0,nn?double(clipn)/double(nn):0.0);
                fflush(stdout);
            }
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
            printf("GRPO_UPDATE step=%u positions=%zu targets=%zu weighted_loss=%.6f lr=%g norm=%.6f\n",
                tr.steps,positions,targets,wloss/double(targets),double(lr_now),norm);
            fflush(stdout);
            if(tr.steps%10u==0)export_ckpt(false);
        }
        export_ckpt(true);
        require(fs::is_regular_file(out/"final.dsb"),"missing root DSB");
        return 0;
    }catch(const std::exception&e){ fprintf(stderr,"GRPO_FAIL %s\n",e.what()); return 1; }
}
