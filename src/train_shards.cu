#define TAO_NO_FFN
// 分片流式训练器（课程式 / 增训）。
//
// 为什么需要它：原 train_noffn_probe 要求把整个 train.bin 一次读入内存
// (read_bpe_pilot 返回全部文档)，0.9B 级语料放不下；且它只认 TLP2 对话格式，
// 读不了代码语料的 TLP3。
//
// 本训练器：
//   1. 按文件名顺序逐个读取分片，任一时刻内存里只有**一个分片**（流式、分片训练）；
//   2. 自动识别 TLP2（对话）与 TLP3（纯文本/代码）；
//   3. 权重与 Adam 状态跨分片持续，支持从既有运行目录**增训**（RESUME_DIR），
//      因此课程三阶段 = 三次带 RESUME_DIR 的调用；
//   4. 每 STEPS_PER_SHARD 次更新换下一分片。
//
// 用法: train_shards SHARD_DIR TOKENIZER.bbp OUT_DIR STEPS_PER_SHARD [SLOTS WIDTH] [RESUME_DIR] [START_SHARD]
#define main unused_probe_graph_main
#include "train_yaoyao_graph_gpuval.cu"
#undef main
#include "shuffled_epoch_batch_plan.hpp"
#include "dual_model_bundle.hpp"
#include "plain_lm_format.hpp"
#include <cmath>             // std::cos for LR cosine decay (TAO_LR_DECAY_START/STEPS)

namespace shardtrain {
using namespace tao::dual;
namespace fs = std::filesystem;
void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }

// 读一个分片，自动识别 TLP2 / TLP3。
std::vector<std::vector<tao::data::Token>> read_shard(const fs::path& p, const std::string& digest) {
    std::ifstream f(p, std::ios::binary);
    require(bool(f), "shard open: " + p.string());
    char magic[4]; f.read(magic, 4);
    require(bool(f), "shard magic read");
    f.seekg(0);
    const std::string m(magic, 4);
    if (m == "TLP2") return tao::data::read_bpe_pilot(f, digest);
    if (m == "TLP3") return tao::data::read_tlp3(f, digest);
    throw std::runtime_error("unknown shard magic in " + p.string());
}
}

int main(int argc, char** argv) {
    try {
        using namespace shardtrain;
        require(argc == 5 || argc == 7 || argc == 8 || argc == 9,
                "usage: train_shards SHARD_DIR TOKENIZER.bbp OUT_DIR STEPS_PER_SHARD [SLOTS WIDTH] [RESUME_DIR] [START_SHARD]");
        const fs::path shard_dir = argv[1];
        const std::string tokpath = argv[2];
        const std::string out_s = argv[3];
        const unsigned steps_per_shard = unsigned(std::stoul(argv[4]));
        const unsigned slot_count = argc >= 7 ? unsigned(std::stoul(argv[5])) : 32u;
        const unsigned graph_width = argc >= 7 ? unsigned(std::stoul(argv[6])) : 32u;
        const std::string resume_dir = argc >= 8 ? std::string(argv[7]) : std::string();
        const unsigned start_shard = argc == 9 ? unsigned(std::stoul(argv[8])) : 0u;
        require(steps_per_shard >= 1 && steps_per_shard <= 1000000u, "steps_per_shard 1..1e6");
        require(slot_count >= 1 && slot_count <= 256, "slots 1..256");
        require(graph_width >= 8 && graph_width <= 512, "width 8..512");

        // 分片清单（按文件名排序 = 产出顺序 = 课程顺序）
        std::vector<fs::path> shards;
        for (const auto& e : fs::directory_iterator(shard_dir)) {
            if (!e.is_regular_file()) continue;
            const std::string n = e.path().filename().string();
            if (n.rfind("shard_", 0) == 0 && e.path().extension() == ".bin") shards.push_back(e.path());
        }
        std::sort(shards.begin(), shards.end());
        require(!shards.empty(), "no shard_*.bin in shard dir");
        require(start_shard < shards.size(), "START_SHARD out of range");

        const fs::path out = out_s;
        require(!out.empty() && !fs::exists(fs::symlink_status(out)), "output directory exists; never overwrite");
        if (!resume_dir.empty())
            require(fs::is_regular_file(fs::path(resume_dir) / "opt_state.bin"), "resume: opt_state.bin missing");

        if (!std::freopen((out_s + ".log").c_str(), "a", stdout)) throw std::runtime_error("log open");
        if (!std::freopen((out_s + ".log").c_str(), "a", stderr)) throw std::runtime_error("log open");
        setvbuf(stdout, nullptr, _IONBF, 0); setvbuf(stderr, nullptr, _IONBF, 0);

        std::string th; tao::text::load_tokenizer(tokpath, th);
        if (!std::getenv("TAO_ALLOW_TOKENIZER"))
            require(th == "18b1c761bbc13d7f29ff99db86b3eb17502d7ba2afe45c549fc4381fffc037fa", "frozen tokenizer mismatch");

        Config cfg{};
        {
            auto ov=[&](const char*n,uint32_t&dst){ if(const char*v=std::getenv(n)){unsigned long x=std::strtoul(v,nullptr,10);
                require(x>=1&&x<=262144,"config override range");dst=uint32_t(x);} };
            ov("TAO_CFG_S",cfg.s); ov("TAO_CFG_D",cfg.d); ov("TAO_CFG_M",cfg.m);
            ov("TAO_CFG_E",cfg.e); ov("TAO_CFG_LAYERS",cfg.layers);
            ov("TAO_CFG_DK",cfg.dk); ov("TAO_CFG_VOCAB",cfg.vocab);
            cfg.validate();
            printf("CONFIG layers=%u d=%u s=%u m=%u e=%u vocab=%u dk=%u\n",
                cfg.layers,cfg.d,cfg.s,cfg.m,cfg.e,cfg.vocab,cfg.dk);
            fflush(stdout);
        }

        SortedGpuTrainer tr(initialize(cfg, 20260912));
        if (!resume_dir.empty()) {
            tr.load_state((fs::path(resume_dir) / "opt_state.bin").string());
            printf("RESUMED steps=%u offload=%d from=%s\n", tr.steps, int(tr.offload_), resume_dir.c_str());
            fflush(stdout);
        }
        SequenceSlots slots(tr, slot_count);
        ReusableBatchGraph replay(tr, slot_count, graph_width);
        DeferredLoss loss(1024);

        printf("SHARD_TRAIN_START shards=%zu start_shard=%u steps_per_shard=%u slots=%u width=%u tokenizer=%s\n",
               shards.size(), start_shard, steps_per_shard, slot_count, graph_width, th.c_str());
        fflush(stdout);

        const bool save_opt = !std::getenv("TAO_OPT_STATE") || std::string(std::getenv("TAO_OPT_STATE")) != "0";
        // 同一步只落盘一次：分片结束与最终收尾可能落在同一步，重复创建会报
        // "checkpoint overwrite"（绝不覆盖是刻意的不变量）。
        unsigned last_exported = std::numeric_limits<unsigned>::max();
        auto export_ckpt=[&](bool finish){
            if (!fs::exists(fs::symlink_status(out))) require(fs::create_directory(out), "out create");
            const unsigned step=tr.steps;
            auto ckpt=out/("step_"+std::to_string(step));
            if (step != last_exported) {
                require(!fs::exists(fs::symlink_status(ckpt)), "checkpoint overwrite");
                require(fs::create_directory(ckpt), "checkpoint create");
                const fs::path staged=ckpt/"final.dsb.tmp";
                CpuModel cpu(tr.graph.c);
                for (auto& kv: cpu.w) kv.second=tr.graph.w.at(kv.first)->value.host();
                save_bundle(cpu, staged.string(), th);
                const fs::path dsb=ckpt/"final.dsb";
                fs::rename(staged, dsb);
                if (save_opt) {
                    tr.save_state((out/"opt_state.bin").string());
                    printf("OPT_STATE step=%u path=%s\n", step, (out/"opt_state.bin").generic_string().c_str());
                }
                printf("CHECKPOINT step=%u path=%s\n", step, ckpt.generic_string().c_str());
                fflush(stdout);
                last_exported = step;
            }
            if (finish) {
                const auto root=out/"final.dsb";
                require(!fs::exists(fs::symlink_status(root)), "root DSB overwrite");
                fs::copy_file(ckpt/"final.dsb", root);
                printf("FINAL step=%u path=%s\n", tr.steps, root.generic_string().c_str());
                fflush(stdout);
            }
        };

        const std::string stoppath = out_s + ".stop";
        // 洗牌种子基值：不同"轮"必须换种子，否则第二轮按相同顺序重放同一批
        // 数据，等于把第一轮再背一遍而不是见到新的数据次序。
        uint64_t seed_base=20260912ull;
        if(const char* e=std::getenv("TAO_SHUFFLE_SEED")){const unsigned long long v=strtoull(e,nullptr,10);if(v>0)seed_base=v;}
        printf("SHUFFLE_SEED=%llu\n",(unsigned long long)seed_base);fflush(stdout);
        // 学习率默认 1e-3；可用 TAO_LR 覆盖（降 LR 续训 / 课程表用）。
        // 注意 warmup 只看全局步数 <20，续训时不会误触发。
        float lr_max=0.001f;
        if (const char* e=std::getenv("TAO_LR")) {
            const float v=strtof(e,nullptr);
            require(v>1e-6f && v<=0.1f, "TAO_LR range");
            lr_max=v;
            printf("LR_OVERRIDE TAO_LR=%g\n", double(v));
            fflush(stdout);
        }
        // 余弦退火（默认关闭）。
        // 触发：设了 TAO_LR_DECAY_START 且 TAO_LR_DECAY_STEPS>0。
        // 从 decay_start 起，按 progress=(steps-decay_start)/decay_steps 走
        //   lr = lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi*progress))
        // 这样分片内有退火，课程三段（分片间 LR 切换）依然按 curriculum.tsv 走。
        // 注意：warmup 只在 steps<20 且 steps<decay_start 时生效；进入退火区段后
        //       退火公式接管（warmup 的"线性上升"只对恒定 LR 区段有意义）。
        float lr_min=1e-5f;
        unsigned lr_decay_start=0, lr_decay_steps=0;
        bool lr_decay=false;
        if(const char* e=std::getenv("TAO_LR_DECAY_START")){
            const long long v=std::strtoll(e,nullptr,10);
            require(v>=0 && v<=1000000000LL, "TAO_LR_DECAY_START range");
            lr_decay_start=(unsigned)v;
        }
        if(const char* e=std::getenv("TAO_LR_DECAY_STEPS")){
            const long long v=std::strtoll(e,nullptr,10);
            require(v>0 && v<=1000000000LL, "TAO_LR_DECAY_STEPS range");
            lr_decay_steps=(unsigned)v;
        }
        if(const char* e=std::getenv("TAO_LR_MIN")){
            const float v=strtof(e,nullptr);
            require(v>=0.f && v<0.1f, "TAO_LR_MIN range");
            lr_min=v;
        }
        if(lr_decay_start>0 && lr_decay_steps>0){
            lr_decay=true;
            printf("LR_DECAY start=%u steps=%u min=%g (cosine)\n", lr_decay_start, lr_decay_steps, double(lr_min));
            fflush(stdout);
        }
        // 梯度裁剪（默认关闭）。
        // 触发：TAO_GRAD_CLIP 设了正数。
        // 实现：在 tr.update 之前若 norm>clip 则把 lr 乘以 clip/norm，等价于把更新步长
        //       限制在 clip/norm 的步长上。update() 内部已有 factor=1/(supervised*norm)
        //       的全局归一化，再叠加这一层就是标准 L2 grad clip 语义。
        float grad_clip=0.f;
        if(const char* e=std::getenv("TAO_GRAD_CLIP")){
            const float v=strtof(e,nullptr);
            require(v>0.f, "TAO_GRAD_CLIP must be positive");
            grad_clip=v;
            printf("GRAD_CLIP=%g\n", double(grad_clip));
            fflush(stdout);
        }

        for (unsigned si=start_shard; si<shards.size(); ++si) {
            printf("SHARD_BEGIN index=%u file=%s\n", si, shards[si].filename().string().c_str());
            fflush(stdout);
            auto docs = read_shard(shards[si], th);
            require(!docs.empty(), "empty shard");
            size_t positions_total=0, targets_total=0;
            for (const auto& d : docs) {
                require(d.size()>=5 && d.size()<=1000000, "shard document length");
                positions_total += d.size()-1;
                for (size_t i=1;i<d.size();++i) targets_total += d[i].loss;
            }
            require(targets_total>0, "empty supervised shard");
            const std::string identity = std::string("shard-train-v1;")+shards[si].filename().string()+";tok="+th+"\n";
            auto base = tao::data::PilotCursor(docs, slot_count, false);
            tao::data::ShuffledEpochCursor epoch(std::move(base), seed_base + uint64_t(si), identity);
            printf("SHARD_LOADED index=%u docs=%zu positions=%zu targets=%zu\n",
                   si, docs.size(), positions_total, targets_total);
            fflush(stdout);

            for (unsigned u=0; u<steps_per_shard; ++u) {
                if (fs::exists(stoppath)) {
                    if (tr.steps%50) export_ckpt(false);
                    printf("PAUSED saved step=%u shard=%u\n", tr.steps, si);
                    return 0;
                }
                size_t positions=0, targets=0;
                const auto before=tr.steps;
                for (unsigned r=0; r<8; ++r) {
                    if (epoch.exhausted()) { check(cudaStreamSynchronize(0)); epoch.begin_next_epoch(true); }
                    auto p=tao::data::take_batch(epoch, graph_width);
                    require(p.timesteps>0, "no work after rollover");
                    replay.run(p, slots);
                    accumulate_loss<<<1,1>>>(replay.data.loss.p, replay.data.loss.n, loss.total, loss.bad);
                    check(cudaGetLastError());
                    positions+=p.positions; targets+=p.supervised;
                }
                require(targets>0, "no supervised update");
                const double train_loss=loss.collect();
                float lr=lr_max;
                if(lr_decay && tr.steps>=lr_decay_start){
                    const unsigned past=tr.steps-lr_decay_start;
                    const float progress=past>=lr_decay_steps ? 1.f : float(past)/float(lr_decay_steps);
                    lr=lr_min+0.5f*(lr_max-lr_min)*(1.f+std::cos(3.14159265358979323846f*progress));
                }else if(tr.steps<20u){
                    lr=lr_max*float(tr.steps+1u)/20.f;   // 线性 warmup（仅在非退火区段）
                }
                float norm=tr.update(targets, lr, grad_clip);   // max_norm=grad_clip；0 表示不裁剪
                if(grad_clip>0.f && norm>grad_clip){
                    printf("CLIPPED step=%u norm=%.6f -> cap=%.6f\n", tr.steps, norm, grad_clip);
                    fflush(stdout);
                }
                check(cudaDeviceSynchronize());
                require(tr.steps==before+1 && std::isfinite(norm), "invalid update");
                printf("UPDATE step=%u shard=%u positions=%zu targets=%zu train_preupdate_NLL=%.6f lr=%.6f norm=%.6f\n",
                    tr.steps, si, positions, targets, train_loss/targets, lr, norm);
                fflush(stdout);
                if (tr.steps%50u==0) export_ckpt(false);
            }
            printf("SHARD_DONE index=%u step=%u\n", si, tr.steps);
            fflush(stdout);
            export_ckpt(false);
        }
        export_ckpt(true);
        require(fs::is_regular_file(out/"final.dsb"), "missing root DSB");
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "SHARD_TRAIN_FAIL %s\n", e.what());
        return 1;
    }
}
