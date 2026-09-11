#define TAO_NO_FFN
// Independent tiny-corpus no-FFN probe. Fresh weights. Does not read or write build/noffn_fresh*.
// Usage: train_noffn_probe TRAIN.bin TOKENIZER.bbp UNIQUE_OUT UPDATES [SLOTS WIDTH] [RESUME_DIR]
//   RESUME_DIR: 含 opt_state.bin 的既有运行目录；权重+Adam 状态+步数从该处恢复，
//   输出仍写入全新的 UNIQUE_OUT（绝不覆盖）。续训时数据游标从新一轮 epoch 开始，
//   故非逐样本确定性的续训；权重与优化器状态是精确恢复的。
#define main unused_probe_graph_main
#include "train_yaoyao_graph_gpuval.cu"
#undef main
#include "shuffled_epoch_batch_plan.hpp"
#include "dual_model_bundle.hpp"

namespace probe {
using namespace tao::dual;
namespace fs = std::filesystem;
void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
}

int main(int argc, char** argv) {
    try {
        using namespace probe;
        require(argc==5||argc==7||argc==8, "usage: train_noffn_probe TRAIN.bin TOKENIZER.bbp UNIQUE_OUT UPDATES [SLOTS WIDTH] [RESUME_DIR]");
        // 并行序列数与 CUDA 图宽度可调。默认 32 / 32 = 加速基线（2026-09-11 转正），可用 argv 覆盖回 4/128。
        // 实测（probe3 语料，RTX 4070 Laptop，slot-tiled 内核）：4/128 -> 5.00 ms/position；32/32 -> 0.427；64/24 -> 0.323。
        // 注意：slots 必须 >=8 且为 8 的倍数，slot-tiled 内核才生效；否则自动回退到 baseline 内核（数值逐位相同）。
        const unsigned slot_count = argc>=7 ? unsigned(std::stoul(argv[5])) : 32u;
        const unsigned graph_width = argc>=7 ? unsigned(std::stoul(argv[6])) : 32u;
        const std::string resume_dir = argc==8 ? std::string(argv[7]) : std::string();
        require(slot_count>=1 && slot_count<=256, "slots 1..256");
        require(graph_width>=8 && graph_width<=512, "width 8..512");
        const std::string logpath=std::string(argv[3])+".log";
        const std::string stoppath=std::string(argv[3])+".stop";
        require(!fs::exists(stoppath), "stop flag present");
        if (!std::freopen(logpath.c_str(),"a",stdout) || !std::freopen(logpath.c_str(),"a",stderr))
            throw std::runtime_error("probe log open");
        setvbuf(stdout, nullptr, _IONBF, 0);
        setvbuf(stderr, nullptr, _IONBF, 0);
        std::string count=argv[4];
        require(!count.empty() && count.size()<=9 && count.find_first_not_of("0123456789")==std::string::npos, "updates syntax");
        unsigned updates=unsigned(std::stoul(count));
        // 长训：上限由 1000 放宽到 1e9（0.9B 级真实训练远超千步）。
        require(updates>=1 && updates<=1000000000u, "updates 1..1e9");
        auto out=fs::path(argv[3]);
        require(!out.empty() && !fs::exists(fs::symlink_status(out)), "output directory exists; never overwrite");
        if (!resume_dir.empty()) {
            require(fs::is_regular_file(fs::path(resume_dir)/"opt_state.bin"), "resume: opt_state.bin missing");
        }
        require(fs::path(argv[1]).filename()=="train.bin", "require train.bin");
        std::string th; tao::text::load_tokenizer(argv[2], th);
        // 换词表（0.9B 真实语料需重训 BPE）时用 TAO_ALLOW_TOKENIZER=1 放行冻结校验。
        if (!std::getenv("TAO_ALLOW_TOKENIZER"))
            require(th=="34463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333", "frozen tokenizer mismatch");
        std::ifstream f(argv[1], std::ios::binary); require(bool(f), "train.bin open");
        std::string raw((std::istreambuf_iterator<char>(f)), {});
        require(!raw.empty() && !f.bad(), "train.bin read");
        const auto newsha=tao::text::sha256(raw);
        std::istringstream newin(raw);
        auto docs=tao::data::read_bpe_pilot(newin, th);
        // 上限由 65536 放宽到 1<<20：真实语料文档数远超旧探测语料。
        // 文档数与单篇长度上限可由环境变量放宽：真实代码语料的单文件远超 512 token。
        // 默认值保持与旧行为逐位一致（不改环境变量时行为不变）。
        size_t max_docs=1u<<20, max_doc_len=512;
        if(const char* e=std::getenv("TAO_MAX_DOCS")){long long v=std::atoll(e); if(v>0)max_docs=size_t(v);}
        if(const char* e=std::getenv("TAO_MAX_DOC_LEN")){long long v=std::atoll(e); if(v>0)max_doc_len=size_t(v);}
        require(docs.size()>=10 && docs.size()<=max_docs, "probe corpus size");
        size_t dataset_targets=0, dataset_positions=0;
        for (const auto& doc: docs) {
            require(doc.size()>=5 && doc.size()<=max_doc_len, "probe document length");
            dataset_positions+=doc.size()-1;
            for (size_t i=1;i<doc.size();++i) dataset_targets+=doc[i].loss;
        }
        require(dataset_targets>0, "empty supervised dataset");
        const uint64_t seed=20260912;
        const std::string identity=std::string("noffn-probe-v1;tiny-closed;fresh;seed20260912;graph4x8x128\ncorpus=")+newsha+"\n";
        // 架构维度覆盖（doc 22 §五）：让「容量类」假设可以 A/B 而不必重新编译。
        // 默认不设任何环境变量时 Config{} 保持原值，行为与之前逐位一致。
        // TAO_CFG_S / TAO_CFG_D / TAO_CFG_M / TAO_CFG_E / TAO_CFG_LAYERS / TAO_CFG_DK / TAO_CFG_VOCAB
        Config cfg{};
        {
            auto ov=[&](const char*n,uint32_t&dst){
                if(const char*v=std::getenv(n)){unsigned long x=std::strtoul(v,nullptr,10);
                    require(x>=1&&x<=262144,"config override range");dst=uint32_t(x);}
            };
            ov("TAO_CFG_S",cfg.s); ov("TAO_CFG_D",cfg.d); ov("TAO_CFG_M",cfg.m);
            ov("TAO_CFG_E",cfg.e); ov("TAO_CFG_LAYERS",cfg.layers);
            ov("TAO_CFG_DK",cfg.dk); ov("TAO_CFG_VOCAB",cfg.vocab);
            cfg.validate();
            printf("CONFIG layers=%u d=%u s=%u m=%u e=%u vocab=%u dk=%u\n",
                cfg.layers,cfg.d,cfg.s,cfg.m,cfg.e,cfg.vocab,cfg.dk);
            // R2：证明激活开关确实生效（doc 24 的 R4 统一）。
#ifdef TAO_TRAIN_FAST_ACT
            printf("ACTIVATION pade-fast  (decoder must use default TAO_FAST_ACT=1)\n");
#else
            printf("ACTIVATION exact      (decoder must use TAO_FAST_ACT=0)\n");
#endif
            fflush(stdout);
        }
        SortedGpuTrainer tr(initialize(cfg, 20260912));
        if (!resume_dir.empty()) {
            tr.load_state((fs::path(resume_dir)/"opt_state.bin").string());
            printf("RESUMED steps=%u offload=%d from=%s\n", tr.steps, int(tr.offload_), resume_dir.c_str());
            fflush(stdout);
        }
        SequenceSlots slots(tr, slot_count);
        tao::data::ShuffledEpochCursor epoch(tao::data::PilotCursor(docs, slot_count, false), seed, identity);
        ReusableBatchGraph replay(tr, slot_count, graph_width);
        DeferredLoss loss(1024);
#ifndef TAO_BASELINE_GEMM
        const char* kernels="slot-tiled";
#else
        const char* kernels="baseline";
#endif
        printf("PROBE_START docs=%zu positions=%zu targets=%zu sha256=%s tokenizer=%s updates=%u lr=0.001 slots=%u width=%u kernels=%s\n",
            docs.size(), dataset_positions, dataset_targets, newsha.c_str(), th.c_str(), updates, slot_count, graph_width, kernels);
        fflush(stdout);
        // TAO_OPT_STATE=0 可关闭优化器状态落盘（0.9B 时每份 ≈ 10.8 GB，按需关）。
        const bool save_opt = !std::getenv("TAO_OPT_STATE") || std::string(std::getenv("TAO_OPT_STATE"))!="0";
        auto export_ckpt=[&](bool finish){
            if (!fs::exists(fs::symlink_status(out))) require(fs::create_directory(out), "output directory create");
            const unsigned step=tr.steps;
            auto ckpt=out/("step_"+std::to_string(step));
            require(!fs::exists(fs::symlink_status(ckpt)), "checkpoint overwrite");
            require(fs::create_directory(ckpt), "checkpoint directory create");
            const fs::path staged=ckpt/"final.dsb.tmp";
            CpuModel cpu(tr.graph.c);
            for (auto& kv: cpu.w) kv.second=tr.graph.w.at(kv.first)->value.host();
            save_bundle(cpu, staged.string(), th);
            const fs::path dsb=ckpt/"final.dsb";
            fs::rename(staged, dsb);
            if (save_opt) {
                tr.save_state((out/"opt_state.bin").string());
                printf("OPT_STATE step=%u path=%s\n", step, (out/"opt_state.bin").generic_string().c_str());
                fflush(stdout);
            }
            printf("CHECKPOINT step=%u path=%s\n", step, ckpt.generic_string().c_str());
            fflush(stdout);
            if (finish) {
                const auto root=out/"final.dsb";
                require(!fs::exists(fs::symlink_status(root)), "root DSB overwrite");
                fs::copy_file(dsb, root);
                printf("FINAL step=%u path=%s\n", tr.steps, root.generic_string().c_str());
                fflush(stdout);
            }
        };
        export_ckpt(false);
        constexpr float lr_max=0.001f;
        for (unsigned u=0; u<updates; ++u) {
            if (fs::exists(stoppath)) {
                if (tr.steps%50) export_ckpt(false);
                printf("PAUSED saved step=%u\n", tr.steps);
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
            const float lr=tr.steps<20u ? lr_max*float(tr.steps+1u)/20.f : lr_max;
            float norm=tr.update(targets, lr);
            check(cudaDeviceSynchronize());
            require(tr.steps==before+1 && std::isfinite(norm), "invalid update");
            printf("UPDATE step=%u epoch=%llu positions=%zu targets=%zu train_preupdate_NLL=%.6f lr=%.6f norm=%.6f\n",
                tr.steps, (unsigned long long)epoch.epoch(), positions, targets, train_loss/targets, lr, norm);
            fflush(stdout);
            if (tr.steps%50u==0 || u+1==updates) export_ckpt(u+1==updates);
        }
        require(fs::is_regular_file(out/"final.dsb"), "missing root DSB");
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "PROBE_FAIL %s\n", e.what());
        return 1;
    }
}
