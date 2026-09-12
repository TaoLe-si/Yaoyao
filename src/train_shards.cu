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
#include "greedy_pipeline_grouped_model.hpp"
#include <cmath>
#include <chrono>

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
        // width 的下界从 8 放宽到 1：GEMM 的 batch 是 slots，width 只是串行展开的
        // 时间步数。显存约正比于 slots*width，所以「大 slots + 小 width」能用同样
        // 显存换到更大的 GEMM batch，吞吐显著更高（d=2048 实测 sl16/wd8=467 tok/s
        // vs sl8/wd16=401）。上界与 train_sft.cu 对齐。
        require(graph_width >= 1 && graph_width <= 4096, "width 1..4096");

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
    // 阶段计时开关（诊断用）：TAO_STAGE_TIMING=1 时每步额外同步两次，只影响日志。
    const bool stage_timing=[](){const char*e=std::getenv("TAO_STAGE_TIMING");if(!e||!*e)return false;return !(e[0]=='0'&&e[1]=='\0');}();
    using stage_clock=std::chrono::steady_clock;
    stage_clock::time_point stage_t0,stage_t1;
    double graph_ms=0.0,upd_ms=0.0;unsigned long long graph_n=0;

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

        // warmup 步数：原硬编码 20 步。峰值 LR 下 20 步即冲到满 LR 太激进，
        // 长训易在开跑阶段发散。改为可配置，未设时保持 20（与旧行为逐位一致）。
        unsigned warmup=20;
        if(const char* e=std::getenv("TAO_WARMUP")){const int v=std::atoi(e);if(v>=1)warmup=(unsigned)v;}

        // 收敛门控（默认关闭）：TAO_CONVERGE=1 时，同一分片持续训练直到该分片收敛才进下一片。
        // 判据：每 TAO_CONV_WINDOW 步算一次该窗口的平均 train NLL；与上一窗口相比的相对改进
        //       低于 TAO_CONV_TOL 即视为「本窗口没学到东西」，连续 TAO_CONV_PATIENCE 个窗口
        //       都如此且已训够 TAO_CONV_MIN 步 → 判定本片收敛。
        // 安全上限：每片最多 TAO_CONV_MAX 步（默认 = steps_per_shard）。
        // 不设 TAO_CONVERGE 时 converge=false，内层循环上界仍是 steps_per_shard，行为与旧版逐位一致。
        // 判据是「统计显著」而不是固定百分比：窗口间噪声实测有 ±0.5~2%，
        // 固定小容差会被噪声主导。故窗口内累积 sum/sumsq → 标准差 → 均值标准误 SE，
        // 要求 (上一窗口均值 - 本窗口均值) > TAO_CONV_Z(默认2) × SE 且相对改进 > TAO_CONV_TOL
        // 才算「本窗口真的还在学」；连续 TAO_CONV_PATIENCE 个窗口不满足 → 判定本片收敛。
        const bool converge = (std::getenv("TAO_CONVERGE") != nullptr);
        unsigned conv_window=200u, conv_patience=3u, conv_min=200u, conv_max=steps_per_shard;
        double conv_tol=0.001, conv_z=2.0;
        // 过拟合下界：训练 NLL 低于此值即判定「已经把本片背下来」，立即停（默认 0.2）。
        // 实测依据：0.2 只在 0.2~3MB 的闭合合成语料上可达（bind 最低 0.0008）；
        // 581MB 真实开放语料 29,065 步里从未低于 0.88，故该下界在真实语料上不会误触发。
        double conv_floor=0.2;
        // 留出集：TAO_HOLDOUT=N 时，从每片**末尾**取 N 篇不参与训练，只用于评估。
        // 有留出集时，收敛判据用**留出 NLL**而不是训练 NLL（训练 NLL 单调下降，必然收敛到过拟合）。
        size_t hold_n=0;
        if (const char* e=std::getenv("TAO_HOLDOUT")) hold_n=(size_t)std::strtoul(e,nullptr,10);
        // 每篇留出文档只评前 N 个受监督 token。留出样本一旦固定，配对差检验用的是**窗口间的差**，
        // 样本量只需足够估计这个差，不必覆盖全文档 —— 这是把评估从 >5 分钟压到数秒的关键。
        unsigned eval_tok_cap=64;
        if (const char* e=std::getenv("TAO_HOLDOUT_TOKENS")) { const int v=std::atoi(e); if(v>=1) eval_tok_cap=(unsigned)v; }
        if (converge) {
            if (const char* e=std::getenv("TAO_CONV_WINDOW")) { const int v=std::atoi(e); if(v>=1) conv_window=(unsigned)v; }
            if (const char* e=std::getenv("TAO_CONV_Z"))      { const double v=std::atof(e); if(v>0.0) conv_z=v; }
            if (const char* e=std::getenv("TAO_CONV_TOL"))    { const double v=std::atof(e); if(v>=0.0) conv_tol=v; }
            if (const char* e=std::getenv("TAO_CONV_PATIENCE")){ const int v=std::atoi(e); if(v>=1) conv_patience=(unsigned)v; }
            if (const char* e=std::getenv("TAO_CONV_MIN"))    { const int v=std::atoi(e); if(v>=0) conv_min=(unsigned)v; }
            if (const char* e=std::getenv("TAO_CONV_MAX"))    { const int v=std::atoi(e); if(v>=1) conv_max=(unsigned)v; }
            if (const char* e=std::getenv("TAO_CONV_FLOOR"))  { const double v=std::atof(e); if(v>0.0) conv_floor=v; }
            printf("CONVERGE_MODE window=%u z=%g tol=%g patience=%u min=%u max=%u floor=%g holdout=%zu holdout_tokens=%u (statistical criterion)\n",
                   conv_window, conv_z, conv_tol, conv_patience, conv_min, conv_max, conv_floor, hold_n, eval_tok_cap);
            fflush(stdout);
        }

        for (unsigned si=start_shard; si<shards.size(); ++si) {
            printf("SHARD_BEGIN index=%u file=%s\n", si, shards[si].filename().string().c_str());
            fflush(stdout);
            auto docs = read_shard(shards[si], th);
            require(!docs.empty(), "empty shard");
            // 留出集切分：末尾 hold_n 篇不参与训练。必须先切再算 positions，日志才诚实。
            std::vector<std::vector<tao::data::Token>> hold_docs;
            if (hold_n > 0 && docs.size() > hold_n*2) {
                hold_docs.assign(docs.end()-(long)hold_n, docs.end());
                docs.resize(docs.size()-hold_n);
            }
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
            printf("SHARD_LOADED index=%u docs=%zu positions=%zu targets=%zu holdout=%zu\n",
                   si, docs.size(), positions_total, targets_total, hold_docs.size());
            fflush(stdout);

            // 留出集评估：拷权重到 CPU，按训练同语义（逐文档重置状态、只统计 loss=true）算每篇的 NLL。
            // 返回**逐文档**向量并保持与 hd 同序 —— 因为每窗口评的是同一批文档，配对差的方差远小于
            // 两独立均值之和，这才是有统计效力的判据。
            unsigned eval_threads=8;
            if (const char* e=std::getenv("TAO_CPU_THREADS")) { const int v=std::atoi(e); if(v>=1) eval_threads=(unsigned)v; }

            auto eval_holdout = [&](const std::vector<std::vector<tao::data::Token>>& hd,
                                    std::vector<double>& out) {
                out.assign(hd.size(), 0.0);
                if (hd.empty()) return;
                CpuModel cpu(tr.graph.c);
                for (auto& kv : cpu.w) kv.second = tr.graph.w.at(kv.first)->value.host();
                GreedyPipelineGroupedModel hm(cpu);
                hm.set_cpu_threads(eval_threads);
                const uint32_t V = cpu.c.vocab;
                for (size_t di=0; di<hd.size(); ++di) {
                    const auto& tk = hd[di];
                    if (tk.size() < 2) continue;
                    auto st = hm.initial();
                    double s = 0.0; size_t n = 0;
                    for (size_t i=1;i<tk.size();++i) {
                        if (n >= eval_tok_cap) break;
                        Vec lg = hm.step((uint32_t)tk[i-1].id, st);
                        if (!tk[i].loss) continue;
                        float mx = lg[0];
                        for (uint32_t k=1;k<V;++k) if (lg[k]>mx) mx=lg[k];
                        double sumexp = 0.0;
                        for (uint32_t k=0;k<V;++k) sumexp += std::exp(double(lg[k]-mx));
                        s += (double(mx) + std::log(sumexp)) - double(lg[(size_t)tk[i].id]);
                        ++n;
                    }
                    if (n) out[di] = s/double(n);
                }
            };

            // 本片的收敛状态（每片重置）
            const unsigned shard_cap = converge ? conv_max : steps_per_shard;
            bool converged = false, overfit_stop = false;
            std::vector<double> conv_samples, conv_prev_vec;
            double conv_prev_mean = 0.0;
            unsigned conv_n = 0u, conv_lowgain = 0u;
            bool conv_have_prev = false;
            auto vmean = [](const std::vector<double>& v){ double s=0.0; for(double x:v)s+=x; return v.empty()?0.0:s/double(v.size()); };
            auto vse   = [&vmean](const std::vector<double>& v){ if(v.size()<2) return 0.0; const double m=vmean(v);
                              double s=0.0; for(double x:v)s+=(x-m)*(x-m); return std::sqrt(s/double(v.size()-1)/double(v.size())); };
            for (unsigned u=0; u<shard_cap; ++u) {
                if (fs::exists(stoppath)) {
                    if (tr.steps%50) export_ckpt(false);
                    printf("PAUSED saved step=%u shard=%u\n", tr.steps, si);
                    return 0;
                }
                size_t positions=0, targets=0;
                const auto before=tr.steps;
                // 计时覆盖 8 次前向+反向（含 loss 归约）；backward 已延迟同步，
                // loss.collect() 的 cudaMemcpy 会把整段图工作收敛到此处。
                if(stage_timing){check(cudaDeviceSynchronize());stage_t0=stage_clock::now();}
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
                double ms_graph=0.0, ms_upd=0.0;
                const double train_loss=loss.collect();
                if(stage_timing){ms_graph=std::chrono::duration<double>(stage_clock::now()-stage_t0).count()*1e3;}
                // 可选余弦退火（默认关闭）：不设 TAO_LR_DECAY_START 时 lr_now==lr_max，
                // 与旧行为逐位一致。用于课程末段退火（论文一致做法）。
                float lr_now=lr_max;
                if(const char* dse=std::getenv("TAO_LR_DECAY_START")){
                    const long long dstart=std::atoll(dse);
                    if(dstart>0 && tr.steps>=(unsigned long long)dstart){
                        const char* dne=std::getenv("TAO_LR_DECAY_STEPS");
                        const long long dspan=dne?std::atoll(dne):1;
                        const char* mne=std::getenv("TAO_LR_MIN");
                        const float lrmin=mne?float(std::atof(mne)):lr_max*0.1f;
                        double tt=double(tr.steps-(unsigned long long)dstart)/double(dspan>0?dspan:1);
                        if(tt>1.0)tt=1.0;
                        lr_now=lrmin+(lr_max-lrmin)*float(0.5*(1.0+std::cos(3.14159265358979323846*tt)));
                    }
                }
                const float lr=tr.steps<warmup ? lr_now*float(tr.steps+1u)/float(warmup) : lr_now;
                if(stage_timing){stage_t1=stage_clock::now();}
                float norm=tr.update(targets, lr);
                if(stage_timing){ms_upd=std::chrono::duration<double>(stage_clock::now()-stage_t1).count()*1e3;graph_ms+=ms_graph;upd_ms+=ms_upd;graph_n++;}
                check(cudaDeviceSynchronize());
                require(tr.steps==before+1 && std::isfinite(norm), "invalid update");
                printf("UPDATE step=%u shard=%u positions=%zu targets=%zu train_preupdate_NLL=%.6f lr=%.6f norm=%.6f",
                    tr.steps, si, positions, targets, train_loss/targets, lr, norm);
                if(stage_timing)printf(" ms_graph=%.1f ms_upd=%.1f ms_project=%.1f ms_graph_sum=%.1f ms_upd_sum=%.1f ms_total=%.1f",
                    ms_graph, ms_upd, tr.last_project_ms, graph_ms, upd_ms, graph_ms+upd_ms);
                printf("\n");
                fflush(stdout);
                if (tr.steps%50u==0) export_ckpt(false);
                if (converge) {
                    conv_samples.push_back(train_loss/targets);
                    if (conv_samples.size() >= conv_window) {
                        const double train_avg = vmean(conv_samples);
                        // 下界（过拟合）：训练 NLL 掉到 TAO_CONV_FLOOR 以下 = 已把本片背下来，立即停。
                        if (train_avg < conv_floor) {
                            printf("SHARD_OVERFIT index=%u step=%u on_shard=%u train_avg=%.6f floor=%g\n",
                                   si, tr.steps, u+1u, train_avg, conv_floor);
                            fflush(stdout);
                            overfit_stop = true; converged = true;
                        }
                        // 判据信号：有留出集就用**留出 NLL**（泛化侧），否则退回训练 NLL。
                        std::vector<double> sigv;
                        if (!hold_docs.empty()) eval_holdout(hold_docs, sigv); else sigv = conv_samples;
                        const double sig = vmean(sigv);
                        if (!overfit_stop && conv_have_prev && sigv.size()==conv_prev_vec.size()) {
                            double gain, need, relg;
                            if (!hold_docs.empty()) {
                                // 配对：同一批留出文档，文档难度在差分中抵消
                                std::vector<double> d(sigv.size());
                                for (size_t i=0;i<sigv.size();++i) d[i]=conv_prev_vec[i]-sigv[i];
                                gain = vmean(d);
                                need = conv_z*vse(d);
                                relg = conv_prev_mean>0.0 ? gain/conv_prev_mean : 0.0;
                            } else {
                                // 独立两窗口（训练侧每步样本互不配对）
                                gain = conv_prev_mean-sig;
                                const double s1=vse(conv_prev_vec), s2=vse(sigv);
                                need = conv_z*std::sqrt(s1*s1+s2*s2);
                                relg = conv_prev_mean>0.0 ? gain/conv_prev_mean : 0.0;
                            }
                            const bool learning = (gain > need) && (relg > conv_tol);
                            if (!learning) ++conv_lowgain; else conv_lowgain = 0u;
                            printf("CONV_PROBE shard=%u step=%u on_shard=%u win=%u train=%.6f %s=%.6f prev=%.6f gain=%+.6f need=%.6f rel=%+.6f lowgain=%u %s\n",
                                   si, tr.steps, u+1u, conv_window, train_avg,
                                   hold_docs.empty() ? "signal" : "holdout", sig,
                                   conv_prev_mean, gain, need, relg, conv_lowgain,
                                   learning ? "learning" : "PLATEAU");
                            fflush(stdout);
                        }
                        conv_prev_vec = sigv; conv_prev_mean = sig; conv_have_prev = true;
                        conv_samples.clear(); conv_n = 0u;
                        if (!overfit_stop && (u+1u) >= conv_min && conv_lowgain >= conv_patience) {
                            printf("SHARD_CONVERGED index=%u step=%u on_shard=%u lowgain=%u signal=%s\n",
                                   si, tr.steps, u+1u, conv_lowgain,
                                   hold_docs.empty() ? "train" : "holdout");
                            fflush(stdout);
                            converged = true;
                        }
                    }
                    if (converged) break;
                }
            }
            if (converge && !converged) {
                printf("SHARD_CAPPED index=%u step=%u cap=%u (reached max steps without convergence)\n",
                       si, tr.steps, shard_cap);
                fflush(stdout);
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
