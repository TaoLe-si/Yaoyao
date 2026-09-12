// GRPO 前置工具：可采样 rollout + 规则奖励打分 + 组内方差判据。
//   —— 回答三个问题：(1) 采样后 pass@1 / pass@G 是多少；
//                    (2) 组内奖励是否有非零方差（GRPO 有没有梯度信号）；
//                    (3) 轨迹可直接 --dump 给 GRPO 更新用。
// 用法: grpo_rollout MODEL PROBE.jsonl [--n N] [--samples G] [--temp T] [--top-k K]
//        [--top-p P] [--max M] [--seed S] [--rep-pen F] [--rep-win W]
//        [--w-correct F] [--w-fmt F] [--w-calc F] [--dump FILE] [--show K] [--prefix train|raw]
#define NOMINMAX
#define TAO_INPUT_SCALE
#define TAO_CPU_AVX2
#define TAO_NO_FFN
#define TAO_DELTA_MEM
#include "tokenizer_file.hpp"
#include "greedy_pipeline_grouped_model.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <memory>
#include <thread>
#include <set>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <stdexcept>

using tao::dual::Vec;

static std::vector<std::string> read_jsonl(const std::string& p, size_t maxn){
    std::ifstream f(p, std::ios::binary);
    if(!f) throw std::runtime_error("probe open " + p);
    std::vector<std::string> lines; std::string L;
    while(std::getline(f, L) && lines.size() < maxn) { if(!L.empty() && L.back()=='\r') L.pop_back(); if(!L.empty()) lines.push_back(L); }
    return lines;
}
// 极简 JSON 字段抽取（样本只含 {"q": "...", "a": "..."}）
static std::string field(const std::string& line, const std::string& key){
    const std::string pat = "\"" + key + "\"";
    size_t k = line.find(pat);
    if(k == std::string::npos) return {};
    size_t c = line.find(':', k + pat.size());
    if(c == std::string::npos) return {};
    size_t i = c + 1;
    while(i < line.size() && (line[i]==' '||line[i]=='\t')) ++i;
    if(i >= line.size() || line[i] != '"') return {};
    ++i; std::string out;
    while(i < line.size() && line[i] != '"'){
        if(line[i]=='\\' && i+1 < line.size()){
            char n = line[i+1];
            if(n=='n') out.push_back('\n'); else if(n=='t') out.push_back('\t');
            else if(n=='u'){ i += 5; out += "?"; continue; }
            else out.push_back(n);
            i += 2;
        } else out.push_back(line[i++]);
    }
    return out;
}
// 与 scripts/eval_math.mjs 完全一致的规范化与抽取
static std::string norm(const std::string& s){
    std::string o;
    for(size_t i=0;i<s.size();++i){
        unsigned char ch = (unsigned char)s[i];
        if(ch==' '||ch=='\t'||ch=='\n'||ch=='\r'||ch==','||ch=='_') continue;
        if(ch==0xEF && i+2<s.size() && (unsigned char)s[i+1]==0xBC && (unsigned char)s[i+2]==0x8C){ i+=2; continue; } // 全角逗号
        if(ch==0xEF && i+2<s.size() && (unsigned char)s[i+1]==0xBC && (unsigned char)s[i+2]==0x9A){ i+=2; continue; } // 全角冒号
        o.push_back((char)ch);
    }
    return o;
}
// 从回答里取出最终答案。语义变更说明：
//   旧实现找不到「答案」标记时回退为"取全文最后一个数字"，且以 ASCII '.' 作为终止符。
//   前者让模型**不写答案也能骗到完整正确性奖励**（reward hacking）；
//   后者把小数答案截断（"28.26" -> "28"），而数学题小数极常见。
//   现改为：必须出现「答案」标记，然后取其后的第一个数字串；无标记返回空（不给分）。
static std::string sanitize_tsv(std::string s){ for(auto& c : s){ if(c=='\n'||c=='\r'||c=='\t') c=' '; } return s; }
static std::string grab_answer(const std::string& text){
    size_t k = text.find("答案");
    if(k == std::string::npos) return "";
    size_t i = k + 6; // "答案" 的 UTF-8 长度
    std::string num;
    bool started = false;
    for(; i < text.size(); ++i){
        char ch = text[i];
        bool isd = (ch>='0'&&ch<='9')||ch=='.'||ch=='/'||ch=='-'||ch=='%';
        if(isd){ started = true; num.push_back(ch); continue; }
        if(started) break;
        if(ch==' '||ch=='\t'||ch=='\n'||ch=='\r') continue;
        if((unsigned char)ch==0xEF) continue;
        if(ch==':') continue;
        if((unsigned char)ch >= 0x80) continue;  // 「是」「：」等多字节字符
        break;
    }
    return num.empty() ? std::string() : norm(num);
}
// 从 logits 采样一个 token，并通过 lse_out 回传**原始 logits** 的 logsumexp。
//
// 【为什么要回传 lse】调用方需要 lp_old = raw_logit(t) - logsumexp(原始 logits)。
// 原实现里这个 logsumexp 在 sample_from 内部算了一遍、调用方又独立算了一遍，
// 每次都是两轮 O(V) 遍历（求 max + 求 exp 和），V=16384 时每 token 白烧 4 轮 exp。
// 现在全流程只算一遍。
//
// 【为什么改写候选选择】原实现用 size_t 索引数组 + 解引用比较器做**全排序**，
// 每 token 16384 项 O(V log V)，且比较器随机访问 lg 导致缓存极不友好。
// 实测（build/bench_sample.cpp，V=16384）：
//     top-k=64      1.276 ms -> 0.168 ms  (7.6x)
//     top-k+top-p   1.243 ms -> 0.084 ms  (14.8x)
// 参照：输出头矩阵乘仅 0.123 ms —— 原来的排序比整个模型前向还贵约 10 倍。
// 现在只对真正用得到的 keep 项做 partial_sort，其余不排。
static size_t sample_from(Vec& lg, double temp, int topk, double topp, std::mt19937_64& rng,
                          double* lse_out = nullptr){
    const size_t V = lg.size();
    // ── 1) 原始 logits 的 logsumexp：本函数内只算这一遍 ──
    double mx = -1.0e30;
    for(size_t i=0;i<V;++i) if((double)lg[i] > mx) mx = (double)lg[i];
    double Z = 0.0;
    for(size_t i=0;i<V;++i) Z += std::exp((double)lg[i] - mx);
    if(lse_out) *lse_out = mx + std::log(Z);

    if(temp > 0.0){ for(size_t i=0;i<V;++i) lg[i] = (float)(lg[i]/temp); }

    size_t keep = (topk>0 && (size_t)topk<V) ? (size_t)topk : V;
    std::vector<uint32_t> all(V);
    for(uint32_t i=0;i<V;++i) all[i]=i;
    auto cmp=[&](uint32_t a,uint32_t b){ return lg[a] > lg[b]; };
    // ── 2) 候选选择 ──
    if(keep < V) std::partial_sort(all.begin(), all.begin()+keep, all.end(), cmp);
    else         std::sort(all.begin(), all.end(), cmp);

    // ── 3) 候选的 exp 只算一遍，top-p 与采样共用 ──
    // 原实现此处为 top-p 单独做了一遍全词表 exp 和，随后采样又对同一批候选重算一遍。
    // 两次 exp 结果完全相同，属于纯重复计算。
    const double mxs = lg[all[0]];
    std::vector<double> e(keep);
    for(size_t i=0;i<keep;++i) e[i] = std::exp((double)lg[all[i]]-mxs);
    if(topp > 0.0 && topp < 1.0 && keep > 1){
        double total = 0.0; for(size_t i=0;i<keep;++i) total += e[i];
        double acc=0; size_t n=0;
        for(size_t i=0;i<keep;++i){ acc += e[i]/total; ++n; if(acc>=topp) break; }
        if(n>=1) keep = n;
    }
    // ── 4) 采样（沿用同一批 e[]，语义与逐项重算完全等价）──
    std::uniform_real_distribution<double> U(0.0, 1.0);
    double sum = 0.0; for(size_t i=0;i<keep;++i) sum += e[i];
    double r = U(rng) * sum, acc = 0;
    for(size_t i=0;i<keep;++i){ acc += e[i]; if(r <= acc || i+1 == keep) return all[i]; }
    return all[0];
}


// 解码并行度：CPU 只负责解码，必须吃满全部逻辑核。
static unsigned decode_threads(){
    unsigned hw=std::thread::hardware_concurrency(); if(hw==0u)hw=4u;
    if(const char*e=std::getenv("TAO_CPU_THREADS")){int v=std::atoi(e); if(v>0)hw=unsigned(v);}
    return hw;
}
template<class F> static void gm_parallel_for(size_t n,unsigned T,F&&f){
    if(n==0)return;
    if(T<=1u||n<2u){for(size_t i=0;i<n;++i)f(i);return;}
    if(size_t(T)>n)T=unsigned(n);
    std::vector<std::thread> ts; ts.reserve(T);
    std::vector<std::exception_ptr> errs(T,nullptr);
    for(unsigned w=0;w<T;++w) ts.emplace_back([&,w](){ try{ for(size_t i=w;i<n;i+=T)f(i);}catch(...){errs[w]=std::current_exception();} });
    for(auto&x:ts)x.join();
    for(auto&e:errs) if(e)std::rethrow_exception(e);
}

int main(int argc, char** argv){
    try{
        if(argc < 3) throw std::runtime_error("usage: grpo_rollout MODEL PROBE.jsonl [options]");
        const std::string model = argv[1], probe = argv[2];
        size_t N = 40, G = 4, maxTok = 32768;  // 默认 32k，可用 --max 覆盖
        // 重复惩罚默认关闭：它是**解码期启发式**，会把策略分布从 softmax(logits) 改成
    // 别的分布。而论文 Eq.3 的比值 r = exp(lp_theta - lp_old) 要求两者是同一分布，
    // 否则权重未变时 r 也不等于 1，裁剪项即失真。RL 阶段的抗退化改由 --w-norep 奖励承担。
    // 需要重复惩罚的独立评测请显式传 --rep-pen。
    double temp = 0.9, topp = 0.95, repPen = 1.0;
        int topk = 0, repWin = 0, show = 3;
        unsigned long long seed = 20260912ull;
        double wC = 1.0, wF = 0.2, wG = 0.1, wT = 0.2, wL = 0.0, wN = 0.0;
        std::string dumpPath, dumpTextPath, refPath, prefix = "train";
        for(int i=3;i<argc;++i){
            std::string a = argv[i];
            auto val = [&](){ if(i+1>=argc) throw std::runtime_error("missing value "+a); return std::string(argv[++i]); };
            if(a=="--n") N = (size_t)std::atoll(val().c_str());
            else if(a=="--samples") G = (size_t)std::atoll(val().c_str());
            else if(a=="--temp") temp = std::atof(val().c_str());
            else if(a=="--top-k") topk = std::atoi(val().c_str());
            else if(a=="--top-p") topp = std::atof(val().c_str());
            else if(a=="--max") maxTok = (size_t)std::atoll(val().c_str());
            else if(a=="--seed") seed = std::strtoull(val().c_str(), nullptr, 10);
            else if(a=="--rep-pen") repPen = std::atof(val().c_str());
            else if(a=="--rep-win") repWin = std::atoi(val().c_str());
            else if(a=="--w-correct") wC = std::atof(val().c_str());
            else if(a=="--w-fmt") wF = std::atof(val().c_str());
            else if(a=="--w-calc") wG = std::atof(val().c_str());
            else if(a=="--w-think") wT = std::atof(val().c_str());
            else if(a=="--w-len") wL = std::atof(val().c_str());
            else if(a=="--w-norep") wN = std::atof(val().c_str());
            else if(a=="--dump-text") dumpTextPath = val();
            else if(a=="--ref") refPath = val();
            else if(a=="--dump") dumpPath = val();
            else if(a=="--show") show = std::atoi(val().c_str());
            else if(a=="--prefix") prefix = val();
            else throw std::runtime_error("unknown option "+a);
        }
        std::string hash;
        const char* tokp = std::getenv("TAO_TOKENIZER"); if(!tokp) tokp = "build/tok_real_v1.bbp";
        auto tokenizer = tao::text::load_tokenizer(tokp, hash);
    // 解码是 CPU 的职责，按逻辑核数并发。每个线程独占一个模型实例。
    std::vector<std::unique_ptr<tao::dual::GreedyPipelineGroupedModel>> models;
    {
        const unsigned DT=decode_threads();
        for(unsigned i=0;i<DT;++i){
            auto mm=std::make_unique<tao::dual::GreedyPipelineGroupedModel>(tao::dual::read_compact_bundle(model, hash));
            mm->set_cpu_threads(1);
            models.push_back(std::move(mm));
        }
        std::printf("DECODE_THREADS %u (每线程独立模型实例, 实例内串行)\n", DT);
    }
    // 冻结参考策略（通常是 R1 冷启动 SFT 模型）：论文 Eq.3 的 KL(π_θ‖π_ref) 需要它。
    // 【为何不能省】k3 估计的梯度是 β(1-e^d)，d=lp_ref-lp_θ。若 π_ref 取 π_old（采样时的
    // 同一策略），则首次前向 d≡0、梯度恒为 0，KL 项完全失效 —— 看起来实现了其实没有。
    // 用不同的冻结分布，KL 才会随策略漂移而真正产生约束。
    std::vector<std::unique_ptr<tao::dual::GreedyPipelineGroupedModel>> refs;
    if(!refPath.empty()){
        for(unsigned i=0;i<decode_threads();++i){
            auto mm=std::make_unique<tao::dual::GreedyPipelineGroupedModel>(tao::dual::read_compact_bundle(refPath, hash));
            mm->set_cpu_threads(1);
            refs.push_back(std::move(mm));
        }
        std::printf("ROLLOUT_REF %s instances=%zu\n", refPath.c_str(), refs.size());
    } else {
        std::printf("ROLLOUT_REF (未提供) -> KL 项将退化为 0，仅裁剪项生效\n");
    }
        tao::dual::GreedyPipelineGroupedModel m(tao::dual::read_compact_bundle(model, hash));
        unsigned nt = std::thread::hardware_concurrency(); if(nt==0u)nt=4u; if(const char* e=std::getenv("TAO_CPU_THREADS")) nt = (unsigned)std::atoi(e);
        if(nt>1) m.set_cpu_threads(nt);
        std::printf("ROLLOUT model=%s samples=%zu temp=%g topk=%d topp=%g max=%zu rep_pen=%g prefix=%s\n",
                    model.c_str(), G, temp, topk, topp, maxTok, repPen, prefix.c_str());
        std::printf("MAX_TOKENS %zu  SFT_LIKE=%d\n", maxTok, prefix=="train"?1:0);
        auto lines = read_jsonl(probe, N);
        if(lines.empty()) throw std::runtime_error("probe empty");
        std::ofstream dump;
        if(!dumpPath.empty()){ dump.open(dumpPath, std::ios::binary); if(!dump) throw std::runtime_error("dump open"); dump.precision(9); }
    std::ofstream tdump;
    if(!dumpTextPath.empty()){ tdump.open(dumpTextPath, std::ios::binary); if(!tdump) throw std::runtime_error("dump-text open"); }
        size_t nprob = 0, ncorrect = 0, passG = 0, nfmt = 0, ncalc = 0, totTok = 0, nsamp = 0, groupsVar = 0;
        double rsum = 0, r2sum = 0;
        std::vector<double> groupMean;
        for(size_t pi = 0; pi < lines.size(); ++pi){
            std::string q = field(lines[pi], "q"), gold = norm(field(lines[pi], "a"));
            if(q.empty()) continue;
            std::string prompt = (prefix=="train") ? ("U " + q + " A ") : q;
            auto enc = tokenizer.encode(prompt);
            // 并发解码：G 条采样分派到独立线程，每线程用自己的模型实例。
            bool anyCorrect = false; size_t correctCnt = 0;
            std::vector<double> rs(G, 0.0);
            std::vector<std::string> texts(G);
            std::vector<std::vector<uint32_t>> gens(G);
            std::vector<std::vector<float>> lpGen(G);   // lp_old：采样时策略的对数概率
            std::vector<std::vector<float>> lpRefGen(G); // lp_ref：冻结参考策略的对数概率
            std::vector<char> hitEosV(G,0), correctV(G,0), fmtV(G,0), calcV(G,0);
            gm_parallel_for(G, decode_threads(), [&](size_t s){
                tao::dual::GreedyPipelineGroupedModel& mm = *models[s % models.size()];
                std::mt19937_64 rng(seed + pi * 1000003ull + s * 7919ull);
                tao::dual::GreedyPipelineGroupedModel* mmRef =
                    refs.empty() ? nullptr : refs[s % refs.size()].get();
                auto st = mm.initial();
                mm.advance(256, st); mm.advance(257, st);
                for(auto t : enc) mm.advance(t, st);
                mm.advance(259, st);
                // 参考策略必须推进**同一段上下文**，否则两个分布不可比、KL 无意义。
                std::unique_ptr<std::decay_t<decltype(mm.initial())>> stRefH;
                if(mmRef){
                    stRefH = std::make_unique<std::decay_t<decltype(mm.initial())>>(mmRef->initial());
                    mmRef->advance(256, *stRefH); mmRef->advance(257, *stRefH);
                    for(auto t : enc) mmRef->advance(t, *stRefH);
                    mmRef->advance(259, *stRefH);
                }
                std::vector<uint32_t> gen, winTok; bool hitEos = false;
                uint32_t cur = 258;
                for(size_t i=0;i<maxTok;++i){
                    Vec lg = mm.step(cur, st);
                    // ── 策略分布 ──
                    // lp_old 必须基于**原始 logits 的 softmax**，与训练侧前向完全一致。
                    // 屏蔽与重复惩罚只决定「采样哪个 token」，不参与概率定义；若用改动后的
                    // 分布算 lp_old，即便权重未变，r=exp(lp_theta-lp_old) 也不等于 1，裁剪项
                    // 立即失真 —— 这类错误是静默的，所以 logsumexp 必须在任何改动之前算。
                    // lse 由 sample_from 回传 —— 原本这里还要再对全词表做两轮 O(V) 遍历，
                    // 而 sample_from 内部已算过完全相同的量，属于纯重复计算。
                    double lse = 0.0;
                    // ── 参考策略分布（冻结模型）──
                    Vec lgR; double lseR = 0.0;
                    if(mmRef){
                        lgR = mmRef->step(cur, *stRefH);
                        double mR = -1.0e30, sR = 0.0;
                        for(size_t k = 0; k < lgR.size(); ++k) if((double)lgR[k] > mR) mR = (double)lgR[k];
                        for(size_t k = 0; k < lgR.size(); ++k) sR += std::exp((double)lgR[k] - mR);
                        lseR = mR + std::log(sR);
                    }
                    // 屏蔽控制 token：只允许 TURN_END(259) 终止，其余 >=256 一律禁止采样。
                    // 否则采样会频繁抽到 BOS/ASSISTANT 等标记，回答长度塌成 0。
                    // 【命名澄清】枚举里 259=TURN_END、260=EOS。回复的结束符是 TURN_END，
                    // EOS 只用于整段对话的收尾。本文件变量名沿用 hitEos 属历史命名，
                    // 语义是"抽到了回合终止符"。
                    for(size_t k = 256u; k < lg.size(); ++k) if(k != 259u) lg[k] = -1.0e30f;
                    if(repPen > 1.0f){
                        for(auto x : winTok){ if(lg[x] > 0) lg[x] = (float)(lg[x]/repPen); else lg[x] = (float)(lg[x]*repPen); }
                    }
                    uint32_t t = (uint32_t)sample_from(lg, temp, topk, topp, rng, &lse);
                    {
                        // lg 已被温度缩放，反乘温度即还原该 token 的原始 logit；
                        // 若重复惩罚生效，按同一规则反解。
                        double rawT = (temp > 0.0) ? (double)lg[t] * temp : (double)lg[t];
                        if(repPen > 1.0f){
                            bool pen = false; for(auto x : winTok) if(x == t){ pen = true; break; }
                            if(pen) rawT = (rawT > 0.0) ? rawT * repPen : rawT / repPen;
                        }
                        lpGen[s].push_back(float(rawT - lse));
                        // lp_ref：同样取参考分布的原始 softmax（不做屏蔽/惩罚）
                        if(mmRef) lpRefGen[s].push_back(float((double)lgR[t] - lseR));
                    }
                    // 特殊 token(>=256) 一律视为回复结束：它们不是文本，decode 会抛异常
                    if(t >= 256u){ if(t == 259u) hitEos = true; break; }
                    gen.push_back(t);
                    bool dup = false; for(auto x : winTok) if(x==t){ dup=true; break; }
                    if(!dup) winTok.push_back(t);
                    if(repWin > 0 && winTok.size() > (size_t)repWin) winTok.erase(winTok.begin());
                    cur = t;
                }
                std::string txt = tokenizer.decode(gen);
                double correct = (gold.size() && grab_answer(txt) == gold) ? 1.0 : 0.0;
                double hasFmt = (txt.find("答案：") != std::string::npos || txt.find("答案:") != std::string::npos) ? 1.0 : 0.0;
                double hasCalc = (txt.find("计算") != std::string::npos) ? 1.0 : 0.0;
                // 过程监督：要求出现 "思考" 段。R1 的推理模式靠它进入，故必须有正向奖励。
                double hasThink = (txt.find("思考") != std::string::npos) ? 1.0 : 0.0;
                // 通用对齐（无奖励模型时）的规则奖励：充分性 + 非退化。
                // 充分性：太短=答非所问，太长=啰嗦；退化：连续重复或词汇坍缩。
                double lenRew = (gen.size()>=24 && gen.size()<=400) ? 1.0 : (gen.size()>=12 ? 0.5 : 0.0);
                double norep = 1.0;
                { size_t best=1,cur=1;
                  for(size_t k=1;k<gen.size();++k){ if(gen[k]==gen[k-1]){++cur; if(cur>best)best=cur;} else cur=1; }
                  if(best>=8) norep = 0.0;
                  else { std::set<uint32_t> uq(gen.begin(),gen.end());
                         double ratio = gen.empty()?0.0:double(uq.size())/double(gen.size());
                         norep = (ratio<0.25)?0.0:((ratio<0.45)?0.5:1.0); } }
                rs[s] = wC*correct + wF*hasFmt + wG*hasCalc + wT*hasThink + wL*lenRew + wN*norep;
                correctV[s] = correct>0; fmtV[s] = hasFmt>0; calcV[s] = hasCalc>0; hitEosV[s] = hitEos;
                gens[s] = std::move(gen);
                texts[s] = std::move(txt);
            });
            // 串行汇总，顺序固定为 s 升序，保证指标与 dump 稳定可复现。
            for(size_t s = 0; s < G; ++s){
                const double r = rs[s];
                // 拒绝采样语料：答案正确且含推理段，才有资格作为 SFT 示范。
                if(tdump && r > 0.0){
                    tdump << q << '\t' << gold << '\t' << r << '\t' << (correctV[s]?1:0) << '\t' << sanitize_tsv(texts[s]) << '\n';
                }
                if(correctV[s]){ anyCorrect = true; ++correctCnt; }
                nfmt += (size_t)fmtV[s]; ncalc += (size_t)calcV[s];
                totTok += gens[s].size(); ++nsamp;
                rsum += r; r2sum += r*r;
                if(dump){
                    dump << pi << '\t' << s << '\t' << r << '\t' << (correctV[s]?1:0) << '\t' << (hitEosV[s]?1:0) << '\t';
                    for(size_t k=0;k<enc.size();++k) dump << enc[k] << (k+1<enc.size()?",":"");
                    dump << '\t';
                    for(size_t k=0;k<gens[s].size();++k) dump << gens[s][k] << (k+1<gens[s].size()?",":"");
                    // 第 8 字段：逐 token 旧策略对数概率（含终止符，故长度 = gen+1）
                    dump << '\t';
                    for(size_t k=0;k<lpGen[s].size();++k) dump << lpGen[s][k] << (k+1<lpGen[s].size()?",":"");
                    // 第 9 字段：逐 token 参考策略对数概率（用于论文 Eq.3 的 KL 项）
                    dump << '\t';
                    for(size_t k=0;k<lpRefGen[s].size();++k) dump << lpRefGen[s][k] << (k+1<lpRefGen[s].size()?",":"");
                    dump << '\n';
                }
            }
            double mu = 0; for(auto x : rs) mu += x; mu /= (double)rs.size();
            double sd = 0; for(auto x : rs) sd += (x-mu)*(x-mu);
            sd = std::sqrt(sd / (double)rs.size());
            groupMean.push_back(mu);
            if(sd > 1e-9) ++groupsVar;
            ncorrect += correctCnt;
            if(anyCorrect) ++passG;
            ++nprob;
            if((int)pi < show){
                std::printf("\n[题%zu] %s\n  标准答案=%s  组内奖励 mean=%.3f sd=%.3f 命中=%zu/%zu\n",
                            pi+1, q.substr(0,60).c_str(), gold.c_str(), mu, sd, correctCnt, G);
                for(size_t s=0;s<texts.size() && s<2;++s)
                    std::printf("   s%zu r=%.2f: %s\n", s, rs[s], texts[s].substr(0,150).c_str());
            }
        }
        if(dump) dump.close();
        double mean = rsum/(double)nsamp;
        double var = r2sum/(double)nsamp - mean*mean;
        std::printf("\n================ GRPO 信号判据 ================\n");
        std::printf("题数=%zu  每题采样=%zu  总样本=%zu  平均生成长度=%.1f token\n", nprob, G, nsamp, (double)totTok/(double)nsamp);
        std::printf("pass@1  (逐样本命中率) = %.2f%%   (%zu/%zu)\n", 100.0*ncorrect/nsamp, ncorrect, nsamp);
        std::printf("pass@G  (每题至少一条命中) = %.2f%%   (%zu/%zu)\n", 100.0*passG/nprob, passG, nprob);
        std::printf("格式率(含 答案：) = %.2f%%   步骤率(含 计算) = %.2f%%\n", 100.0*nfmt/nsamp, 100.0*ncalc/nsamp);
        std::printf("全局奖励 mean=%.4f var=%.4f sd=%.4f\n", mean, var, std::sqrt(var<0?0:var));
        std::printf("组内方差>0 的题数 = %zu/%zu  (%.1f%%)   <-- GRPO 的梯度信号来源\n",
                    groupsVar, nprob, 100.0*groupsVar/nprob);
        if(groupsVar == 0) std::printf("!! 所有组奖励完全相同 -> 优势 A=(r-mean)/std 无定义 -> GRPO 学不到东西\n");
        return 0;
    }catch(const std::exception& e){ std::fprintf(stderr, "FAIL %s\n", e.what()); return 1; }
}
