// 诊断解码器：定位并修复「退化重复」。
//
// 与 h2r_cpu.exe 的差异仅在于**解码策略**：本工具用 model.step() 取完整 logits，
// 因此可以施加重复惩罚 / n-gram 阻断 / top-k / 温度，再 argmax。
// 默认参数（--rep-penalty 1 --ngram 0 --top-k 0 --temp 1）必须与 h2r_cpu.exe 逐字一致，
// 这是判定「重复是解码问题还是架构问题」的对照基线。
//
// 构建（与 h2r_cpu.exe 同 flag）：
//   cl /nologo /O2 /EHsc /std:c++17 /arch:AVX2 /DTAO_NO_FFN /DTAO_DELTA_MEM /I src \
//      src\decode_diag.cpp /Febuild\decode_diag.exe /Fobuild\decode_diag.obj bcrypt.lib
//
// 用法: decode_diag MODEL.dsb [选项]   从 stdin 读 prompt，/reset 重置，/quit 退出
#define NOMINMAX
#define TAO_INPUT_SCALE
#define TAO_CPU_AVX2
#include "tokenizer_file.hpp"
#include "greedy_pipeline_grouped_model.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using tao::dual::GreedyPipelineGroupedModel;
using tao::dual::LayerState;
using tao::dual::Vec;

namespace {

struct Opts {
    double rep_penalty = 1.0;   // >1 惩罚已出现过的 token（CTRL 式）
    int    ngram       = 0;     // >0 阻断会重复 n-gram 的 token
    int    top_k       = 0;     // >0 仅保留前 k
    double temp        = 1.0;
    int    max_tokens  = 64;
    int    dump        = 0;     // >0 打印前若干步的 top-k logits
    bool   stats       = false; // 打印每步最大 logit 与熵
    int    window      = 0;     // >0 仅对最近 window 个 token 施加重复惩罚
};

const char* TOKEN_NOTE(uint32_t t) {
    switch (t) {
        case 256: return "<BOS>";
        case 257: return "<USER>";
        case 258: return "<ASSISTANT>";
        case 259: return "<TURN_END>";
        case 260: return "<EOS>";
        default:  return nullptr;
    }
}

struct StepInfo { uint32_t tok; float maxlogit; double entropy; };

// 施加惩罚/阻断后返回选中的 token；同时回填诊断信息。
uint32_t pick(Vec& logits, const std::vector<uint32_t>& gen, const Opts& o, StepInfo& info) {
    const size_t V = logits.size();
    const size_t from = (o.window > 0 && gen.size() > size_t(o.window)) ? gen.size() - size_t(o.window) : 0;

    if (o.rep_penalty != 1.0) {
        for (size_t i = from; i < gen.size(); ++i) {
            const uint32_t t = gen[i];
            if (t < V) { float& z = logits[t]; z = (z > 0.f) ? float(z / o.rep_penalty) : float(z * o.rep_penalty); }
        }
    }

    std::vector<char> blocked;
    if (o.ngram > 0 && int(gen.size()) >= o.ngram) {
        blocked.assign(V, 0);
        const int N = o.ngram;
        const size_t pre = size_t(N - 1);
        for (size_t i = 0; i + pre < gen.size(); ++i) {
            bool match = true;
            for (size_t j = 0; j < pre; ++j)
                if (gen[gen.size() - pre + j] != gen[i + j]) { match = false; break; }
            if (!match) continue;
            const size_t nxt = i + pre;
            if (nxt < gen.size() && gen[nxt] < V) blocked[gen[nxt]] = 1;
        }
        bool any = false;
        for (size_t t = 0; t < V; ++t) if (!blocked[t] && std::isfinite(logits[t])) { any = true; break; }
        if (!any) blocked.clear();   // 全被阻断则放弃阻断，避免死锁
    }

    if (o.temp > 0.0 && o.temp != 1.0) for (auto& z : logits) z = float(z / o.temp);

    // 诊断：最大 logit 与熵（在惩罚之后、top-k 之前）
    double mx = -1e30, sum = 0.0;
    int finite = 0;
    for (size_t t = 0; t < V; ++t) {
        if (!blocked.empty() && blocked[t]) continue;
        if (!std::isfinite(logits[t])) continue;
        ++finite;
        if (logits[t] > mx) mx = logits[t];
    }
    for (size_t t = 0; t < V; ++t) {
        if (!blocked.empty() && blocked[t]) continue;
        if (!std::isfinite(logits[t])) continue;
        sum += std::exp(double(logits[t]) - mx);
    }
    info.maxlogit = float(mx);
    info.entropy = (finite > 0 && sum > 0) ? (std::log(sum) + mx * 0.0) : 0.0;
    {   // 真实熵
        double H = 0.0;
        for (size_t t = 0; t < V; ++t) {
            if (!blocked.empty() && blocked[t]) continue;
            if (!std::isfinite(logits[t])) continue;
            const double p = std::exp(double(logits[t]) - mx) / sum;
            if (p > 0) H -= p * std::log(p);
        }
        info.entropy = H;
    }

    // top-k 掩码
    std::vector<char> keep;
    if (o.top_k > 0 && size_t(o.top_k) < V) {
        std::vector<uint32_t> idx;
        idx.reserve(V);
        for (size_t t = 0; t < V; ++t) {
            if (!blocked.empty() && blocked[t]) continue;
            if (!std::isfinite(logits[t])) continue;
            idx.push_back(uint32_t(t));
        }
        std::partial_sort(idx.begin(), idx.begin() + std::min<size_t>(o.top_k, idx.size()), idx.end(),
                          [&](uint32_t a, uint32_t b) { return logits[a] > logits[b]; });
        keep.assign(V, 0);
        for (size_t i = 0; i < std::min<size_t>(o.top_k, idx.size()); ++i) keep[idx[i]] = 1;
    }

    uint32_t best = 0;
    float bv = -3.4e38f;
    bool found = false;
    for (size_t t = 0; t < V; ++t) {
        if (!blocked.empty() && blocked[t]) continue;
        if (!keep.empty() && !keep[t]) continue;
        if (!std::isfinite(logits[t])) continue;
        if (logits[t] > bv) { bv = logits[t]; best = uint32_t(t); found = true; }
    }
    if (!found) throw std::runtime_error("no admissible token");
    info.tok = best;
    return best;
}

void print_topk(const Vec& logits, const std::string& tag, int k) {
    std::vector<uint32_t> idx(logits.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = uint32_t(i);
    const size_t kk = std::min<size_t>(size_t(k), idx.size());
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                      [&](uint32_t a, uint32_t b) { return logits[a] > logits[b]; });
    std::printf("  %s top%zu:", tag.c_str(), kk);
    for (size_t i = 0; i < kk; ++i)
        std::printf(" [%u]=%.3f", idx[i], logits[idx[i]]);
    std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error("usage: decode_diag MODEL.dsb [--rep-penalty F] [--ngram N] [--top-k K] [--temp T] [--max M] [--dump K] [--stats] [--window W]");
        const std::string model_path = argv[1];
        Opts o;
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            auto val = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
                return std::string(argv[++i]);
            };
            if (a == "--rep-penalty") o.rep_penalty = std::stod(val());
            else if (a == "--ngram")  o.ngram = std::stoi(val());
            else if (a == "--top-k")  o.top_k = std::stoi(val());
            else if (a == "--temp")   o.temp = std::stod(val());
            else if (a == "--max")    o.max_tokens = std::stoi(val());
            else if (a == "--dump")   o.dump = std::stoi(val());
            else if (a == "--window") o.window = std::stoi(val());
            else if (a == "--stats")  o.stats = true;
            else throw std::runtime_error("unknown option: " + a);
        }
        if (o.max_tokens < 1 || o.max_tokens > 4096) throw std::runtime_error("--max 1..4096");

        std::string hash;
        auto tokenizer = tao::text::load_tokenizer("build/formal_tokenizer.bbp", hash);
        GreedyPipelineGroupedModel model(tao::dual::read_compact_bundle(model_path, hash));
        auto state = model.initial();
        bool fresh = true;
        std::fprintf(stderr, "READY rep_penalty=%g ngram=%d top_k=%d temp=%g max=%d window=%d\n",
                     o.rep_penalty, o.ngram, o.top_k, o.temp, o.max_tokens, o.window);

        std::string prompt;
        size_t request = 0;
        while (std::getline(std::cin, prompt)) {
            if (prompt == "/quit") break;
            if (prompt == "/reset") { state = model.initial(); fresh = true; std::cout << "RESET\n"; continue; }
            if (prompt.size() > 8192) { std::cout << "ERROR prompt too long\n"; continue; }

            auto encoded = tokenizer.encode(prompt);
            if (fresh) { model.advance(256, state); fresh = false; }
            model.advance(257, state);
            for (auto t : encoded) model.advance(t, state);
            model.advance(259, state);

            std::vector<uint32_t> gen;
            std::vector<StepInfo> steps;
            uint32_t cur = 258;
            int end = -1;
            for (int i = 0; i < o.max_tokens; ++i) {
                Vec logits = model.step(cur, state);
                if (o.dump > 0 && int(gen.size()) < o.dump)
                    print_topk(logits, "step" + std::to_string(gen.size()), 8);
                StepInfo info{};
                const uint32_t next = pick(logits, gen, o, info);
                steps.push_back(info);
                if (next == 259 || next == 260) { model.advance(next, state); end = int(next); break; }
                gen.push_back(next);
                cur = next;
            }
            if (end < 0) model.advance(259, state);

            // 文本 token 的 id 有两种：<256 是原始字节，>=261 是 BPE 合并 token。
            // 必须走 tokenizer.decode，不能直接 char(t)（否则合并 token 会输出垃圾）。
            std::string out = tokenizer.decode(gen);
            std::cout << "BEGIN_REPLY " << ++request << "\n" << (out.empty() ? "[empty]" : out) << "\n";
            std::cout << "END_REPLY tokens=" << gen.size() << " end=" << end
                      << " truncated=" << (end < 0 ? 1 : 0);
            if (o.stats && !steps.empty()) {
                double H = 0, mx = -1e30;
                for (const auto& s : steps) { H += s.entropy; mx = std::max(mx, double(s.maxlogit)); }
                std::cout << " mean_entropy=" << (H / steps.size()) << " max_logit=" << mx;
            }
            std::cout << std::endl;
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
