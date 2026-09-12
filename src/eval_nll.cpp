// eval_nll.cpp -- 独立的留出集 NLL / 困惑度评测器（纯 CPU，多核）。
//
// 为什么单独做一个：训练器虽然每 150 步会打印一次 CONV_PROBE 留出 NLL，但那是
// 训练过程的副产品。评测命令应当能在任意检查点上独立复现，且不占用 GPU
// （训练正在用），所以这里走完整的 CPU 推理栈：
//   CpuModel + GreedyPipelineGroupedModel（行并行，set_cpu_threads）
//
// 语义与训练器的留出评估完全一致：逐文档重置状态；对第 i 个 token 用第 i-1 个
// token 前向后取目标 token 的负对数似然；不做任何截断、采样或平滑。
//
// 用法：
//   eval_nll --model build\L1_pretrain\step_5000\final.dsb \
//            --text data\holdout_zh.txt --tokenizer build\tok_v2.bbp \
//            --threads 16 [--max-docs N] [--max-tokens N] [--json]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <algorithm>

#include "byte_bpe.hpp"
#include "tokenizer_file.hpp"
#include "dual_model_bundle.hpp"
#include "dual_state_cpu.hpp"
#include "greedy_pipeline_grouped_model.hpp"

using namespace tao::dual;   // CpuModel / Vec / LayerState / load_bundle / GreedyPipelineGroupedModel
using namespace tao::text;   // ByteBpe / load_tokenizer

static std::string opt(int argc, char** argv, const char* name, const char* dflt = "") {
    for (int i = 1; i + 1 < argc; ++i) if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    return dflt;
}

int main(int argc, char** argv) {
    try {
        const std::string model = opt(argc, argv, "--model");
        const std::string textp = opt(argc, argv, "--text");
        const std::string tokp  = opt(argc, argv, "--tokenizer", "build/tok_v2.bbp");
        if (model.empty() || textp.empty()) {
            std::fprintf(stderr,
                "usage: eval_nll --model M.dsb --text F.txt [--tokenizer T] "
                "[--threads N] [--max-docs N] [--max-tokens N] [--json]\n");
            return 2;
        }
        unsigned threads = (unsigned)(std::max)(1, std::atoi(opt(argc, argv, "--threads", "8").c_str()));
        size_t max_docs  = std::strtoull(opt(argc, argv, "--max-docs", "0").c_str(), nullptr, 10);
        size_t max_tok   = std::strtoull(opt(argc, argv, "--max-tokens", "0").c_str(), nullptr, 10);
        const bool json  = std::strcmp(opt(argc, argv, "--json", "0").c_str(), "1") == 0;

        std::string fp;
        ByteBpe bpe = load_tokenizer(tokp, fp);
        std::fprintf(stderr, "EVAL_TOKENIZER fingerprint=%s\n", fp.c_str());
        CpuModel cpu = load_bundle(model, fp);
        // 打印模型身份，便于和训练日志对照
        std::fprintf(stderr,
            "EVAL_MODEL layers=%u d=%u s=%u m=%u dk=%u vocab=%u\n",
            cpu.c.layers, cpu.c.d, cpu.c.s, cpu.c.m, cpu.c.dk, cpu.c.vocab);

        GreedyPipelineGroupedModel hm(cpu);
        hm.set_cpu_threads(threads);

        // 语料：一行一篇文档（UTF-8）。空行跳过。
        std::ifstream f(textp, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open --text");
        std::vector<std::string> docs;
        { std::string line;
          while (std::getline(f, line)) {
              if (!line.empty() && line.back() == '\r') line.pop_back();
              if (line.empty()) continue;
              docs.push_back(std::move(line));
              if (max_docs && docs.size() >= max_docs) break;
          } }

        const uint32_t V = cpu.c.vocab;
        double nll_sum = 0.0;
        size_t ntok = 0, ndoc = 0, skipped = 0;
        std::vector<double> per_doc;

        for (const auto& d : docs) {
            std::vector<uint32_t> ids = bpe.encode(d);
            if (ids.size() < 2) { ++skipped; continue; }
            auto st = hm.initial();
            double s = 0.0; size_t n = 0;
            for (size_t i = 1; i < ids.size(); ++i) {
                Vec lg = hm.step((uint32_t)ids[i - 1], st);
                if (lg.size() != V) throw std::runtime_error("logit width != vocab");
                float mx = lg[0];
                for (uint32_t k = 1; k < V; ++k) if (lg[k] > mx) mx = lg[k];
                double se = 0.0;
                for (uint32_t k = 0; k < V; ++k) se += std::exp(double(lg[k]) - double(mx));
                s += (double(mx) + std::log(se)) - double(lg[(size_t)ids[i]]);
                ++n;
                if (max_tok && ntok + n >= max_tok) break;
            }
            if (!n) { ++skipped; continue; }
            nll_sum += s; ntok += n; ++ndoc;
            per_doc.push_back(s / double(n));
            if (max_tok && ntok >= max_tok) break;
        }

        if (!ndoc) throw std::runtime_error("no evaluable document");
        const double mean_nll = nll_sum / double(ntok);
        const double ppl = std::exp(mean_nll);
        std::vector<double> sorted = per_doc;
        std::sort(sorted.begin(), sorted.end());
        auto pct = [&](double p) {
            if (sorted.empty()) return 0.0;
            size_t i = (size_t)(p * double(sorted.size() - 1));
            return sorted[std::min(i, sorted.size() - 1)];
        };

        if (json) {
            std::printf("{\"model\":\"%s\",\"text\":\"%s\",\"docs\":%zu,\"skipped\":%zu,"
                        "\"tokens\":%zu,\"nll\":%.6f,\"ppl\":%.4f,"
                        "\"doc_nll_p50\":%.6f,\"doc_nll_p90\":%.6f}\n",
                        model.c_str(), textp.c_str(), ndoc, skipped, ntok,
                        mean_nll, ppl, pct(0.5), pct(0.9));
        } else {
            std::printf("EVAL_NLL model=%s text=%s docs=%zu skipped=%zu tokens=%zu "
                        "nll=%.6f ppl=%.4f doc_nll_p50=%.6f doc_nll_p90=%.6f\n",
                        model.c_str(), textp.c_str(), ndoc, skipped, ntok,
                        mean_nll, ppl, pct(0.5), pct(0.9));
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "EVAL_FAIL %s\n", e.what());
        return 1;
    }
}
