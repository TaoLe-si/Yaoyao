// entropy_ref.cpp -- 留出文本上的一元/二元参考下界。
//
// 为什么需要它：报告的 NLL=6.75 单独看没有意义 —— 必须知道"不学任何结构的模型"
// 能拿到多少。若一元模型就有 6.8，说明模型几乎没学到东西；若一元是 9 而模型是 6.75，
// 才说明它确实抓住了结构。前一半文档拟合、后一半文档评测，避免自评。
//
// 用法：entropy_ref --text data\holdout_zh.txt --tokenizer build\tok_v2.bbp
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <stdexcept>
#include "byte_bpe.hpp"
#include "tokenizer_file.hpp"

using namespace tao::text;

int main(int argc, char** argv) {
    try {
        std::string textp = "data/holdout_zh.txt", tokp = "build/tok_v2.bbp";
        for (int i = 1; i + 1 < argc; ++i) {
            if (!std::strcmp(argv[i], "--text")) textp = argv[++i];
            else if (!std::strcmp(argv[i], "--tokenizer")) tokp = argv[++i];
        }
        std::string fp;
        ByteBpe bpe = load_tokenizer(tokp, fp);
        // 256 字节 + 7 个特殊标记 + BPE 合并
        const uint32_t V = (uint32_t)(263 + bpe.merges.size());
        std::fprintf(stderr, "ENTROPY_TOKENIZER fingerprint=%s vocab=%u\n", fp.c_str(), V);

        std::vector<std::string> docs;
        { std::ifstream f(textp, std::ios::binary);
          if (!f) throw std::runtime_error("cannot open --text");
          std::string line;
          while (std::getline(f, line)) {
              if (!line.empty() && line.back() == '\r') line.pop_back();
              if (line.empty()) continue;
              docs.push_back(std::move(line));
          } }

        const size_t half = docs.size() / 2;
        std::vector<std::vector<uint32_t>> train, test;
        for (size_t i = 0; i < docs.size(); ++i) {
            auto ids = bpe.encode(docs[i]);
            if (ids.size() < 2) continue;
            if (i < half) train.push_back(std::move(ids));
            else          test.push_back(std::move(ids));
        }

        std::vector<double> uni(V, 0.0);
        double Tun = 0.0;
        std::vector<double> row(V, 0.0);
        std::map<uint64_t, double> bi;
        for (const auto& d : train)
            for (size_t i = 0; i < d.size(); ++i) {
                const uint32_t v = d[i];
                if (v < V) { uni[v] += 1.0; Tun += 1.0; }
                if (i >= 1) {
                    const uint32_t u = d[i - 1];
                    if (u < V && v < V) { bi[((uint64_t)u << 32) | v] += 1.0; row[u] += 1.0; }
                }
            }

        const double a = 1.0;   // add-one
        double nllU = 0.0, nllB = 0.0;
        size_t nU = 0, nB = 0;
        uint32_t best = 0;
        for (uint32_t i = 1; i < V; ++i) if (uni[i] > uni[best]) best = i;
        for (const auto& d : test)
            for (size_t i = 0; i < d.size(); ++i) {
                const uint32_t v = d[i];
                if (v >= V) continue;
                nllU += -std::log((uni[v] + a) / (Tun + a * (double)V));  ++nU;
                if (i >= 1) {
                    const uint32_t u = d[i - 1];
                    if (u < V) {
                        auto it = bi.find(((uint64_t)u << 32) | v);
                        const double c = (it == bi.end()) ? 0.0 : it->second;
                        nllB += -std::log((c + a) / (row[u] + a * (double)V));
                        ++nB;
                    }
                }
            }

        const double mu = nllU / (double)nU, mb = nllB / (double)nB;
        const double ptop = (uni[best] + a) / (Tun + a * (double)V);
        std::printf("ENTROPY text=%s docs=%zu train_docs=%zu test_docs=%zu\n",
                    textp.c_str(), docs.size(), train.size(), test.size());
        std::printf("ENTROPY_VOCAB V=%u distinct_in_train=%zu\n", V, [&]{ size_t c=0; for(double x:uni) if(x>0)++c; return c; }());
        std::printf("ENTROPY unigram_nll=%.6f ppl=%.2f  tokens=%zu\n", mu, std::exp(mu), nU);
        std::printf("ENTROPY bigram_nll=%.6f ppl=%.2f  tokens=%zu\n", mb, std::exp(mb), nB);
        std::printf("ENTROPY constant_top_nll=%.6f  top_token=%u\n", -std::log(ptop), best);
        return 0;
    } catch (const std::exception& e) {
        std::printf("ENTROPY_FAIL %s\n", e.what());
        return 1;
    }
}
