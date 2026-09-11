// build_corpus —— 语料处理模块的驱动：原始语料 → 理好的训练语料（分片 + 清单 + 课程表）。
//
// 用法:
//   build_corpus --out DIR --tokenizer TOK [选项] INPUT [INPUT...]
//
// 选项:
//   --format auto|dialogue|plain|jsonl   输入格式（默认 auto 按扩展名/内容判定）
//   --stage NAME                         本批语料的课程阶段标签（core|mixed|interfere）
//   --seal FILE                          封存实体清单（每行一个，评测 held-out 用）
//   --docs-per-shard N                   每片文档数（默认 20000）
//   --whole-file                         plain 模式下整文件作为一档
//   --no-near-dedup                      关闭近似去重
//   --near-threshold F                   近似去重阈值（默认 0.8）
//   --min-bytes N                        最小文档字节数（默认 64）
//   --max-repeat F                       退化重复上限（默认 0.35）
//   --dialogue-out                       输出 TLP2 对话格式（默认 TLP3 纯文本）
//   --dry-run                            只统计不产出
//
// 说明：输出目录必须为空，绝不覆盖既有产物。
#define NOMINMAX
#include "corpus_pipeline.hpp"
#include <iostream>

using namespace tao::corpus;

static void usage() {
    std::cerr <<
        "usage: build_corpus --out DIR --tokenizer TOK [--format auto|dialogue|plain|jsonl]\n"
        "                    [--stage NAME] [--seal FILE] [--docs-per-shard N] [--whole-file]\n"
        "                    [--doc-sep MARKER] [--preserve-layout]\n"
        "                    [--no-near-dedup] [--near-threshold F] [--min-bytes N]\n"
        "                    [--max-repeat F] [--dialogue-out] [--dry-run] INPUT [INPUT...]\n";
}

int main(int argc, char** argv) {
    try {
        PipelineConfig pc;
        bool dry=false, dialogue_out=false;
        for (int i=1;i<argc;++i) {
            const std::string a=argv[i];
            auto next=[&](const char* what)->std::string{
                if (i+1>=argc) throw std::runtime_error(std::string("missing value for ")+what);
                return std::string(argv[++i]);
            };
            if (a=="--out") pc.out_dir=next("--out");
            else if (a=="--tokenizer") { /* 位置在下方统一读取 */ pc.inputs.push_back("__TOK__"+next("--tokenizer")); }
            else if (a=="--format") pc.input_format=next("--format");
            else if (a=="--stage") pc.stage=next("--stage");
            else if (a=="--seal") pc.seal_file=next("--seal");
            else if (a=="--docs-per-shard") pc.docs_per_shard=size_t(std::stoull(next("--docs-per-shard")));
            else if (a=="--whole-file") pc.whole_file=true;
            else if (a=="--doc-sep") pc.doc_sep=next("--doc-sep");
            else if (a=="--preserve-layout") pc.preserve_layout=true;
            else if (a=="--no-near-dedup") pc.near_dedup=false;
            else if (a=="--near-threshold") pc.near_dup_threshold=std::stod(next("--near-threshold"));
            else if (a=="--min-bytes") pc.quality.min_bytes=size_t(std::stoull(next("--min-bytes")));
            else if (a=="--max-repeat") pc.quality.max_repeat=std::stod(next("--max-repeat"));
            else if (a=="--dialogue-out") dialogue_out=true;
            else if (a=="--dry-run") dry=true;
            else if (a=="--help" || a=="-h") { usage(); return 0; }
            else if (a.size()>1 && a[0]=='-') throw std::runtime_error("unknown option: "+a);
            else pc.inputs.push_back(a);
        }
        // 分离 tokenizer 占位
        fs::path tok;
        std::vector<fs::path> inputs;
        for (const auto& s : pc.inputs) {
            const std::string str=s.string();
            if (str.rfind("__TOK__",0)==0) tok=str.substr(7);
            else inputs.push_back(s);
        }
        pc.inputs=std::move(inputs);
        require(!pc.out_dir.empty(), "--out required");
        require(!tok.empty(), "--tokenizer required");
        require(!pc.inputs.empty(), "at least one INPUT required");
        require(pc.docs_per_shard>=1 && pc.docs_per_shard<=1000000, "--docs-per-shard 1..1000000");
        require(pc.stage=="core"||pc.stage=="mixed"||pc.stage=="interfere", "--stage core|mixed|interfere");
        if (dry) pc.emit=false;

        std::string digest;
        auto bpe=tao::text::load_tokenizer(tok.string(), digest);
        std::cout << "CORPUS_TOKENIZER path=" << tok.generic_string()
                  << " digest=" << digest << " merges=" << bpe.merges.size() << "\n";
        const auto st=run_pipeline(pc, bpe, digest, dialogue_out);
        print_stats(st);
        if (!dry) std::cout << "CORPUS_DONE out=" << pc.out_dir.generic_string()
                            << " format=" << (dialogue_out?"TLP2":"TLP3") << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "CORPUS_FAILED " << e.what() << "\n";
        usage();
        return 1;
    }
}
