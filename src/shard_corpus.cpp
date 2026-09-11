// 分片工具：把一个大的 TLP2 语料切成 N 个等大小分片，并产出清单与课程表模板。
// 用途见 doc 20 §三：0.9B 级语料无法整文件读入内存，必须分片流式。
//
// 用法: shard_corpus SRC.bin TOKENIZER.bbp OUT_DIR DOCS_PER_SHARD [STAGE_TAG]
// 输出: OUT_DIR/shard_00000.bin ... (每个都是完整、独立、字节兼容的 TLP2)
//       OUT_DIR/shard_manifest.tsv   分片清单（序号/文件/docs/tokens/supervised/sha256/阶段）
//       OUT_DIR/curriculum.tsv       课程表模板（core / mixed / interfere 三段，按分片数三等分）
// 保护: OUT_DIR 必须不存在或为空，绝不覆盖既有产物。
#define NOMINMAX
#include "tokenizer_file.hpp"
#include "language_data_contract.hpp"
#include "bpe_pilot_reader.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <iterator>
#include <stdexcept>
#include <cstdio>
namespace fs = std::filesystem;
using namespace tao::data;
static void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
static void le32(std::ostream& out, uint32_t n) { for (int i=0;i<4;++i) out.put(char(n>>(8*i))); }
static std::string digits(size_t i, int w) {
    std::ostringstream o; o<<std::setw(w)<<std::setfill('0')<<i; return o.str();
}
int main(int argc, char** argv) {
    try {
        require(argc==5||argc==6, "usage: shard_corpus SRC.bin TOKENIZER.bbp OUT_DIR DOCS_PER_SHARD [STAGE_TAG]");
        const fs::path src=fs::absolute(argv[1]).lexically_normal();
        const fs::path tok=fs::absolute(argv[2]).lexically_normal();
        const fs::path outdir=fs::absolute(argv[3]).lexically_normal();
        const size_t per=size_t(std::stoull(argv[4]));
        const std::string stage=argc==6?std::string(argv[5]):std::string("core");
        require(fs::is_regular_file(src), "source corpus missing");
        require(fs::is_regular_file(tok), "tokenizer missing");
        require(per>=1 && per<=1000000, "docs per shard 1..1000000");
        if (fs::exists(outdir)) {
            require(fs::is_directory(outdir), "output path is not a directory");
            require(fs::is_empty(outdir), "output directory must be empty; never overwrite");
        } else {
            require(fs::create_directories(outdir), "output directory create");
        }
        std::string digest; tao::text::load_tokenizer(tok.string(), digest);
        require(digest.size()==64, "tokenizer digest length");
        std::ifstream in(src, std::ios::binary); require(bool(in), "source corpus open");
        BpePilotShardReader reader(in, digest);
        std::ofstream man(outdir/"shard_manifest.tsv", std::ios::binary);
        require(bool(man), "manifest open");
        man << "shard\tfile\tdocs\ttokens\tsupervised\tsha256\tstage\n";
        std::vector<std::vector<Token>> docs;
        std::vector<std::string> shard_files;
        uint64_t grand_docs=0, grand_tokens=0, grand_sup=0;
        size_t index=0;
        while (true) {
            const size_t got=reader.read(per, docs);
            if (!got) break;
            const std::string name="shard_"+digits(index,5)+".bin";
            const fs::path tmp=outdir/(name+".partial");
            const fs::path dst=outdir/name;
            std::ofstream out(tmp, std::ios::binary); require(bool(out), "shard open");
            out.write("TLP2",4); out << digest;
            uint64_t tokens=0, sup=0;
            for (const auto& t: docs) {
                require(t.size()>=3 && t.size()<=1000000, "record length");
                uint64_t s=0; le32(out, uint32_t(t.size()));
                for (const auto& x: t) { out.put(char(x.id&255)); out.put(char(x.id>>8)); out.put(char(x.loss)); s+=x.loss; }
                require(s>0, "zero supervision in document");
                tokens+=t.size(); sup+=s;
            }
            out.close(); require(bool(out), "shard close");
            // Windows 上不能重命名仍有打开句柄的文件：必须让 ifstream 先离开作用域。
            // （此前该 bug 使本工具永远无法产出分片。）
            std::string blob;
            {
                std::ifstream hf(tmp, std::ios::binary);
                require(bool(hf), "shard reopen");
                blob.assign((std::istreambuf_iterator<char>(hf)), std::istreambuf_iterator<char>());
                require(!blob.empty() && !hf.bad(), "shard hash read");
            }
            const auto sha=tao::text::sha256(blob);
            fs::rename(tmp, dst);
            man << index << '\t' << name << '\t' << got << '\t' << tokens << '\t' << sup
                << '\t' << sha << '\t' << stage << '\n';
            shard_files.push_back(name);
            grand_docs+=got; grand_tokens+=tokens; grand_sup+=sup;
            std::cout << "SHARD " << index << " file=" << name << " docs=" << got
                      << " tokens=" << tokens << " supervised=" << sup << " sha256=" << sha << "\n";
            ++index;
        }
        man.close(); require(bool(man), "manifest close");
        require(index>0, "no shards produced");
        // 课程表模板：按分片数三等分。用户可自行编辑每片步数与学习率。
        {
            const size_t n=shard_files.size();
            // 三段切分：core = [0, a)，mixed = [a, b)，interfere = [b, n)。
            const size_t a=std::max<size_t>(1,(n+2)/3);
            const size_t b=std::max(a,(2*n+2)/3);
            const size_t core_hi=a-1;
            const size_t mix_lo=a, mix_hi=(b<=n?b:n)-1;
            const size_t int_lo=b;
            std::ofstream cur(outdir/"curriculum.tsv", std::ios::binary);
            require(bool(cur), "curriculum open");
            cur << "# stage\tshard_lo\tshard_hi\tsteps_per_shard\tlr\n";
            cur << "core\t0\t" << core_hi << "\t400\t0.001\n";
            if (mix_lo<=mix_hi) cur << "mixed\t" << mix_lo << "\t" << mix_hi << "\t300\t0.0008\n";
            if (int_lo<n) cur << "interfere\t" << int_lo << "\t" << (n-1) << "\t300\t0.0005\n";
            cur.close(); require(bool(cur), "curriculum close");
        }
        std::cout << "SHARD_CORPUS_COMPLETE shards=" << index << " docs=" << grand_docs
                  << " tokens=" << grand_tokens << " supervised=" << grand_sup
                  << " out=" << outdir.generic_string() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "SHARD_CORPUS_FAILED " << e.what() << "; do not consume partial outputs\n";
        return 1;
    }
}
