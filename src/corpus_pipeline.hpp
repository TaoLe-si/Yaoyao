#pragma once
// 语料处理模块（独立）：输入原始语料 → 输出「理好的」训练语料。
//
// 设计目标（doc 20）：
//   1. **原始 → 规范**：多种原始格式（纯文本 / jsonl / 对话 txt）归一为内部文档表示；
//   2. **去重**：精确去重（sha256）+ 近似去重（MinHash-LSH，词 shingle）；
//   3. **质量过滤**：长度、可打印字符比、退化重复、控制字符；
//   4. **封存（seal）**：把评测用实体（名字等）从语料中剔除，保证 held-out 成立（铁律 R6）；
//   5. **课程标注**：core / mixed / interfere 三阶段标签，供课程式分段训练；
//   6. **分片产出**：TLP3（纯文本，供代码/自然语言）或 TLP2（对话）分片 + 清单 + 课程表。
//
// 本模块只依赖标准库与 tokenizer_file.hpp / language_data_contract.hpp，
// 不依赖 CUDA，可单独构建（cl.exe 即可）。
#include "tokenizer_file.hpp"
#include "language_data_contract.hpp"
#include "bpe_pilot_reader.hpp"
#include "plain_lm_format.hpp"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <thread>
#include <exception>
#include <cstdlib>

// Windows 遗留宏 near/far 会把普通标识符展开掉（本项目曾因此报 “语法错误: '.'”）。
#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif

namespace tao::corpus {

namespace fs = std::filesystem;
using tao::data::Message;
using tao::data::Token;

// ---------- 内部表示 ----------
struct RawDoc {
    std::string text;        // 纯文本模式：正文
    std::vector<Message> turns;  // 对话模式：U/A 轮次（非空时优先）
    std::string source;      // 来源文件
    std::string stage = "core";
    bool dialogue = false;
};

struct Stats {
    size_t ingested=0, dropped_short=0, dropped_long=0, dropped_quality=0,
           dropped_exact_dup=0, dropped_near_dup=0, dropped_sealed=0, kept=0;
    size_t bytes_in=0, bytes_out=0;
    std::map<std::string,size_t> per_stage;
};

inline void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }

// 语料处理的并行度。默认为全部逻辑核（CPU 处理必须吃满多核）；
// TAO_CORPUS_THREADS 可覆盖，设为 1 即回到纯串行。
inline unsigned corpus_threads() {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0u) hw = 4u;
    unsigned n = hw;
    if (const char* e = std::getenv("TAO_CORPUS_THREADS")) {
        const int v = std::atoi(e);
        if (v > 0) n = unsigned(v);
    }
    return n;
}

// 把 [0,n) 静态划分给 threads 个线程。任何线程抛出的异常在 join 后重抛，
// 保证与串行版本同样的失败语义。n==0 或 threads<=1 时退化为串行。
template <class F>
inline void parallel_for(size_t n, unsigned threads, F&& f) {
    if (n == 0) return;
    if (threads <= 1u || n < 2u) { for (size_t i = 0; i < n; ++i) f(i); return; }
    if (size_t(threads) > n) threads = unsigned(n);
    std::vector<std::thread> ts; ts.reserve(threads);
    std::vector<std::exception_ptr> errs(threads, nullptr);
    for (unsigned w = 0; w < threads; ++w) {
        ts.emplace_back([&, w]() {
            try { for (size_t i = w; i < n; i += threads) f(i); }
            catch (...) { errs[w] = std::current_exception(); }
        });
    }
    for (auto& t : ts) t.join();
    for (auto& e : errs) if (e) std::rethrow_exception(e);
}

// ---------- 归一化 ----------
// 折叠空白、去掉控制字符（保留 \n 由调用方决定），并做 UTF-8 合法性粗检。
// preserve_layout=true 时保留换行与缩进（代码语料必需：折叠空白会摧毁缩进语义，
// 使模型永远学不会合法代码）。制表符展开为 4 空格，行尾空格剔除，连续空行折叠为一行。
inline std::string normalize_text(const std::string& in, bool preserve_layout=false) {
    if (preserve_layout) {
        std::string out, line; out.reserve(in.size());
        bool prev_blank=false;
        auto flush=[&](bool more){
            while(!line.empty() && line.back()==' ') line.pop_back();
            out+=line; line.clear();
            if(more) out.push_back('\n');
        };
        for (size_t i=0;i<in.size();++i) {
            unsigned char c=(unsigned char)in[i];
            if (c=='\r') continue;
            if (c=='\n') {
                const bool blank = line.find_first_not_of(" \t")==std::string::npos;
                flush(!(blank && prev_blank));
                prev_blank=blank; continue;
            }
            if (c=='\t') { line+="    "; continue; }
            if (c<0x20 || c==0x7f) continue;
            line.push_back(char(c));
        }
        flush(false);
        return out;
    }
    std::string out; out.reserve(in.size());
    bool prev_space=false;
    for (size_t i=0;i<in.size();++i) {
        unsigned char c=(unsigned char)in[i];
        if (c=='\r') continue;
        if (c=='\n' || c=='\t') { if(!prev_space){out.push_back(' ');prev_space=true;} continue; }
        if (c<0x20 || c==0x7f) continue;                 // 其他控制字符丢弃
        if (c==' ') { if(!prev_space){out.push_back(' ');prev_space=true;} continue; }
        out.push_back(char(c)); prev_space=false;
    }
    while(!out.empty() && out.back()==' ') out.pop_back();
    return out;
}

// 退化重复检测：最长连续重复单元的占比。用于剔除 "小何何何何…" 类样本。
inline double max_repeat_ratio(const std::string& s) {
    if (s.size()<8) return 0.0;
    // 逐单元长度检查前 8 种周期，取覆盖比最大者
    double best=0.0;
    for (size_t unit=1; unit<=8 && unit*3<=s.size(); ++unit) {
        size_t run=0, i=0;
        while (i+unit<=s.size()) {
            if (s.compare(i,unit,s,0,unit)==0) { run+=unit; i+=unit; }
            else { i+=1; run=0; }
            best=std::max(best, double(run)/double(s.size()));
        }
    }
    return best;
}

inline double printable_ratio(const std::string& s) {
    if (s.empty()) return 0.0;
    size_t ok=0;
    for (unsigned char c : s) if (c>=0x20 || c=='\n' || c=='\t') ++ok;
    return double(ok)/double(s.size());
}

struct QualityConfig {
    size_t min_bytes=64;
    size_t max_bytes=2000000;
    double max_repeat=0.35;        // 退化重复上限
    double min_printable=0.90;
    bool enabled=true;
};

inline bool passes_quality(const std::string& s, const QualityConfig& qc, Stats& st) {
    if (!qc.enabled) return true;
    if (s.size()<qc.min_bytes) { ++st.dropped_short; return false; }
    if (s.size()>qc.max_bytes) { ++st.dropped_long; return false; }
    if (printable_ratio(s)<qc.min_printable) { ++st.dropped_quality; return false; }
    if (max_repeat_ratio(s)>qc.max_repeat) { ++st.dropped_quality; return false; }
    return true;
}

// ---------- 近似去重：MinHash + LSH 分带 ----------
class NearDedup {
public:
    explicit NearDedup(double threshold=0.8, size_t bands=16, size_t rows=4)
        : bands_(bands), rows_(rows), threshold_(threshold) {
        require(bands*rows<=64, "minhash bands*rows <= 64");
        for (size_t i=0;i<bands*rows;++i) seeds_.push_back(uint64_t(0x9e3779b97f4a7c15ull)*(i+1)+0x1234567ull);
    }
    // 返回 true 表示与已见文档近似重复（应丢弃）
    bool duplicate(const std::string& text) {
        const auto sig=signature(text);
        bool dup=false;
        for (size_t b=0;b<bands_ && !dup;++b) {
            const std::string key=band_key(sig,b);
            auto it=seen_.find(key);
            if (it==seen_.end()) { seen_.emplace(key,sig); continue; }
            // 同带命中 → 精确比较签名相似度
            size_t same=0;
            for (size_t i=0;i<sig.size();++i) if (sig[i]==it->second[i]) ++same;
            if (double(same)/double(sig.size())>=threshold_) dup=true;
        }
        if (!dup) for (size_t b=0;b<bands_;++b) seen_.emplace(band_key(sig,b),sig);
        return dup;
    }
private:
    std::vector<uint64_t> signature(const std::string& text) const {
        const size_t K=5;
        std::vector<std::string> shingles;
        std::istringstream is(text); std::string w;
        std::vector<std::string> words;
        while (is>>w) words.push_back(w);
        if (words.size()<K) { shingles.push_back(text); }
        else for (size_t i=0;i+K<=words.size();++i) {
            std::string s;
            for (size_t j=0;j<K;++j) { if(j) s+=' '; s+=words[i+j]; }
            shingles.push_back(std::move(s));
        }
        std::vector<uint64_t> sig(bands_*rows_, UINT64_MAX);
        for (const auto& sh : shingles) {
            const uint64_t h0=fnv1a(sh);
            for (size_t i=0;i<seeds_.size();++i) {
                uint64_t h=h0^seeds_[i];
                h*=0x100000001b3ull; h^=h>>29;
                sig[i]=std::min(sig[i],h);
            }
        }
        return sig;
    }
    static uint64_t fnv1a(const std::string& s) {
        uint64_t h=1469598103934665603ull;
        for (unsigned char c : s) { h^=c; h*=1099511628211ull; }
        return h;
    }
    std::string band_key(const std::vector<uint64_t>& sig, size_t band) const {
        std::string k; k.reserve(rows_*9);
        for (size_t r=0;r<rows_;++r) { k+=std::to_string(sig[band*rows_+r]); k+='|'; }
        return k;
    }
    size_t bands_, rows_;
    double threshold_;
    std::vector<uint64_t> seeds_;
    std::unordered_map<std::string,std::vector<uint64_t>> seen_;
};

// ---------- 封存实体 ----------
class Sealer {
public:
    void add(const std::string& entity) { if(!entity.empty()) entities_.push_back(entity); }
    void load(const fs::path& p) {
        std::ifstream f(p); require(bool(f), "seal file open");
        std::string line;
        while (std::getline(f,line)) {
            while(!line.empty() && (line.back()=='\r'||line.back()==' ')) line.pop_back();
            if (!line.empty() && line[0]!='#') add(line);
        }
    }
    bool contains(const std::string& text) const {
        for (const auto& e : entities_) if (text.find(e)!=std::string::npos) return true;
        return false;
    }
    size_t size() const { return entities_.size(); }
private:
    std::vector<std::string> entities_;
};

// ---------- 摄取 ----------
// 对话 txt：DOC / "U ..." / "A ..." 交替（与 export_noffn_probe 同一格式）
inline std::vector<RawDoc> ingest_dialogue_txt(const fs::path& p, const std::string& stage) {
    std::ifstream in(p, std::ios::binary); require(bool(in), "dialogue open: "+p.string());
    std::string raw((std::istreambuf_iterator<char>(in)), {});
    require(!in.bad(), "dialogue read");
    if (raw.size()>=3 && (unsigned char)raw[0]==0xef && (unsigned char)raw[1]==0xbb && (unsigned char)raw[2]==0xbf)
        raw.erase(0,3);
    std::vector<RawDoc> docs; std::vector<Message> cur;
    auto flush=[&](){
        if (!cur.empty()) {
            require(cur.size()%2==0 && !cur.front().assistant, "dialogue role order");
            RawDoc d; d.turns=std::move(cur); d.source=p.filename().string();
            d.stage=stage; d.dialogue=true; docs.push_back(std::move(d)); cur.clear();
        }
    };
    std::istringstream lines(raw); std::string line;
    while (std::getline(lines,line)) {
        if (!line.empty() && line.back()=='\r') line.pop_back();
        if (line.empty() || line[0]=='#') continue;
        if (line=="DOC") { flush(); continue; }
        require(line.size()>=3 && (line.compare(0,2,"U ")==0 || line.compare(0,2,"A ")==0), "dialogue line prefix");
        const bool assistant = line[0]=='A';
        require(cur.empty()? !assistant : assistant==((cur.size()%2)==1), "dialogue alternating");
        cur.push_back({assistant, line.substr(2)});
    }
    flush();
    return docs;
}

// 纯文本 txt：以空行为段落分隔（代码文件建议整文件一档，故也支持整文件单档）
inline std::vector<RawDoc> ingest_plain_txt(const fs::path& p, const std::string& stage, bool whole_file,
                                          const std::string& doc_sep = {}, bool preserve_layout = false) {
    std::ifstream in(p, std::ios::binary); require(bool(in), "plain open: "+p.string());
    std::string raw((std::istreambuf_iterator<char>(in)), {});
    require(!in.bad(), "plain read");
    if (raw.size()>=3 && (unsigned char)raw[0]==0xef && (unsigned char)raw[1]==0xbb && (unsigned char)raw[2]==0xbf)
        raw.erase(0,3);
    std::vector<RawDoc> docs;
    auto push=[&](const std::string& t){
        const auto n=normalize_text(t, preserve_layout);
        if (n.empty()) return;
        RawDoc d; d.text=n; d.source=p.filename().string(); d.stage=stage; d.dialogue=false;
        docs.push_back(std::move(d));
    };
    if (whole_file) { push(raw); return docs; }
    std::istringstream is(raw); std::string line, buf;
    while (std::getline(is,line)) {
        if (!line.empty() && line.back()=='\r') line.pop_back();
        if (!doc_sep.empty()) {
            // 标记分隔模式：一行恰好等于 doc_sep 表示上一篇结束（代码一文件一档）。
            if (line==doc_sep) { push(buf); buf.clear(); continue; }
            if(!buf.empty()) buf+='\n'; buf+=line; continue;
        }
        if (line.empty()) { push(buf); buf.clear(); }
        else { if(!buf.empty()) buf+='\n'; buf+=line; }
    }
    push(buf);
    return docs;
}

// jsonl：每行一个对象，取 "text" 字段；或 "prompt"/"response" 组成对话
inline std::vector<RawDoc> ingest_jsonl(const fs::path& p, const std::string& stage) {
    std::ifstream in(p, std::ios::binary); require(bool(in), "jsonl open: "+p.string());
    std::vector<RawDoc> docs; std::string line;
    auto field=[&](const std::string& s, const std::string& key)->std::string{
        const std::string pat="\""+key+"\"";
        auto k=s.find(pat); if (k==std::string::npos) return {};
        auto c=s.find(':', k+pat.size()); if (c==std::string::npos) return {};
        auto q=s.find('"', c+1); if (q==std::string::npos) return {};
        std::string out;
        for (size_t i=q+1;i<s.size();++i) {
            if (s[i]=='\\' && i+1<s.size()) {
                const char e=s[i+1];
                if (e=='n') out+='\n'; else if (e=='t') out+='\t';
                else if (e=='"'||e=='\\'||e=='/') out+=e;
                else if (e=='u') { i+=5; out+='?'; }
                else out+=e;
                ++i; continue;
            }
            if (s[i]=='"') break;
            out+=s[i];
        }
        return out;
    };
    while (std::getline(in,line)) {
        if (line.empty() || line[0]=='#') continue;
        const auto text=field(line,"text");
        const auto prompt=field(line,"prompt"), response=field(line,"response");
        RawDoc d; d.source=p.filename().string(); d.stage=stage;
        if (!text.empty()) { d.text=normalize_text(text); }
        else if (!prompt.empty() || !response.empty()) {
            d.dialogue=true;
            if(!prompt.empty()) d.turns.push_back({false,normalize_text(prompt)});
            if(!response.empty()) d.turns.push_back({true,normalize_text(response)});
        } else continue;
        if (d.dialogue && d.turns.size()%2!=0) d.turns.push_back({true,""});
        docs.push_back(std::move(d));
    }
    return docs;
}

// ---------- 产出分片 ----------
struct EmitConfig {
    fs::path out_dir;
    std::string digest;
    size_t docs_per_shard=20000;
    bool dialogue=false;      // true → TLP2；false → TLP3
};

// 一篇文档的编码结果：对话模式填 turns（每条消息一个 id 序列），纯文本模式填 text。
struct EncodedDoc {
    std::vector<std::vector<uint32_t>> turns;
    std::vector<uint32_t> text;
};

// 把已整理的文档写成分片 + 清单 + 课程表。返回分片数。
inline size_t emit_shards(const std::vector<RawDoc>& docs, const tao::text::ByteBpe& bpe,
                          const EmitConfig& ec, Stats& st) {
    require(!docs.empty(), "no documents to emit");
    require(ec.digest.size()==64, "digest");
    if (fs::exists(ec.out_dir)) {
        require(fs::is_directory(ec.out_dir), "out is not a directory");
        require(fs::is_empty(ec.out_dir), "out dir must be empty; never overwrite");
    } else require(fs::create_directories(ec.out_dir), "out dir create");
    // 单线程先完成惰性校验：之后 encode()/decode() 只读 merges，可安全并行。
    bpe.validate();
    const unsigned nthreads = corpus_threads();
    std::cout << "CORPUS_THREADS " << nthreads << "\n";
    std::ofstream man(ec.out_dir/"shard_manifest.tsv", std::ios::binary);
    require(bool(man), "manifest open");
    man << "shard\tfile\tdocs\ttokens\tsupervised\tstage\tsha256\n";
    size_t index=0, i=0;
    std::vector<std::string> names;
    while (i<docs.size()) {
        const size_t end=std::min(docs.size(), i+ec.docs_per_shard);
        std::ostringstream nm; nm<<"shard_"<<std::setw(5)<<std::setfill('0')<<index<<".bin";
        const fs::path tmp=ec.out_dir/(nm.str()+".partial");
        const fs::path dst=ec.out_dir/nm.str();
        std::ofstream out(tmp, std::ios::binary); require(bool(out), "shard open");
        if (ec.dialogue) { out.write("TLP2",4); out<<ec.digest; }
        else tao::data::write_tlp3_header(out, ec.digest);
        // 先并行编码本片全部文档，再串行写出。encode() 是只读 merges 的纯函数，
        // 故并行结果与串行逐字节相同；写出仍按原顺序，分片 sha256 不变。
        const size_t cnt=end-i;
        std::vector<EncodedDoc> enc(cnt);
        parallel_for(cnt, nthreads, [&](size_t k){
            const auto& d=docs[i+k];
            EncodedDoc& e=enc[k];
            if (d.dialogue) {
                e.turns.reserve(d.turns.size());
                for (const auto& m : d.turns) {
                    auto ids=bpe.encode(m.utf8);
                    require(bpe.decode(ids)==m.utf8, "BPE roundtrip");
                    for (auto id : ids) require(id<16384 && (id<256||id>=tao::data::FIRST_MERGE), "BPE text ID");
                    e.turns.push_back(std::move(ids));
                }
            } else {
                e.text=bpe.encode(d.text);
                require(bpe.decode(e.text)==d.text, "BPE roundtrip (plain)");
                for (auto id : e.text) require(id<16384, "BPE id range");
            }
        });
        uint64_t tokens=0, sup=0;
        std::string stage=docs[i].stage;
        for (size_t k=i;k<end;++k) {
            const auto& d=docs[k];
            const auto& e=enc[k-i];
            if (d.dialogue) {
                std::vector<Token> t{{tao::data::BOS,false}};
                for (size_t mi=0; mi<d.turns.size(); ++mi) {
                    const auto& m=d.turns[mi];
                    t.push_back({m.assistant?tao::data::ASSISTANT:tao::data::USER,false});
                    for (auto id : e.turns[mi]) t.push_back({int(id), m.assistant});
                    t.push_back({tao::data::TURN_END, m.assistant});
                }
                require(t.size()>=5 && t.size()<=1000000, "record length");
                uint64_t s=0; tao::data::write_le32(out, uint32_t(t.size()));
                for (const auto& x : t) { out.put(char(x.id&255)); out.put(char(x.id>>8)); out.put(char(x.loss)); s+=x.loss; }
                require(s>0, "zero supervision");
                tokens+=t.size(); sup+=s;
            } else {
                // encode() 返回 vector<uint32_t>，TLP3 写入接口取 vector<int>，显式转换并校验范围。
                std::vector<int> wide; wide.reserve(e.text.size());
                for (auto id : e.text) wide.push_back(int(id));
                tao::data::write_tlp3_record(out, wide);
                tokens+=wide.size()+2; sup+=wide.size()+1;   // BOS 不监督，正文与 TURN_END 监督
            }
        }
        out.close(); require(bool(out), "shard close");
        // Windows 上不能重命名仍有打开句柄的文件：必须让 ifstream 先离开作用域。
        // （此前该 bug 使 emit_shards 永远无法产出分片，整条流水线从未真正跑通。）
        std::string blob;
        {
            std::ifstream hf(tmp, std::ios::binary);
            require(bool(hf), "shard reopen");
            blob.assign((std::istreambuf_iterator<char>(hf)), std::istreambuf_iterator<char>());
            require(!blob.empty() && !hf.bad(), "shard hash read");
        }
        const auto sha=tao::text::sha256(blob);
        fs::rename(tmp, dst);
        man << index << '\t' << nm.str() << '\t' << (end-i) << '\t' << tokens << '\t' << sup
            << '\t' << stage << '\t' << sha << '\n';
        names.push_back(nm.str());
        std::cout << "SHARD " << index << " docs=" << (end-i) << " tokens=" << tokens
                  << " supervised=" << sup << " stage=" << stage << " sha256=" << sha << "\n";
        i=end; ++index;
    }
    man.close(); require(bool(man), "manifest close");
    // 课程表：按分片数三段切分（用户可再手改步数与学习率）
    {
        const size_t n=names.size();
        const size_t a=std::max<size_t>(1,(n+2)/3), b=std::max(a,(2*n+2)/3);
        std::ofstream cur(ec.out_dir/"curriculum.tsv", std::ios::binary);
        require(bool(cur), "curriculum open");
        cur << "# stage\tshard_lo\tshard_hi\tsteps_per_shard\tlr\n";
        cur << "core\t0\t" << (a-1) << "\t400\t0.001\n";
        if (a<=std::min(b,n)-1) cur << "mixed\t" << a << "\t" << (std::min(b,n)-1) << "\t300\t0.0008\n";
        if (b<n) cur << "interfere\t" << b << "\t" << (n-1) << "\t300\t0.0005\n";
        cur.close(); require(bool(cur), "curriculum close");
    }
    (void)st;
    return index;
}

// ---------- 顶层：原始语料 → 理好的语料 ----------
struct PipelineConfig {
    std::vector<fs::path> inputs;         // 原始文件
    std::string input_format="auto";      // auto | dialogue | plain | jsonl
    bool whole_file=false;                // plain 模式下整文件一档
    std::string doc_sep;                  // plain 模式下文档分隔标记行
    bool preserve_layout=false;           // 保留换行/缩进（代码语料必需）
    fs::path out_dir;
    fs::path seal_file;                   // 可选：封存实体清单
    std::string stage="core";
    size_t docs_per_shard=20000;
    QualityConfig quality;
    bool near_dedup=true;
    double near_dup_threshold=0.8;
    bool emit=true;
};

inline Stats run_pipeline(const PipelineConfig& pc, const tao::text::ByteBpe& bpe,
                          const std::string& digest, bool dialogue_out) {
    Stats st;
    Sealer sealer;
    if (!pc.seal_file.empty()) sealer.load(pc.seal_file);
    NearDedup near_dedup(pc.near_dup_threshold);
    std::unordered_set<std::string> exact;
    std::vector<RawDoc> kept;
    for (const auto& p : pc.inputs) {
        require(fs::is_regular_file(p), "input missing: "+p.string());
        std::string fmt=pc.input_format;
        if (fmt=="auto") {
            const auto ext=p.extension().string();
            if (ext==".jsonl") fmt="jsonl";
            else {
                // 含 "DOC" 行或 "U "/"A " 行首 → 对话格式
                std::ifstream probe(p, std::ios::binary);
                std::string head(4096,'\0');
                probe.read(&head[0], 4096);
                head.resize(size_t(probe.gcount()));
                fmt = (head.find("\nDOC\n")!=std::string::npos || head.compare(0,4,"DOC\n")==0) ? "dialogue" : "plain";
            }
        }
        std::vector<RawDoc> docs;
        if (fmt=="dialogue") docs=ingest_dialogue_txt(p, pc.stage);
        else if (fmt=="jsonl") docs=ingest_jsonl(p, pc.stage);
        else docs=ingest_plain_txt(p, pc.stage, pc.whole_file, pc.doc_sep, pc.preserve_layout);
        for (auto& d : docs) {
            ++st.ingested;
            const std::string probe = d.dialogue ? [&]{ std::string s; for(const auto&m:d.turns) s+=m.utf8; return s; }() : d.text;
            st.bytes_in += probe.size();
            if (!passes_quality(probe, pc.quality, st)) continue;
            if (sealer.contains(probe)) { ++st.dropped_sealed; continue; }
            const auto h=tao::text::sha256(probe);
            if (!exact.insert(h).second) { ++st.dropped_exact_dup; continue; }
            if (pc.near_dedup && near_dedup.duplicate(probe)) { ++st.dropped_near_dup; continue; }
            st.bytes_out += probe.size();
            ++st.per_stage[d.stage];
            kept.push_back(std::move(d));
        }
    }
    st.kept=kept.size();
    if (pc.emit) {
        EmitConfig ec; ec.out_dir=pc.out_dir; ec.digest=digest;
        ec.docs_per_shard=pc.docs_per_shard; ec.dialogue=dialogue_out;
        emit_shards(kept, bpe, ec, st);
    }
    return st;
}

inline void print_stats(const Stats& st) {
    std::cout << "CORPUS_STATS ingested=" << st.ingested << " kept=" << st.kept
              << " short=" << st.dropped_short << " long=" << st.dropped_long
              << " quality=" << st.dropped_quality << " exact_dup=" << st.dropped_exact_dup
              << " near_dup=" << st.dropped_near_dup << " sealed=" << st.dropped_sealed
              << " bytes_in=" << st.bytes_in << " bytes_out=" << st.bytes_out << "\n";
    for (const auto& kv : st.per_stage) std::cout << "CORPUS_STAGE " << kv.first << " docs=" << kv.second << "\n";
}
}
