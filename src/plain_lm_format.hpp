#pragma once
// TLP3 —— 纯语言建模（plain LM）语料格式。
//
// 【为什么需要】TLP2 是**对话专用**格式：reader 强制 role 序列
// (BOS / USER|ASSISTANT / 正文 / TURN_END ...)，正文 token 的 loss 由 role 决定。
// 代码与自然语言预训练语料没有 role，用 TLP2 表达不了（会被 reader 拒绝）。
//
// 【为什么下游不用改】PilotCursor 的校验只有三条：
//   front().id==BOS && back().id==TURN_END && size>=3
// 且 take_batch 只用 doc[i].id 与 doc[i].loss。因此把纯文本文档编码成
//   BOS(256) + [文本 token, loss=true] + TURN_END(259, loss=true)
// 就能被现有训练器（PilotCursor / ShuffledEpochCursor / ReusableBatchGraph）直接消费，
// **训练器零改动**。
//
// 记录布局（与 TLP2 一致，仅去掉 role 约束）：
//   header: "TLP3" + 64 字节 tokenizer digest = 68 字节
//   每条:   u32 token 数(小端) + 每 token (u16 id 小端, u8 loss)
#include "language_data_contract.hpp"
#include "pilot_reader.hpp"
#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace tao::data {

inline void write_le32(std::ostream& out, uint32_t n) { for (int i=0;i<4;++i) out.put(char(n>>(8*i))); }

inline void write_tlp3_header(std::ostream& out, const std::string& digest) {
    if (digest.size()!=64) throw std::invalid_argument("tokenizer digest must be 64 chars");
    out.write("TLP3",4); out << digest;
}

// 把一段纯文本 token 序列写成一条 TLP3 记录（自动补 BOS / TURN_END）。
inline void write_tlp3_record(std::ostream& out, const std::vector<int>& ids) {
    if (ids.empty()) throw std::invalid_argument("empty plain document");
    const size_t n = ids.size()+2;
    if (n>1000000) throw std::runtime_error("record length");
    write_le32(out, uint32_t(n));
    out.put(char(BOS&255)); out.put(char(BOS>>8)); out.put(char(0));
    for (int id : ids) {
        if (id<0 || id>=16384) throw std::runtime_error("plain token id range");
        out.put(char(id&255)); out.put(char(id>>8)); out.put(char(1));   // 全文监督
    }
    out.put(char(TURN_END&255)); out.put(char(TURN_END>>8)); out.put(char(1));
}

// 读一条 TLP3 记录；EOF 返回 false。产出形状满足 PilotCursor 的边界要求。
inline bool read_tlp3_doc(std::istream& in, std::vector<Token>& t) {
    if (in.peek()==std::char_traits<char>::eof()) return false;
    uint32_t n=0; for (int i=0;i<4;++i) n|=getbyte(in)<<(8*i);
    if (n<3 || n>1000000) throw std::runtime_error("plain record length");
    t.clear(); t.reserve(n);
    for (uint32_t i=0;i<n;++i) {
        unsigned lo=getbyte(in), hi=getbyte(in), mask=getbyte(in);
        int id=int(lo|(hi<<8));
        if (id>=16384 || mask>1) throw std::runtime_error("plain token");
        t.push_back({id, bool(mask)});
    }
    if (t.front().id!=BOS || t.back().id!=TURN_END)
        throw std::runtime_error("plain record boundary");
    return true;
}

inline std::vector<std::vector<Token>> read_tlp3(std::istream& in, const std::string& expected) {
    char header[68]; in.read(header,68);
    if (!in || std::string(header,4)!="TLP3" || std::string(header+4,64)!=expected)
        throw std::runtime_error("TLP3 identity");
    std::vector<std::vector<Token>> docs; std::vector<Token> t;
    while (read_tlp3_doc(in,t)) docs.push_back(t);
    if (in.bad()) throw std::runtime_error("TLP3 IO");
    return docs;
}

// 分片流式读取 TLP3（0.9B 级语料用）。
class Tlp3ShardReader {
public:
    Tlp3ShardReader(std::istream& in, std::string expected)
        : in_(in), expected_(std::move(expected)) {}
    size_t read(size_t max_docs, std::vector<std::vector<Token>>& out) {
        ensure_header();
        if (!max_docs) throw std::invalid_argument("shard read width");
        out.clear(); out.reserve(max_docs);
        std::vector<Token> t;
        while (out.size()<max_docs && read_tlp3_doc(in_,t)) out.push_back(t);
        if (in_.bad()) throw std::runtime_error("TLP3 shard IO");
        return out.size();
    }
    bool eof() { ensure_header(); return in_.peek()==std::char_traits<char>::eof(); }
private:
    void ensure_header() {
        if (header_read_) return;
        char header[68]; in_.read(header,68);
        if (!in_ || std::string(header,4)!="TLP3" || std::string(header+4,64)!=expected_)
            throw std::runtime_error("TLP3 shard identity");
        header_read_=true;
    }
    std::istream& in_;
    std::string expected_;
    bool header_read_=false;
};
}
