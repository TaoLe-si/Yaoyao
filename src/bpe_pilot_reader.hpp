#pragma once
#include "pilot_reader.hpp"
namespace tao::data {

// 读一篇文档；EOF 处返回 false（不抛异常）。文档体解析与 read_bpe_pilot 逐条一致。
inline bool read_bpe_pilot_doc(std::istream&in,std::vector<Token>&t){
    if(in.peek()==std::char_traits<char>::eof())return false;
    uint32_t n=0;for(int i=0;i<4;++i)n|=getbyte(in)<<(8*i);
    if(n<3||n>1000000)throw std::runtime_error("record length");
    t.clear();t.reserve(n);
    for(uint32_t i=0;i<n;++i){unsigned lo=getbyte(in),hi=getbyte(in),mask=getbyte(in);
        int id=int(lo|(hi<<8));if(id>=16384||mask>1)throw std::runtime_error("token");
        t.push_back({id,bool(mask)});}
    if(t[0].id!=BOS||t[0].loss)throw std::runtime_error("BOS");
    size_t p=1;
    while(p<t.size()){
        if((t[p].id!=USER&&t[p].id!=ASSISTANT)||t[p].loss)throw std::runtime_error("role");
        bool a=t[p++].id==ASSISTANT;
        while(p<t.size()&&(t[p].id<256||t[p].id>=261)){if(t[p++].loss!=a)throw std::runtime_error("mask");}
        if(p>=t.size()||t[p].id!=TURN_END||t[p].loss!=a)throw std::runtime_error("turn");
        ++p;
    }
    return true;
}

inline std::vector<std::vector<Token>> read_bpe_pilot(std::istream&in,const std::string&expected){char header[68];in.read(header,68);if(!in||std::string(header,4)!="TLP2"||std::string(header+4,64)!=expected)throw std::runtime_error("BPE dataset identity");std::vector<std::vector<Token>>docs;std::vector<Token>t;while(read_bpe_pilot_doc(in,t))docs.push_back(t);return docs;}

// ---- 分片流式读取 ----
// 语义：对一个 TLP2 流连续调用 read()，每次最多取 max_docs 篇，绝不一次性载入全部。
// header 只在首次调用时校验。返回本次实际读取的文档数（0 = 已到 EOF）。
// 用途：0.9B 级语料无法整文件读入内存（doc 20 §三）。
class BpePilotShardReader {
public:
    BpePilotShardReader(std::istream&in,std::string expected)
        : in_(in), expected_(std::move(expected)) {}
    size_t read(size_t max_docs,std::vector<std::vector<Token>>&out){
        ensure_header();
        out.clear();
        if(!max_docs)throw std::invalid_argument("shard read width");
        out.reserve(max_docs);
        std::vector<Token>t;
        while(out.size()<max_docs&&read_bpe_pilot_doc(in_,t))out.push_back(t);
        if(in_.bad())throw std::runtime_error("shard IO");
        return out.size();
    }
    bool eof(){
        ensure_header();
        return in_.peek()==std::char_traits<char>::eof();
    }
private:
    void ensure_header(){
        if(header_read_)return;
        char header[68];in_.read(header,68);
        if(!in_||std::string(header,4)!="TLP2"||std::string(header+4,64)!=expected_)
            throw std::runtime_error("BPE shard identity");
        header_read_=true;
    }
    std::istream&in_;
    std::string expected_;
    bool header_read_=false;
};
}
