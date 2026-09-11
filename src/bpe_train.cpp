// Byte-level BPE trainer with a hard digit constraint.
//
// WHY: numbers were opaque single tokens in the old tokenizer (0-100 each became
// one token), so arithmetic could only ever be table lookup -- the model never saw
// place value. Controlled experiment (pure-arithmetic corpus) gave in-corpus 14/15
// but composition generalization 0/15 for '+'. Fixing this requires that every
// decimal digit is ALWAYS its own token, so "123" is [1][2][3] and the model can
// learn carry/place-value structure.
//
// CONSTRAINT: no merge may involve an ASCII digit byte (0x30-0x39). Digits are
// therefore never absorbed into a larger token, in any context.
//
// Algorithm: incremental BPE. Linked list over the corpus plus per-pair occurrence
// lists and counts, with a lazily-invalidated max-heap. Cost is proportional to the
// number of merges actually performed, not merges x corpus.
//
// Usage: bpe_train <out.bbp> <nmerges> <max_mb> <text1> [text2 ...]
#define NOMINMAX
#include "tokenizer_file.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <queue>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
using namespace tao::text;

static const uint32_t SENT = 0xFFFFFFFFu;   // barrier: no merge crosses it
static inline bool is_digit(uint32_t t){ return t>=0x30u && t<=0x39u; }

int main(int argc,char**argv){
  try{
    if(argc<5) throw std::runtime_error("usage: bpe_train OUT.bbp NMERGES MAX_MB TEXT...");
    const std::string outpath=argv[1];
    const size_t nmerges=size_t(strtoull(argv[2],nullptr,10));
    const size_t max_mb=size_t(strtoull(argv[3],nullptr,10));
    if(nmerges<1||nmerges>16123) throw std::runtime_error("nmerges out of range (1..16123)");
    const size_t cap=max_mb*1024ull*1024ull;

    // ---- read lines, concatenating with barriers ----
    std::vector<uint32_t> tok; tok.reserve(cap+1024);
    size_t lines=0, bytes=0;
    for(int a=4;a<argc && tok.size()<cap;++a){
      std::ifstream f(argv[a],std::ios::binary);
      if(!f){ std::fprintf(stderr,"WARN cannot open %s\n",argv[a]); continue; }
      std::string line;
      while(std::getline(f,line) && tok.size()<cap){
        if(!line.empty() && line.back()=='\r') line.pop_back();
        if(line.empty()) continue;
        for(unsigned char c: line) tok.push_back(uint32_t(c));
        tok.push_back(SENT);
        ++lines; bytes+=line.size();
      }
    }
    if(tok.size()<1024) throw std::runtime_error("corpus too small");
    const size_t N=tok.size();
    std::fprintf(stderr,"CORPUS lines=%zu bytes=%zu symbols=%zu cap_mb=%zu\n",lines,bytes,N,max_mb);

    // ---- linked list ----
    std::vector<int64_t> prev(N), next(N);
    for(size_t i=0;i<N;++i){ prev[i]=(i==0?-1:int64_t(i-1)); next[i]=(i+1==N?-1:int64_t(i+1)); }

    std::unordered_map<uint64_t,uint32_t> cnt;        // pair -> live occurrence count
    std::unordered_map<uint64_t,std::vector<uint32_t>> pos; // pair -> positions (may hold stale)
    cnt.reserve(N/4+16); pos.reserve(N/4+16);
    using Item=std::pair<uint32_t,uint64_t>;          // (count, key)
    std::priority_queue<Item> heap;

    auto key=[](uint32_t a,uint32_t b){ return (uint64_t(a)<<32)|uint64_t(b); };
    auto add_pair=[&](uint32_t a,uint32_t b,uint32_t at){
      if(a==SENT||b==SENT) return;
      if(is_digit(a)||is_digit(b)) return;            // <-- digit constraint
      const uint64_t k=key(a,b);
      uint32_t &c=cnt[k]; ++c;
      pos[k].push_back(at);
      heap.push({c,k});
    };
    auto del_pair=[&](uint32_t a,uint32_t b){
      if(a==SENT||b==SENT) return;
      if(is_digit(a)||is_digit(b)) return;
      auto it=cnt.find(key(a,b));
      if(it!=cnt.end() && it->second>0) --it->second;
    };

    for(size_t i=0;i+1<N;++i) add_pair(tok[i],tok[i+1],uint32_t(i));

    // ---- merge loop ----
    std::vector<std::pair<uint32_t,uint32_t>> merges;
    merges.reserve(nmerges);
    size_t stale_skips=0;
    while(merges.size()<nmerges){
      uint64_t k=0; bool have=false;
      while(!heap.empty()){
        Item top=heap.top();
        auto it=cnt.find(top.second);
        if(it!=cnt.end() && it->second==top.first && top.first>0){ k=top.second; have=true; break; }
        heap.pop(); ++stale_skips;
      }
      if(!have) break;
      const uint32_t a=uint32_t(k>>32), b=uint32_t(k&0xFFFFFFFFu);
      heap.pop();
      const uint32_t newid=uint32_t(261+merges.size());
      merges.push_back({a,b});
      cnt.erase(k);
      std::vector<uint32_t> ps; 
      { auto it=pos.find(k); if(it!=pos.end()){ ps.swap(it->second); pos.erase(it);} }
      size_t done=0;
      for(uint32_t i: ps){
        if(tok[i]!=a) continue;
        int64_t j=next[i];
        if(j<0||tok[size_t(j)]!=b) continue;
        const int64_t L=prev[i], R=next[size_t(j)];
        if(L>=0) del_pair(tok[size_t(L)],tok[i]);
        if(R>=0) del_pair(tok[size_t(j)],tok[size_t(R)]);
        tok[i]=newid; next[i]=R;
        if(R>=0) prev[size_t(R)]=int64_t(i);
        if(L>=0) add_pair(tok[size_t(L)],newid,uint32_t(L));
        if(R>=0) add_pair(newid,tok[size_t(R)],uint32_t(i));
        ++done;
      }
      if(merges.size()%1000==0||merges.size()==1)
        std::fprintf(stderr,"MERGE %zu/%zu pair=(%u,%u) count=%u applied=%zu\n",
                     merges.size(),nmerges,a,b,ps.size(),done);
    }

    ByteBpe b; b.merges=merges; b.validate();
    std::ofstream o(outpath,std::ios::binary);
    if(!o) throw std::runtime_error("cannot write output");
    const std::string ser=serialize_tokenizer(b);
    o.write(ser.data(),std::streamsize(ser.size()));
    o.close(); if(!o) throw std::runtime_error("write failed");

    // ---- self-check: digits must stay single tokens ----
    auto enc=[&](const std::string&s){ return b.encode(s); };
    struct { const char* s; size_t want; } checks[] = {
      {"123",3},{"7",1},{"2026",4},{"3.14",4},{"12+34",5}
    };
    bool ok=true;
    for(auto&c:checks){
      auto ids=enc(c.s);
      if(ids.size()!=c.want){ ok=false; std::fprintf(stderr,"SELFCHECK_FAIL %s -> %zu ids (want %zu)\n",c.s,ids.size(),c.want); }
      if(b.decode(ids)!=std::string(c.s)){ ok=false; std::fprintf(stderr,"SELFCHECK_ROUNDTRIP_FAIL %s\n",c.s); }
    }
    for(const auto&m:merges) if(is_digit(m.first)||is_digit(m.second)){ ok=false; break; }
    std::printf("BPE_TRAIN_COMPLETE merges=%zu bytes=%zu lines=%zu digit_isolated=%s stale_skips=%zu\n",
                merges.size(),ser.size(),lines,ok?"YES":"NO",stale_skips);
    return ok?0:1;
  }catch(const std::exception&e){ std::fprintf(stderr,"BPE_TRAIN_FAILED %s\n",e.what()); return 1; }
}
