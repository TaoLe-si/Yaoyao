// Byte-level BPE trainer with a hard digit constraint.
#include "language_data_contract.hpp"
#include <thread>

// CPU 处理必须吃满多核。这里的并行只作用于「初始配对计数」这一纯统计阶段：
// 它按区间划分互不重叠，结果与串行逐位等价。
static unsigned bpe_threads(){
  unsigned hw=std::thread::hardware_concurrency(); if(hw==0u)hw=4u;
  if(const char*e=std::getenv("TAO_CPU_THREADS")){int v=std::atoi(e); if(v>0)hw=unsigned(v);}
  return hw;
}

// 说明：BPE 的合并循环是 16121 步严格串行的全局状态变更，且堆中每个配对依赖
// "递增计数条目" 的历史语义（计数下降后仍需被重新选中），因此本工具不并行化。
// 语料处理与数据加载的多核并行在 corpus_pipeline.hpp / train_sft.cu / grpo_rollout.cpp 中实现。
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
    // 装载保持串行：这是磁盘 I/O 而非 CPU 计算，且按行截断的语义（cap 前判断、
    // 可能超出一个整行）必须原样保留，否则语料内容会变。真正的 CPU 热点是下面的配对计数。
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

    {
      // 初始计数并行化。等价性要点：串行版对每个 key 依次压入 (1,k),(2,k),...,(c,k)，
      // 因此堆里是「递增条目」的多重集 —— 计数因合并下降后，历史条目仍能让该 pair
      // 被重新选中。这里必须先并行统计出 c，再原样压入同一个多重集，否则合并路径分叉。
      const unsigned T=bpe_threads();
      std::vector<std::unordered_map<uint64_t,uint32_t>> lc(T);
      std::vector<std::unordered_map<uint64_t,std::vector<uint32_t>>> lp(T);
      for(unsigned w=0;w<T;++w){ lc[w].reserve((N/4u)/T+16u); }
      std::vector<std::thread> ts; ts.reserve(T);
      std::vector<std::exception_ptr> errs(T,nullptr);
      for(unsigned w=0;w<T;++w){
        ts.emplace_back([&,w](){
          try{
            const size_t b=w*(N-1)/T, e=(w+1)*(N-1)/T;
            auto&c=lc[w]; auto&ps=lp[w];
            for(size_t i=b;i<e;++i){
              const uint32_t a=tok[i], bb=tok[i+1];
              if(a==SENT||bb==SENT) continue;
              if(is_digit(a)||is_digit(bb)) continue;
              const uint64_t k=key(a,bb);
              ++c[k];
              ps[k].push_back(uint32_t(i));
            }
          }catch(...){ errs[w]=std::current_exception(); }
        });
      }
      for(auto&x:ts)x.join();
      for(auto&e:errs) if(e)std::rethrow_exception(e);
      // 合并：计数求和；位置按区间升序拼接（各线程区间连续且递增，故整体仍是升序）。
      for(unsigned w=0;w<T;++w){
        for(auto&kv:lc[w]) cnt[kv.first]+=kv.second;
        for(auto&kv:lp[w]) { auto&d=pos[kv.first]; d.insert(d.end(),kv.second.begin(),kv.second.end()); }
      }
      // 重建与串行完全相同的堆多重集 {(1,k),(2,k),...,(c,k)}。
      for(auto&kv:cnt){
        const uint32_t c=kv.second; const uint64_t k=kv.first;
        for(uint32_t v=1;v<=c;++v) heap.push({v,k});
      }
    }

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
      const uint32_t newid=uint32_t(tao::data::FIRST_MERGE+merges.size());
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
