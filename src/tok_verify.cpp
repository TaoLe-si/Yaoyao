// Verifies a tokenizer: digit isolation, round-trip, and corpus encode rate.
// Usage: tok_verify <tok.bbp> [text...]
#define NOMINMAX
#include "tokenizer_file.hpp"
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <thread>
#include <atomic>
#include <chrono>
#include <stdexcept>
using namespace tao::text;
static bool is_digit_tok(uint32_t t){ return t>=0x30u && t<=0x39u; }
int main(int argc,char**argv){
  try{
    if(argc<2) throw std::runtime_error("usage: tok_verify TOK.bbp [text...]");
    std::string fp; auto b=load_tokenizer(argv[1],fp);
    b.validate();
    std::printf("TOKENIZER merges=%zu sha256=%s\n",b.merges.size(),fp.c_str());
    bool allok=true;
    // 1) every merge operand must be a non-digit
    for(size_t i=0;i<b.merges.size();++i){
      if(is_digit_tok(b.merges[i].first)||is_digit_tok(b.merges[i].second)){
        std::printf("  DIGIT_MERGE_VIOLATION at %zu: (%u,%u)\n",i,b.merges[i].first,b.merges[i].second);
        allok=false; break;
      }
    }
    std::printf("  no_merge_involves_digit = %s\n", allok?"YES":"NO");
    // 2) digit isolation across contexts
    const char* probes[]={"0","7","42","123","2026","3.14","12+34","100-37","7*8","9999","a1b2c3","1,234","x=105"};
    for(const char* p:probes){
      auto ids=b.encode(p);
      size_t ndig=0; for(const char* q=p;*q;++q) if(*q>='0'&&*q<='9') ndig++;
      size_t digtok=0; for(auto id:ids) if(id<256 && is_digit_tok(id)) digtok++;
      const auto ref=b.encode_ref(p);
      const bool match=ids==ref;
      const bool ok = (digtok==ndig) && (b.decode(ids)==std::string(p)) && match;
      if(!ok) allok=false;
      std::printf("  %-8s ids=%-3zu digit_tokens=%zu/%zu roundtrip=%s ref=%s %s\n",
        p,ids.size(),digtok,ndig,(b.decode(ids)==std::string(p))?"ok":"BAD",match?"ok":"BAD",ok?"":"  <== FAIL");
    }
    // 3) encode the provided texts: ref match on a sample, then 1-vs-N thread throughput
    auto run_pool=[&](const std::vector<std::string>& lines, unsigned T){
      std::atomic<size_t> next{0};
      std::vector<std::thread> pool;
      pool.reserve(T);
      for(unsigned t=0;t<T;++t) pool.emplace_back([&]{
        for(;;){
          const size_t i=next.fetch_add(1,std::memory_order_relaxed);
          if(i>=lines.size()) break;
          b.encode(lines[i]);
        }
      });
      for(auto& th:pool) th.join();
    };
    for(int i=2;i<argc;++i){
      std::ifstream f(argv[i],std::ios::binary); if(!f){ std::printf("  cannot open %s\n",argv[i]); continue; }
      std::vector<std::string> lines; std::string line; size_t bytes=0;
      while(std::getline(f,line)){
        if(!line.empty() && line.back()=='\r') line.pop_back();
        if(line.empty()) continue;
        bytes+=line.size(); lines.push_back(std::move(line));
      }
      size_t toks=0, nref=lines.size()<64?lines.size():64, nbad=0;
      for(size_t k=0;k<nref;++k){
        auto ids=b.encode(lines[k]);
        toks+=ids.size();
        if(ids!=b.encode_ref(lines[k])) ++nbad;
      }
      for(size_t k=nref;k<lines.size();++k) toks+=b.encode(lines[k]).size();
      if(nbad){ allok=false; std::printf("  FILE %s REF_MISMATCH %zu/%zu\n",argv[i],nbad,nref); }
      std::printf("  FILE %s lines=%zu bytes=%zu tokens=%zu  bytes/token=%.3f ref_ok=%zu/%zu\n",
        argv[i],lines.size(),bytes,toks,double(bytes)/double(toks?toks:1),nref-nbad,nref);
      if(lines.size()>=256){
        const unsigned hw=std::thread::hardware_concurrency();
        const unsigned nt=hw&&hw<16u?hw:8u;
        using clock=std::chrono::steady_clock;
        const auto t1=clock::now(); run_pool(lines,1u);
        const auto t8=clock::now(); run_pool(lines,nt);
        const auto t9=clock::now();
        const double s1=std::chrono::duration<double>(t8-t1).count();
        const double sn=std::chrono::duration<double>(t9-t8).count();
        std::printf("  ENCODE_SCALE threads=1 %.3fs  threads=%u %.3fs  speedup=%.2fx\n",
          s1,nt,sn,sn>0?s1/sn:0);
      }
    }
    std::printf("TOK_VERIFY %s\n", allok?"OK":"FAILED");
    return allok?0:1;
  }catch(const std::exception&e){ std::fprintf(stderr,"TOK_VERIFY_FAILED %s\n",e.what()); return 1; }
}
