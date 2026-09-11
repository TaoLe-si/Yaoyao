// Verifies a tokenizer: digit isolation, round-trip, and corpus encode rate.
// Usage: tok_verify <tok.bbp> [text...]
#define NOMINMAX
#include "tokenizer_file.hpp"
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
using namespace tao::text;
static bool is_digit_tok(uint32_t t){ return t>=0x30u && t<=0x39u; }
int main(int argc,char**argv){
  try{
    if(argc<2) throw std::runtime_error("usage: tok_verify TOK.bbp [text...]");
    std::string fp; auto b=load_tokenizer(argv[1],fp);
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
      const bool ok = (digtok==ndig) && (b.decode(ids)==std::string(p));
      if(!ok) allok=false;
      std::printf("  %-8s ids=%-3zu digit_tokens=%zu/%zu roundtrip=%s %s\n",
        p,ids.size(),digtok,ndig,(b.decode(ids)==std::string(p))?"ok":"BAD",ok?"":"  <== FAIL");
    }
    // 3) encode the provided texts and report tokens/byte
    for(int i=2;i<argc;++i){
      std::ifstream f(argv[i],std::ios::binary); if(!f){ std::printf("  cannot open %s\n",argv[i]); continue; }
      std::string line; size_t bytes=0,toks=0,lines=0;
      while(std::getline(f,line)){ if(line.empty())continue; auto ids=b.encode(line); bytes+=line.size(); toks+=ids.size(); ++lines; }
      std::printf("  FILE %s lines=%zu bytes=%zu tokens=%zu  bytes/token=%.3f\n",
        argv[i],lines,bytes,toks,double(bytes)/double(toks?toks:1));
    }
    std::printf("TOK_VERIFY %s\n", allok?"OK":"FAILED");
    return allok?0:1;
  }catch(const std::exception&e){ std::fprintf(stderr,"TOK_VERIFY_FAILED %s\n",e.what()); return 1; }
}
