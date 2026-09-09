#include "byte_bpe_incremental.hpp"
#include <random>
#include <cstdio>
int main(){try{using namespace tao::text;std::mt19937 rng(831);size_t cases=0;for(int n=0;n<40;++n){std::vector<std::string>corpus{"aaaaaaa","abababab","banana banana",""};for(int j=0;j<8;++j){std::string s;for(int k=0;k<32;++k)s.push_back(char('a'+rng()%6));corpus.push_back(s);}ByteBpe ref;ref.fit(corpus,64);auto fast=fit_incremental(corpus,64);if(ref.merges!=fast.merges)throw std::runtime_error("merge mismatch");for(auto&s:corpus)if(fast.decode(fast.encode(s))!=s)throw std::runtime_error("roundtrip");++cases;}printf("PASS incremental/reference complete merge tables cases=%zu overlap/ties/boundaries\n",cases);return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
