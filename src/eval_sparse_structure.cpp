#define main probe_main
#include "diagnose_sparse_write.cpp"
#undef main
#include "token_write_hard.hpp"
#include <fstream>
int main(){for(unsigned seed:{42u,123u,2026u,7u,999u}){Net net(seed);std::ifstream f("build/sparse_write_adam_"+std::to_string(seed)+".weights",std::ios::binary);if(!f.read((char*)net.w.data(),sizeof(net.w)))return 1;tao::TokenWriteHard hard(net.w);for(int mode=0;mode<3;++mode){std::mt19937 rng(81273);int correct=0;for(int i=0;i<256;++i){auto s=make(rng,8);std::vector<int>tokens;for(size_t t=0;t<s.tokens.size();t+=3){if(mode==0)tokens.insert(tokens.end(),{s.tokens[t],s.tokens[t+1],12});if(mode==1)tokens.insert(tokens.end(),{s.tokens[t+1],s.tokens[t],12});if(mode==2)tokens.insert(tokens.end(),{s.tokens[t],12,s.tokens[t+1],12});}hard.reset();for(int t:tokens)hard.step(s.query,t);correct+=hard.memory==s.target;}printf("seed=%u grammar=%d hard_accuracy=%.6f\n",seed,mode,correct/256.);}}}
