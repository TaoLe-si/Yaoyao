#define main legacy_probe_main
#ifdef EXPAND
#include "diagnose_expand_write4.cpp"
#else
#include "diagnose_deep16_write4.cpp"
#endif
#undef main
#include <set>
#include <sstream>
#include <fstream>
std::string key(const Seq&s){std::ostringstream out;
out<<s.query<<":";
for(int t:s.tokens)out<<t<<",";
return out.str();
}
Seq grammar(Seq s,std::mt19937&r,int mode){std::vector<int>out;
for(size_t t=0;
t<s.tokens.size();
t+=3){int object=s.tokens[t],owner=s.tokens[t+1];
bool reverse=mode==1||(mode==0&&r()%2);
if(mode==2)out.insert(out.end(),{object,12,owner,12});
else if(reverse)out.insert(out.end(),{owner,object,12});
else out.insert(out.end(),{object,owner,12});
}s.tokens=out;
return s;
}
bool allowed(const Seq&s){for(size_t t=0;
t<s.tokens.size();
t+=3)if((s.tokens[t+1]-4)%4==s.tokens[t])return false;
return true;
}
Seq training(std::mt19937&r,int noise){auto s=make(r,noise);
for(size_t t=0;
t<s.tokens.size();
t+=3)while((s.tokens[t+1]-4)%4==s.tokens[t])s.tokens[t+1]=4+r()%8;
for(size_t t=s.tokens.size();
t>0;
t-=3)if(s.tokens[t-3]==s.query){s.target=s.tokens[t-2]-4;
break;
}if(!allowed(s))throw std::runtime_error("split");
return s;
}
Seq heldout(std::mt19937&r,int noise){auto s=training(r,noise);
int owner=s.query+4*(r()%2);
for(size_t t=s.tokens.size();
t>0;
t-=3)if(s.tokens[t-3]==s.query){s.tokens[t-2]=4+owner;
break;
}s.target=owner;
return s;
}


#include "residual_write_hard.hpp"
#include <chrono>
int main(){try{
#ifdef EXPAND
constexpr int E=32;const char*name="expand";
#else
constexpr int E=16;const char*name="deep16";
#endif
for(unsigned seed:{42u,123u,2026u,7u,999u}){Net net(seed);std::ifstream f(std::string("build/")+name+"_write4_"+std::to_string(seed)+".weights",std::ios::binary);if(!f.read((char*)net.w.data(),sizeof(net.w))||f.peek()!=EOF)throw std::runtime_error("weights");tao::ResidualWriteHard<E> hard(net.w);int agree=0,correct=0;for(int syntax=0;syntax<3;++syntax){std::mt19937 rng(938241+syntax);for(int n=0;n<64;++n){auto s=grammar(heldout(rng,128),rng,syntax);int ref;net.run(s,nullptr,nullptr,&ref);hard.reset();for(int t:s.tokens)hard.step(s.query,t);if(hard.memory!=ref)throw std::runtime_error("parity");++agree;correct+=hard.memory==s.target;}}std::mt19937 rng(193);auto s=grammar(heldout(rng,32),rng,0);double times[7];volatile int sink=0;for(int r=0;r<7;++r){auto start=std::chrono::steady_clock::now();for(int n=0;n<200;++n){hard.reset();for(int t:s.tokens)sink+=hard.step(s.query,t);}times[r]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/(200*s.tokens.size());}std::sort(times,times+7);printf("model=%s seed=%u parity=%d/192 correct=%d/192 median_us=%.6f sink=%d\n",name,seed,agree,correct,times[3],int(sink));}return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
