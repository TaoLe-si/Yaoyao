#define main legacy_probe_main
#include "diagnose_commit_boundary4.cpp"
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
int begin=int(out.size());
bool reverse=mode==1||(mode==0&&r()%2);
if(mode==2)out.insert(out.end(),{object,12,owner,12});
else if(reverse)out.insert(out.end(),{owner,object,12});
else out.insert(out.end(),{object,owner,12});
s.starts.insert(s.starts.end(),out.size()-begin,begin);
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

#include <map>
#include <array>
using Window=std::array<int,4>;
#include "stream_commit4.hpp"
#include <chrono>
int main(){try{std::array<std::array<int,2>,81>counts{};std::mt19937 rng(640131);for(int n=0;n<10000;++n){auto s=grammar(training(rng,rng()%3),rng,rng()%3);int win=80;for(size_t i=0;i<s.tokens.size();++i){win=(win%27)*3+(s.tokens[i]<4?0:s.tokens[i]<12?1:2);bool end=i+1==s.tokens.size()||s.starts[i+1]!=s.starts[i];++counts[win][end];}}std::array<int,81>table;table.fill(-1);for(int i=0;i<81;++i){if(counts[i][0]&&counts[i][1])throw std::runtime_error("boundary conflict");if(counts[i][0]+counts[i][1])table[i]=counts[i][1]>0;}
for(unsigned seed:{7u,123u,2026u}){Net net(seed);std::ifstream f("build/commit_boundary4_"+std::to_string(seed)+".weights",std::ios::binary);if(!f.read((char*)net.w.data(),sizeof(net.w))||f.peek()!=EOF)throw std::runtime_error("weights");tao::StreamCommit4 stream(net.w,table);int parity=0,correct=0;size_t prefixes=0;for(int syntax=0;syntax<3;++syntax){std::mt19937 data(736182+syntax);for(int n=0;n<128;++n){auto s=grammar(heldout(data,128),data,syntax);stream.reset();int expected=-1;for(size_t i=0;i<s.tokens.size();++i){bool end=i+1==s.tokens.size()||s.starts[i+1]!=s.starts[i];int result=stream.step(s.query,s.tokens[i]);if(end){Seq prefix=s;prefix.tokens.resize(i+1);prefix.starts.resize(i+1);net.run(prefix,nullptr,nullptr,&expected);if(stream.memory==-1&&expected==0)expected=-1;}if(result!=expected)throw std::runtime_error("prefix parity");++prefixes;}if(stream.unknown)throw std::runtime_error("unseen boundary");++parity;correct+=stream.memory==s.target;}}
std::mt19937 data(911);auto s=grammar(heldout(data,32),data,0);double times[7];volatile int sink=0;for(int r=0;r<7;++r){auto start=std::chrono::steady_clock::now();for(int n=0;n<400;++n){stream.reset();for(int t:s.tokens)sink+=stream.step(s.query,t);}times[r]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/(400*s.tokens.size());}std::sort(times,times+7);printf("seed=%u sequences=%d correct=%d prefix_checks=%zu us_token=%.6f sink=%d\n",seed,parity,correct,prefixes,times[3],int(sink));}return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
