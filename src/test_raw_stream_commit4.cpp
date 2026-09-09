#define main legacy_probe_main
#include "diagnose_raw_commit4.cpp"
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
#include "raw_stream_commit4.hpp"
#include <chrono>
int main(){try{for(unsigned seed:{7u,123u,2026u,42u,999u}){Net net(seed);std::ifstream f("build/raw_commit4_"+std::to_string(seed)+".weights",std::ios::binary);if(!f.read((char*)net.w.data(),sizeof(net.w)))throw std::runtime_error("memoryweights");unsigned bs=seed==7?999:seed==123?42:seed==2026?7:seed==42?123:2026;std::array<double,217>bw;std::ifstream bf("build/raw_boundary_"+std::to_string(bs)+".weights",std::ios::binary);if(!bf.read((char*)bw.data(),sizeof(bw)))throw std::runtime_error("boundaryweights");tao::RawStreamCommit4 stream(net.w,bw);int ok=0,changedok=0,bounderr=0;for(int syntax=0;syntax<3;++syntax){std::mt19937 rng(122719+syntax);for(int n=0;n<128;++n){auto base=heldout(rng,128);auto other=base;int target=(base.target+4)%8;for(size_t i=other.tokens.size();i>0;i-=3)if(other.tokens[i-3]==other.query){other.tokens[i-2]=4+target;break;}other.target=target;auto layout=rng;auto s=grammar(base,rng,syntax);auto changed=grammar(other,layout,syntax);for(int variant=0;variant<2;++variant){auto&seq=variant?changed:s;stream.reset();for(size_t i=0;i<seq.tokens.size();++i){size_t old=stream.commits;int previous=stream.memory;stream.step(seq.query,seq.tokens[i]);bool end=i+1==seq.tokens.size()||seq.starts[i+1]!=seq.starts[i];bounderr+=(stream.commits!=old)!=end;if(stream.commits==old&&stream.memory!=previous)throw std::runtime_error("earlywrite");}if(variant)changedok+=stream.memory==seq.target;else ok+=stream.memory==seq.target;}}}
std::mt19937 rng(811);auto s=grammar(heldout(rng,32),rng,0);double times[7];volatile int sink=0;for(int r=0;r<7;++r){auto start=std::chrono::steady_clock::now();for(int n=0;n<400;++n){stream.reset();for(int t:s.tokens)sink+=stream.step(s.query,t);}times[r]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/(400*s.tokens.size());}std::sort(times,times+7);printf("memoryseed=%u boundaryseed=%u correct=%d/384 changed=%d/384 boundaryerrors=%d median_us=%.6f sink=%d\n",seed,bs,ok,changedok,bounderr,times[3],int(sink));}return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
