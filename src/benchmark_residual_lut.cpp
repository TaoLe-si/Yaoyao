#define main legacy_probe_main
#include "diagnose_expand_write4.cpp"
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
int split_shift=0;
bool allowed(const Seq&s){for(size_t t=0;
t<s.tokens.size();
t+=3)if((s.tokens[t+1]-4)%4==(s.tokens[t]+split_shift)%4)return false;
return true;
}
Seq training(std::mt19937&r,int noise){auto s=make(r,noise);
for(size_t t=0;
t<s.tokens.size();
t+=3)while((s.tokens[t+1]-4)%4==(s.tokens[t]+split_shift)%4)s.tokens[t+1]=4+r()%8;
for(size_t t=s.tokens.size();
t>0;
t-=3)if(s.tokens[t-3]==s.query){s.target=s.tokens[t-2]-4;
break;
}if(!allowed(s))throw std::runtime_error("split");
return s;
}
Seq heldout(std::mt19937&r,int noise){auto s=training(r,noise);
int owner=(s.query+split_shift)%4+4*(r()%2);
for(size_t t=s.tokens.size();
t>0;
t-=3)if(s.tokens[t-3]==s.query){s.tokens[t-2]=4+owner;
break;
}s.target=owner;
return s;
}

#include "residual_write_threshold.hpp"

#include "residual_packed_lut.hpp"
#include <chrono>
template<class Model> double measure(Model&m,const Seq&s){volatile int sink=0;double times[7];for(int r=0;r<7;++r){auto start=std::chrono::steady_clock::now();for(int i=0;i<200;++i){m.reset();for(int t:s.tokens)sink+=m.step(s.query,t);}times[r]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/(200*s.tokens.size());}std::sort(times,times+7);if(sink==0)throw std::runtime_error("sink");return times[3];}
int main(){try{for(unsigned seed:{7u,123u,31415u}){Net n(seed);std::ifstream f("build/qat_spaced_0_91357_1000_partition_0_"+std::to_string(seed)+".weights",std::ios::binary);if(!f.read((char*)n.w.data(),sizeof(n.w)))throw std::runtime_error("weights");tao::ResidualWriteThreshold<32>d(n.w);PackedResidualLut x(n.w);std::mt19937 r(193);auto s=grammar(heldout(r,32),r,0);double td=measure(d,s),tf=measure(x,s);printf("seed=%u FP64_us=%.6f packed_us=%.6f ratio=%.3f\n",seed,td,tf,tf/td);}return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
