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

#include "residual_packed.hpp"
int main(){try{for(unsigned seed:{7u,123u,31415u})for(int shift:{0,1,2,3}){split_shift=shift;Net net(seed);std::ifstream f("build/qat_spaced_0_91357_1000_partition_"+std::to_string(shift)+"_"+std::to_string(seed)+".weights",std::ios::binary);if(!f.read((char*)net.w.data(),sizeof(net.w))||f.peek()!=EOF)throw std::runtime_error("weights");for(double threshold:{0.}){PackedResidual m(net.w);int correct=0,updated=0,parity=0;for(int syntax=0;syntax<3;++syntax){std::mt19937 rng(583921+syntax);for(int i=0;i<64;++i){auto base=heldout(rng,128);auto other=base;other.target=(base.target+4)%8;for(size_t j=other.tokens.size();j>0;j-=3)if(other.tokens[j-3]==base.query){other.tokens[j-2]=4+other.target;break;}auto layout=rng;auto s=grammar(base,rng,syntax),changed=grammar(other,layout,syntax);m.reset();for(int t:s.tokens)m.step(s.query,t);correct+=m.memory==s.target;if(threshold==0){int ref;net.run(s,nullptr,nullptr,&ref);if(ref!=m.memory)throw std::runtime_error("parity");++parity;}m.reset();for(int t:changed.tokens)m.step(changed.query,t);updated+=m.memory==changed.target;}}printf("seed=%u shift=%d threshold=%+.1f correct=%d/192 update=%d/192 parity=%d\n",seed,shift,threshold,correct,updated,parity);}}return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
