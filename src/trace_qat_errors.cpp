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

#include "residual_write_hard.hpp"
int main(){try{for(unsigned seed:{7u,123u,31415u})for(int shift:{0,1,2,3}){split_shift=shift;Net net(seed);std::ifstream f("build/qat_spaced_0_91357_1000_partition_"+std::to_string(shift)+"_"+std::to_string(seed)+".weights",std::ios::binary);if(!f.read((char*)net.w.data(),sizeof(net.w)))throw std::runtime_error("weights");tao::ResidualWriteHard<32> hard(net.w);for(int syntax=0;syntax<3;++syntax){int errors=0,miss=0,wrong=0,polluted=0;int byowner[8]={};std::mt19937 rng(481273+syntax);for(int n=0;n<128;++n){auto base=heldout(rng,128);size_t last=0;for(size_t j=0;j<base.tokens.size();j+=3)if(base.tokens[j]==base.query)last=j/3;Seq s{base.query,base.target,{}};std::vector<size_t>ends;for(size_t j=0;j<base.tokens.size();j+=3){int o=base.tokens[j],who=base.tokens[j+1];bool rev=syntax==1||(syntax==0&&rng()%2);if(syntax==2)s.tokens.insert(s.tokens.end(),{o,12,who,12});else if(rev)s.tokens.insert(s.tokens.end(),{who,o,12});else s.tokens.insert(s.tokens.end(),{o,who,12});ends.push_back(s.tokens.size());}hard.reset();size_t start=last?ends[last-1]:0,before=0;int after=0;size_t writes=0;for(size_t j=0;j<s.tokens.size();++j){if(j==start)before=hard.writes;hard.step(s.query,s.tokens[j]);if(j+1==ends[last]){after=hard.memory;writes=hard.writes-before;}}int ref;net.run(s,nullptr,nullptr,&ref);if(ref!=hard.memory)throw std::runtime_error("parity");if(ref!=s.target){++errors;++byowner[s.target];if(after==s.target)++polluted;else if(!writes)++miss;else ++wrong;}}printf("seed=%u shift=%d syntax=%d errors=%d/128 missed=%d wrote_wrong=%d polluted=%d owner_errors=",seed,shift,syntax,errors,miss,wrong,polluted);for(int x:byowner)printf("%d,",x);printf("\n");}}return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
