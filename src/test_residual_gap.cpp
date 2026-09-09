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

int main(){try{for(unsigned seed:{7u,123u,31415u}){split_shift=0;Net net(seed);std::ifstream f("build/role_spaced_0_91357_1000_partition_0_"+std::to_string(seed)+".weights",std::ios::binary);if(!f.read((char*)net.w.data(),sizeof(net.w)))throw std::runtime_error("weights");for(int gap:{0,1,3,8,32}){int ok=0,update=0;std::mt19937 rng(719381);for(int n=0;n<128;++n){auto b=heldout(rng,32);Seq s{b.query,b.target,{}},changed=s;size_t last=0;for(size_t j=0;j<b.tokens.size();j+=3)if(b.tokens[j]==b.query)last=j;for(size_t j=0;j<b.tokens.size();j+=3){s.tokens.push_back(b.tokens[j]);for(int k=0;k<gap;++k)s.tokens.push_back(12);s.tokens.push_back(b.tokens[j+1]);s.tokens.push_back(12);}changed=s;changed.target=(s.target+4)%8;changed.tokens[(last/3)*(gap+3)+gap+1]=4+changed.target;int a;net.run(s,nullptr,nullptr,&a);ok+=a==s.target;net.run(changed,nullptr,nullptr,&a);update+=a==changed.target;}printf("seed=%u gap=%d original=%d/128 updated=%d/128\n",seed,gap,ok,update);}}return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
