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

int main(){try{
#ifdef EXPAND
const char*name="expand";
#else
const char*name="deep16";
#endif
for(unsigned seed:{42u,123u,2026u,7u,999u}){Net net(seed);std::ifstream f(std::string("build/")+name+"_write4_"+std::to_string(seed)+".weights",std::ios::binary);if(!f.read((char*)net.w.data(),sizeof(net.w))||f.peek()!=EOF)throw std::runtime_error("weights");for(int syntax=0;syntax<3;++syntax){std::mt19937 rng(948123+syntax);int softok=0,hardok=0,updateok=0;std::set<std::string>seen;double ce=0;for(int n=0;n<128;){auto base=heldout(rng,128);auto altered=base;altered.target=(base.target+4)%8;for(size_t t=altered.tokens.size();t>0;t-=3)if(altered.tokens[t-3]==base.query){altered.tokens[t-2]=4+altered.target;break;}auto layout=rng;auto s=grammar(base,rng,syntax);auto changed=grammar(altered,layout,syntax);if(!seen.insert(key(s)).second)continue;int a,b;ce+=net.run(s,nullptr,&a,&b);softok+=a==s.target;hardok+=b==s.target;net.run(changed,nullptr,nullptr,&b);updateok+=b==changed.target;++n;}printf("model=%s seed=%u syntax=%d noise128 soft=%d/128 hard=%d/128 update=%d/128 CE=%.6f\n",name,seed,syntax,softok,hardok,updateok,ce/128);}}return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
