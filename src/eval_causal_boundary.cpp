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
int main(){try{std::map<Window,std::array<int,2>>table;std::mt19937 rng(640131);for(int n=0;n<10000;++n){auto s=grammar(training(rng,rng()%3),rng,rng()%3);Window win{12,12,12,12};for(size_t i=0;i<s.tokens.size();++i){std::rotate(win.begin(),win.begin()+1,win.end());win[3]=s.tokens[i]<4?0:(s.tokens[i]<12?4:12);bool end=i+1==s.tokens.size()||s.starts[i+1]!=s.starts[i];++table[win][end];}}
int conflicts=0;for(auto&kv:table)conflicts+=kv.second[0]&&kv.second[1];printf("boundary_table windows=%zu conflicts=%d\n",table.size(),conflicts);
for(unsigned seed:{7u,123u,2026u}){Net net(seed);std::ifstream file("build/commit_boundary4_"+std::to_string(seed)+".weights",std::ios::binary);if(!file.read((char*)net.w.data(),sizeof(net.w)))throw std::runtime_error("weights");for(int syntax=0;syntax<3;++syntax){int missed=0,wrongBoundary=0,oracleok=0,predok=0;size_t total=0;std::mt19937 test(839912+syntax);for(int n=0;n<128;++n){auto s=grammar(heldout(test,128),test,syntax);int ref;net.run(s,nullptr,nullptr,&ref);oracleok+=ref==s.target;Window win{12,12,12,12};int start=0;auto derived=s;for(size_t i=0;i<s.tokens.size();++i){std::rotate(win.begin(),win.begin()+1,win.end());win[3]=s.tokens[i]<4?0:(s.tokens[i]<12?4:12);auto it=table.find(win);bool prediction=false;if(it==table.end())++missed;else prediction=it->second[1]>it->second[0];bool truth=i+1==s.tokens.size()||s.starts[i+1]!=s.starts[i];wrongBoundary+=prediction!=truth;derived.starts[i]=start;if(prediction)start=int(i+1);++total;}int pred;net.run(derived,nullptr,nullptr,&pred);predok+=pred==s.target;}printf("seed=%u syntax=%d token_n=%zu unseen_windows=%d boundary_errors=%d oracle=%d/128 derived=%d/128\n",seed,syntax,total,missed,wrongBoundary,oracleok,predok);}}return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
