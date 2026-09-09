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
int main(int argc,char**argv){try{unsigned seed=argc>1?std::stoul(argv[1]):42;
std::string optimizer=argc>2?argv[2]:"adam";
Net net(seed);
int init=argc>3?std::stoi(argv[3]):0;
if(init==1)for(auto&w:net.w)w*=.5;
if(init==2){for(int h=0;
h<H;
++h)net.w[H*I+h]*=.1;
net.w[H*I+H]=-1.;
}if(init==3){for(auto&w:net.w)w*=.5;
net.w[H*I+H]=-1.;
}printf("initialization=%d\n",init);
std::mt19937 data_rng(58191),order(12345);
std::vector<Seq>train;
std::set<std::string>seen;
while(train.size()<512){auto s=grammar(training(data_rng,data_rng()%3),data_rng,int(data_rng()%3));
if(seen.insert(key(s)).second)train.push_back(s);
}W gradient{},m{},v{};
int steps=optimizer=="adam"?3000:1200;
for(int step=1;
step<=steps;
++step){gradient.fill(0);
int batch=optimizer=="adam"?32:512;
for(int j=0;
j<batch;
++j){auto fresh=grammar(training(data_rng,data_rng()%3),data_rng,int(data_rng()%3));
seen.insert(key(fresh));
net.run(fresh,&gradient);
}for(int i=0;
i<N;
++i){double g=gradient[i]/batch+.0001*net.w[i];
if(optimizer=="adam"){m[i]=.9*m[i]+.1*g;
v[i]=.999*v[i]+.001*g*g;
net.w[i]-=.003*(m[i]/(1-std::pow(.9,step)))/(std::sqrt(v[i]/(1-std::pow(.999,step)))+1e-8);
}else net.w[i]-=.15*g;
}}double tr=0;
int tok=0;
for(auto&s:train){int a;
tr+=net.run(s,nullptr,nullptr,&a);
tok+=a==s.target;
}printf("seed=%u optimizer=%s trainhard=%.6f CE=%.9f\n",seed,optimizer.c_str(),tok/512.,tr/512);
for(int held:{0,1})for(int syntax:{0,1,2})for(int noise:{2,32}){std::mt19937 test(28181+noise);
std::set<std::string>unique;
int n=0,softok=0,hardok=0,overlap=0;
double ce=0;
int wanted=noise==0?0:256;
while(n<wanted){auto s=grammar(held?heldout(test,noise):training(test,noise),test,syntax);
auto k=key(s);
if(seen.count(k)){++overlap;
continue;
}if(!unique.insert(k).second)continue;
int a,b;
ce+=net.run(s,nullptr,&a,&b);
softok+=a==s.target;
hardok+=b==s.target;
++n;
}printf("held=%d syntax=%d noise=%d n=%d soft=%.6f hard=%.6f CE=%.9f rejected_train_overlap=%d\n",held,syntax,noise,n,double(softok)/n,double(hardok)/n,ce/n,overlap);
}std::ofstream out("build/commit_boundary4_"+std::to_string(seed)+".weights",std::ios::binary);
out.write((char*)net.w.data(),sizeof(net.w));
if(!out)throw std::runtime_error("weights");
return 0;
}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());
return 1;
}}
