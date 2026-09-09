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
int role_mode=0;int exposure=0;std::mt19937 injection(82419);unsigned long long counts[4][8]={},roles[3][4][8]={};
Seq exposed(std::mt19937&r,int noise){auto s=training(r,noise);size_t last=0;std::vector<size_t>irrelevant;for(size_t j=0;j<s.tokens.size();j+=3){if(s.tokens[j]==s.query)last=j;else irrelevant.push_back(j);}if(int(injection()%10000)<exposure){if(role_mode==0||!irrelevant.empty()){size_t j=role_mode==0?last:irrelevant[injection()%irrelevant.size()];s.tokens[j+1]=4+(s.tokens[j]+split_shift)%4+4*(injection()%2);}}s.target=s.tokens[last+1]-4;for(size_t j=0;j<s.tokens.size();j+=3){int role=j==last?0:(s.tokens[j]==s.query?1:2);++counts[s.tokens[j]][s.tokens[j+1]-4];++roles[role][s.tokens[j]][s.tokens[j+1]-4];}return s;}
int main(int argc,char**argv){try{unsigned seed=argc>1?std::stoul(argv[1]):42;
std::string optimizer=argc>2?argv[2]:"adam";
split_shift=argc>4?std::stoi(argv[4]):0;
if(split_shift<0||split_shift>3)throw std::runtime_error("shift");
exposure=argc>5?std::stoi(argv[5]):0;if(exposure<0||exposure>10000)throw std::runtime_error("exposure");
role_mode=argc>7?std::stoi(argv[7]):0;if(role_mode<0||role_mode>1)throw std::runtime_error("role");
unsigned data_seed=argc>6?std::stoul(argv[6]):58191;
injection.seed(data_seed==58191?82419:data_seed+24228);
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
std::mt19937 data_rng(data_seed),order(12345);
std::vector<Seq>train;
std::set<std::string>seen;
while(train.size()<512){auto s=grammar(exposed(data_rng,data_rng()%3),data_rng,int(data_rng()%3));
if(seen.insert(key(s)).second)train.push_back(s);
}for(auto&row:counts)for(auto&c:row)c=0;for(auto&role:roles)for(auto&row:role)for(auto&c:row)c=0;
W gradient{},m{},v{};
int steps=optimizer=="adam"?3000:1200;
for(int step=1;
step<=steps;
++step){gradient.fill(0);
int batch=optimizer=="adam"?32:512;
for(int j=0;
j<batch;
++j){auto fresh=grammar(exposed(data_rng,data_rng()%3),data_rng,int(data_rng()%3));
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
}printf("exposure_bp=%d pair_counts=",exposure);for(auto&row:counts)for(auto c:row)printf("%llu,",c);printf("\n");
for(int role=0;role<3;++role){printf("role=%d rare_pair_counts=",role);for(int object=0;object<4;++object)for(int owner=0;owner<8;++owner)if(owner%4==(object+split_shift)%4)printf("%llu,",roles[role][object][owner]);printf("\n");}
std::ofstream out("build/role_only_"+std::to_string(role_mode)+"_"+std::to_string(data_seed)+"_"+std::to_string(exposure)+"_partition_"+std::to_string(split_shift)+"_"+std::to_string(seed)+".weights",std::ios::binary);
out.write((char*)net.w.data(),sizeof(net.w));
if(!out)throw std::runtime_error("weights");
return 0;
}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());
return 1;
}}
