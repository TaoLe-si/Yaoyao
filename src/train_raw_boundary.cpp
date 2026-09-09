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
struct Boundary {
 std::array<double,217>w{};
 Boundary(unsigned seed){std::mt19937 r(seed);std::normal_distribution<double>d(0,.1);for(auto&x:w)x=d(r);}
 double run(const std::array<int,4>&x,int label,std::array<double,217>*g=nullptr,int*prediction=nullptr)const{
 double h[4];for(int j=0;j<4;++j){double a=w[j*53+52];for(int k=0;k<4;++k)a+=w[j*53+k*13+x[k]];h[j]=std::tanh(a);}
 double z=w[216];for(int j=0;j<4;++j)z+=w[212+j]*h[j];double p=z>=0?1/(1+std::exp(-z)):std::exp(z)/(1+std::exp(z));if(prediction)*prediction=z>=0;
 if(g){double dz=p-label;(*g)[216]+=dz;for(int j=0;j<4;++j){(*g)[212+j]+=dz*h[j];double da=dz*w[212+j]*(1-h[j]*h[j]);(*g)[j*53+52]+=da;for(int k=0;k<4;++k)(*g)[j*53+k*13+x[k]]+=da;}}
 return std::max(z,0.)-label*z+std::log1p(std::exp(-std::abs(z)));
 }
};
int main(int argc,char**argv){try{unsigned seed=argc>1?std::stoul(argv[1]):42;Boundary b(seed);std::array<double,217>g{},m{},v{};std::array<int,4>x{1,5,12,2};b.run(x,0,&g);double err=0;for(int i=0;i<217;++i){double old=b.w[i],eps=1e-6;b.w[i]=old+eps;double hi=b.run(x,0);b.w[i]=old-eps;double lo=b.run(x,0);b.w[i]=old;err=std::max(err,std::abs((hi-lo)/(2*eps)-g[i]));}if(err>1e-7)throw std::runtime_error("gradient");
std::mt19937 rng(412391);for(int step=1;step<=2000;++step){g.fill(0);int count=0;for(int n=0;n<8;++n){auto s=grammar(training(rng,rng()%3),rng,rng()%3);x={12,12,12,12};for(size_t t=0;t<s.tokens.size();++t){std::rotate(x.begin(),x.begin()+1,x.end());x[3]=s.tokens[t];bool end=t+1==s.tokens.size()||s.starts[t+1]!=s.starts[t];b.run(x,end,&g);++count;}}for(int i=0;i<217;++i){double d=g[i]/count;m[i]=.9*m[i]+.1*d;v[i]=.999*v[i]+.001*d*d;b.w[i]-=.003*(m[i]/(1-std::pow(.9,step)))/(std::sqrt(v[i]/(1-std::pow(.999,step)))+1e-8);}}
size_t total=0,wrong=0;int badseq=0;for(int syntax=0;syntax<3;++syntax){std::mt19937 test(632811+syntax);for(int n=0;n<128;++n){auto s=grammar(heldout(test,128),test,syntax);x={12,12,12,12};bool bad=false;for(size_t t=0;t<s.tokens.size();++t){std::rotate(x.begin(),x.begin()+1,x.end());x[3]=s.tokens[t];bool end=t+1==s.tokens.size()||s.starts[t+1]!=s.starts[t];int pred;b.run(x,end,nullptr,&pred);wrong+=pred!=end;bad|=pred!=end;++total;}badseq+=bad;}}
std::ofstream out("build/raw_boundary_"+std::to_string(seed)+".weights",std::ios::binary);out.write((char*)b.w.data(),sizeof(b.w));if(!out)throw std::runtime_error("write");printf("seed=%u params217 grad=%.12g boundary_errors=%zu/%zu badseq=%d/384\n",seed,err,wrong,total,badseq);return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
