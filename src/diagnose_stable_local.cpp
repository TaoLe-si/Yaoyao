#include "tao_ternary.hpp"
#include <array>
#include <string>
#include <random>
#include <cstdio>
#include <cmath>
#include <algorithm>
using State=std::array<int32_t,64>;
struct Chain{std::array<State,16>emb;Chain(unsigned seed){std::mt19937 r(seed);for(auto&e:emb)for(auto&v:e)v=int(r()%3)-1;}void push(State&s,int id,int mode)const{auto&e=emb[id];if(mode>=2){State old=s;for(int d=0;d<64;++d){int v=old[(d+63)%64]+e[d];s[d]=mode==2?tao::ternary::mod3(v):int8_t(v);}return;}if(mode==0){for(int d=0;d<64;++d)s[d]=tao::ternary::mod3(s[d]+e[d]);return;}for(int d=0;d<32;++d)s[d]=tao::ternary::mod3(s[d]+e[d]+s[32+d]*e[32+d]);for(int d=0;d<32;++d)s[32+d]=tao::ternary::mod3(s[32+d]+e[32+d]+s[(d+1)%32]*e[d]);}void pop(State&s,int id,int mode)const{auto&e=emb[id];if(mode>=2){State old=s;for(int d=0;d<64;++d){int v=old[(d+1)%64]-e[(d+1)%64];s[d]=mode==2?tao::ternary::mod3(v):int8_t(v);}return;}if(mode==0){for(int d=0;d<64;++d)s[d]=tao::ternary::mod3(s[d]-e[d]);return;}for(int d=0;d<32;++d)s[32+d]=tao::ternary::mod3(s[32+d]-e[32+d]-s[(d+1)%32]*e[d]);for(int d=0;d<32;++d)s[d]=tao::ternary::mod3(s[d]-e[d]-s[32+d]*e[32+d]);}};
struct Sample{State x;int y;bool test;};
struct Net {
 std::vector<double>w,u;
 Net(unsigned seed):w(24*65),u(8*25){std::mt19937 rng(seed);std::normal_distribution<double>d(0,.08);for(auto&x:w)x=d(rng);for(auto&x:u)x=d(rng);}
 double run(const Sample&s,double scale,std::vector<double>*gw=nullptr,std::vector<double>*gu=nullptr,int*pred=nullptr)const{
 double h[24],z[8],mx=-1e300,sum=0;for(int i=0;i<24;++i){double a=w[i*65+64];for(int d=0;d<64;++d)a+=w[i*65+d]*s.x[d]*scale;h[i]=std::tanh(a);}
 int best=0;for(int v=0;v<8;++v){z[v]=u[v*25+24];for(int i=0;i<24;++i)z[v]+=u[v*25+i]*h[i];if(z[v]>mx){mx=z[v];best=v;}}for(double x:z)sum+=std::exp(x-mx);double loss=mx+std::log(sum)-z[s.y];if(pred)*pred=best;
 if(gw){double dh[24]={};for(int v=0;v<8;++v){double dz=std::exp(z[v]-mx)/sum-(v==s.y);(*gu)[v*25+24]+=dz;for(int i=0;i<24;++i){(*gu)[v*25+i]+=dz*h[i];dh[i]+=dz*u[v*25+i];}}for(int i=0;i<24;++i){double da=dh[i]*(1-h[i]*h[i]);(*gw)[i*65+64]+=da;for(int d=0;d<64;++d)(*gw)[i*65+d]+=da*s.x[d]*scale;}}return loss;
 }
};
#include <map>
#include <chrono>
State encode(const Chain&c,const std::vector<int>&tokens,int mode){State state{};int previous=15,older=15;for(int token:tokens){if(mode==0){State old=state;for(int d=0;d<64;++d)state[d]=old[(d+63)%64]+c.emb[token][d];}else for(int d=0;d<64;++d){int u=c.emb[token][d]*c.emb[previous][(d+7)%64];if(mode==2)u*=c.emb[older][(d+19)%64];state[d]+=u;}older=previous;previous=token;}return state;}
std::vector<int> event(int a,int b,int object,int query,int noise){std::vector<int>s{a,12,8+object,13,b};for(int n=0;n<noise;++n)s.push_back(8+(object+n+1)%4);s.push_back(query?15:14);return s;}
int main(int argc,char**argv){try{unsigned seed=argc>1?std::stoul(argv[1]):42;Chain c(seed);for(int mode=0;mode<3;++mode){std::vector<Sample>data;int ntrain=0,ntest=0;for(int a=0;a<8;++a)for(int b=0;b<8;++b)if(a!=b)for(int o=0;o<4;++o)for(int q=0;q<2;++q){bool test=(std::min(a,b)+std::max(a,b)+o)%4==0;data.push_back({encode(c,event(a,b,o,q,0),mode),q?a:b,test});(test?ntest:ntrain)++;}Net net(seed+999);std::vector<double>gw(net.w.size()),gu(net.u.size());for(int step=0;step<1200;++step){std::fill(gw.begin(),gw.end(),0);std::fill(gu.begin(),gu.end(),0);for(auto&s:data)if(!s.test)net.run(s,1/std::sqrt(6.),&gw,&gu);for(size_t i=0;i<gw.size();++i)net.w[i]-=.15*(gw[i]/ntrain+.001*net.w[i]);for(size_t i=0;i<gu.size();++i)net.u[i]-=.15*(gu[i]/ntrain+.001*net.u[i]);}for(int noise:{0,1,3,16}){int correct=0,total=0;double ce=0;for(int a=0;a<8;++a)for(int b=0;b<8;++b)if(a!=b)for(int o=0;o<4;++o)if((std::min(a,b)+std::max(a,b)+o)%4==0)for(int q=0;q<2;++q){Sample s{encode(c,event(a,b,o,q,noise),mode),q?a:b,true};int pred;ce+=net.run(s,1/std::sqrt(6.),nullptr,nullptr,&pred);correct+=pred==s.y;++total;}printf("seed=%u mode=%d noise=%d n=%d accuracy=%.6f CE=%.9f\n",seed,mode,noise,total,double(correct)/total,ce/total);}
// Two event blocks separated by enough identical padding to reset finite local context.
int collisions=0,total=0;for(int a=0;a<8;++a)for(int b=0;b<8;++b)if(a!=b){std::vector<int>A{8,12,a,15,15,15},B{8,12,b,15,15,15};auto x=A;x.insert(x.end(),B.begin(),B.end());x.push_back(14);auto y=B;y.insert(y.end(),A.begin(),A.end());y.push_back(14);collisions+=encode(c,x,mode)==encode(c,y,mode);++total;}printf("LATEST_EVENT seed=%u mode=%d conflicting_collisions=%d/%d\n",seed,mode,collisions,total);
std::vector<int>longseq;for(int i=0;i<1000;++i)longseq.push_back(i%16);volatile int sink=0;double times[5];for(int round=0;round<5;++round){auto start=std::chrono::steady_clock::now();for(int rep=0;rep<200;++rep){longseq[0]=rep%16;sink+=encode(c,longseq,mode)[0];}times[round]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/200000;}std::sort(times,times+5);printf("CPU mode=%d D64 update_us=%.6f sink=%d\n",mode,times[2],int(sink));}return 0;}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
