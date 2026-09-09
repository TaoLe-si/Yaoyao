#include "tao_ternary.hpp"
#include <array>
#include <string>
#include <random>
#include <cstdio>
#include <cmath>
#include <algorithm>
using State=std::array<int8_t,64>;
struct Chain{std::array<State,16>emb;Chain(unsigned seed){std::mt19937 r(seed);for(auto&e:emb)for(auto&v:e)v=int(r()%3)-1;}void push(State&s,int id,int mode)const{auto&e=emb[id];if(mode>=2){State old=s;for(int d=0;d<64;++d){int v=old[(d+63)%64]+e[d];s[d]=mode==2?tao::ternary::mod3(v):int8_t(v);}return;}if(mode==0){for(int d=0;d<64;++d)s[d]=tao::ternary::mod3(s[d]+e[d]);return;}for(int d=0;d<32;++d)s[d]=tao::ternary::mod3(s[d]+e[d]+s[32+d]*e[32+d]);for(int d=0;d<32;++d)s[32+d]=tao::ternary::mod3(s[32+d]+e[32+d]+s[(d+1)%32]*e[d]);}void pop(State&s,int id,int mode)const{auto&e=emb[id];if(mode>=2){State old=s;for(int d=0;d<64;++d){int v=old[(d+1)%64]-e[(d+1)%64];s[d]=mode==2?tao::ternary::mod3(v):int8_t(v);}return;}if(mode==0){for(int d=0;d<64;++d)s[d]=tao::ternary::mod3(s[d]-e[d]);return;}for(int d=0;d<32;++d)s[32+d]=tao::ternary::mod3(s[32+d]-e[32+d]-s[(d+1)%32]*e[d]);for(int d=0;d<32;++d)s[d]=tao::ternary::mod3(s[d]-e[d]-s[32+d]*e[32+d]);}};
struct Sample{State x;int y;int split;};
struct Net {
 std::vector<double>w,u;
 Net(unsigned seed):w(24*65),u(8*25){std::mt19937 rng(seed);std::normal_distribution<double>d(0,.08);for(auto&x:w)x=d(rng);for(auto&x:u)x=d(rng);}
 double run(const Sample&s,double scale,std::vector<double>*gw=nullptr,std::vector<double>*gu=nullptr,int*pred=nullptr)const{
 double h[24],z[8],mx=-1e300,sum=0;for(int i=0;i<24;++i){double a=w[i*65+64];for(int d=0;d<64;++d)a+=w[i*65+d]*s.x[d]*scale;h[i]=std::tanh(a);}
 int best=0;for(int v=0;v<8;++v){z[v]=u[v*25+24];for(int i=0;i<24;++i)z[v]+=u[v*25+i]*h[i];if(z[v]>mx){mx=z[v];best=v;}}for(double x:z)sum+=std::exp(x-mx);double loss=mx+std::log(sum)-z[s.y];if(pred)*pred=best;
 if(gw){double dh[24]={};for(int v=0;v<8;++v){double dz=std::exp(z[v]-mx)/sum-(v==s.y);(*gu)[v*25+24]+=dz;for(int i=0;i<24;++i){(*gu)[v*25+i]+=dz*h[i];dh[i]+=dz*u[v*25+i];}}for(int i=0;i<24;++i){double da=dh[i]*(1-h[i]*h[i]);(*gw)[i*65+64]+=da;for(int d=0;d<64;++d)(*gw)[i*65+d]+=da*s.x[d]*scale;}}return loss;
 }
};
int main(int argc,char**argv){try{unsigned seed=argc>1?unsigned(std::stoul(argv[1])):42;Chain c(seed);std::vector<Sample>data;int count[3]={};for(int a=0;a<8;++a)for(int b=0;b<8;++b)if(a!=b)for(int o=0;o<4;++o){unsigned hash=(unsigned(std::min(a,b)*8+std::max(a,b))*73856093u)^(unsigned(o)*19349663u)^919u;bool held=hash%4==0;for(int layout=0;layout<2;++layout)for(int query=0;query<2;++query)for(int noise:{0,1,3}){if(noise==3&&!held)continue;State s{};std::vector<int>seq=layout==0?std::vector<int>{a,12,8+o,13,b}:std::vector<int>{b,13,8+o,12,a};for(int n=0;n<noise;++n)seq.push_back(8+(o+n+1)%4);seq.push_back(query?15:14);for(int id:seq)c.push(s,id,3);int split=noise==3?2:held?1:0;data.push_back({s,query?a:b,split});++count[split];}}
for(int qat=0;qat<2;++qat){Net master(seed+999),effective=master;auto quantize=[&](){effective=master;if(qat)for(int layer=0;layer<2;++layer){auto&w=layer?effective.u:effective.w;int stride=layer?25:65;double scale=layer?.5:.25;for(size_t i=0;i<w.size();++i)if(i%stride!=stride-1)w[i]=w[i]>scale/2?scale:w[i]<-scale/2?-scale:0;}};
std::vector<double>gw(master.w.size()),gu(master.u.size());for(int step=0;step<1200;++step){quantize();std::fill(gw.begin(),gw.end(),0);std::fill(gu.begin(),gu.end(),0);for(auto&s:data)if(s.split==0)effective.run(s,1/std::sqrt(6.),&gw,&gu);for(int layer=0;layer<2;++layer){auto&w=layer?master.u:master.w;auto&g=layer?gu:gw;double scale=layer?.5:.25;int stride=layer?25:65;for(size_t i=0;i<w.size();++i){bool matrix=i%stride!=stride-1;double grad=g[i]/count[0];if(qat&&matrix&&std::abs(w[i])>scale)grad=0;w[i]-=.15*(grad+.001*w[i]);if(qat&&matrix)w[i]=std::max(-scale,std::min(scale,w[i]));}}}quantize();int correct[3]={};double ce[3]={};for(auto&s:data){int pred;ce[s.split]+=effective.run(s,1/std::sqrt(6.),nullptr,nullptr,&pred);correct[s.split]+=pred==s.y;}for(int split=0;split<3;++split)printf("seed=%u head=%s split=%d n=%d accuracy=%.6f CE=%.9f\n",seed,qat?"ternary_QAT":"float",split,count[split],double(correct[split])/count[split],ce[split]/count[split]);}
return 0;}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
