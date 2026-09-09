#include "tao_ternary.hpp"
#include <array>
#include <string>
#include <random>
#include <cstdio>
#include <cmath>
#include <algorithm>
using State=std::array<int8_t,64>;
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
int main(int argc,char**argv){try{unsigned seed=argc>1?unsigned(std::stoul(argv[1])):42;Chain c(seed);for(int mode=2;mode<=3;++mode){std::vector<Sample>data;for(int a=0;a<8;++a)for(int b=0;b<8;++b)if(a!=b)for(int o=0;o<4;++o)for(int layout=0;layout<2;++layout)for(int query=0;query<2;++query){State s{};std::vector<int>seq=layout==0?std::vector<int>{a,12,8+o,13,b}:std::vector<int>{b,13,8+o,12,a};if(layout)seq.insert(seq.end(),{15,8+(o+1)%4,15});seq.push_back(query?15:14);for(int id:seq)c.push(s,id,mode);data.push_back({s,query?a:b,(std::min(a,b)+std::max(a,b)+o)%4==0});}
 Net net(seed+999);double scale=mode==3?1/std::sqrt(6.):1.;std::vector<double>gw(net.w.size()),gu(net.u.size());net.run(data[0],scale,&gw,&gu);double maxerr=0;
 for(int layer=0;layer<2;++layer){auto&weights=layer?net.u:net.w;auto&gradient=layer?gu:gw;for(int idx:{0,24,100,199}){double old=weights[idx],eps=1e-6;weights[idx]=old+eps;double hi=net.run(data[0],scale);weights[idx]=old-eps;double lo=net.run(data[0],scale);weights[idx]=old;maxerr=std::max(maxerr,std::abs((hi-lo)/(2*eps)-gradient[idx]));}}if(maxerr>1e-7)throw std::runtime_error("gradient failure");
 for(int step=0;step<1200;++step){std::fill(gw.begin(),gw.end(),0);std::fill(gu.begin(),gu.end(),0);for(auto&s:data)if(!s.test)net.run(s,scale,&gw,&gu);for(size_t i=0;i<gw.size();++i)net.w[i]-=.15*(gw[i]/672+.001*net.w[i]);for(size_t i=0;i<gu.size();++i)net.u[i]-=.15*(gu[i]/672+.001*net.u[i]);}
 int correct[2]={};double ce[2]={};for(auto&s:data){int pred;ce[s.test]+=net.run(s,scale,nullptr,nullptr,&pred);correct[s.test]+=pred==s.y;}printf("seed=%u mode=%s params=%zu grad_error=%.12g train_acc=%.6f test_acc=%.6f train_CE=%.9f test_CE=%.9f\n",seed,mode==2?"mod3":"wide",net.w.size()+net.u.size(),maxerr,double(correct[0])/672,double(correct[1])/224,ce[0]/672,ce[1]/224);
}return 0;}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
