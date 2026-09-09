#include "pilot_reader.hpp"
#include <array>
#include <random>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <chrono>
using namespace tao::data;
constexpr int H=16,V=261,R=V*H,O=R+H*H,N=O+V*H;
using State=std::array<double,H>;using Weights=std::array<double,N>;
struct Cache{State prev,h;std::array<double,V>p;int input,target;bool loss;};
struct Rnn{Weights w{};Rnn(){std::mt19937 rng(713);std::normal_distribution<double>d(0,.08);for(auto&x:w)x=d(rng);}
Cache forward(int input,int target,bool loss,const State&prev)const{Cache c{};c.prev=prev;c.input=input;c.target=target;c.loss=loss;for(int j=0;j<H;++j){double z=w[input*H+j];for(int k=0;k<H;++k)z+=w[R+j*H+k]*prev[k];c.h[j]=std::tanh(z);}double mx=-1e100,sum=0;for(int v=0;v<V;++v){double z=0;for(int j=0;j<H;++j)z+=w[O+v*H+j]*c.h[j];c.p[v]=z;mx=std::max(mx,z);}for(double&x:c.p){x=std::exp(x-mx);sum+=x;}for(double&x:c.p)x/=sum;return c;}
void backward(const std::vector<Cache>&cs,Weights&g)const{State carry{};for(int t=int(cs.size())-1;t>=0;--t){const auto&c=cs[t];State dh=carry;for(int v=0;v<V;++v){double dz=c.loss?c.p[v]-(v==c.target):0;for(int j=0;j<H;++j){g[O+v*H+j]+=dz*c.h[j];dh[j]+=dz*w[O+v*H+j];}}carry.fill(0);for(int j=0;j<H;++j){double dz=dh[j]*(1-c.h[j]*c.h[j]);g[c.input*H+j]+=dz;for(int k=0;k<H;++k){g[R+j*H+k]+=dz*c.prev[k];carry[k]+=dz*w[R+j*H+k];}}}}
};
int main(){try{Rnn net; // Exact smooth gradient check before corpus training.
std::vector<Cache>cs;State h{};for(int i=0;i<5;++i){cs.push_back(net.forward(i+10,i+11,i>=2,h));h=cs.back().h;}Weights grad{};net.backward(cs,grad);auto loss=[&](){State s{};double l=0;for(int i=0;i<5;++i){auto c=net.forward(i+10,i+11,i>=2,s);s=c.h;if(c.loss)l-=std::log(c.p[c.target]);}return l;};double worst=0;for(int p=0;p<N;++p){double old=net.w[p],eps=1e-5;net.w[p]=old+eps;double a=loss();net.w[p]=old-eps;double b=loss();net.w[p]=old;worst=std::max(worst,std::abs((a-b)/(2*eps)-grad[p]));}if(worst>1e-7)throw std::runtime_error("gradient");printf("gradient parameters=%d max_error=%.3g\n",N,worst);
std::ifstream f("build/pilot_train.bin",std::ios::binary),vf("build/pilot_validation.bin",std::ios::binary);auto train=read_pilot(f),val=read_pilot(vf);auto evaluate=[&](){double l=0;size_t n=0;for(const auto&t:val){State s{};for(size_t i=1;i<t.size();++i){auto c=net.forward(t[i-1].id,t[i].id,t[i].loss,s);s=c.h;if(c.loss){l-=std::log(c.p[c.target]);++n;}}}return l/n;};printf("validation_before=%.6f\n",evaluate());auto start=std::chrono::steady_clock::now();size_t updates=0;for(const auto&t:train){State state{};for(size_t b=1;b<t.size();b+=32){std::vector<Cache>batch;int count=0;for(size_t i=b;i<t.size()&&i<b+32;++i){auto c=net.forward(t[i-1].id,t[i].id,t[i].loss,state);state=c.h;count+=c.loss;batch.push_back(c);}if(!count)continue;Weights g{};net.backward(batch,g);double norm=0;for(double&x:g){x/=count;norm+=x*x;}double factor=.03/std::max(1.,std::sqrt(norm));for(int j=0;j<N;++j)net.w[j]-=factor*g[j];++updates;}}double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();printf("validation_after=%.6f updates=%zu train_seconds=%.3f\n",evaluate(),updates,seconds);std::ofstream out("build/pilot_rnn16.weights",std::ios::binary);out.write((char*)net.w.data(),sizeof(net.w));if(!out)throw std::runtime_error("write");return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
