#include "tao_joint_selector.hpp"
#include <random>
#include <cstdio>
#include <cstring>
#include <chrono>
using namespace tao::joint;
void need(bool x){if(!x)throw std::runtime_error("test failure");}
int main(){try{std::mt19937 rng(77);std::uniform_real_distribution<double>dist(-2,2);for(int n=0;n<=16;++n)for(int t=0;t<60;++t){Input x; x.original.resize(31);x.values.assign(n,Vector(31));x.logits.resize(1+2*n);for(auto&v:x.original)v=dist(rng);for(auto&row:x.values)for(auto&v:row)v=dist(rng);for(auto&v:x.logits)v=dist(rng);auto a=forward(x);auto b=hard_forward(x);need(a.choice==b.choice&&a.hard.size()==b.value.size());need(std::memcmp(a.hard.data(),b.value.data(),31*sizeof(double))==0);}
int fetches=0;Vector original{-0.,1};auto skip=hard_lazy(original,{9,0,0},1,[&](size_t){++fetches;return Vector{2,3};});need(fetches==0&&std::memcmp(skip.value.data(),original.data(),16)==0);auto add=hard_lazy(original,{0,0,9},1,[&](size_t i){need(i==0);++fetches;return Vector{2,3};});need(fetches==1&&add.value==Vector({.5,1.75}));auto tie=hard_lazy(original,{0,0,0},1,[&](size_t){throw std::runtime_error("must not fetch");return Vector{};});need(tie.choice==0);
Input x;x.original.resize(256);x.values.assign(16,Vector(256));x.logits.resize(33);for(auto&v:x.original)v=dist(rng);for(auto&r:x.values)for(auto&v:r)v=dist(rng);for(auto&v:x.logits)v=dist(rng);volatile double sink=0;double times[3][9];for(int warm=0;warm<100;++warm){sink+=forward(x).hard[0];sink+=hard_forward(x).value[0];}
for(int round=0;round<9;++round)for(int order=0;order<3;++order){int mode=(round+order)%3;auto start=std::chrono::steady_clock::now();for(int i=0;i<2000;++i){x.logits[0]=double(i%5)*.001;if(mode==0)sink+=forward(x).hard[0];else if(mode==1)sink+=hard_forward(x).value[0];else sink+=hard_lazy(x.original,x.logits,x.values.size(),[&](size_t j)->const Vector&{return x.values[j];}).value[0];}times[mode][round]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/2000;}
for(int m=0;m<3;++m){std::sort(times[m],times[m]+9);printf("BENCH mode=%d min_us=%.6f median_us=%.6f max_us=%.6f\n",m,times[m][0],times[m][4],times[m][8]);}printf("PASS1020 bitwise hard/reference cases;skip fetch0,read fetch1,tieSkip;checksum=%.9f\n",double(sink));return 0;}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
