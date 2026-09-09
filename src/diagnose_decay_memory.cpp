#include <array>
#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <stdexcept>
// Explicit owner features isolate retention tradeoff. Not language learning.
int main(){try{for(double lambda:{0.,.5,.9,.99,.999,1.}){for(int noise:{0,1,8,64,512,4096}){int correct=0,total=0;double targetmass=0;for(int a=0;a<8;++a)for(int b=0;b<8;++b)if(a!=b)for(int update=0;update<2;++update){std::array<double,8>m{};auto push=[&](int owner){for(auto&x:m)x*=lambda;if(owner>=0)m[owner]+=1;};push(a);if(update)push(b);int target=update?b:a;int distractor=(target+1)%8;for(int i=0;i<noise;++i)push(distractor);correct+=std::max_element(m.begin(),m.end())-m.begin()==target;targetmass+=m[target];++total;}printf("lambda=%.3f noise=%d n=%d repeated_irrelevant_accuracy=%.6f target_mass=%.12g\n",lambda,noise,total,double(correct)/total,targetmass/total);}double ab0=lambda,ab1=1,ba0=1,ba1=lambda;printf("lambda=%.3f swap_distance=%.12g\n",lambda,std::abs(ab0-ba0)+std::abs(ab1-ba1));}
// All channels same update rule; query given only to oracle selecting scale after stream.
std::array<double,5>scales{.5,.9,.99,.999,1.};for(int noise:{0,1,8,64,512}){int any=0,all=0,total=0;for(int a=0;a<8;++a)for(int b=0;b<8;++b)if(a!=b){int count=0;for(double l:scales){std::array<double,8>m{};m[a]=l;m[b]=1;for(int i=0;i<noise;++i){for(auto&v:m)v*=l;m[(b+1)%8]+=1;}count+=std::max_element(m.begin(),m.end())-m.begin()==b;}any+=count>0;all+=count==5;++total;}printf("multiscale noise=%d even_oracle_best_channel=%.6f all_channels=%.6f\n",noise,double(any)/total,double(all)/total);}
// Zero-content filler still shrinks absolute retained signal forlambda<1.
for(double l:{.9,.99,.999})printf("zero_input lambda=%.3f residual_after4096=%.12g\n",l,std::pow(l,4096));return 0;}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
