#include <cstdint>
#include "pilot_reader.hpp"
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <array>
using namespace tao::data;
int main(){try{std::vector<std::vector<Token>> data[3];const char*names[]={"train","validation","test"};for(int k=0;k<3;++k){std::ifstream f(std::string("build/pilot_")+names[k]+".bin",std::ios::binary);data[k]=read_pilot(f);size_t n=0,m=0;for(auto&t:data[k])for(auto x:t){++n;m+=x.loss;}printf("validated %s docs=%zu tokens=%zu loss=%zu\n",names[k],data[k].size(),n,m);}
std::array<std::array<double,261>,261> count{};std::array<double,261>total{};for(auto&t:data[0])for(size_t i=1;i<t.size();++i)if(t[i].loss){++count[t[i-1].id][t[i].id];++total[t[i-1].id];}
for(int k=0;k<2;++k){double loss=0;size_t n=0;for(auto&t:data[k])for(size_t i=1;i<t.size();++i)if(t[i].loss){loss-=std::log((count[t[i-1].id][t[i].id]+1)/(total[t[i-1].id]+261));++n;}printf("%s bigram NLL=%.6f bits/supervised-token=%.6f n=%zu uniform_NLL=%.6f\n",names[k],loss/n,loss/n/std::log(2.),n,std::log(261.));}
for(auto bad:{std::string("BAD!"),std::string("TLP1x")}){bool rejected=false;try{std::istringstream s(bad);read_pilot(s);}catch(const std::runtime_error&){rejected=true;}if(!rejected)throw std::runtime_error("negative test");}printf("PASS malformed header/truncated length rejected\n");return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
