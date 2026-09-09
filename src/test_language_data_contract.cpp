#include "language_data_contract.hpp"
#include <cstdio>
using namespace tao::data;
void check(bool b){if(!b)throw std::runtime_error("contract");}
int main(){try{auto t=encode({{false,"Q<EOS>"},{true,"A\n"}});check(t.back().id==EOS&&t.back().loss);size_t losses=0;for(auto x:t)losses+=x.loss;check(losses==4);
 for(size_t width:{1u,2u,7u,100u}){auto c=chunks(t,width);size_t pos=0,ends=0;for(size_t i=0;i<c.size();++i){auto&e=c[i];check(e.reset==(i==0));for(size_t j=0;j<e.input.size();++j,++pos){check(e.input[j]==t[pos].id&&e.target[j]==t[pos+1].id&&e.loss[j]==t[pos+1].loss);ends+=e.target[j]==EOS;}}check(pos==t.size()-1&&ends==1);}
 auto u=encode({{false,"unfinished prompt"}});check(!u.back().loss);auto c=chunks(u,3);check(c[0].reset);std::string all;for(int i=0;i<256;++i)all.push_back(char(i));auto bytes=encode({{true,all}});for(int i=0;i<256;++i)check(bytes[i+2].id==i);check(bytes.size()==260);
 bool rejected=false;try{chunks(t,0);}catch(const std::invalid_argument&){rejected=true;}check(rejected);printf("PASS byte coverage, target masks, chunk continuity, real EOS, reset contract\n");return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
