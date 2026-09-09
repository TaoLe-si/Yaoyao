#include "tao_ternary.hpp"
#include <cstdio>
#include <random>
#include <climits>
namespace tt=tao::ternary;
void require(bool b){if(!b)throw std::runtime_error("test failure");}
template<class F>void rejects(F f){bool ok=false;try{f();}catch(const std::invalid_argument&){ok=true;}require(ok);}
int main(){try{
 for(int a=-1;a<=1;++a)for(int b=-1;b<=1;++b)require(tt::subtract(tt::add(a,b),b)==a);
 for(int x:{INT_MIN,INT_MAX,-5,-4,-3,-2,-1,0,1,2,3,4,5}){auto y=tt::mod3(x);require(tt::valid(y)&&((int64_t(x)-y)%3==0));}
 require(tt::pack({0,1,-1,0})==std::vector<uint8_t>{0x24});
 std::mt19937 rng(42);for(size_t n=0;n<1025;++n){std::vector<int8_t>a(n),b(n);int64_t ref=0;for(size_t i=0;i<n;++i){a[i]=int(rng()%3)-1;b[i]=int(rng()%3)-1;ref+=a[i]*b[i];}auto p=tt::pack(a);require(tt::unpack(p,n)==a);require(tt::dot(p,tt::pack(b),n)==ref);}
 rejects([]{tt::pack({2});});rejects([]{tt::unpack({3},1);});rejects([]{tt::unpack({4},1);});rejects([]{tt::unpack({},1);});rejects([]{tt::unpack({0},0);});
 auto plus=tt::pack(std::vector<int8_t>(100000,1));require(tt::dot(plus,plus,100000)==100000);require(tt::add(1,1)==-1);
 auto row=tt::quantize({-9,-1,-.5f,0,.5f,1,9},1);require(tt::unpack(row.codes,row.count)==std::vector<int8_t>({-1,-1,0,0,0,1,1}));require(tt::dequantize(row)==std::vector<float>({-1,-1,0,0,0,1,1}));
 require(tt::dot_float(tt::pack({1,-1,0}),{2,3,99},2)==-2);
 for(float s:{0.f,-1.f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()})rejects([&]{tt::quantize({0},s);});
 rejects([]{tt::quantize({std::numeric_limits<float>::quiet_NaN()},1);});rejects([]{tt::dot_float(tt::pack({0}),{std::numeric_limits<float>::infinity()},1);});
 printf("PASS ternary v1 encoding, lengths0..1024 roundtrip/dot, reserved/padding/length rejection, int boundaries, mod3 inverse,100000 wide sum, scale/tie/nonfinite checks\n");return 0;
 }catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
