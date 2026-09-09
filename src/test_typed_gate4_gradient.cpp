#define main probe_main
#include "diagnose_typed_gate4.cpp"
#undef main
#include <fstream>
int main(){try{double err=0;for(unsigned seed:{42u,123u,2026u}){Net net(seed);std::mt19937 rng(997);auto s=make(rng,1);W g{};net.run(s,&g);for(int i=0;i<N;++i){double old=net.w[i],eps=1e-6;net.w[i]=old+eps;double hi=net.run(s);net.w[i]=old-eps;double lo=net.run(s);net.w[i]=old;err=std::max(err,std::abs((hi-lo)/(2*eps)-g[i]));}}if(err>1e-7)throw std::runtime_error("gradient");printf("PASS sparse BPTT3195 coordinates maxerr=%.12g away from clipping kinks\n",err);return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
