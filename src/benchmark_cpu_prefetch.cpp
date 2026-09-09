#define TAO_INPUT_SCALE
#define TAO_CPU_AVX2
#include "byte_cpu_model.hpp"
#include "prefetch_cpu_model.hpp"
#include <chrono>
#include <algorithm>
#include <cstdio>
int main(){using namespace tao::dual;try{std::string p="build/yaoyao_graph_step_360.dsb",h="34463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333";ByteCpuModel a(read_compact_bundle(p,h));PrefetchCpuModel b(read_compact_bundle(p,h));auto sa=a.initial(),sb=b.initial();unsigned t=256;for(int i=0;i<32;++i){auto x=a.step(t,sa),y=b.step(t,sb);if(x!=y)throw std::runtime_error("logits");for(size_t l=0;l<sa.size();++l)if(sa[l].s!=sb[l].s||sa[l].m!=sb[l].m)throw std::runtime_error("state");t=std::max_element(x.begin(),x.end())-x.begin();}puts("PASS prefetch32 fullstate/logits exact");auto bench=[](auto&m){auto s=m.initial();unsigned t=256;auto start=std::chrono::steady_clock::now();for(int i=0;i<128;++i){auto y=m.step(t,s);t=std::max_element(y.begin(),y.end())-y.begin();}return 128/std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};for(int i=0;i<2;++i){double x,y;if(!i){x=bench(a);y=bench(b);}else{y=bench(b);x=bench(a);}printf("baseline=%.3f prefetch=%.3f ratio=%.3f\n",x,y,y/x);}return 0;}catch(const std::exception&e){puts(e.what());return 1;}}
