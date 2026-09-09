#define TAO_INPUT_SCALE
#define TAO_CPU_AVX2
#include "dual_model_bundle.hpp"
#include "cpu_ternary_avx512.hpp"
#define CpuTernaryRows CpuTernaryRows512
#include "byte_cpu_model.hpp"
#undef CpuTernaryRows
#include <chrono>
#include <cstdio>
#include <algorithm>
int main(){using namespace tao::dual;try{auto base=load_bundle("build/yaoyao_graph_step_360.dsb","34463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333");ByteCpuModel model(base);auto a=base.initial(),b=model.initial();unsigned token=256;for(int t=0;t<32;++t){auto x=base.step(token,a),y=model.step(token,b);if(x!=y)throw std::runtime_error("logits differ");for(size_t l=0;l<a.size();++l)if(a[l].s!=b[l].s||a[l].m!=b[l].m)throw std::runtime_error("state differs");token=std::max_element(x.begin(),x.end())-x.begin();}puts("PASS full logits/states32 exact AVX512");for(bool candidate:{false,true,true,false}){use_cpu_avx512=candidate;auto state=model.initial();unsigned id=256;auto start=std::chrono::steady_clock::now();for(int t=0;t<128;++t){model.four_rows=false;auto y=model.step(id,state);id=std::max_element(y.begin(),y.end())-y.begin();}printf("backend=%s forced128 tps=%.3f\n",candidate?"AVX512":"AVX2",128/std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());}return 0;}catch(const std::exception&e){puts(e.what());return 1;}}
