#define TAO_INPUT_SCALE
#define TAO_CPU_AVX2
#include "profile_cpu_row_executor.hpp"
#include "cpu_row_parallel_executor.hpp"
#define CpuRowParallelExecutor ProfileRowExecutor
#include "row_parallel_byte_cpu_model.hpp"
#undef CpuRowParallelExecutor
#include <algorithm>
#include <cstdio>
int main(){using namespace tao::dual;RowParallelByteCpuModel m(read_compact_bundle("build/yaoyao_graph_step_360.dsb","34463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333"));auto s=m.initial();unsigned t=256;auto start=std::chrono::steady_clock::now();for(int i=0;i<128;++i){auto y=m.step(t,s);t=std::max_element(y.begin(),y.end())-y.begin();}double total=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();printf("PROFILE total_seconds=%.6f\n",total);}
