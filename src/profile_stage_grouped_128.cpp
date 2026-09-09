#define NOMINMAX
#define TAO_CPU_AVX2
#define TAO_INPUT_SCALE
#include "profile_stage_grouped_byte_cpu_model.hpp"
#include <algorithm>
#include <cstdio>
using namespace tao::dual;
int main(int argc,char**argv){try{
 if(argc>3)throw std::runtime_error("usage: profile_stage_grouped_128 [bundle [tokenizer_hash]]");
 const char*path=argc>1?argv[1]:"build/yaoyao_graph_step_360.dsb";
 const char*hash=argc>2?argv[2]:"34463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333";
 // Calibration only, no model steps: timer+counter floor, NOT an overhead correction.
 ProfileStageGroupedByteCpuModel::Bucket calibration;
 constexpr size_t calibration_scopes=10000;
 const auto cs=ProfileStageGroupedByteCpuModel::Clock::now();
 for(size_t i=0;i<calibration_scopes;++i){ProfileStageGroupedByteCpuModel::Timer timer(calibration);}
 const double calibration_seconds=std::chrono::duration<double>(ProfileStageGroupedByteCpuModel::Clock::now()-cs).count();
 const double ns_per_scope=1e9*calibration_seconds/calibration_scopes;
 std::printf("CALIBRATION empty_timer_scopes=%zu seconds=%.9f ns_per_scope=%.3f NOT_OVERHEAD_CORRECTION=1\n",calibration_scopes,calibration_seconds,ns_per_scope);
 const ProfileStageGroupedByteCpuModel model(read_compact_bundle(path,hash));
 auto state=model.initial();unsigned token=256;unsigned long long trace=1469598103934665603ull;
 const auto start=ProfileStageGroupedByteCpuModel::Clock::now();
 for(int i=0;i<128;++i){auto logits=model.step(token,state);for(float v:logits)if(!std::isfinite(v))throw std::runtime_error("nonfinite logits");token=unsigned(std::max_element(logits.begin(),logits.end())-logits.begin());trace=(trace^token)*1099511628211ull;}
 const double elapsed=std::chrono::duration<double>(ProfileStageGroupedByteCpuModel::Clock::now()-start).count();
 std::printf("RUN positions=128 seconds=%.9f tps=%.3f final_token=%u trace=%llu threads=2 deployed=0\n",elapsed,128/elapsed,token,trace);
 model.print_profile();
 const size_t scopes=model.total.calls+model.group_s.calls+model.group_m.calls+model.group_read.calls+model.ff_linear.calls+model.head_linear.calls+model.norm_time.calls+model.add_time.calls+model.nonlinear.calls;
 std::printf("CAVEAT empty_scope_floor_seconds=%.9f floor_pct_step=%.3f excludes_executor_and_schedule_perturbation=1\n",scopes*ns_per_scope*1e-9,model.total.seconds?100*scopes*ns_per_scope*1e-9/model.total.seconds:0);
 std::puts("CAVEAT leaf buckets disjoint; step contains leaves; executor dispatch/wait nested inside projection buckets: DO NOT ADD. Wait includes worker compute/imbalance and scheduling, not pure overhead. Timers perturb synchronization/cache/inlining; calibration is not measured production overhead. RUN also includes finite checks/greedy outside STEP. Destructor PROFILE repeats nested executor totals, not extra work.");
 return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
