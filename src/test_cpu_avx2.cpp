#define NOMINMAX
#define TAO_INPUT_SCALE
#define TAO_CPU_AVX2
#include "tokenizer_file.hpp"
#include "dual_model_bundle.hpp"
#include <chrono>
#include <cstdio>
int main(){try{std::string h;tao::text::load_tokenizer("build/formal_tokenizer.bbp",h);auto m=tao::dual::load_bundle("build/yaoyao_arch2_step_24.dsb",h);std::vector<tao::dual::Vec> ref;auto state=m.initial();m.fast=false;auto t=std::chrono::steady_clock::now();for(unsigned i=0;i<64;++i)ref.push_back(m.step(i==0?256:261+(i*179)%16000,state));double scalar=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();auto expected=state;state=m.initial();m.fast=true;float err=0;t=std::chrono::steady_clock::now();for(unsigned i=0;i<64;++i){auto y=m.step(i==0?256:261+(i*179)%16000,state);for(size_t j=0;j<y.size();++j)err=std::max(err,std::abs(y[j]-ref[i][j]));}double fast=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();float se=0;for(size_t l=0;l<state.size();++l){for(size_t i=0;i<state[l].s.size();++i)se=std::max(se,std::abs(state[l].s[i]-expected[l].s[i]));for(size_t i=0;i<state[l].m.size();++i)se=std::max(se,std::abs(state[l].m[i]-expected[l].m[i]));}printf("full_model64positions scalar_seconds=%.6f avx_seconds=%.6f scalar_tps=%.3f avx_tps=%.3f logits_max_error=%.9g state_max_error=%.9g\n",scalar,fast,64/scalar,64/fast,err,se);if(!std::isfinite(err)||err>.001||se>.0001)return 1;return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
