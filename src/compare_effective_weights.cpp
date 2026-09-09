#define NOMINMAX
#define TAO_INPUT_SCALE
#include "tokenizer_file.hpp"
#include "dual_model_bundle.hpp"
#include <cstdio>
int main(int argc,char**argv){try{if(argc!=3)throw std::runtime_error("two bundles required");std::string h;tao::text::load_tokenizer("build/formal_tokenizer.bbp",h);auto a=tao::dual::load_bundle(argv[1],h),b=tao::dual::load_bundle(argv[2],h);for(auto&t:tao::dual::schema(a.c)){if(!t.ternary)continue;auto&x=a.w.at(t.name);auto&y=b.w.at(t.name);if(x.size()!=y.size())throw std::runtime_error("shape");size_t symbols=0,changed=0;double ss=0,base=0;for(size_t i=0;i<x.size();++i){symbols+=((x[i]>0)-(x[i]<0))!=((y[i]>0)-(y[i]<0));changed+=x[i]!=y[i];double d=double(y[i])-x[i];ss+=d*d;base+=double(x[i])*x[i];}printf("%s n=%zu symbol_changes=%zu symbol_fraction=%.7f effective_changes=%zu delta_rms=%.9f relative_l2=%.7f\n",t.name.c_str(),x.size(),symbols,double(symbols)/x.size(),changed,sqrt(ss/x.size()),sqrt(ss/(base+1e-30)));}return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
