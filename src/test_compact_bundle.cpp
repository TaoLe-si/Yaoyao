#define TAO_INPUT_SCALE
#include "cpu_compact_bundle.hpp"
#include <cstdio>
int main(){using namespace tao::dual;try{std::string h="34463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333",p="build/yaoyao_graph_step_360.dsb";auto a=load_bundle(p,h);auto b=read_compact_bundle(p,h);size_t n=0;for(auto&t:schema(a.c)){auto&w=a.w.at(t.name);if(t.ternary){auto&m=b.matrices.at(t.name);for(size_t i=0;i<w.size();++i)if(w[i]!=float(m.q[i])*m.scale[i/t.cols])return 1;}else if(w!=b.vectors.at(t.name))return 2;n+=w.size();}puts(("PASS direct compact load exact parameters="+std::to_string(n)).c_str());return 0;}catch(const std::exception&e){puts(e.what());return 3;}}
