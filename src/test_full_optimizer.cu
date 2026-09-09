#define TAO_INPUT_SCALE
#include "dual_state_sorted_trainer.cuh"
#include "dual_state_initialization.hpp"
#include <fstream>
#include <chrono>
#include <cstdio>
__global__ void fixture(float*g,size_t n){size_t i=size_t(blockIdx.x)*256+threadIdx.x;if(i<n)g[i]=float(int(i%29)-14)*.01f;}
int main(int argc,char**argv){try{if(argc!=2)return 2;using namespace tao::dual;SortedGpuTrainer tr(initialize(Config{},713));for(auto&t:tr.spec){auto&g=tr.graph.w.at(t.name)->grad;fixture<<<(g.n+255)/256,256>>>(g.p,g.n);}check(cudaDeviceSynchronize());auto t=std::chrono::steady_clock::now();float norm=tr.update(148,.0005f);double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();std::ofstream f(argv[1],std::ios::binary);for(auto&spec:tr.spec){for(auto*p:{tr.master.at(spec.name).get(),tr.moment.at(spec.name).get(),tr.variance.at(spec.name).get(),&tr.graph.w.at(spec.name)->value}){auto v=p->host();f.write(reinterpret_cast<const char*>(v.data()),v.size()*4);}}f.close();if(!f)return 3;printf("FULL_OPTIMIZER tensors=%zu norm=%.9g update_seconds=%.6f dump=%s\n",tr.spec.size(),norm,seconds,argv[1]);return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
