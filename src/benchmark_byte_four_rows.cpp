#define NOMINMAX
#define TAO_CPU_AVX2
#define TAO_INPUT_SCALE
#include "dual_model_bundle.hpp"
#include "byte_cpu_model.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
using namespace tao::dual;
static void equal(const Vec&a,const Vec&b){
    if(a.size()!=b.size())throw std::runtime_error("size mismatch");
    for(size_t i=0;i<a.size();++i)
        if(!std::isfinite(a[i])||!std::isfinite(b[i])||std::memcmp(&a[i],&b[i],sizeof(float)))
            throw std::runtime_error("nonfinite or bitwise mismatch");
}
static unsigned greedy(const Vec&v){return unsigned(std::max_element(v.begin(),v.end())-v.begin());}
static void kernels(){
    unsigned cases=0;
    for(size_t rows: {1u,2u,3u,4u,5u,7u,8u,9u})
    for(size_t cols: {1u,7u,8u,9u,15u,16u,17u,31u,32u,33u,128u,256u}){
        Vec w(rows*cols),input(cols+1),a(rows),b(rows);
        for(size_t j=0;j<input.size();++j)input[j]=float(int((j*37)%101)-50)*0.037f;
        for(size_t r=0;r<rows;++r)for(size_t j=0;j<cols;++j)
            w[r*cols+j]=r%4==0?0.f:float(int((j*13+r)%3)-1)*(0.031f*float(r+1));
        CpuTernaryRows p(w,rows,cols);
        // Offset input exercises unaligned AVX2 loads.
        for(size_t r=0;r<rows;++r)a[r]=p.dot(r,input.data()+1);
        p.matvec_four_rows(input.data()+1,b.data());equal(a,b);++cases;
    }
    std::printf("CHECK kernel_cases=%u bitwise_equal=1\n",cases);
}
static double bench(ByteCpuModel&model,bool four){
    model.four_rows=four;auto state=model.initial();unsigned token=256;
    const auto start=std::chrono::steady_clock::now();
    for(int i=0;i<128;++i)token=greedy(model.step(token,state));
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::printf("RUN backend=%s tokens=128 seconds=%.9f tps=%.3f final_token=%u\n",four?"four_rows":"single_row",seconds,128/seconds,token);
    return 128/seconds;
}
int main(){try{
    kernels();
    ByteCpuModel model(load_bundle("build/yaoyao_graph_step_360.dsb","34463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333"));
    auto a=model.initial(),b=model.initial();unsigned token=256;
    for(int i=0;i<32;++i){
        model.four_rows=false;auto x=model.step(token,a);
        model.four_rows=true;auto y=model.step(token,b);equal(x,y);
        for(size_t l=0;l<a.size();++l){equal(a[l].s,b[l].s);equal(a[l].m,b[l].m);}
        if(greedy(x)!=greedy(y))throw std::runtime_error("argmax mismatch");
        token=greedy(x);
    }
    std::printf("CHECK recurrent_steps=32 logits_states_bitwise_equal=1 argmax_equal=1 threads=1\n");
    // Same weights, reset state, full greedy forward+argmax; no EOS early stop.
    // Alternate order to reduce bias; excludes load/packing and initial allocation.
    double baseline[4],candidate[4];
    for(int i=0;i<4;++i){
        if(i%2){candidate[i]=bench(model,true);baseline[i]=bench(model,false);}
        else{baseline[i]=bench(model,false);candidate[i]=bench(model,true);}
    }
    std::sort(baseline,baseline+4);std::sort(candidate,candidate+4);
    const double base=(baseline[1]+baseline[2])/2,cand=(candidate[1]+candidate[2])/2;
    std::printf("SUMMARY baseline_median_tps=%.3f candidate_median_tps=%.3f ratio=%.6f deployed=0\n",base,cand,cand/base);
    return 0;
}catch(const std::exception&e){std::printf("FAIL %s\n",e.what());return 1;}}
