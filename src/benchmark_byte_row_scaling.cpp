#define NOMINMAX
#define TAO_CPU_AVX2
#define TAO_INPUT_SCALE
#include "byte_cpu_model.hpp"
#include "configurable_row_byte_cpu_model.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <atomic>
using namespace tao::dual;
static void equal(const Vec&a,const Vec&b){
    if(a.size()!=b.size())throw std::runtime_error("size mismatch");
    for(size_t i=0;i<a.size();++i)
        if(!std::isfinite(a[i])||!std::isfinite(b[i])||std::memcmp(&a[i],&b[i],sizeof(float)))
            throw std::runtime_error("nonfinite or bitwise mismatch at element "+std::to_string(i));
}
static unsigned greedy(const Vec&v){return unsigned(std::max_element(v.begin(),v.end())-v.begin());}
static void executor_checks(size_t threads){
    CpuConfigurableRowExecutor executor(threads,262144);
    for(int mode=0;mode<3;++mode){
        bool caught=false;
        try {executor.run(1024,256,[&](size_t begin,size_t){
            if(mode==2||(mode==0&&begin==0)||(mode==1&&begin!=0))throw std::runtime_error("expected");
        });}catch(const std::runtime_error&){caught=true;}
        if(!caught)throw std::runtime_error("exception propagation failed");
        std::atomic<size_t> count{0};
        executor.run(1024,256,[&](size_t begin,size_t end){count.fetch_add(end-begin);});
        if(count!=1024)throw std::runtime_error("executor recovery failed");
    }
    for(size_t rows:{1u,3u,1023u,1024u,1025u}){
        const size_t cols=257;Vec weights(rows*cols),x(cols+1),a(rows),b(rows);
        for(size_t j=0;j<x.size();++j)x[j]=float(int(j%31)-15)*0.037f;
        for(size_t r=0;r<rows;++r)for(size_t j=0;j<cols;++j)
            weights[r*cols+j]=float(int((r+j)%3)-1)*0.031f;
        CpuTernaryRows p(weights,rows,cols);
        for(size_t r=0;r<rows;++r)a[r]=p.dot(r,x.data()+1);
        executor.run(rows,cols,[&](size_t begin,size_t end){for(size_t r=begin;r<end;++r)b[r]=p.dot(r,x.data()+1);});
        equal(a,b);
    }
    std::printf("CHECK executor_exception_recovery=1 kernel_bitwise_equal=1\n");
}
template<class Model> static double bench(Model&model,const char*name){
    auto state=model.initial();unsigned token=256;
    const auto start=std::chrono::steady_clock::now();
    for(int i=0;i<128;++i)token=greedy(model.step(token,state));
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::printf("RUN backend=%s tokens=128 seconds=%.9f tps=%.3f final_token=%u\n",name,seconds,128/seconds,token);
    return 128/seconds;
}
int main(){try{
    executor_checks(2);executor_checks(4);
    const char* path="build/yaoyao_graph_step_360.dsb";
    // Existing bundle API validates this tokenizer identity plus payload checksum.
    const char* identity="34463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333";
    ByteCpuModel baseline(read_compact_bundle(path,identity));
    double means[2][2]{};
    for(size_t ti=0;ti<2;++ti)for(size_t hi=0;hi<2;++hi){
        const size_t threads=ti==0?2:4,threshold=hi==0?262144:1048576;
        // One live pool only. Destroy and join before the next configuration.
        ConfigurableRowByteCpuModel candidate(read_compact_bundle(path,identity),threads,threshold);
        std::printf("CONFIG threads=%zu threshold=%zu\n",threads,threshold);
        size_t scheduled=0,total=0;uint64_t scheduled_elements=0;
        for(const auto&kv:candidate.packed){
            const auto&p=kv.second;++total;
            if(p.rows>=threads&&p.cols&&p.rows>=threshold/p.cols+(threshold%p.cols!=0)){
                ++scheduled;scheduled_elements+=uint64_t(p.rows)*p.cols;
            }
        }
        std::printf("SHAPES parallel_matvecs_per_step=%zu total_matvecs_per_step=%zu parallel_elements=%llu\n",scheduled,total,(unsigned long long)scheduled_elements);
        auto a=baseline.initial(),b=candidate.initial();unsigned token=256;
        for(int i=0;i<32;++i){
            auto x=baseline.step(token,a),y=candidate.step(token,b);equal(x,y);
            for(size_t l=0;l<a.size();++l){equal(a[l].s,b[l].s);equal(a[l].m,b[l].m);}
            if(greedy(x)!=greedy(y))throw std::runtime_error("argmax mismatch");
            token=greedy(x);
        }
        std::printf("CHECK threads=%zu threshold=%zu recurrent_steps=32 full_logits_states_bitwise_equal=1\n",threads,threshold);
        double base[2],cand[2];
        base[0]=bench(baseline,"single_row");cand[0]=bench(candidate,"configurable_row");
        cand[1]=bench(candidate,"configurable_row");base[1]=bench(baseline,"single_row");
        const double bmean=(base[0]+base[1])/2,cmean=(cand[0]+cand[1])/2;
        means[ti][hi]=cmean;
        std::printf("SUMMARY threads=%zu threshold=%zu pairs=2 tokens_per_trial=128 baseline_mean_tps=%.3f candidate_mean_tps=%.3f ratio=%.6f pair0_ratio=%.6f pair1_ratio=%.6f deployed=0\n",threads,threshold,bmean,cmean,cmean/bmean,cand[0]/base[0],cand[1]/base[1]);
    }
    std::printf("SCALING four_over_two_low=%.6f four_over_two_high=%.6f high_over_low_two=%.6f high_over_low_four=%.6f\n",means[1][0]/means[0][0],means[1][1]/means[0][1],means[0][1]/means[0][0],means[1][1]/means[1][0]);
    // Timing alone cannot separate bandwidth saturation from scheduler contention.
    // Higher threshold winning suggests dispatch overhead; four threads losing
    // suggests overhead/contention/saturation rather than useful core scaling.
    return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
catch(...){std::fprintf(stderr,"FAIL unknown exception\n");return 1;}}
