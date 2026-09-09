#define TAO_CPU_AVX2
#define TAO_INPUT_SCALE
#include "dual_model_bundle.hpp"
#include "byte_cpu_model.hpp"
#include "signmask_byte_cpu_model.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <random>
#include <limits>

static volatile float sink = 0;
static void equal(float a, float b) {
    if (!std::isfinite(a) || !std::isfinite(b) || std::memcmp(&a,&b,sizeof(float)))
        throw std::runtime_error("bitwise finite comparison failed");
}
static void equal_vec(const std::vector<float>& a,const std::vector<float>& b) {
    if(a.size()!=b.size()) throw std::runtime_error("comparison shape");
    for(size_t i=0;i<a.size();++i) equal(a[i],b[i]);
}
template<class M> static double dot_time(const M& m,std::vector<float>& x,int repeats) {
    auto start=std::chrono::steady_clock::now();
    for(int k=0;k<repeats;++k) {
        x[0]=float(k%31-15)*0.03125f; // inhibit loop-invariant dot hoisting
        for(size_t r=0;r<m.rows;++r) sink=m.dot(r,x.data());
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
}
static void synthetic() {
    std::mt19937 rng(7321);
    size_t checks=0;
    std::vector<size_t> widths;
    for(size_t c=1;c<=80;++c) widths.push_back(c);
    for(size_t c: {127u,128u,129u,255u,256u,257u,511u,512u,513u,1023u,1024u,1025u}) widths.push_back(c);
    for(size_t c:widths) {
        const size_t rows=9;
        std::vector<float> w(rows*c), storage(c+1);
        float* x=storage.data()+1; // deliberately not guaranteed AVX aligned
        for(size_t j=0;j<c;++j) x[j]=float(int(rng()%20001)-10000)*0.00013f;
        for(size_t r=0;r<rows;++r) for(size_t j=0;j<c;++j) {
            int q=r==0?0:(r==1?-1:(r==2?1:int(rng()%3)-1));
            float scale=r==3?0.1234567f:(r==4?1e-20f:(r==5?1e10f:float(r+1)*0.03125f));
            w[r*c+j]=float(q)*scale;
        }
        CpuTernaryRows byte(w,rows,c); CpuTernarySignMaskRows packed(w,rows,c);
        for(size_t r=0;r<rows;++r) {
            for(size_t j=0;j<c;++j) if(packed.q[r*c+j]!=byte.q[r*c+j]) throw std::runtime_error("symbol mismatch");
            equal(byte.dot(r,x),packed.dot(r,x)); ++checks;
        }
    }
    bool rejected=false;
    try { CpuTernarySignMaskRows bad(std::vector<float>{1,2},1,2); }
    catch(const std::runtime_error&) {rejected=true;}
    if(!rejected) throw std::runtime_error("invalid ternary accepted");
    std::printf("CHECK synthetic bitwise rows=%zu widths=%zu tails=0..7 PASS\n",checks,widths.size());
    // New-kernel edge checks: signed zeros, subnormal scale, DAZ/FTZ.
    const unsigned saved=_mm_getcsr();
    for(unsigned mode: {0u,0x8040u}) {
        _mm_setcsr((saved & ~0x8040u)|mode);
        try {
            for(size_t c=1;c<=17;++c) {
                for(float scale: {0.1234567f,std::numeric_limits<float>::min(),std::numeric_limits<float>::denorm_min()}) {
                    // Set exact bits without performing subnormal arithmetic.
                    std::vector<float> w(c),x(c);
                    for(size_t j=0;j<c;++j) {
                        uint32_t bits;std::memcpy(&bits,&scale,4);
                        if(j%3==0)bits=0x80000000u;
                        else if(j%3==1)bits|=0x80000000u;
                        std::memcpy(&w[j],&bits,4);
                        x[j]=j%2?-0.0f:0.75f;
                    }
                    CpuTernaryRows byte(w,1,c);CpuTernarySignMaskRows candidate(w,1,c);
                    equal(byte.dot(0,x.data()),candidate.dot(0,x.data()));
                }
            }
        }catch(...){_mm_setcsr(saved);throw;}
    }
    _mm_setcsr(saved);
    for(float bad: {std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        bool rejected=false;
        try {CpuTernarySignMaskRows candidate(std::vector<float>{bad},1,1);}
        catch(const std::runtime_error&){rejected=true;}
        if(!rejected)throw std::runtime_error("nonfinite accepted");
    }
    std::puts("CHECK signmask signed-zero/subnormal/DAZ-FTZ/nonfinite PASS");
}
template<class M> static double model_time(const M& m,unsigned first,int steps) {
    auto state=m.initial(); unsigned token=first;
    auto start=std::chrono::steady_clock::now();
    for(int i=0;i<steps;++i) {
        auto y=m.step(token,state);
        token=unsigned(std::max_element(y.begin(),y.end())-y.begin());
        sink=y[token];
    }
    return steps/std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
}
int main(int argc,char** argv) {
    try {
        synthetic();
        if(argc==1) {std::puts("Full-model optional: benchmark_ternary_signmask <bundle.dsb> <tokenizer_sha256> [steps=128]");return 0;}
        if(argc<3 || argc>4) throw std::runtime_error("expected bundle, tokenizer hash, optional steps");
        int steps=argc==4?std::stoi(argv[3]):128;
        if(steps<=0) throw std::runtime_error("steps must be positive");
        using namespace tao::dual;
        auto src=load_bundle(argv[1],argv[2]);
        ByteCpuModel byte(src); SignMaskByteCpuModel packed(src);
        src.w.clear(); // release unused FP32 reference before timing
        unsigned first=byte.c.vocab>256?256:0, token=first;
        auto a=byte.initial(),b=packed.initial();
        for(int i=0;i<32;++i) {
            auto x=byte.step(token,a),y=packed.step(token,b);
            equal_vec(x,y);
            for(size_t l=0;l<a.size();++l) {equal_vec(a[l].s,b[l].s);equal_vec(a[l].m,b[l].m);}
            token=unsigned(std::max_element(x.begin(),x.end())-x.begin());
            if(token!=unsigned(std::max_element(y.begin(),y.end())-y.begin())) throw std::runtime_error("argmax mismatch");
        }
        std::printf("CHECK full-model bitwise logits/state/argmax steps=32 PASS byte_weight_bytes=%zu packed_weight_bytes=%zu\n",byte.weight_bytes(),packed.weight_bytes());
        model_time(byte,first,4); model_time(packed,first,4);
        for(int trial=0;trial<4;++trial) {
            double bt,pt;
            if(trial%2) {pt=model_time(packed,first,steps);bt=model_time(byte,first,steps);}
            else {bt=model_time(byte,first,steps);pt=model_time(packed,first,steps);}
            std::printf("FULL_GREEDY trial=%d forced_steps=%d byte_tps=%.3f signmask_tps=%.3f signmask_over_byte=%.4f\n",trial,steps,bt,pt,pt/bt);
        }
        return 0;
    } catch(const std::exception& e) {std::printf("FAIL %s\n",e.what());return 1;}
}
