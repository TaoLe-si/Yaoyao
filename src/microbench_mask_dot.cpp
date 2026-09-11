// 路线 B：利用三值结构的掩码点积，对照现状 AVX2-float 内核。
//   dot = Σ_{w=+1} x − Σ_{w=−1} x
// 权重存成两个位图（plus / minus），每行 cols/8 字节 × 2 = 128 字节（cols=512），
// 相比 int8 的 512 字节省 4×。
//
// 精确性设计：参考内核的每 lane 累加顺序是
//   lane L += (w[L]*scale)*x[L]; lane L += (w[8+L]*scale)*x[8+L]; ...
// 本实现先算 xv = x*scale 再掩码加减，因 (±scale)*x == ±(scale*x)（IEEE 乘法可交换），
// 且 lane 顺序一致，故逐位相等。校验和必须完全相同。
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <immintrin.h>

// ---- 参考内核：照抄 cpu_pipeline_rows.hpp ----
static float dot_avx2_float(const int8_t* q,const float* x,size_t cols,float scale){
    __m256 sum=_mm256_setzero_ps(),alpha=_mm256_set1_ps(scale);size_t j=0;
    for(;j+16<=cols;j+=16){
        auto w0=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(q+j)))),alpha);
        auto w1=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(q+j+8)))),alpha);
        sum=_mm256_add_ps(sum,_mm256_mul_ps(w0,_mm256_loadu_ps(x+j)));
        sum=_mm256_add_ps(sum,_mm256_mul_ps(w1,_mm256_loadu_ps(x+j+8)));
    }
    for(;j+8<=cols;j+=8){
        __m256 w=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(q+j)))),alpha);
        sum=_mm256_add_ps(sum,_mm256_mul_ps(w,_mm256_loadu_ps(x+j)));
    }
    alignas(32)float lane[8];_mm256_store_ps(lane,sum);float z=0;for(float v:lane)z+=v;
    for(;j<cols;++j)z+=(float(q[j])*scale)*x[j];
    return z;
}

// ---- 路线 B：掩码内核 ----
static float dot_mask(const uint8_t* plus,const uint8_t* minus,const float* x,size_t cols,float scale){
    __m256 sum=_mm256_setzero_ps(),alpha=_mm256_set1_ps(scale);
    size_t j=0;
    for(;j+8<=cols;j+=8){
        __m256 xv=_mm256_mul_ps(_mm256_loadu_ps(x+j),alpha);
        __mmask8 kp=plus[j>>3],km=minus[j>>3];
        // 与参考内核一致：两路都以 sum 为累加器，顺序为 j 递增
        sum=_mm256_mask_add_ps(sum,kp,sum,xv);
        sum=_mm256_mask_sub_ps(sum,km,sum,xv);
    }
    alignas(32)float lane[8];_mm256_store_ps(lane,sum);float z=0;for(float v:lane)z+=v;
    for(;j<cols;++j){
        if(plus[j>>3]&(1u<<(j&7)))z+=scale*x[j];
        else if(minus[j>>3]&(1u<<(j&7)))z-=scale*x[j];
    }
    return z;
}


// ---- 路线 A：int8 量化激活 + VNNI vpdpbusd（精确的 u8 x s8 + 零点修正）----
// 激活 x 量化: a = round(x / xscale)，a in [-127,127]；存为 u8 时加 128 偏移。
// dot_int = Σ (a_s8+128)*w_s8 = Σ a_s8*w + 128*Σ w
// 故 dot = (vpdpbusd(a_u8, w_s8) - 128*rowsum) * xscale * scale
static int32_t dot_vnni_exact(const int8_t* q,const uint8_t* au8,size_t cols){
    __m512i acc=_mm512_setzero_si512();size_t j=0;
    for(;j+64<=cols;j+=64){
        __m512i w=_mm512_loadu_si512((const void*)(q+j));
        __m512i a=_mm512_loadu_si512((const void*)(au8+j));
        acc=_mm512_dpbusd_epi32(acc,a,w);
    }
    alignas(64)int32_t lane[16];_mm512_store_si512((void*)lane,acc);
    int32_t z=0;for(int v:lane)z+=v;
    for(;j<cols;++j)z+=int32_t(q[j])*int32_t(au8[j])-128*int32_t(q[j]);
    return z;
}

int main(int argc,char**argv){
    size_t rows=argc>1?strtoull(argv[1],nullptr,10):16384;
    size_t cols=argc>2?strtoull(argv[2],nullptr,10):512;
    int reps=argc>3?atoi(argv[3]):20;
    printf("=== 掩码三值内核 vs 参考  行=%zu 列=%zu ===\n",rows,cols);
    if(cols%8){printf("列必须是 8 的倍数\n");return 1;}
    const size_t rb=cols/8;
    std::vector<int8_t> Q(rows*cols);
    std::vector<uint8_t> P(rows*rb),M(rows*rb);
    std::vector<float> X(cols);
    uint32_t st=7;auto rnd=[&](){st=st*1103515245u+12345u;return (st>>16)&0x7fff;};
    for(auto&v:X)v=float(int(rnd()%2001)-1000)*0.01f;      // 有正有负的浮点激活
    std::vector<float> scale(rows);
    for(size_t r=0;r<rows;++r){
        scale[r]=0.01f+float(r%17)*0.001f;
        for(size_t j=0;j<cols;++j){
            int t=int(rnd()%3)-1;                            // -1/0/+1
            Q[r*cols+j]=int8_t(t);
            if(t>0)P[r*rb+(j>>3)]|=uint8_t(1u<<(j&7));
            else if(t<0)M[r*rb+(j>>3)]|=uint8_t(1u<<(j&7));
        }
    }
    // --- 逐位相等验证 ---
    size_t diff=0;double maxrel=0;
    for(size_t r=0;r<rows;++r){
        float a=dot_avx2_float(Q.data()+r*cols,X.data(),cols,scale[r]);
        float b=dot_mask(P.data()+r*rb,M.data()+r*rb,X.data(),cols,scale[r]);
        if(memcmp(&a,&b,sizeof(float))!=0){
            ++diff;double d=std::fabs(double(a)-double(b))/std::max(1e-30,std::fabs(double(a)));
            if(d>maxrel)maxrel=d;
        }
    }
    printf("  逐位相等检查: %zu / %zu 行不同",diff,rows);
    if(diff)printf("   最大相对差 %.3e",maxrel);
    printf("\n\n");

    auto bench=[&](const char*name,auto fn){
        double sink=0;for(size_t r=0;r<rows;++r)sink+=fn(r);
        auto t0=std::chrono::steady_clock::now();
        double acc=0;
        for(int k=0;k<reps;++k)for(size_t r=0;r<rows;++r)acc+=fn(r);
        auto t1=std::chrono::steady_clock::now();
        double sec=std::chrono::duration<double>(t1-t0).count();
        double mac=double(rows)*cols*reps;
        printf("  %-20s %8.3f ms/pass   %7.2f G MAC/s   (校验 %.6f)\n",
               name,sec*1000/reps,mac/sec/1e9,acc);
        return mac/sec;
    };
    double a=bench("参考 AVX2 float",[&](size_t r){return dot_avx2_float(Q.data()+r*cols,X.data(),cols,scale[r]);});
    double b=bench("路线B 掩码",[&](size_t r){return float(dot_mask(P.data()+r*rb,M.data()+r*rb,X.data(),cols,scale[r]));});

    // --- 路线 A 的速度与数值误差 ---
    {
        std::vector<uint8_t> A(rows*cols);std::vector<int32_t> rowsum(rows,0);
        std::vector<float> xsc(rows);
        for(size_t r=0;r<rows;++r){
            float mx=0;for(size_t j=0;j<cols;++j)mx=std::max(mx,std::fabs(X[j]));
            float s=mx>0?mx/127.0f:1.0f;xsc[r]=s;
            int32_t rs=0;
            for(size_t j=0;j<cols;++j){
                int a=int(std::lround(X[j]/s));if(a>127)a=127;if(a<-127)a=-127;
                A[r*cols+j]=uint8_t(a+128);
                rs+=Q[r*cols+j];
            }
            rowsum[r]=rs;
        }
        double maxrel=0,sumrel=0;size_t worst=0;int argmax_ref=-1,argmax_a=-1;float best=-1e30f;
        std::vector<double> errs;
        for(size_t r=0;r<rows;++r){
            float ref=dot_avx2_float(Q.data()+r*cols,X.data(),cols,scale[r]);
            int32_t di=dot_vnni_exact(Q.data()+r*cols,A.data()+r*cols,cols);
            float got=float(double(di-128*rowsum[r])*double(xsc[r])*double(scale[r]));
            double d=std::fabs(double(ref)-double(got))/std::max(1e-30,std::fabs(double(ref)));
            errs.push_back(d);
            if(d>maxrel){maxrel=d;worst=r;}
            sumrel+=d;
        }
        std::sort(errs.begin(),errs.end());
        double sink=0;auto t0=std::chrono::steady_clock::now();
        for(int k=0;k<reps;++k)for(size_t r=0;r<rows;++r)sink+=dot_vnni_exact(Q.data()+r*cols,A.data()+r*cols,cols);
        auto t1=std::chrono::steady_clock::now();
        double sec=std::chrono::duration<double>(t1-t0).count();
        double mac=double(rows)*cols*reps;
        printf("\n=== 路线 A（int8 量化激活 + VNNI，零点修正）===\n");
        printf("  速度 %.3f ms/pass  %.2f G MAC/s  (校验 %.0f)\n",sec*1000/reps,mac/sec/1e9,sink);
        printf("  相对误差: 中位 %.3e  90分位 %.3e  最大 %.3e (行 %zu)\n",
               errs[errs.size()/2],errs[errs.size()*9/10],maxrel,worst);
        printf("  (参考内核 %.2f G MAC/s 见上)\n",a);
    }

    printf("\n  路线B/参考 = %.2f×\n",b/a);
    printf("  每行字节: 参考 %zu  路线B %zu  (省 %.1f×)\n",cols,2*rb,double(cols)/(2*rb));
    return 0;
}
