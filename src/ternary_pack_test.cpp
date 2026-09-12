// ternary_pack_test.cpp -- 2-bit 打包三值内核的独立校验。
// 用朴素参考实现对拍 dot / matvec_four_rows，并覆盖非 32 倍数列数的尾部路径。
//
// 注意：x 的生成必须写 float(int(rng()%2000)-1000)。若写成
// float((rng()%2000)-1000)，% 的结果是 unsigned，减 1000 会下溢到 4.29e9，
// 除以 250 得到 1.7e7 量级的"随机数"，于是绝对容差失效、内核会被误判为错。
#include "cpu_pipeline_rows.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
static int fails=0;
static void check(const char*what,size_t rows,size_t cols){
    std::mt19937 rng(12345u+unsigned(rows*1000+cols));
    std::vector<float> w(rows*cols);
    for(size_t i=0;i<rows;++i){
        const float a=(i%3+1)*0.5f;
        for(size_t j=0;j<cols;++j)w[i*cols+j]=float(int(rng()%3)-1)*a;
    }
    PipelineRows p(w,rows,cols);
    std::vector<float> x(cols);
    for(size_t j=0;j<cols;++j)x[j]=float(int(rng()%2000)-1000)/250.0f;
    std::vector<float> ref(rows,0),got(rows,0),mv(rows,0);
    for(size_t i=0;i<rows;++i){
        float z=0;for(size_t j=0;j<cols;++j)z+=(float(p.at(i,j))*p.scale[i])*x[j];
        ref[i]=z;got[i]=p.dot(i,x.data());
    }
    p.matvec_four_rows(x.data(),mv.data());
    // 相对容差：float32 逐项累加的舍入随 |ref| 增长。
    double worst1=0,worst2=0;
    for(size_t i=0;i<rows;++i){
        const double s=1.0+std::fabs(double(ref[i]));
        worst1=std::max(worst1,std::fabs(double(got[i]-ref[i]))/s);
        worst2=std::max(worst2,std::fabs(double(mv[i]-ref[i]))/s);
    }
    const bool ok=(worst1<1e-5)&&(worst2<1e-5);
    if(!ok)++fails;
    std::printf("[%s] %-10s rows=%2zu cols=%5zu  相对误差 dot=%.2e mv4=%.2e\n",
        ok?"PASS":"FAIL",what,rows,cols,worst1,worst2);
}
int main(){
    std::printf("stride(3200)=%zu 字节/行（int8 常驻需 3200）-> 常驻比 %.1f%%\n",
        CpuTernaryRows::stride_for(3200),100.0*CpuTernaryRows::stride_for(3200)/3200.0);
    check("31x31",31,31);      // 非 32 倍数，无完整块
    check("8x64",8,64);
    check("5x100",5,100);      // cols 非 32 倍数 -> 走尾部
    check("1x3200",1,3200);
    check("4x3200",4,3200);
    check("7x1600",7,1600);
    check("16x400",16,400);
    check("33x33",33,33);
    {   // 位序与打包密度
        std::vector<float> w(8,0); w[0]=1; w[3]=-1; w[7]=1;
        PipelineRows p(w,1,8);
        const bool ok=(p.q.size()==2)&&p.at(0,0)==1&&p.at(0,1)==0&&p.at(0,2)==0&&p.at(0,3)==-1&&p.at(0,7)==1;
        if(!ok)++fails;
        std::printf("[%s] 位序/密度 8 权重 -> %zu 字节\n",ok?"PASS":"FAIL",p.q.size());
    }
    std::printf(fails?"TERNARY_PACK_FAIL %d\n":"TERNARY_PACK_OK 全部通过\n",fails);
    return fails?1:0;
}
