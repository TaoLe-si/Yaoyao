// 加宽验证：dm_forward 的新二维网格 vs 宿主机参考实现，逐元素比对。
//
// 为什么需要这个探针：有限差分校验器对"前向错误"是盲的 —— FD 与解析梯度
// 共用同一份前向，前向即使全错，只要它是确定性的，FD 仍会与解析梯度吻合。
// 而 dm_forward 从 <<<slots,dv>>> 改成二维网格后，唯一的风险正是
// 「某些行没被任何线程写到」或「行号映射错位」，这类错误只能靠
// 与独立实现的逐元素对照来发现。
//
// 覆盖点：dv 取 512/1024（旧路径区）、1040/1280/2048（新路径区，含非 256 整数倍）。
#define NOMINMAX
#define TAO_DELTA_MEM
#define TAO_NO_FFN
#include "delta_mem_kernels.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <string>
using namespace tao::dual;

static float* dalloc(size_t n){ float*p=nullptr; check(cudaMalloc(&p,n*4)); return p; }

static int one(unsigned dv,unsigned dk,unsigned slots,unsigned seed){
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.f,1.f);
    const size_t nM=size_t(slots)*dv*dk, nK=size_t(slots)*dk, nV=size_t(slots)*dv;
    std::vector<float> hM(nM),hK(nK),hQ(nK),hV(nV),hB(slots);
    for(auto&x:hM)x=u(rng); for(auto&x:hK)x=u(rng); for(auto&x:hQ)x=u(rng);
    for(auto&x:hV)x=u(rng); for(auto&x:hB)x=0.1f+0.8f*std::fabs(u(rng));

    float*dM=dalloc(nM),*dK=dalloc(nK),*dQ=dalloc(nK),*dV=dalloc(nV),*dB=dalloc(slots);
    float*dMo=dalloc(nM),*dO=dalloc(nV),*dA=dalloc(nV);
    check(cudaMemcpy(dM,hM.data(),nM*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dK,hK.data(),nK*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dQ,hQ.data(),nK*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dV,hV.data(),nV*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dB,hB.data(),slots*4,cudaMemcpyHostToDevice));
    // 用哨兵预填充 Mout/o/a：若新网格漏写某行，哨兵会残留并被数值比对抓到。
    {   const float SEN=123456.0f;
        std::vector<float> f(nM,SEN); check(cudaMemcpy(dMo,f.data(),nM*4,cudaMemcpyHostToDevice));
        std::vector<float> g(nV,SEN); check(cudaMemcpy(dO,g.data(),nV*4,cudaMemcpyHostToDevice));
        check(cudaMemcpy(dA,g.data(),nV*4,cudaMemcpyHostToDevice)); }
    dm_forward<<<dm_grid(slots,dv),DM_THREADS>>>(dM,dK,dQ,dV,dB,dMo,dO,dA,dv,dk);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());

    std::vector<float> gMo(nM),gO(nV),gA(nV);
    check(cudaMemcpy(gMo.data(),dMo,nM*4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(gO.data(),dO,nV*4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(gA.data(),dA,nV*4,cudaMemcpyDeviceToHost));

    // ---- 宿主机参考实现（独立重写，不复用设备代码）----
    std::vector<float> eMo(nM),eO(nV),eA(nV);
    for(unsigned s=0;s<slots;++s)for(unsigned i=0;i<dv;++i){
        const size_t si=size_t(s)*dv+i;
        const float* src=&hM[si*dk];
        float acc=0; for(unsigned j=0;j<dk;++j)acc+=src[j]*hK[size_t(s)*dk+j];
        const float b=hB[s];
        const float g=b*(hV[si]-acc);
        float* dst=&eMo[si*dk];
        for(unsigned j=0;j<dk;++j)dst[j]=src[j]+g*hK[size_t(s)*dk+j];
        float rd=0; for(unsigned j=0;j<dk;++j)rd+=dst[j]*hQ[size_t(s)*dk+j];
        eO[si]=rd; eA[si]=acc;
    }
    auto cmp=[&](const char*tag,const std::vector<float>&a,const std::vector<float>&b)->size_t{
        double worst=0; size_t bad=0, sent=0;
        for(size_t i=0;i<a.size();++i){
            if(a[i]==123456.0f)++sent;                      // 哨兵残留 = 该元素未被写入
            const double d=std::fabs(double(a[i])-double(b[i]));
            if(d>worst)worst=d;
            if(d>1e-4*(1.0+std::fabs(double(b[i]))))++bad;
        }
        printf("    %-4s 最大绝对差=%.3e  超差元素=%zu/%zu  哨兵残留=%zu\n",tag,worst,bad,a.size(),sent);
        // 判据用「相对超差元素数 + 哨兵残留」而非绝对最大值：
        // o 要累加 dk 项，绝对误差随 dk 线性增长（dk=512 时达 1.5e-4），
        // 但相对量级仍在 1e-7 量级 —— 绝对的阈值会把规模效应误判为错误。
        return bad+sent;
    };
    printf("  dv=%-5u dk=%-4u slots=%u  grid=(%u,%u) block=%u\n",
           dv,dk,slots,(dv+DM_THREADS-1)/DM_THREADS,(dv+DM_THREADS-1)/DM_THREADS,DM_THREADS);
    const size_t w1=cmp("Mout",gMo,eMo), w2=cmp("o",gO,eO), w3=cmp("a",gA,eA);
    cudaFree(dM);cudaFree(dK);cudaFree(dQ);cudaFree(dV);cudaFree(dB);
    cudaFree(dMo);cudaFree(dO);cudaFree(dA);
    return (w1==0&&w2==0&&w3==0)?0:1;
}

int main(){
    setvbuf(stdout,NULL,_IONBF,0);
    printf("== 加宽验证：dm_forward 二维网格 vs 宿主机参考实现 ==\n");
    struct Case{unsigned dv,dk;};
    const Case cs[]={{512,64},{1024,128},{1040,128},{1152,144},{1280,160},{2048,256},{4096,512}};
    int bad=0;
    for(auto c:cs){ if(one(c.dv,c.dk,3,7+c.dv)){bad++; printf("    ^^ 该配置失败\n");} }
    printf(bad==0?"结论: 全部通过 —— 新网格与参考实现逐元素一致，无漏写\n"
                 :"结论: %d 个配置不一致\n",bad);
    return bad?1:0;
}
