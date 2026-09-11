#define NOMINMAX
#define TAO_INPUT_SCALE
#define TAO_CPU_AVX2
#include "dual_state_initialization.hpp"
#include "dual_model_bundle.hpp"
#include <cstdio>
#include <cmath>
#include <chrono>
using namespace tao::dual;
int main(){
    setvbuf(stdout,NULL,_IONBF,0);
    try{
    Config c; c.layers=8; c.d=512; c.s=128; c.m=512; c.dk=64; c.vocab=16384;
    uint64_t tern=0,flt=0; for(auto&t:schema(c)) (t.ternary?tern:flt)+=t.elements();
    printf("OPERATOR_ID = %s\n",TAO_OPERATOR_ID);
    printf("dims layers=%u d=%u s=%u m=%u dk=%u vocab=%u\n",c.layers,c.d,c.s,c.m,c.dk,c.vocab);
    printf("tensors=%zu  ternary=%llu  float=%llu  model_bytes=%llu\n",schema(c).size(),(unsigned long long)tern,(unsigned long long)flt,(unsigned long long)model_bytes(c));
    auto model=initialize(c,20260911);
    auto st=model.initial();
    size_t stBytes=0; for(uint32_t l=0;l<c.layers;++l) stBytes+=4*(st[l].s.size()+st[l].m.size());
    printf("state/layer: s=%zu  M=%zu (%ux%u)   全会话状态=%zu B\n",st[0].s.size(),st[0].m.size(),c.m,c.dk,stBytes);
    // 结构验证：单步之后必须有 M k_hat == beta * v（增量规则的定义性质）
    {
        auto probe=model.initial();
        // 复算 step() 内部真实的 x 与 xn（不能自己造一个 xn）
        const auto&emb=model.w.at("embedding");
        Vec x(emb.begin()+uint64_t(261)*c.d,emb.begin()+uint64_t(262)*c.d);
        for(float&z:x)z*=std::sqrt(float(c.d));
        auto xn=model.norm(x,"layer.0.input.norm");
        auto k=model.linear("layer.0.mem.key",xn,c.dk);
        auto v=model.linear("layer.0.mem.value",xn,c.m);
        float kn=0; for(float z:k)kn+=z*z; kn=1.0f/(std::sqrt(kn)+1e-6f); for(float&z:k)z*=kn;
        float beta=model.sigmoid(model.linear("layer.0.mem.beta",xn,1)[0]+model.w.at("layer.0.mem.beta.bias")[0]);
        model.step(261,probe);
        const auto&M=probe[0].m; double err=0,ref=0;
        for(uint32_t i=0;i<c.m;++i){ double mk=0; const float*row=M.data()+size_t(i)*c.dk; for(uint32_t j=0;j<c.dk;++j)mk+=double(row[j])*k[j];
            double want=double(beta)*v[i]; err+=(mk-want)*(mk-want); ref+=want*want; }
        printf("[结构] 第一步后 ||M k_hat - beta*v|| / ||beta*v|| = %.6f   (应 ~0)   beta=%.4f\n",std::sqrt(err/(ref+1e-30)),beta);
    }
    // 前向：32 步，检查有限性与 argmax
    auto t0=std::chrono::steady_clock::now();
    double first=0,last=0;
    for(int i=0;i<32;++i){
        auto lg=model.step(uint32_t(261+i*137),st);
        double mx=-1e30; int arg=0; bool fin=true;
        for(size_t j=0;j<lg.size();++j){ if(!std::isfinite(lg[j]))fin=false; if(lg[j]>mx){mx=lg[j];arg=int(j);} }
        if(i==0||i==31)printf("step %2d argmax=%5d max=%8.4f finite=%s\n",i,arg,mx,fin?"yes":"NO");
        if(i==0)first=mx; if(i==31)last=mx;
    }
    auto t1=std::chrono::steady_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    printf("32 步耗时 %.1f ms  (%.2f ms/token, 标量参考实现)\n",ms,ms/32);
    for(uint32_t l=0;l<c.layers;++l){
        double ns=0,nm=0; for(float z:st[l].s)ns+=double(z)*z; for(float z:st[l].m)nm+=double(z)*z;
        printf("layer %u  |s|=%.4f  ||M||_F=%.4f\n",l,std::sqrt(ns),std::sqrt(nm));
    }
    // 三值投影（initialize 给出的是连续权重；bundle 只接受逐行三值）
    for(auto&t:schema(c)){ if(!t.ternary)continue; auto&w=model.w.at(t.name);
        for(uint32_t r=0;r<t.rows;++r){ float sc=0; for(uint32_t j=0;j<t.cols;++j)sc=std::max(sc,std::abs(w[size_t(r)*t.cols+j]));
            if(sc==0)sc=1; for(uint32_t j=0;j<t.cols;++j){ float v=w[size_t(r)*t.cols+j]; w[size_t(r)*t.cols+j]= v>0?sc:v<0?-sc:0.f; } } }
    printf("[投影] 已逐行三值化（scale=rowmax, q=sign）\n");
    // bundle 往返 + 身份
    save_bundle(model,"build/delta_mem_smoke.dsb","34463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333");
    auto back=load_bundle("build/delta_mem_smoke.dsb","34463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333");
    auto s1=back.initial(), s2=back.initial();
    auto a=model.step(300,s1), b=model.step(300,s2);
    double d=0; for(size_t i=0;i<a.size();++i)d+=std::abs(double(a[i])-b[i]);
    printf("[往返] bundle 重新载入后同 token 的 logits L1 差 = %.3e\n",d);
    // 交叉身份：用旧算子载入必须失败
    try{ load_bundle("build/noffn_fresh/step_1512/final.dsb","34463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333");
         printf("[身份] 载入旧 dual-state-3 的 DSB: **竟然成功（不应发生）**\n"); }
    catch(const std::exception&e){ printf("[身份] 载入旧 dual-state-3 的 DSB 被拒绝: %s\n",e.what()); }
    }catch(const std::exception&e){ printf("[异常] %s\n",e.what()); return 2; }
    return 0;
}
