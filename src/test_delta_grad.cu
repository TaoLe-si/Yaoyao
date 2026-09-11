// H2R 增量规则记忆：训练路径反向的有限差分校验。
// 用 3 个时间步让 M 真正跨步累积，从而同时校验 dM 在时间上的回传。
#include "dual_state_cpu.hpp"
#include "batch_train_graph.cuh"
#include <cstdio>
#include <cmath>
#include <map>
#include <random>
using namespace tao::dual;
int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    try{
        Config c;c.layers=1;c.d=16;c.s=8;c.m=16;c.e=16;c.vocab=272;c.dk=4;c.validate();
        const unsigned slots=2,steps=(argc>1?unsigned(atoi(argv[1])):3u);
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> u(-1.f,1.f);
        CpuModel cpu(c);
        std::map<std::string,bool> tern;
        for(auto&t:schema(c))tern[t.name]=t.ternary;
        for(auto&kv:cpu.w)
            for(size_t i=0;i<kv.second.size();++i)
                kv.second[i]=tern[kv.first]?(u(rng)>0?1.f:-1.f):u(rng)*0.6f;
        Tape up;std::map<std::string,Node> shared;
        for(auto&kv:cpu.w)shared[kv.first]=up.leaf(kv.second);
        auto mkdev=[&](const std::vector<uint32_t>&v){
            auto d=std::make_shared<Device>(v.size());
            check(cudaMemcpy(d->p,v.data(),v.size()*4,cudaMemcpyHostToDevice));return d;};
        std::vector<uint32_t> idx(size_t(steps)*slots),msk(size_t(steps)*slots,1),rst(size_t(steps)*slots,0);
        for(size_t t=0;t<steps;++t)for(unsigned s=0;s<slots;++s)idx[t*slots+s]=uint32_t(1+((t*7+s*3)%29));
        for(unsigned s=0;s<slots;++s)rst[s]=1;                       // 第一步走 restart 分支
        auto pidx=mkdev(idx),pmsk=mkdev(msk),prst=mkdev(rst);
        std::vector<float> coef(size_t(c.vocab)*slots);
        for(size_t i=0;i<coef.size();++i)coef[i]=u(rng);
        auto forward=[&](BatchTrainGraph&g){
            Node y;
            for(unsigned t=0;t<steps;++t)y=g.step_device(pidx,pmsk,prst,size_t(t)*slots);
            return y;};
        // 解析梯度
        BatchTrainGraph g(c,slots,shared);
        Node y=forward(g);
        check(cudaMemcpy(y->grad.p,coef.data(),coef.size()*4,cudaMemcpyHostToDevice));
        g.tape.backward();
        // 损失：固定系数与 logits 的内积
        auto evalLoss=[&]()->double{
            BatchTrainGraph gg(c,slots,shared);
            Node yy=forward(gg);
            Vec lg(size_t(c.vocab)*slots);
            check(cudaMemcpy(lg.data(),yy->value.p,lg.size()*4,cudaMemcpyDeviceToHost));
            double L=0;for(size_t i=0;i<lg.size();++i)L+=double(coef[i])*lg[i];
            return L;};
        printf("== H2R 训练路径梯度校验  layers=1 d=16 s=8 m=8 dk=%u slots=%u steps=%u ==\n",c.dk,slots,steps);
        const char* names[]={"embedding","layer.0.mem.key","layer.0.mem.query","layer.0.mem.value",
            "layer.0.mem.beta","layer.0.mem.beta.bias","layer.0.read.s","layer.0.read.norm",
            "layer.0.s.candidate.x","layer.0.s.gate.bias","layer.0.input.norm","final.norm","vocab.bias"};
        const float h=5e-3f;
        double worst=0;const char* worstName="";
        for(const char* nm:names){
            auto it=shared.find(nm);
            if(it==shared.end()){printf("缺失张量 %s\n",nm);continue;}
            Node w=it->second;size_t n=w->value.n;
            Vec host(n),ana(n);
            check(cudaMemcpy(host.data(),w->value.p,n*4,cudaMemcpyDeviceToHost));
            check(cudaMemcpy(ana.data(),w->grad.p,n*4,cudaMemcpyDeviceToHost));
            size_t big=0;double mag=0;
            for(size_t i=0;i<n;++i)if(std::fabs(ana[i])>mag){mag=std::fabs(ana[i]);big=i;}
            std::vector<size_t> probe{0,n/2,big};
            for(size_t i:probe){
                float orig=host[i];
                host[i]=orig+h;check(cudaMemcpy(w->value.p,host.data(),n*4,cudaMemcpyHostToDevice));
                double Lp=evalLoss();
                host[i]=orig-h;check(cudaMemcpy(w->value.p,host.data(),n*4,cudaMemcpyHostToDevice));
                double Lm=evalLoss();
                host[i]=orig;check(cudaMemcpy(w->value.p,host.data(),n*4,cudaMemcpyHostToDevice));
                double fd=(Lp-Lm)/(2*h);
                // 有限差分的噪声底 ≈ eps*|L|/h + h^2*|L'''|，绝对量级 ~2e-4，故用绝对容差判定。
                double err=std::fabs(fd-double(ana[i]));
                double tol=2e-3+0.05*std::fabs(fd);
                const bool ok=err<=tol;
                if(!ok&&err/tol>worst){worst=err/tol;worstName=nm;}
                printf("  %-26s idx=%-5zu 解析=%+.7e  有限差分=%+.7e  绝对误差=%.2e  容差=%.2e %s\n",
                       nm,i,double(ana[i]),fd,err,tol,ok?"OK":"超标");
            }
        }
        printf("最差超标倍数 = %.3f  (%s)   [<=1 表示全部落在容差内]\n",worst,worstName);
        printf(worst<=1.0?"结论: 通过 —— 反向化简与有限差分一致\n"
                          :"结论: 不一致 —— 反向有误\n");
        return worst<=1.0?0:1;
    }catch(const std::exception&e){printf("异常: %s\n",e.what());return 1;}
}
