#define NOMINMAX
#define TAO_INPUT_SCALE
#define TAO_CPU_AVX2
#include "tokenizer_file.hpp"
#include "greedy_resident_lifecycle.hpp"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
// 零训练诊断：上下文绑定失败 = 写不进状态，还是读不出状态？
// 全部走 GreedyResidentSession::reply，与会话评测完全同一语义。
using namespace tao::dual;
static double nrm(const Vec&v){double s=0;for(float x:v)s+=double(x)*x;return std::sqrt(s);}
static double dist(const Vec&a,const Vec&b){double s=0;for(size_t i=0;i<a.size();++i){double d=double(a[i])-double(b[i]);s+=d*d;}return std::sqrt(s);}
static double cosine(const Vec&a,const Vec&b){double d=0,na=0,nb=0;for(size_t i=0;i<a.size();++i){d+=double(a[i])*b[i];na+=double(a[i])*a[i];nb+=double(b[i])*b[i];}return d/(std::sqrt(na)*std::sqrt(nb)+1e-30);}
int main(int argc,char**argv){
    const char* path=argc>1?argv[1]:"build/noffn_probe2_run/step_113/final.dsb";
    std::string hash;
    auto tok=tao::text::load_tokenizer("build/formal_tokenizer.bbp",hash);
    GreedyPipelineGroupedModel model(read_compact_bundle(path,hash));
    model.set_cpu_threads(4);
    const std::string PAIR_A="我的名字叫小何，我喜欢银色。请记住。";
    const std::string PAIR_B="我的名字叫小张，我喜欢紫色。请记住。";
    const std::string CTRL ="今天天气不错，我们去公园散步吧。";
    const std::string ASK  ="我叫什么名字？喜欢什么颜色？";
    auto enc=[&](const std::string&s){return tok.encode(s);};
    auto show=[&](const std::vector<uint32_t>&g){std::string out;for(size_t i=0;i<g.size();++i){try{out+=tok.decode({g[i]});}catch(...){out+="<"+std::to_string(g[i])+">";}}return out;};
    printf("DB=%s\nA=%s\nB=%s\nCTRL=%s\nASK=%s\n",path,PAIR_A.c_str(),PAIR_B.c_str(),CTRL.c_str(),ASK.c_str());
    // ---- 写入检查：只喂 USER+文本+TURN_END，取状态 ----
    auto prime=[&](const std::string&text){
        std::vector<LayerState> st=model.initial();
        model.advance(256,st);model.advance(257,st);
        for(auto t:enc(text))model.advance(t,st);
        model.advance(259,st);
        return st;
    };
    auto pA=prime(PAIR_A),pB=prime(PAIR_B),pC=prime(CTRL);
    printf("\n[写入] 只看「USER+文本+TURN_END」后的状态\n");
    printf("layer\t|s|\tds(A,B)/|s|\tdm(A,B)/|m|\tds(A,C)/|s|\tdm(A,C)/|m|\tcos(m_A,m_B)\tcos(m_A,m_C)\n");
    for(uint32_t l=0;l<model.c.layers;++l){
        double ns=nrm(pA[l].s),nm=nrm(pA[l].m);
        printf("%u\t%.3f\t%.4f\t%.4f\t%.4f\t%.4f\t%.6f\t%.6f\n",l,ns,
          dist(pA[l].s,pB[l].s)/(ns+1e-30),dist(pA[l].m,pB[l].m)/(nm+1e-30),
          dist(pA[l].s,pC[l].s)/(ns+1e-30),dist(pA[l].m,pC[l].m)/(nm+1e-30),
          cosine(pA[l].m,pB[l].m),cosine(pA[l].m,pC[l].m));
    }
    // ---- 读出检查：完整走一轮问答，再问同一句话 ----
    auto session=[&](const std::string&pair){
        GreedyResidentSession s(model);
        auto r1=s.reply(enc(pair));
        std::vector<LayerState> after_turn=s.state;
        auto r2=s.reply(enc(ASK));
        return std::make_tuple(r1.ids,after_turn,r2.ids);
    };
    auto [r1A,stA,ansA]=session(PAIR_A);
    auto [r1B,stB,ansB]=session(PAIR_B);
    auto [r1C,stC,ansC]=session(CTRL);
    printf("\n[读出] 第一轮回复\n  A -> %s\n  B -> %s\n  C(无关) -> %s\n",
        show(r1A).c_str(),show(r1B).c_str(),show(r1C).c_str());
    printf("\n[读出] 同一问题「%s」的回答\n  A -> %s\n  B -> %s\n  C -> %s\n",
        ASK.c_str(),show(ansA).c_str(),show(ansB).c_str(),show(ansC).c_str());
    printf("\n[读出] 换事实值后，回答是否变化: A vs B = %s   (无上下文 C 作对照)\n",
        (ansA==ansB?"**完全相同**":"不同"));
    printf("[读出] 提问前状态差: dm(A,B)/|m| 末层=%.4f   ds(A,B)/|s| 末层=%.4f\n",
        dist(stA[model.c.layers-1].m,stB[model.c.layers-1].m)/(nrm(stA[model.c.layers-1].m)+1e-30),
        dist(stA[model.c.layers-1].s,stB[model.c.layers-1].s)/(nrm(stA[model.c.layers-1].s)+1e-30));
    printf("[读出] 回答 token ids A=[");for(auto t:ansA)printf("%u ",t);printf("] B=[");for(auto t:ansB)printf("%u ",t);printf("]\n");
    return 0;
}
