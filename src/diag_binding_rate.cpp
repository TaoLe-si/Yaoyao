#define NOMINMAX
#define TAO_INPUT_SCALE
#define TAO_CPU_AVX2
#include "tokenizer_file.hpp"
#include "greedy_resident_lifecycle.hpp"
#include <cstdio>
#include <string>
#include <vector>
using namespace tao::dual;
struct P{const char* n;const char* c;int nv;int cv;};
int main(int argc,char**argv){
    const char* path=argc>1?argv[1]:"build/noffn_probe2_run/step_113/final.dsb";
    std::string hash;
    auto tok=tao::text::load_tokenizer("build/formal_tokenizer.bbp",hash);
    GreedyPipelineGroupedModel model(read_compact_bundle(path,hash));
    model.set_cpu_threads(4);
    // nv/cv = 名字/颜色是否在训练语料词表内（1=在, 0=不在）
    // 关键项：语料内名字 × 语料内颜色 的「交叉组合」——语料里从未出现过该配对
    std::vector<P> tests={
        {"小王","紫色",1,1},{"小张","蓝色",1,1},{"小林","红色",1,1},{"阿明","黄色",1,1},
        {"李华","橙色",1,1},{"小陈","棕色",1,1},{"小刘","青色",1,1},{"小赵","绿色",1,1},
        {"小杨","白色",1,1},{"小吴","粉色",1,1},
        {"小何","紫色",0,1},{"小马","红色",0,1},{"小孙","蓝色",0,1},
        {"小王","银色",1,0},{"小张","金色",1,0},
        {"小何","银色",0,0},{"郭静","灰色",0,1},{"孙丽","绿色",0,1},
    };
    const std::string ASK="我叫什么名字？喜欢什么颜色？";
    auto enc=[&](const std::string&s){return tok.encode(s);};
    auto show=[&](const std::vector<uint32_t>&g){std::string o;for(auto t:g){try{o+=tok.decode({t});}catch(...){o+="?";}}return o;};
    printf("DB=%s\n",path);
    printf("inject\t| 回答\t| 名字抄对\t颜色抄对\t| 名字在词表 颜色在词表\n");
    int tot=0,nOK=0,cOK=0,both=0,bothTot=0;
    int gn=0,gc=0,gb=0,gbt=0;   // 语料内名字
    int un=0,uc=0,ub=0,ubt=0;   // 语料外名字
    for(auto&p:tests){
        std::string text=std::string("我的名字叫")+p.n+"，我喜欢"+p.c+"。请记住。";
        GreedyResidentSession s(model);
        s.reply(enc(text));
        auto r=s.reply(enc(ASK));
        std::string a=show(r.ids);
        bool nOk=a.find(p.n)!=std::string::npos, cOk=a.find(p.c)!=std::string::npos;
        printf("%s/%s\t| %s\t| %s\t%s\n",p.n,p.c,a.c_str(),nOk?"OK":"NO",cOk?"OK":"NO");
        tot++; nOK+=nOk; cOK+=cOk; if(nOk&&cOk)both++;
        if(p.nv){gbt++;gn+=nOk;gc+=cOk;if(nOk&&cOk)gb++;}
        else    {ubt++;un+=nOk;uc+=cOk;if(nOk&&cOk)ub++;}
    }
    printf("\n== 汇总 ==\n");
    printf("总计          : 名字 %d/%d   颜色 %d/%d   两者都对 %d/%d\n",nOK,tot,cOK,tot,both,tot);
    printf("名字在词表内  : 名字 %d/%d   颜色 %d/%d   两者都对 %d/%d\n",gn,gbt,gc,gbt,gb,gbt);
    printf("名字不在词表内: 名字 %d/%d   颜色 %d/%d   两者都对 %d/%d\n",un,ubt,uc,ubt,ub,ubt);
    return 0;
}
