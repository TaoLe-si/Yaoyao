#define NOMINMAX
#define TAO_INPUT_SCALE
#define TAO_CPU_AVX2
#include "tokenizer_file.hpp"
#include "greedy_resident_lifecycle.hpp"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
using namespace tao::dual;
static double dist(const Vec&a,const Vec&b){double s=0;for(size_t i=0;i<a.size();++i){double d=double(a[i])-double(b[i]);s+=d*d;}return std::sqrt(s);}
// 无混淆判定：固定名字只变颜色（或固定颜色只变名字），多措辞取多样本。
// LOO 最近质心能解出「变的那一维」= 信息确实进了状态 -> 瓶颈在读出。
int main(int argc,char**argv){
    const char* path=argc>1?argv[1]:"build/noffn_probe2_run/step_113/final.dsb";
    std::string hash;
    auto tok=tao::text::load_tokenizer("build/formal_tokenizer.bbp",hash);
    GreedyPipelineGroupedModel model(read_compact_bundle(path,hash));
    model.set_cpu_threads(4);
    std::vector<std::string> colors={"蓝色","红色","绿色","黄色","紫色","白色","黑色","橙色","粉色","灰色","棕色","青色"};
    std::vector<std::string> names ={"小林","小王","小张","阿明","李华","小陈","小刘","小杨","小赵","小周","小吴","小徐"};
    auto enc=[&](const std::string&s){return tok.encode(s);};
    // 4 种措辞，保证同类多样本
    auto variants=[&](const std::string&n,const std::string&c){
        return std::vector<std::string>{
            "我的名字叫"+n+"，我喜欢"+c+"。请记住。",
            "我叫"+n+"，我喜欢"+c+"，记住。",
            "请记住：我的名字叫"+n+"，我喜欢"+c+"。",
            "记住，我叫"+n+"，我喜欢"+c+"。",
        };
    };
    const uint32_t L=model.c.layers, NV=4;
    struct Res{};
    auto collect=[&](bool varyColor){
        size_t K=varyColor?colors.size():names.size();
        std::vector<std::vector<LayerState>> snap(K*NV);
        for(size_t k=0;k<K;++k)for(size_t v=0;v<NV;++v){
            std::string n=varyColor?"小林":names[k];
            std::string c=varyColor?colors[k]:"蓝色";
            auto vs=variants(n,c);
            GreedyResidentSession s(model);
            s.reply(enc(vs[v]));
            snap[k*NV+v]=s.state;
        }
        return snap;
    };
    auto report=[&](const char*label,std::vector<std::vector<LayerState>>&snap,size_t K){
        printf("\n=== %s  (类别数=%zu, 每类 %u 个措辞样本) ===\n",label,K,NV);
        printf("层\t同类平均距离\t异类平均距离\t比值\tLOO最近质心\t随机基线\n");
        for(uint32_t l=0;l<L;++l){
            double same=0,cross=0;long ns=0,nc=0;
            for(size_t a=0;a<snap.size();++a)for(size_t b=a+1;b<snap.size();++b){
                double d=dist(snap[a][l].m,snap[b][l].m);
                if(a/NV==b/NV){same+=d;++ns;}else{cross+=d;++nc;}
            }
            int ok=0;
            for(size_t i=0;i<snap.size();++i){
                double best=1e300;size_t bc=999;
                for(size_t c=0;c<K;++c){
                    Vec cent(snap[i][l].m.size(),0.f);int cnt=0;
                    for(size_t j=0;j<snap.size();++j){if(j==i||j/NV!=c)continue;for(size_t q=0;q<cent.size();++q)cent[q]+=snap[j][l].m[q];++cnt;}
                    if(!cnt)continue; for(auto&x:cent)x/=float(cnt);
                    double d=dist(snap[i][l].m,cent); if(d<best){best=d;bc=c;}
                }
                if(bc==i/NV)++ok;
            }
            printf("%u\t%.4f\t\t%.4f\t\t%.3f\t%d/%zu (%.0f%%)\t%.1f%%\n",l,same/ns,cross/nc,(same/ns)/(cross/nc),ok,snap.size(),100.0*ok/snap.size(),100.0/K);
        }
    };
    printf("DB=%s\n",path);
    auto sc=collect(true);  report("只变颜色（名字固定=小林）",sc,colors.size());
    auto sn=collect(false); report("只变名字（颜色固定=蓝色）",sn,names.size());
    printf("\n判读：LOO 明显高于基线 => 该维度信息确实存在于状态下（瓶颈在读出，不在写入）\n");
    return 0;
}
