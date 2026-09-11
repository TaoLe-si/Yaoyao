// bench_v1 宿主：读取 scripts/gen_bench.mjs 生成的 spec.tsv，逐条在全新 session 里跑完所有话轮，
// 按类别统计成功率并给出 Wilson 95% 置信区间。
// 目的：把"绑定/复制"这类能力终点的测量分辨率从 n=18 提到 n=100~200，
//       使"改进是否真实"可判定 —— n=18 时 10/18 vs 13/18 的 Fisher p 高达 0.49，测不出任何东西。
// 用法: diag_bench <model.dsb> [spec.tsv] [--limit N]
#define NOMINMAX
#define TAO_INPUT_SCALE
#define TAO_CPU_AVX2
#include "tokenizer_file.hpp"
#include "greedy_resident_lifecycle.hpp"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
using namespace tao::dual;

struct Item{std::string cat,id,mode;std::vector<std::string> expect;std::vector<std::string> turns;};

static std::vector<std::string> split(const std::string&s,char d){
    std::vector<std::string> o;std::string cur;
    for(char c:s){if(c==d){o.push_back(cur);cur.clear();}else cur+=c;}
    o.push_back(cur);return o;
}
// Wilson 95% 区间
static void wilson(int k,int n,double&lo,double&hi){
    if(n<=0){lo=hi=0;return;}
    const double z=1.959964,p=double(k)/n,z2=z*z;
    const double den=1.0+z2/n, ctr=p+z2/(2.0*n), rad=z*std::sqrt(p*(1.0-p)/n+z2/(4.0*double(n)*n));
    lo=ctr-rad; hi=ctr+rad;
    if(lo<0)lo=0; if(hi>1)hi=1;
}

int main(int argc,char**argv){
    if(argc<2){printf("usage: diag_bench <model.dsb> [spec.tsv] [--limit N]\n");return 2;}
    const char* path=argv[1];
    const char* spec=argc>2&&argv[2][0]!='-'?argv[2]:"data/bench_v1/spec.tsv";
    int limit=0;
    for(int i=2;i<argc;++i) if(std::string(argv[i])=="--limit"&&i+1<argc) limit=std::atoi(argv[i+1]);

    std::ifstream in(spec);
    if(!in){printf("BENCH_ERROR cannot open %s\n",spec);return 2;}
    std::vector<Item> items;
    std::string line;
    while(std::getline(in,line)){
        if(line.empty()||line[0]=='#')continue;
        auto f=split(line,'\t');
        if(f.size()<5)continue;
        Item it;it.cat=f[0];it.id=f[1];it.mode=f[2];
        it.expect=split(f[3],';');
        for(size_t i=4;i<f.size();++i) it.turns.push_back(f[i]);
        items.push_back(it);
        if(limit&&int(items.size())>=limit)break;
    }
    if(items.empty()){printf("BENCH_ERROR no items\n");return 2;}

    std::string hash;
    auto tok=tao::text::load_tokenizer("build/formal_tokenizer.bbp",hash);
    GreedyPipelineGroupedModel model(read_compact_bundle(path,hash));
    model.set_cpu_threads(4);
    auto enc=[&](const std::string&s){return tok.encode(s);};
    auto show=[&](const std::vector<uint32_t>&g){std::string o;for(auto t:g){try{o+=tok.decode({t});}catch(...){o+="?";}}return o;};

    printf("BENCH_START model=%s spec=%s items=%zu\n",path,spec,items.size());
    std::map<std::string,int> tot,ok;
    std::vector<std::string> fails;
    for(auto&it:items){
        GreedyResidentSession s(model);
        std::string a;
        for(size_t t=0;t<it.turns.size();++t) a=show(s.reply(enc(it.turns[t])).ids);
        bool good;
        if(it.mode=="order"){
            size_t pos=0;good=true;
            for(auto&e:it.expect){auto p=a.find(e,pos);if(p==std::string::npos){good=false;break;}pos=p+e.size();}
        }else{
            good=true;for(auto&e:it.expect) if(a.find(e)==std::string::npos){good=false;break;}
        }
        tot[it.cat]++; if(good)ok[it.cat]++;
        if(!good&&fails.size()<40){
            std::string exp;for(size_t e=0;e<it.expect.size();++e){if(e)exp+=";";exp+=it.expect[e];}
            fails.push_back(it.cat+" "+it.id+" 期望="+exp+" 得到="+a);
        }
    }

    printf("\n类别            成功/总数    成功率    Wilson95%%CI\n");
    int GT=0,GO=0;
    for(auto&kv:tot){
        int n=kv.second,k=ok[kv.first];double lo,hi;wilson(k,n,lo,hi);
        printf("%-15s %4d/%-4d   %6.2f%%   [%5.2f%%, %5.2f%%]\n",kv.first.c_str(),k,n,100.0*k/n,100*lo,100*hi);
        GT+=n;GO+=k;
    }
    {double lo,hi;wilson(GO,GT,lo,hi);
     printf("%-15s %4d/%-4d   %6.2f%%   [%5.2f%%, %5.2f%%]\n","总计",GO,GT,100.0*GO/GT,100*lo,100*hi);}
    printf("BENCH_RESULT model=%s total=%d ok=%d\n",path,GT,GO);
    if(!fails.empty()){
        printf("\n失败样例:\n");
        for(auto&f:fails)printf("  %s\n",f.c_str());
    }
    return 0;
}
