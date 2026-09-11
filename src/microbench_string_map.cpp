// 量「字符串拼接 + std::map<std::string> 查找」的纯开销，判断是否值得改成预解析索引。
// 键名与真实模型一致，条目数按 layer.0..7 × 约 25 个张量名构造。
#include <cstdio>
#include <cstdint>
#include <string>
#include <map>
#include <vector>
#include <chrono>

struct Row { std::vector<int8_t> q; std::vector<float> scale; };

int main(){
    // 真实模型里每个层用到的张量名（来自 greedy_pipeline_grouped_model.hpp）
    const char* names[] = {
        "s.candidate.x","s.candidate.s","s.gate.x","s.gate.s",
        "s.candidate.bias","s.gate.bias",
        "m.candidate.x","m.candidate.s","m.candidate.m",
        "m.gate.x","m.gate.s","m.gate.m",
        "m.candidate.bias","m.gate.bias",
        "read.s","read.m","read.norm",
        "x.norm","final.norm","s.norm",
        "tok.embed","vocab.bias","embedding"
    };
    std::map<std::string,Row> packed;
    for(int l=0;l<8;++l)
        for(const char* n:names)
            packed["layer."+std::to_string(l)+"."+n] = Row{};
    printf("map 条目数 = %zu\n",packed.size());

    // 复现热路径：每 token 约 100 次「前缀拼接 + at」
    const char* suffix[] = {"s.candidate.x","s.gate.x","read.s","read.m","m.candidate.x","m.gate.x"};
    volatile size_t sink=0;
    const int per_token=100, tokens=20000;
    auto run=[&](int mode){
        auto t0=std::chrono::steady_clock::now();
        for(int k=0;k<tokens;++k){
            for(int i=0;i<per_token;++i){
                if(mode==0){
                    std::string key="layer."+std::to_string(i&7)+"."+suffix[i%6];
                    sink+=packed.find(key)->second.q.size();
                }else{
                    // 候选方案：预解析后的整数索引（模拟数组下标）
                    static Row dummy;
                    sink+=dummy.q.size();
                }
            }
        }
        auto t1=std::chrono::steady_clock::now();
        return std::chrono::duration<double>(t1-t0).count()/tokens*1e6;   // µs/token
    };
    double a=run(0), b=run(1);
    printf("  「拼接+map.at」 %8.3f µs/token\n",a);
    printf("  「整数索引」   %8.3f µs/token\n",b);
    printf("  ⇒ 差值 = %.3f µs/token\n",a-b);
    printf("\n  参照：当前 8 线程单 token 约 350 µs（2858 tps）\n");
    printf("  ⇒ 该开销占 %.2f%%\n",(a-b)/350*100);
    return 0;
}
