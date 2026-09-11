// 内存带宽实测 + 解码期实际带宽核算。
//
// 背景：§14 判定「输出头非带宽受限」是在 1371 tps 下测的（约 30 GB/s）。
// 现在吞吐已达 3209 tps，权重总量 21.76 MB => 约 70 GB/s。
// **绑定约束可能已经翻转。** 若确为带宽受限，则 2-bit 打包（逐位精确、不需重训）
// 重新成为有效杠杆 —— 它此前是在低吞吐（计算受限）时被否证的。
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <vector>
#include <cstring>
#include <thread>
#include <atomic>
#include <immintrin.h>

static double now_s(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}

// 单线程流式读取（每 64 字节取一个 cacheline，排除预取器过度帮助）
static double read_bw(const uint8_t* p,size_t bytes,int threads){
    std::atomic<size_t> total{0};
    std::vector<std::thread> ts;
    auto t0=std::chrono::steady_clock::now();
    for(int t=0;t<threads;++t)ts.emplace_back([&,t]{
        size_t chunk=bytes/threads, off=t*chunk;
        __m256i acc=_mm256_setzero_si256();
        for(size_t i=off;i<off+chunk;i+=32){
            acc=_mm256_add_epi8(acc,_mm256_load_si256((const __m256i*)(p+i)));
        }
        alignas(32)uint8_t l[32];_mm256_store_si256((__m256i*)l,acc);
        size_t s=0;for(int k=0;k<32;++k)s+=l[k];
        total+=s;
    });
    for(auto&x:ts)x.join();
    auto t1=std::chrono::steady_clock::now();
    double sec=std::chrono::duration<double>(t1-t0).count();
    if(total.load()==0xdeadbeefULL)printf("");
    return bytes/sec/1e9;
}

int main(){
    const size_t MB=256;
    std::vector<uint8_t> buf(MB*1024*1024);
    for(size_t i=0;i<buf.size();++i)buf[i]=(uint8_t)(i*2654435761u>>24);
    // 预热（把页映射进去）
    {volatile uint8_t s=0;for(size_t i=0;i<buf.size();i+=4096)s+=buf[i];(void)s;}

    printf("=== 流式读取带宽（块 %zu MB）===\n",MB);
    for(int t : {1,2,4,8})printf("  %d 线程: %6.2f GB/s\n",t,read_bw(buf.data(),buf.size(),t));

    printf("\n=== 解码各配置的实际权重带宽需求 ===\n");
    // 权重字节：R3+R4 = 2 Full + 6 Reuse。Reuse 不读 group<6>（4 个 512x512 + 2 个 512x128）
    // 完整 8 层三元权重 + 输出头
    const double layer_all = 8.0*(128.0*512+128*128+128*512+128*128      // s 分支 4 个
                                 +512.0*512*4+512*128*2                  // m 分支 6 个
                                 +512.0*128+512*512);                    // read 2 个
    const double head = 16384.0*512;
    const double mb=1e6;
    printf("  全 8 层三元权重 %.2f MB, 输出头 %.2f MB, 合计 %.2f MB\n",
           layer_all/mb,head/mb,(layer_all+head)/mb);
    struct C{const char*n;double mbytes;double tps;};
    C cs[]={{"R0 基线(全 8 层, 无复用)",(layer_all+head)/mb,990.0},
            {"R3+R4 (2 Full + 6 Reuse)",(2.0/8*layer_all + 6.0/8*(layer_all-8.0*(512.0*512*4+512*128*2))+head)/mb,3209.0}};
    // Reuse 层只付 s 分支 + read 分支
    double reuse_layer = 128.0*512+128*128+128*512+128*128 + 512.0*128+512*512;
    double full_layer  = layer_all/8.0;
    double r34 = (2*full_layer + 6*reuse_layer + head)/mb;
    cs[1].mbytes=r34;
    for(auto&c:cs)printf("  %-28s %6.2f MB/token x %6.0f tps = %6.2f GB/s\n",c.n,c.mbytes,c.tps,c.mbytes*c.tps/1000.0);
    printf("\n  R3+R4 构成: 2 Full %.2f MB + 6 Reuse %.2f MB + 输出头 %.2f MB\n",
           (2*full_layer)/mb,(6*reuse_layer)/mb,head/mb);
    printf("\n  若改为 2-bit 打包（字节 /4）:\n");
    printf("    R3+R4 权重 %.2f MB/token  =>  3209 tps 时 %6.2f GB/s\n",
           r34/4,(r34/4)*3209.0/1000.0);
    return 0;
}
