// Engram 查表微基准（13 号文档 §六 第 1 步；对应 04 号文档 §二/§六）
// 目的：验证「预取不是优化项而是先决条件」这一判断，并定出表规模上限。
// 测三件事：
//   1) 有/无软件预取 的每次查表延迟（串行 vs 重叠）
//   2) 4 KB 普通页 vs 2 MB 大页（TLB 抖动）
//   3) 表规模 33.5 MB / 1.6 GB 的差异
// 用法: microbench_engram_lookup [orders] [heads] [dim_per_head] [table_rows] [lookups] [--hugepage]
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>
#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#endif

static inline uint64_t splitmix64(uint64_t&x){x+=0x9E3779B97F4A7C15ull;uint64_t z=x;
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ull;z=(z^(z>>27))*0x94D049BB133111EBull;return z^(z>>31);}

int main(int argc,char**argv){
    int orders=2,heads=4,dimPerHead=16;                 // 64 B/条目 => 一个 n-gram 一条 cache line
    uint64_t rows=1ull<<18;                              // 2^18 = 262144
    uint64_t lookups=2000000;
    bool huge=false, doPrefetch=true, doSerial=true;
    if(argc>1)orders=atoi(argv[1]);
    if(argc>2)heads=atoi(argv[2]);
    if(argc>3)dimPerHead=atoi(argv[3]);
    if(argc>4)rows=strtoull(argv[4],nullptr,10);
    if(argc>5)lookups=strtoull(argv[5],nullptr,10);
    for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--hugepage")huge=true;
        if(a=="--no-prefetch")doPrefetch=false;if(a=="--no-serial")doSerial=false;}

    const size_t entryBytes=size_t(heads)*dimPerHead;    // 每 n-gram 一条（E1：头间交错）
    const size_t totalBytes=size_t(orders)*rows*entryBytes;
    printf("=== Engram 查表微基准 ===\n");
    printf("阶数=%d 头数=%d 每头维=%d => 条目=%zu B\n",orders,heads,dimPerHead,entryBytes);
    printf("每表行数=2^%.0f (%llu)  表总大小=%.1f MB (%zu B)\n",
           log2((double)rows),rows,totalBytes/1048576.0,totalBytes);

    // 分配：普通页 or 2 MB 大页
    void* base=nullptr; bool gotHuge=false;
#ifdef _WIN32
    if(huge){
        SIZE_T want=totalBytes;
        base=VirtualAlloc(nullptr,want,MEM_RESERVE|MEM_COMMIT|MEM_LARGE_PAGES,PAGE_READWRITE);
        if(base){gotHuge=true;}
        else printf("[警告] 大页分配失败 (GetLastError=%lu) —— 需要 SeLockMemoryPrivilege\n",GetLastError());
    }
    if(!base) base=VirtualAlloc(nullptr,totalBytes,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE);
#else
    if(posix_memalign(&base,2<<20,totalBytes))return 1;
#endif
    if(!base){printf("分配失败\n");return 1;}
    printf("页类型=%s  实际=%s\n",huge?"请求 2MB":"请求 4KB",gotHuge?"已获 2MB 大页":"普通页");

    // 先触碰每一页，避免测量里混入缺页中断
    for(size_t off=0;off<totalBytes;off+=4096)((volatile char*)base)[off]=1;

    // 随机键（模拟不可预测的 n-gram 序列；真实场景键有局部性，此处是最坏情况）
    std::vector<uint64_t> keys(lookups);
    uint64_t st=12345; for(auto&k:keys)k=splitmix64(st)%rows;

    volatile uint64_t sink=0;
    auto bench=[&](const char*name,bool prefetch){
        auto t0=std::chrono::steady_clock::now();
        uint64_t acc=0;
        for(uint64_t i=0;i<lookups;++i){
            if(prefetch&&i+1<lookups){
                // 预取下一次查表：地址只依赖 token id => 可提前算出（04 号 §二）
                for(int o=0;o<orders;++o){
                    const char* p=(const char*)base+(size_t(o)*rows+keys[i+1])*entryBytes;
#ifdef _WIN32
                    _mm_prefetch(p,_MM_HINT_T0);
#else
                    __builtin_prefetch(p,0,3);
#endif
                }
            }
            // 本步真正读取：全部头（同一 cache line）
            for(int o=0;o<orders;++o){
                const uint8_t* p=(const uint8_t*)base+(size_t(o)*rows+keys[i])*entryBytes;
                for(int h=0;h<heads;++h) acc+=p[size_t(h)*dimPerHead];
            }
        }
        auto t1=std::chrono::steady_clock::now();
        sink+=acc;
        double ns=std::chrono::duration<double,std::nano>(t1-t0).count();
        printf("  %-22s %8.1f ns/token   %7.1f ns/次访问   上限 %6.0f token/s\n",
               name,ns/lookups,ns/(lookups*orders),1e9/(ns/lookups));
        return ns/lookups;
    };

    printf("\n--- 串行查表（每 token 依次访问 %d 个 n-gram 表）---\n",orders);
    double no=0,yes=0;
    if(doSerial){
        no=bench("无预取",false);
        if(doPrefetch)yes=bench("软件预取",true);
    }
    printf("\n--- 交错查表（2 个 token 的访问重叠，模拟 decode 流水）---\n");
    {
        auto t0=std::chrono::steady_clock::now();
        uint64_t acc=0; const uint64_t half=lookups/2;
        for(uint64_t i=0;i<half;++i){
            if(doPrefetch){
                for(int o=0;o<orders;++o){
                    const char* p=(const char*)base+(size_t(o)*rows+keys[half+i])*entryBytes;
#ifdef _WIN32
                    _mm_prefetch(p,_MM_HINT_T0);
#endif
                }
            }
            for(int o=0;o<orders;++o){
                const uint8_t* p=(const uint8_t*)base+(size_t(o)*rows+keys[i])*entryBytes;
                for(int h=0;h<heads;++h) acc+=p[size_t(h)*dimPerHead];
            }
        }
        auto t1=std::chrono::steady_clock::now();
        sink+=acc;
        double ns=std::chrono::duration<double,std::nano>(t1-t0).count();
        printf("  %-22s %8.1f ns/token   %7.1f ns/次访问   上限 %6.0f token/s\n",
               doPrefetch?"预取下一 token":"无预取",ns/half,ns/(half*orders),1e9/(ns/half));
    }
    if(doSerial&&doPrefetch&&no>0)
        printf("\n预取带来的改善: %.2f×  (04 号文档预测：串行无预取约 2.4 us/token 上限)\n",no/yes);
    printf("(sink=%llu)\n",(unsigned long long)sink);
#ifdef _WIN32
    VirtualFree(base,0,MEM_RELEASE);
#else
    free(base);
#endif
    return 0;
}
