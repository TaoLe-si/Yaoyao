// Query-only Windows CPU topology probe. No model, CUDA, affinity setters,
// priority setters, worker threads, or benchmark. Build from x64 Native Tools:
// cl /nologo /std:c++17 /EHsc /W4 probe_cpu_topology_windows.cpp /Fe:probe_cpu_topology_windows.exe
#ifndef _WIN32
#error This probe requires 64-bit Windows.
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static_assert(sizeof(KAFFINITY) == 8, "Build x64: avoid WOW64 topology folding");
using Record = SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX;
struct Core { unsigned id; BYTE efficiency; std::vector<GROUP_AFFINITY> masks; };
struct Cache { BYTE level; DWORD bytes; GROUP_AFFINITY mask; };
static unsigned bits(KAFFINITY m) { unsigned n=0; for(;m;m&=m-1) ++n; return n; }
static KAFFINITY first_bit(KAFFINITY m) { return m & (~m + 1); }
static unsigned bit_index(KAFFINITY m) { unsigned n=0; while((m & 1)==0) {m>>=1; ++n;} return n; }
static std::runtime_error win_error(const char* api) {
    return std::runtime_error(std::string(api)+" failed: Win32="+std::to_string(GetLastError()));
}
static std::vector<unsigned char> query(LOGICAL_PROCESSOR_RELATIONSHIP relation) {
    DWORD size=0;
    if(GetLogicalProcessorInformationEx(relation,nullptr,&size) ||
       GetLastError()!=ERROR_INSUFFICIENT_BUFFER) throw win_error("GLPIEx size");
    for(unsigned attempt=0;attempt<4;++attempt) {
        std::vector<unsigned char> data(size);
        DWORD used=size;
        if(GetLogicalProcessorInformationEx(relation,reinterpret_cast<Record*>(data.data()),&used)) {
            data.resize(used); return data;
        }
        if(GetLastError()!=ERROR_INSUFFICIENT_BUFFER) throw win_error("GLPIEx query");
        size=used;
    }
    throw std::runtime_error("Topology changed repeatedly; retry later");
}
template<class F> static void visit(const std::vector<unsigned char>& data,F fn) {
    size_t offset=0;
    constexpr size_t header=offsetof(Record,Processor);
    while(offset<data.size()) {
        if(data.size()-offset<header) throw std::runtime_error("Truncated topology header");
        const auto* r=reinterpret_cast<const Record*>(data.data()+offset);
        if(r->Size<header || r->Size>data.size()-offset) throw std::runtime_error("Invalid topology record size");
        fn(*r); offset+=r->Size;
    }
}
static void print_mask(const GROUP_AFFINITY& a) {
    std::printf(" group=%u mask=0x%016llx LPs=",unsigned(a.Group),static_cast<unsigned long long>(a.Mask));
    for(unsigned b=0;b<64;++b) if(a.Mask & (KAFFINITY(1)<<b)) std::printf("%u,",b);
}
int main() try {
    std::puts("READ ONLY: topology and this probe's placement only; no scheduling changes.");
    const WORD groups=GetActiveProcessorGroupCount();
    if(!groups) throw win_error("GetActiveProcessorGroupCount");
    std::printf("active_groups=%u active_LPs=%lu\n",unsigned(groups),GetActiveProcessorCount(ALL_PROCESSOR_GROUPS));
    std::vector<Core> cores;
    unsigned total=0;
    visit(query(RelationProcessorCore),[&](const Record& r) {
        constexpr size_t base=offsetof(Record,Processor)+offsetof(PROCESSOR_RELATIONSHIP,GroupMask);
        if(r.Relationship!=RelationProcessorCore || r.Size<base) throw std::runtime_error("Invalid core record");
        const auto& p=r.Processor;
        if(!p.GroupCount || r.Size<base+size_t(p.GroupCount)*sizeof(GROUP_AFFINITY)) throw std::runtime_error("Invalid core masks");
        Core c{static_cast<unsigned>(cores.size()),p.EfficiencyClass,{}};
        std::printf("core=%u SMT_flag=%u efficiency_class=%u",c.id,unsigned((p.Flags & LTP_PC_SMT)!=0),unsigned(c.efficiency));
        for(unsigned i=0;i<p.GroupCount;++i) { c.masks.push_back(p.GroupMask[i]); print_mask(p.GroupMask[i]); total+=bits(p.GroupMask[i].Mask); }
        std::puts(""); cores.push_back(c);
    });
    std::printf("physical_cores=%zu core_logical_processors=%u\n",cores.size(),total);
    std::vector<Cache> caches;
    visit(query(RelationCache),[&](const Record& r) {
        constexpr size_t need=offsetof(Record,Cache)+offsetof(CACHE_RELATIONSHIP,GroupMask)+sizeof(GROUP_AFFINITY);
        if(r.Relationship!=RelationCache || r.Size<need) throw std::runtime_error("Invalid cache record");
        const auto& c=r.Cache;
        if(c.Level<2) return;
        std::printf("cache L%u type=%u bytes=%lu",unsigned(c.Level),unsigned(c.Type),c.CacheSize);
        print_mask(c.GroupMask); std::puts("");
        // Single-group recommendation only; multi-group cache records are not used.
        if(groups==1 && c.Type==CacheUnified) caches.push_back({c.Level,c.CacheSize,c.GroupMask});
    });
    GROUP_AFFINITY thread{};
    if(!GetThreadGroupAffinity(GetCurrentThread(),&thread)) throw win_error("GetThreadGroupAffinity");
    std::printf("probe_thread_affinity:"); print_mask(thread); std::puts("");
    PROCESSOR_NUMBER current{}; GetCurrentProcessorNumberEx(&current);
    std::printf("probe_sample_location: group=%u LP=%u (not decoder placement)\n",unsigned(current.Group),unsigned(current.Number));
    DWORD_PTR allowed=0,system=0;
    if(!GetProcessAffinityMask(GetCurrentProcess(),&allowed,&system)) throw win_error("GetProcessAffinityMask");
    std::printf("probe_process_mask=0x%016llx system_mask=0x%016llx\n",static_cast<unsigned long long>(allowed),static_cast<unsigned long long>(system));
    if(groups!=1 || !allowed) {
        std::puts("NO AUTOMATIC PAIR: multi-group/ambiguous process eligibility; inspect topology manually."); return 0;
    }
    // Detect overlapping core records, rather than risk recommending SMT siblings.
    KAFFINITY seen=0;
    for(const auto& c:cores) for(const auto& m:c.masks) {
        if(m.Group!=0 || (seen & m.Mask)) throw std::runtime_error("Unexpected/overlapping core masks; no pair");
        seen|=m.Mask;
    }
    for(const auto& cache:caches) if(cache.level==3) {
        for(size_t i=0;i<cores.size();++i) for(size_t j=i+1;j<cores.size();++j) {
            if(cores[i].efficiency!=cores[j].efficiency) continue;
            KAFFINITY a=0,b=0;
            for(const auto& m:cores[i].masks) a|=m.Mask & allowed & cache.mask.Mask;
            for(const auto& m:cores[j].masks) b|=m.Mask & allowed & cache.mask.Mask;
            if(!a || !b) continue;
            a=first_bit(a); b=first_bit(b);
            std::printf("CANDIDATE ONLY: shared L3, distinct cores. caller core=%u group=0 LP=%u mask=0x%016llx; persistent worker core=%u group=0 LP=%u mask=0x%016llx\n",cores[i].id,bit_index(a),static_cast<unsigned long long>(a),cores[j].id,bit_index(b),static_cast<unsigned long long>(b));
            std::puts("Not a least-busy/preferred-core ranking. Recheck eligibility in the CPU benchmark process, including CPU Sets/job restrictions. This probe does not inspect other processes or change affinity.");
            return 0;
        }
    }
    std::puts("NO AUTOMATIC PAIR: no eligible same-class distinct-core pair sharing a reported L3.");
    return 0;
} catch(const std::exception& e) {
    std::fprintf(stderr,"topology probe: %s\n",e.what()); return 1;
}
