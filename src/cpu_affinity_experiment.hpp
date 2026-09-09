#pragma once
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <vector>
#include <stdexcept>
#include <string>
#include <cstdio>
#include "cpu_row_parallel_executor.hpp"
namespace tao::dual {
struct AffinitySkip:std::runtime_error {using std::runtime_error::runtime_error;};
inline void affinity_require(bool ok,const char* why){if(!ok)throw AffinitySkip(std::string(why)+" Win32="+std::to_string(GetLastError()));}
// Conservative policy: refuse any explicit CPU Sets selection (even an eligible one).
// Require both ID and mask query APIs; older Windows fails closed, not silently permissive.
inline void affinity_eligible(unsigned lp) {
 static_assert(sizeof(KAFFINITY)==8,"x64 required");
 affinity_require(lp<64 && GetActiveProcessorGroupCount()==1,"single group required");
 auto k=GetModuleHandleW(L"kernel32.dll");
 using IDs=BOOL(WINAPI*)(HANDLE,PULONG,ULONG,PULONG);
 using Masks=BOOL(WINAPI*)(HANDLE,PGROUP_AFFINITY,USHORT,PUSHORT);
 auto pids=reinterpret_cast<IDs>(GetProcAddress(k,"GetProcessDefaultCpuSets"));
 auto tids=reinterpret_cast<IDs>(GetProcAddress(k,"GetThreadSelectedCpuSets"));
 auto pmasks=reinterpret_cast<Masks>(GetProcAddress(k,"GetProcessDefaultCpuSetMasks"));
 auto tmasks=reinterpret_cast<Masks>(GetProcAddress(k,"GetThreadSelectedCpuSetMasks"));
 affinity_require(pids&&tids&&pmasks&&tmasks,"CPU Sets eligibility APIs unavailable");
 ULONG n=0; USHORT m=0;
 affinity_require(pids(GetCurrentProcess(),nullptr,0,&n)&&n==0,"explicit/unknown process CPU Sets");
 affinity_require(tids(GetCurrentThread(),nullptr,0,&n)&&n==0,"explicit/unknown thread CPU Sets");
 affinity_require(pmasks(GetCurrentProcess(),nullptr,0,&m)&&m==0,"explicit/unknown process CPU Set masks");
 affinity_require(tmasks(GetCurrentThread(),nullptr,0,&m)&&m==0,"explicit/unknown thread CPU Set masks");
 using Info=BOOL(WINAPI*)(PSYSTEM_CPU_SET_INFORMATION,ULONG,PULONG,HANDLE,ULONG);
 auto info=reinterpret_cast<Info>(GetProcAddress(k,"GetSystemCpuSetInformation"));
 affinity_require(info!=nullptr,"system CPU Sets API unavailable");
 ULONG bytes=0; BOOL ok=info(nullptr,0,&bytes,GetCurrentProcess(),0);
 affinity_require((ok||GetLastError()==ERROR_INSUFFICIENT_BUFFER)&&bytes>0,"CPU Sets size");
 std::vector<unsigned char> buf(bytes);
 affinity_require(info(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buf.data()),bytes,&bytes,GetCurrentProcess(),0)!=0,"CPU Sets query");
 bool found=false;
 for(size_t off=0;off<bytes;) {
  affinity_require(bytes-off>=sizeof(ULONG)*2,"CPU Sets header");
  auto p=reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buf.data()+off);
  affinity_require(p->Size>=sizeof(ULONG)*2&&p->Size<=bytes-off,"CPU Sets record");
  if(p->Type==CpuSetInformation) {
   affinity_require(p->Size>=sizeof(SYSTEM_CPU_SET_INFORMATION),"CPU Sets structure");
   if(p->CpuSet.Group==0&&p->CpuSet.LogicalProcessorIndex==lp) {
    affinity_require(!found,"duplicate CPU Set"); found=true;
    affinity_require(!p->CpuSet.Parked&&(!p->CpuSet.Allocated||p->CpuSet.AllocatedToTargetProcess),"target parked or allocated elsewhere");
   }
  } off+=p->Size;
 }
 affinity_require(found,"target missing from CPU Sets");
 DWORD_PTR process=0,system=0; GROUP_AFFINITY thread{};
 affinity_require(GetProcessAffinityMask(GetCurrentProcess(),&process,&system)!=0,"process affinity query");
 affinity_require(GetThreadGroupAffinity(GetCurrentThread(),&thread)!=0,"thread affinity query");
 KAFFINITY bit=KAFFINITY(1)<<lp;
 affinity_require(thread.Group==0&&(thread.Mask&bit)&&(process&bit)&&(system&bit),"target excluded by existing affinity");
}
inline void affinity_distinct(unsigned a,unsigned b) {
 affinity_require(a<64&&b<64&&a!=b,"invalid LP pair");
 DWORD bytes=0;
 GetLogicalProcessorInformationEx(RelationProcessorCore,nullptr,&bytes);
 affinity_require(GetLastError()==ERROR_INSUFFICIENT_BUFFER&&bytes>0,"core topology size");
 std::vector<unsigned char> buf(bytes);
 affinity_require(GetLogicalProcessorInformationEx(RelationProcessorCore,reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()),&bytes)!=0,"core topology query");
 int ca=-1,cb=-1,index=0;
 for(size_t off=0;off<bytes;++index) {
  affinity_require(bytes-off>=offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,Processor),"core header");
  auto p=reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data()+off);
  const size_t base=offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,Processor)+offsetof(PROCESSOR_RELATIONSHIP,GroupMask);
  affinity_require(p->Size>=base&&p->Size<=bytes-off,"core record");
  affinity_require(p->Relationship==RelationProcessorCore&&p->Processor.GroupCount==1&&p->Size>=base+sizeof(GROUP_AFFINITY),"single-group core record required");
  const auto mask=p->Processor.GroupMask[0];
  affinity_require(mask.Group==0,"unexpected core group");
  if(mask.Mask&(KAFFINITY(1)<<a)){affinity_require(ca<0,"duplicate caller core");ca=index;}
  if(mask.Mask&(KAFFINITY(1)<<b)){affinity_require(cb<0,"duplicate worker core");cb=index;}
  off+=p->Size;
 }
 affinity_require(ca>=0&&cb>=0&&ca!=cb,"pair must belong to distinct physical cores");
 std::printf("TOPOLOGY caller_LP=%u core=%d worker_LP=%u core=%d (not least-busy ranking)\n",a,ca,b,cb);
}
class AffinityThreadGuard {
 GROUP_AFFINITY old_{}; bool active_=false;
 public:
 void pin(unsigned lp){affinity_eligible(lp);GROUP_AFFINITY target{};target.Mask=KAFFINITY(1)<<lp;
 affinity_require(SetThreadGroupAffinity(GetCurrentThread(),&target,&old_)!=0,"thread pin failed");active_=true;
 GROUP_AFFINITY actual{};
 affinity_require(GetThreadGroupAffinity(GetCurrentThread(),&actual)&&actual.Group==0&&actual.Mask==target.Mask,"thread pin readback failed");}
 void restore(){if(active_){if(!SetThreadGroupAffinity(GetCurrentThread(),&old_,nullptr))throw std::runtime_error("FATAL thread affinity restore failed");active_=false;
 GROUP_AFFINITY actual{};if(!GetThreadGroupAffinity(GetCurrentThread(),&actual)||actual.Group!=old_.Group||actual.Mask!=old_.Mask)throw std::runtime_error("FATAL restore readback failed");}}
 ~AffinityThreadGuard(){if(active_)try{restore();}catch(...){std::fputs("FATAL affinity restoration failed\n",stderr);std::terminate();}}
};
// Original executor dispatch/arithmetic/threshold unchanged. Control messages use
// its synchronous borrowed-task handshake, outside timed regions.
class CpuAffinityExperimentExecutor {
 CpuRowParallelExecutor base_;
 AffinityThreadGuard caller_,worker_; // worker guard is operated only on worker
 DWORD owner_=0; bool enabled_=false;
 template<class F> void on_worker(F fn){base_.run(CpuRowParallelExecutor::minimum_elements,1,[&](size_t begin,size_t){if(begin!=0)fn();});}
 public:
 static constexpr size_t minimum_elements=CpuRowParallelExecutor::minimum_elements;
 void enable(unsigned a,unsigned b){if(enabled_)throw std::runtime_error("already pinned");owner_=GetCurrentThreadId();affinity_distinct(a,b);
 try{on_worker([&]{worker_.pin(b);});caller_.pin(a);enabled_=true;}
 catch(...){auto e=std::current_exception();on_worker([&]{worker_.restore();});caller_.restore();std::rethrow_exception(e);}}
 void disable(){if(!enabled_)return;if(owner_!=GetCurrentThreadId())throw std::runtime_error("affinity owner thread changed");
 on_worker([&]{worker_.restore();});caller_.restore();enabled_=false;}
 ~CpuAffinityExperimentExecutor(){try{disable();}catch(...){std::terminate();}}
 template<class F> void run(size_t rows,size_t cols,F fn){base_.run(rows,cols,fn);}
};
}
