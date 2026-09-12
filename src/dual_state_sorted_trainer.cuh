#pragma once
#include "dual_projection_sorted.cuh"
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <new>
#include <vector>
#include <chrono>
#ifdef TAO_GPU_HEALTH
#include "gpu_health_reduce.cuh"
#include "gpu_adam_device.cuh"
#endif
namespace tao::dual {

inline bool trainer_env_flag(const char*name,bool fallback){
  const char*v=std::getenv(name);
  if(!v||!*v)return fallback;
  return !(v[0]=='0'&&v[1]=='\0');
}

// 【主机侧优化器状态必须页锁定】可分页(pageable)内存的 H2D/D2H 会被驱动先拷进
// 内部暂存区再 DMA，实测有效带宽只有 ~2.5 GB/s；而每步要搬 3.3 GB（3 状态 × 2 方向
// + project 的 master 上载），于是 ms_update 里几乎全是等这个搬运。改为
// cudaHostAlloc 页锁定后 DMA 直传，实测提升约 3~4 倍。内存仍归主机 RAM，不占显存。
template<class T> struct PinnedAlloc {
  using value_type=T;
  PinnedAlloc()=default;
  template<class U> PinnedAlloc(const PinnedAlloc<U>&){}
  T* allocate(std::size_t n){
    void*p=nullptr;
    if(cudaHostAlloc(&p,n*sizeof(T),cudaHostAllocDefault)!=cudaSuccess){
      cudaGetLastError();throw std::bad_alloc();}
    return static_cast<T*>(p);
  }
  void deallocate(T*p,std::size_t){if(p)cudaFreeHost(p);}
  template<class U> bool operator==(const PinnedAlloc<U>&)const{return true;}
  template<class U> bool operator!=(const PinnedAlloc<U>&)const{return false;}
};
using PinnedVec=std::vector<float,PinnedAlloc<float>>;

// SortedGpuTrainer —— 三值投影 + AdamW 训练器。
//
// 【权重驻内存·默认开启】权威的 master/moment/variance 常驻主机 RAM（本机 47 GB），
//   显存只保留「生效权重 + 梯度」和按张量复用的 staging 缓冲。
//   这是本项目既定架构：「GPU 只负责训练，CPU 负责解码，权重不放显存、放入内存」。
//   置 TAO_OPT_OFFLOAD=0 可退回「优化器状态全放显存」的对照口径。
//
// 【为什么用批量搬运而不是 cudaHostAlloc(Mapped)】实测（d=1024，5 步）：
//     · 前向权重走主机映射：15.36 s/步、132 tok/s、GPU 94.4%
//     · 前向权重走显存    ： 1.41 s/步、1439 tok/s、GPU 65.6%
//   主机映射让每个 GEMM 经 PCIe 随机读权重，慢 10.9 倍（GPU 利用率反而更高，
//   因为流处理器在等内存）。project_sorted 的三值投影是随机访问，同理会更慢。
//   因此：前向权重必须驻显存；优化器状态驻内存但用**批量 cudaMemcpy** 搬运
//   （顺序大块传输远快于内核逐元素走 PCIe）。
//   实测 d=3200 dk=400 sl=8 wd=8：批量搬运 4.04 s/步，直接映射 6.12 s/步。
//
// 【续训】save_state/load_state 落盘 master/moment/variance + steps，
//   使长训可中断续跑（旧版恒从零初始化，中断即全丢）。
struct SortedGpuTrainer {
TrainGraph graph;std::map<std::string,std::shared_ptr<Device>>master,moment,variance,scales;std::vector<TensorSpec>spec;unsigned steps=0;
bool offload_=false;
std::map<std::string,PinnedVec>master_h_,moment_h_,variance_h_;
std::shared_ptr<Device>stage_,stage_m_,stage_v_;

explicit SortedGpuTrainer(const CpuModel&cpu):graph(cpu),spec(schema(cpu.c)){
  offload_=trainer_env_flag("TAO_OPT_OFFLOAD",true);
  for(auto&t:spec)if(t.ternary)scales[t.name]=std::make_shared<Device>(t.rows);
  if(!offload_){
    for(auto&t:spec){
      master[t.name]=std::make_shared<Device>(cpu.w.at(t.name));
      moment[t.name]=std::make_shared<Device>(Vec(t.elements(),0.f));
      variance[t.name]=std::make_shared<Device>(Vec(t.elements(),0.f));
    }
  }else{
    // 显存只放「最大单张量」大小的复用缓冲，而不是全量优化器状态。
    size_t biggest=0;
    for(auto&t:spec)biggest=std::max(biggest,t.elements());
    stage_=std::make_shared<Device>(biggest);
    stage_m_=std::make_shared<Device>(biggest);
    stage_v_=std::make_shared<Device>(biggest);
    for(auto&t:spec){
      const auto&w=cpu.w.at(t.name);
      master_h_.emplace(t.name,PinnedVec(w.begin(),w.end()));
      moment_h_.emplace(t.name,PinnedVec(t.elements(),0.f));
      variance_h_.emplace(t.name,PinnedVec(t.elements(),0.f));
    }
  }
  project();
}

double last_project_ms=0.0;
void project(){
  const auto pt0=std::chrono::steady_clock::now();
#ifdef TAO_GPU_HEALTH
  GpuHealth health;
#endif
  for(auto&t:spec){
    const size_t n=t.elements();
    auto q=graph.w.at(t.name);
    if(!offload_){
      auto a=master.at(t.name);
#ifdef TAO_GPU_HEALTH
      health.add(*a);
#else
      for(float v:a->host())if(!std::isfinite(v))throw std::runtime_error("nonfinite master");
#endif
      if(t.ternary){project_sorted(*a,q->value,*scales.at(t.name),t.rows,t.cols);check(cudaGetLastError());
#ifdef TAO_GPU_HEALTH
        health.add(*scales.at(t.name),true);
#else
        for(float v:scales.at(t.name)->host())if(!std::isfinite(v)||v<=0)throw std::runtime_error("scale");
#endif
      }else check(cudaMemcpy(q->value.p,a->p,n*4,cudaMemcpyDeviceToDevice));
    }else{
      const auto&h=master_h_.at(t.name);
#ifndef TAO_GPU_HEALTH
      for(float v:h)if(!std::isfinite(v))throw std::runtime_error("nonfinite master");
#endif
      check(cudaMemcpy(stage_->p,h.data(),n*4,cudaMemcpyHostToDevice));
#ifdef TAO_GPU_HEALTH
      // GPU 侧校验 master：语义等价于主机扫描（有限），但训练步内零 CPU。
      health.add(*stage_);
#endif
      if(t.ternary){project_sorted(*stage_,q->value,*scales.at(t.name),t.rows,t.cols);check(cudaGetLastError());
#ifdef TAO_GPU_HEALTH
        health.add(*scales.at(t.name),true);
#else
        for(float v:scales.at(t.name)->host())if(!std::isfinite(v)||v<=0)throw std::runtime_error("scale");
#endif
      }else check(cudaMemcpy(q->value.p,stage_->p,n*4,cudaMemcpyDeviceToDevice));
    }
  }
#ifdef TAO_GPU_HEALTH
  health.finish();
#endif
  last_project_ms=std::chrono::duration<double>(std::chrono::steady_clock::now()-pt0).count()*1e3;
}

// Discard temporal graph while preserving state values and accumulated parameter gradients.
void detach(){auto copy=[](Node old){auto n=std::make_shared<GradNode>(old->value.n);check(cudaMemcpy(n->value.p,old->value.p,old->value.n*4,cudaMemcpyDeviceToDevice));return n;};for(auto&n:graph.s)n=copy(n);for(auto&n:graph.m)n=copy(n);graph.tape.reverse.clear();}

void zero_grad(){for(auto&kv:graph.w)check(cudaMemset(kv.second->grad.p,0,kv.second->grad.n*4));}

float update(size_t supervised,float lr=.0003f){
  if(!supervised)return 0;
  double norm=0;
#ifdef TAO_GPU_HEALTH
  GpuHealth health;for(auto&t:spec)health.add(graph.w.at(t.name)->grad,false,double(supervised));
#else
  for(auto&t:spec)for(float g:graph.w.at(t.name)->grad.host()){if(!std::isfinite(g))throw std::runtime_error("gradient");double z=double(g)/supervised;norm+=z*z;}
  norm=std::sqrt(norm);
#endif
  const float factor=float(1./supervised/std::max(1.,norm));
  ++steps;
  const float bc1=1-std::pow(.9f,float(steps)),bc2=1-std::pow(.999f,float(steps));
  for(auto&t:spec){
    const size_t n=t.elements();
    const unsigned blocks=unsigned((n+127)/128);
    auto g=graph.w.at(t.name)->grad.p;
    if(!offload_){
      auto a=master.at(t.name);
#ifdef TAO_GPU_HEALTH
      ds_adamw_device<<<blocks,128>>>(a->p,g,moment.at(t.name)->p,variance.at(t.name)->p,int(n),health.sum(),health.bad(),double(supervised),lr,t.ternary?.01f:0.f,bc1,bc2);
#else
      ds_adamw<<<blocks,128>>>(a->p,g,moment.at(t.name)->p,variance.at(t.name)->p,int(n),factor,lr,t.ternary?.01f:0.f,bc1,bc2);
#endif
    }else{
      // 优化器状态驻内存：批量上载 -> GPU 上原地 AdamW -> 批量回写。
      // 顺序大块传输，避免内核逐元素走 PCIe（实测快 1.5 倍）。
      auto&mh=master_h_.at(t.name);auto&mo=moment_h_.at(t.name);auto&va=variance_h_.at(t.name);
      check(cudaMemcpy(stage_->p,mh.data(),n*4,cudaMemcpyHostToDevice));
      check(cudaMemcpy(stage_m_->p,mo.data(),n*4,cudaMemcpyHostToDevice));
      check(cudaMemcpy(stage_v_->p,va.data(),n*4,cudaMemcpyHostToDevice));
#ifdef TAO_GPU_HEALTH
      ds_adamw_device<<<blocks,128>>>(stage_->p,g,stage_m_->p,stage_v_->p,int(n),health.sum(),health.bad(),double(supervised),lr,t.ternary?.01f:0.f,bc1,bc2);
#else
      ds_adamw<<<blocks,128>>>(stage_->p,g,stage_m_->p,stage_v_->p,int(n),factor,lr,t.ternary?.01f:0.f,bc1,bc2);
#endif
      check(cudaMemcpy(mh.data(),stage_->p,n*4,cudaMemcpyDeviceToHost));
      check(cudaMemcpy(mo.data(),stage_m_->p,n*4,cudaMemcpyDeviceToHost));
      check(cudaMemcpy(va.data(),stage_v_->p,n*4,cudaMemcpyDeviceToHost));
    }
    check(cudaGetLastError());
  }
#ifdef TAO_GPU_HEALTH
  norm=std::sqrt(health.finish());
#endif
  check(cudaDeviceSynchronize());
  project();zero_grad();
  return float(norm);
}

// ---- 优化器状态存档（续训用）----
// 格式: magic "TAOOPT02" | u32 steps | u32 spec_count
//       每个张量: u64 name_len, name, u64 elements, master[] f32, moment[] f32, variance[] f32
void save_state(const std::string&path)const{
  std::ofstream f(path,std::ios::binary|std::ios::trunc);
  if(!f)throw std::runtime_error("opt state open for write");
  f.write("TAOOPT02",8);
  const uint32_t sc=uint32_t(spec.size());
  f.write(reinterpret_cast<const char*>(&steps),4);
  f.write(reinterpret_cast<const char*>(&sc),4);
  for(const auto&t:spec){
    const uint64_t nl=uint64_t(t.name.size()),ne=uint64_t(t.elements());
    f.write(reinterpret_cast<const char*>(&nl),8);
    f.write(t.name.data(),std::streamsize(nl));
    f.write(reinterpret_cast<const char*>(&ne),8);
    if(!offload_){
      const Vec a=master.at(t.name)->host(),b=moment.at(t.name)->host(),c=variance.at(t.name)->host();
      f.write(reinterpret_cast<const char*>(a.data()),std::streamsize(ne*4));
      f.write(reinterpret_cast<const char*>(b.data()),std::streamsize(ne*4));
      f.write(reinterpret_cast<const char*>(c.data()),std::streamsize(ne*4));
    }else{
      const auto&A=master_h_.at(t.name);const auto&B=moment_h_.at(t.name);const auto&C=variance_h_.at(t.name);
      f.write(reinterpret_cast<const char*>(A.data()),std::streamsize(ne*4));
      f.write(reinterpret_cast<const char*>(B.data()),std::streamsize(ne*4));
      f.write(reinterpret_cast<const char*>(C.data()),std::streamsize(ne*4));
    }
  }
  if(!f)throw std::runtime_error("opt state write failed");
}

// 返回是否成功恢复；张量名/尺寸不匹配即抛错（绝不静默错配）。
bool load_state(const std::string&path){
  std::ifstream f(path,std::ios::binary);
  if(!f)return false;
  char magic[8];f.read(magic,8);
  if(std::memcmp(magic,"TAOOPT02",8)!=0)throw std::runtime_error("opt state magic mismatch");
  uint32_t sc=0;
  f.read(reinterpret_cast<char*>(&steps),4);
  f.read(reinterpret_cast<char*>(&sc),4);
  if(sc!=uint32_t(spec.size()))throw std::runtime_error("opt state tensor count mismatch");
  for(const auto&t:spec){
    uint64_t nl=0,ne=0;
    f.read(reinterpret_cast<char*>(&nl),8);
    std::string name(size_t(nl),'\0');
    f.read(name.data(),std::streamsize(nl));
    f.read(reinterpret_cast<char*>(&ne),8);
    if(name!=t.name||ne!=uint64_t(t.elements()))throw std::runtime_error("opt state tensor mismatch: "+name);
    // 注意：不能写 Vec a(size_t(ne)) —— 那会被解析成函数声明（most vexing parse）。
    const size_t cnt=size_t(ne);
    PinnedVec a(cnt),b(cnt),c(cnt);
    f.read(reinterpret_cast<char*>(a.data()),std::streamsize(ne*4));
    f.read(reinterpret_cast<char*>(b.data()),std::streamsize(ne*4));
    f.read(reinterpret_cast<char*>(c.data()),std::streamsize(ne*4));
    if(!f)throw std::runtime_error("opt state truncated: "+name);
    if(!offload_){
      check(cudaMemcpy(master.at(t.name)->p,a.data(),ne*4,cudaMemcpyHostToDevice));
      check(cudaMemcpy(moment.at(t.name)->p,b.data(),ne*4,cudaMemcpyHostToDevice));
      check(cudaMemcpy(variance.at(t.name)->p,c.data(),ne*4,cudaMemcpyHostToDevice));
    }else{
      master_h_[t.name]=std::move(a);moment_h_[t.name]=std::move(b);variance_h_[t.name]=std::move(c);
    }
  }
  check(cudaDeviceSynchronize());
  project();
  return true;
}
};
}
