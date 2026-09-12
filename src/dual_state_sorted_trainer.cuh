#pragma once
#include "dual_projection_sorted.cuh"
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
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

// SortedGpuTrainer —— 三值投影 + AdamW 训练器。
//
// 【显存口径】默认每参数 20 B fp32：
//   master(4) + moment(4) + variance(4) + grad(4) + 生效权重 graph.w.value(4)
//   0.9B 参数 => ≈18 GB，8 GB 卡装不下。
//
// 【主存替代】TAO_OPT_OFFLOAD=1 时，master/moment/variance 常驻主机内存
//   （本机 47 GB），GPU 只保留 grad + 生效权重 = 8 B/参数 => 0.9B ≈ 7.2 GB。
//   用可复用的 staging 缓冲按张量搬运；数值语义与关闭时逐位一致
//   （同一 ds_adamw 内核、同一张量顺序、同一 default stream 定序）。
//
// 【续训】save_state/load_state 落盘 master/moment/variance + steps，
//   使长训可中断续跑（旧版恒从零初始化，中断即全丢）。
struct SortedGpuTrainer {
TrainGraph graph;std::map<std::string,std::shared_ptr<Device>>master,moment,variance,scales;std::vector<TensorSpec>spec;unsigned steps=0;
bool offload_=false;
std::map<std::string,Vec>master_h_,moment_h_,variance_h_;
std::shared_ptr<Device>stage_,stage_m_,stage_v_;

explicit SortedGpuTrainer(const CpuModel&cpu):graph(cpu),spec(schema(cpu.c)){
  offload_=trainer_env_flag("TAO_OPT_OFFLOAD",false);
  for(auto&t:spec)if(t.ternary)scales[t.name]=std::make_shared<Device>(t.rows);
  if(!offload_){
    for(auto&t:spec){
      master[t.name]=std::make_shared<Device>(cpu.w.at(t.name));
      moment[t.name]=std::make_shared<Device>(Vec(t.elements()));
      variance[t.name]=std::make_shared<Device>(Vec(t.elements()));
    }
  }else{
    size_t biggest=0;
    for(auto&t:spec)biggest=std::max(biggest,t.elements());
    stage_=std::make_shared<Device>(biggest);
    stage_m_=std::make_shared<Device>(biggest);
    stage_v_=std::make_shared<Device>(biggest);
    for(auto&t:spec){
      master_h_[t.name]=cpu.w.at(t.name);
      moment_h_[t.name]=Vec(t.elements(),0.f);
      variance_h_[t.name]=Vec(t.elements(),0.f);
    }
  }
  project();
}

void project(){
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
      if(t.ternary){project_sorted(*stage_,q->value,*scales.at(t.name),t.rows,t.cols);check(cudaGetLastError());
#ifndef TAO_GPU_HEALTH
        for(float v:scales.at(t.name)->host())if(!std::isfinite(v)||v<=0)throw std::runtime_error("scale");
#endif
      }else check(cudaMemcpy(q->value.p,stage_->p,n*4,cudaMemcpyDeviceToDevice));
    }
  }
#ifdef TAO_GPU_HEALTH
  health.finish();
#endif
}

// Discard temporal graph while preserving state values and accumulated parameter gradients.
void detach(){auto copy=[](Node old){auto n=std::make_shared<GradNode>(old->value.n);check(cudaMemcpy(n->value.p,old->value.p,old->value.n*4,cudaMemcpyDeviceToDevice));return n;};for(auto&n:graph.s)n=copy(n);for(auto&n:graph.m)n=copy(n);graph.tape.reverse.clear();}

void zero_grad(){for(auto&kv:graph.w)check(cudaMemset(kv.second->grad.p,0,kv.second->grad.n*4));}

float update(size_t supervised,float lr=.0003f,float max_norm=0.f){
  if(!supervised)return 0;
  double norm=0;
#ifdef TAO_GPU_HEALTH
  GpuHealth health;for(auto&t:spec)health.add(graph.w.at(t.name)->grad,false,double(supervised));
#else
  for(auto&t:spec)for(float g:graph.w.at(t.name)->grad.host()){if(!std::isfinite(g))throw std::runtime_error("gradient");double z=double(g)/supervised;norm+=z*z;}
#endif
  norm=std::sqrt(norm);
  // 梯度裁剪（L2 范数）：norm>max_norm 时把 factor 里的 norm 替换为 max_norm。
  // 因为 Adam 内核用 factor 把梯度 pre-scale 到 unit-norm 区，再乘 lr；
  // 把 norm 替换成 max_norm 等价于把整批梯度的有效 L2 视为 max_norm —— 标准 L2 grad clip 语义。
  // max_norm<=0 时退化到原行为（factor=1/(supervised*max(norm,1))）。
  const float norm_for_scale=max_norm>0.f && norm>max_norm ? max_norm : float(std::max(1.,norm));
  const float factor=float(1./supervised/norm_for_scale);
  ++steps;
  const float bc1=1-std::pow(.9f,float(steps)),bc2=1-std::pow(.999f,float(steps));
  for(auto&t:spec){
    const size_t n=t.elements();
    const unsigned blocks=unsigned((n+127)/128);
    auto g=graph.w.at(t.name)->grad.p;
    if(!offload_){
      auto a=master.at(t.name);
#ifdef TAO_GPU_HEALTH
      ds_adamw_device<<<blocks,128>>>(a->p,g,moment.at(t.name)->p,variance.at(t.name)->p,int(n),health.sum,health.bad,double(supervised),lr,t.ternary?.01f:0.f,bc1,bc2);
#else
      ds_adamw<<<blocks,128>>>(a->p,g,moment.at(t.name)->p,variance.at(t.name)->p,int(n),factor,lr,t.ternary?.01f:0.f,bc1,bc2);
#endif
    }else{
      auto&mh=master_h_.at(t.name);auto&mo=moment_h_.at(t.name);auto&va=variance_h_.at(t.name);
      check(cudaMemcpy(stage_->p,mh.data(),n*4,cudaMemcpyHostToDevice));
      check(cudaMemcpy(stage_m_->p,mo.data(),n*4,cudaMemcpyHostToDevice));
      check(cudaMemcpy(stage_v_->p,va.data(),n*4,cudaMemcpyHostToDevice));
#ifdef TAO_GPU_HEALTH
      ds_adamw_device<<<blocks,128>>>(stage_->p,g,stage_m_->p,stage_v_->p,int(n),health.sum,health.bad,double(supervised),lr,t.ternary?.01f:0.f,bc1,bc2);
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
    Vec a,b,c;
    if(!offload_){
      a=master.at(t.name)->host();b=moment.at(t.name)->host();c=variance.at(t.name)->host();
    }else{
      a=master_h_.at(t.name);b=moment_h_.at(t.name);c=variance_h_.at(t.name);
    }
    f.write(reinterpret_cast<const char*>(a.data()),std::streamsize(ne*4));
    f.write(reinterpret_cast<const char*>(b.data()),std::streamsize(ne*4));
    f.write(reinterpret_cast<const char*>(c.data()),std::streamsize(ne*4));
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
    Vec a(cnt),b(cnt),c(cnt);
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
