#pragma once
#include "dual_state_cuda_backward.cuh"
#include <functional>
#include "gpu_linear_dx_tiled.cuh"
#include "gpu_rms_parallel.cuh"
#include "gpu_matvec_warp.cuh"
#ifdef TAO_DELTA_MEM
#include "delta_mem_kernels.cuh"
#endif
namespace tao::dual {
inline void tape_copy(float*dst,const float*src,size_t bytes){
#ifdef TAO_ASYNC_D2D
check(cudaMemcpyAsync(dst,src,bytes,cudaMemcpyDeviceToDevice,0));
#else
check(cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToDevice));
#endif
}
#ifdef TAO_GRAD_ALLOCATION_OBSERVER
inline std::function<void(Device&)> grad_allocation_observer;
inline void observe_grad(Device&g){if(grad_allocation_observer)grad_allocation_observer(g);}
#else
inline void observe_grad(Device&){}
#endif
#ifdef TAO_DEVICE_ZERO_GRAD
struct GradNode{Device value,grad;explicit GradNode(const Vec&v):value(v),grad(v.size()){check(cudaMemsetAsync(grad.p,0,grad.n*sizeof(float)));observe_grad(grad);}explicit GradNode(size_t n):value(n),grad(n){check(cudaMemsetAsync(grad.p,0,grad.n*sizeof(float)));observe_grad(grad);}};
#else
struct GradNode{Device value,grad;explicit GradNode(const Vec&v):value(v),grad(Vec(v.size())){}explicit GradNode(size_t n):value(n),grad(Vec(n)){} };
#endif
using Node=std::shared_ptr<GradNode>;
__global__ void ds_scaled_copy(const float*x,float*y,int n,float scale,bool accumulate){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=x[i]*scale;if(accumulate)y[i]+=v;else y[i]=v;}};
__global__ void ds_add_direct(const float*a,const float*b,float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=a[i]+b[i];}
__global__ void ds_add_pair_grad(const float*dy,float*da,float*db,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){da[i]+=dy[i];db[i]+=dy[i];}}
struct Tape {
Node scaled(Node x,float scale){auto y=std::make_shared<GradNode>(x->value.n);ds_scaled_copy<<<(x->value.n+127)/128,128>>>(x->value.p,y->value.p,int(x->value.n),scale,false);reverse.push_back([=](){ds_scaled_copy<<<(x->value.n+127)/128,128>>>(y->grad.p,x->grad.p,int(x->value.n),scale,true);});return y;}
std::vector<std::function<void()>> reverse;
Node leaf(const Vec&v){return std::make_shared<GradNode>(v);}
#ifdef TAO_DELTA_MEM
// ---- H2R 增量规则矩阵记忆：按 slot 数参数化，slots==1 即单序列训练 ----
Node l2norm(Node k,unsigned slots,unsigned dk){
  if(!slots||!dk||k->value.n!=size_t(slots)*dk)throw std::runtime_error("l2norm shape");
  auto y=std::make_shared<GradNode>(k->value.n);
  auto norm=std::make_shared<Device>(size_t(slots));
  dm_normalize_fwd<<<slots,1>>>(k->value.p,y->value.p,norm->p,dk);
  check(cudaGetLastError());
  reverse.push_back([=](){dm_normalize_bwd<<<slots,dk>>>(y->value.p,y->grad.p,norm->p,k->grad.p,dk);
                          check(cudaGetLastError());});
  return y;
}
Node sigmoid(Node x){
  if(!x->value.n)throw std::runtime_error("sigmoid shape");
  auto y=std::make_shared<GradNode>(x->value.n);
  const int n=int(x->value.n);
  dm_sigmoid_fwd<<<(n+127)/128,128>>>(x->value.p,y->value.p,n);
  check(cudaGetLastError());
  reverse.push_back([=](){dm_sigmoid_bwd<<<(n+127)/128,128>>>(y->value.p,y->grad.p,x->grad.p,n);
                          check(cudaGetLastError());});
  return y;
}
// 返回 {新 M, o}；o 由写入之后的 M' 计算（先写后读）。
std::pair<Node,Node> delta(Node M,Node khat,Node q,Node v,Node beta,
                           unsigned slots,unsigned dv,unsigned dk){
  if(!slots||!dv||!dk||dv>1024)throw std::runtime_error("delta shape");
  if(M->value.n!=size_t(slots)*dv*dk||khat->value.n!=size_t(slots)*dk||q->value.n!=size_t(slots)*dk||
     v->value.n!=size_t(slots)*dv||beta->value.n!=slots)throw std::runtime_error("delta shape");
  auto Mout=std::make_shared<GradNode>(M->value.n);
  auto o=std::make_shared<GradNode>(size_t(slots)*dv);
  auto a=std::make_shared<Device>(size_t(slots)*dv);
  auto c=std::make_shared<Device>(size_t(slots));
  auto e=std::make_shared<Device>(size_t(slots));
  auto p=std::make_shared<Device>(size_t(slots)*dv);
  auto du=std::make_shared<Device>(size_t(slots)*dv);
  dm_forward<<<slots,dv>>>(M->value.p,khat->value.p,q->value.p,v->value.p,beta->value.p,
                           Mout->value.p,o->value.p,a->p,dv,dk);
  check(cudaGetLastError());
  reverse.push_back([=](){
      dm_proj_bwd<<<slots,dv>>>(Mout->grad.p,khat->value.p,p->p,dv,dk);
      dm_scalars_bwd<<<slots,1>>>(q->value.p,khat->value.p,o->grad.p,v->value.p,a->p,p->p,
                                  beta->value.p,c->p,e->p,du->p,beta->grad.p,dv,dk);
      dm_dM_dv_bwd<<<slots,dv>>>(Mout->grad.p,khat->value.p,q->value.p,o->grad.p,du->p,beta->value.p,
                                 M->grad.p,v->grad.p,dv,dk);
      dm_dkhat_dq_bwd<<<slots,dk>>>(M->value.p,khat->value.p,q->value.p,o->grad.p,v->value.p,a->p,
                                    du->p,e->p,beta->value.p,Mout->grad.p,khat->grad.p,q->grad.p,dv,dk);
      check(cudaGetLastError());});
  return {Mout,o};
}
#endif
Node add(Node a,Node b){if(a->value.n!=b->value.n)throw std::runtime_error("add");auto y=std::make_shared<GradNode>(a->value.n);
#ifdef TAO_DIRECT_ADD
ds_add_direct<<<(a->value.n+127)/128,128>>>(a->value.p,b->value.p,y->value.p,a->value.n);
#else
tape_copy(y->value.p,a->value.p,a->value.n*4);ds_add<<<(a->value.n+127)/128,128>>>(y->value.p,b->value.p,a->value.n);
#endif
reverse.push_back([=](){
#ifdef TAO_FUSED_ADD_BACKWARD
ds_add_pair_grad<<<(a->value.n+127)/128,128>>>(y->grad.p,a->grad.p,b->grad.p,a->value.n);
#else
ds_add<<<(a->value.n+127)/128,128>>>(a->grad.p,y->grad.p,a->value.n);ds_add<<<(b->value.n+127)/128,128>>>(b->grad.p,y->grad.p,b->value.n);
#endif
});return y;}
Node linear(Node w,Node x,unsigned rows){if(w->value.n!=rows*x->value.n)throw std::runtime_error("linear");auto y=std::make_shared<GradNode>(rows);training_matvec(w->value.p,x->value.p,y->value.p,rows,x->value.n);reverse.push_back([=](){
#ifdef TAO_TILED_DX
ds_dx_tiled<<<(x->value.n+31)/32,dim3(32,8)>>>(w->value.p,y->grad.p,x->grad.p,rows,x->value.n);
#else
ds_linear_dx<<<(x->value.n+127)/128,128>>>(w->value.p,y->grad.p,x->grad.p,rows,x->value.n);
#endif
ds_linear_dw<<<(w->value.n+127)/128,128>>>(x->value.p,y->grad.p,w->grad.p,rows,x->value.n);});return y;}
Node norm(Node x,Node g){auto y=std::make_shared<GradNode>(x->value.n);
#ifdef TAO_PARALLEL_RMS
rms_parallel<<<1,256>>>(x->value.p,g->value.p,y->value.p,x->value.n);
#else
rms<<<1,1>>>(x->value.p,g->value.p,y->value.p,x->value.n);
#endif
reverse.push_back([=](){
#ifdef TAO_PARALLEL_RMS
rms_back_parallel<<<1,256>>>(x->value.p,g->value.p,y->grad.p,x->grad.p,g->grad.p,x->value.n);
#else
ds_rms_backward<<<1,1>>>(x->value.p,g->value.p,y->grad.p,x->grad.p,g->grad.p,x->value.n);
#endif
});return y;}
Node update(Node old,Node u,Node g){auto y=std::make_shared<GradNode>(old->value.n);tape_copy(y->value.p,old->value.p,old->value.n*4);ds_update<<<(old->value.n+127)/128,128>>>(y->value.p,u->value.p,g->value.p,old->value.n);reverse.push_back([=](){ds_update_backward<<<(old->value.n+127)/128,128>>>(old->value.p,u->value.p,g->value.p,y->grad.p,old->grad.p,u->grad.p,g->grad.p,old->value.n);});return y;}
Node silu(Node x){auto y=std::make_shared<GradNode>(x->value.n);tape_copy(y->value.p,x->value.p,x->value.n*4);ds_silu<<<(x->value.n+127)/128,128>>>(y->value.p,x->value.n);reverse.push_back([=](){ds_silu_backward<<<(x->value.n+127)/128,128>>>(x->value.p,y->grad.p,x->grad.p,x->value.n);});return y;}
Node embedding(Node w,unsigned token,unsigned d){auto y=std::make_shared<GradNode>(d);tape_copy(y->value.p,w->value.p+size_t(token)*d,d*4);reverse.push_back([=](){ds_add<<<(d+127)/128,128>>>(w->grad.p+size_t(token)*d,y->grad.p,d);});return y;}
void backward(){for(auto it=reverse.rbegin();it!=reverse.rend();++it){(*it)();check(cudaGetLastError());}
#ifndef TAO_DEFER_BACKWARD_SYNC
check(cudaDeviceSynchronize());
#endif
}
};
struct TrainGraph {
Config c;Tape tape;std::map<std::string,Node>w;std::vector<Node>s,m;
explicit TrainGraph(const CpuModel&cpu):c(cpu.c){for(auto&kv:cpu.w)w[kv.first]=tape.leaf(kv.second);for(unsigned l=0;l<c.layers;++l){s.push_back(tape.leaf(Vec(c.s)));m.push_back(tape.leaf(Vec(size_t(c.memory_size()))));}}
Node step(unsigned token){if(token>=c.vocab)throw std::invalid_argument("token");auto x=tape.embedding(w.at("embedding"),token,c.d);
#ifdef TAO_INPUT_SCALE
x=tape.scaled(x,std::sqrt(float(c.d)));
#endif
for(unsigned l=0;l<c.layers;++l){auto p="layer."+std::to_string(l)+".";auto xn=tape.norm(x,w.at(p+"input.norm"));auto branch=[&](std::string name,bool mem){unsigned n=mem?c.m:c.s;auto z=tape.add(tape.linear(w.at(p+name+".x"),xn,n),tape.linear(w.at(p+name+".s"),s[l],n));if(mem)z=tape.add(z,tape.linear(w.at(p+name+".m"),m[l],n));return tape.add(z,w.at(p+name+".bias"));};auto u=branch("s.candidate",false),a=branch("s.gate",false);s[l]=tape.update(s[l],u,a);
#ifdef TAO_DELTA_MEM
{const unsigned dk=c.dk,dv=c.m;
 auto kk=tape.linear(w.at(p+"mem.key"),xn,dk),qq=tape.linear(w.at(p+"mem.query"),xn,dk),vv=tape.linear(w.at(p+"mem.value"),xn,dv);
 auto khat=tape.l2norm(kk,1,dk);
 auto beta=tape.sigmoid(tape.add(tape.linear(w.at(p+"mem.beta"),xn,1),w.at(p+"mem.beta.bias")));
 auto dm=tape.delta(m[l],khat,qq,vv,beta,1,dv,dk);
 m[l]=dm.first;
 x=tape.add(x,tape.norm(tape.add(tape.linear(w.at(p+"read.s"),s[l],c.d),dm.second),w.at(p+"read.norm")));}
#else
auto v=branch("m.candidate",true),g=branch("m.gate",true);m[l]=tape.update(m[l],v,g);auto r=tape.add(tape.linear(w.at(p+"read.s"),s[l],c.d),tape.linear(w.at(p+"read.m"),m[l],c.d));x=tape.add(x,tape.norm(r,w.at(p+"read.norm")));
#endif
#ifndef TAO_NO_FFN
auto f=tape.silu(tape.linear(w.at(p+"ff.up"),tape.norm(x,w.at(p+"ff.norm")),c.e));x=tape.add(x,tape.linear(w.at(p+"ff.down"),f,c.d));
#endif
}return tape.add(tape.linear(w.at("embedding"),tape.norm(x,w.at("final.norm")),c.vocab),w.at("vocab.bias"));}
};
}
