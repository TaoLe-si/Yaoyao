#pragma once
#include "dual_state_autograd.cuh"
#include "gpu_batched_matvec.cuh"
#include "gpu_batched_backward.cuh"
#include "gpu_batch_bias.cuh"
#include "gpu_batch_rms.cuh"
#include "gpu_batch_select.cuh"
#include "gpu_batch_embedding.cuh"
#ifndef TAO_BASELINE_GEMM
#include "gpu_batched_matvec_slots.cuh"
#endif
#ifdef TAO_BATCH_CUBLAS
#include "gpu_batch_cublas.cuh"
#endif
namespace tao::dual {
struct BatchTape: Tape {
#ifdef TAO_BATCH_CUBLAS
std::shared_ptr<BatchBlas> blas=std::make_shared<BatchBlas>();
#endif
Node zero_device(size_t count){auto y=std::make_shared<GradNode>(count);check(cudaMemsetAsync(y->value.p,0,count*sizeof(float),0));return y;}
// Device indices must originate from validated GpuBatchPlan; no hot-path host readback.
Node embedding_device(Node w,std::shared_ptr<Device>index,size_t offset,unsigned slots,unsigned d){if(!index||!slots||!d||w->value.n%d||offset>index->n||slots>index->n-offset)throw std::runtime_error("embedding slice shape");auto y=std::make_shared<GradNode>(size_t(d)*slots);batch_embed<<<(y->value.n+127)/128,128>>>(w->value.p,index->p+offset,y->value.p,d,slots);reverse.push_back([=](){batch_embed_back<<<(d+127)/128,128>>>(y->grad.p,index->p+offset,w->grad.p,d,slots);});return y;}
Node select_device(Node a,Node b,std::shared_ptr<Device>mask,size_t offset,unsigned slots){if(!mask||!slots||offset>mask->n||slots>mask->n-offset||a->value.n!=b->value.n||a->value.n%slots)throw std::runtime_error("mask slice shape");auto y=std::make_shared<GradNode>(a->value.n);size_t n=a->value.n/slots;batch_select_forward<<<(a->value.n+127)/128,128>>>(a->value.p,b->value.p,mask->p+offset,y->value.p,n,a->value.n);reverse.push_back([=](){batch_select_backward<<<(a->value.n+127)/128,128>>>(y->grad.p,mask->p+offset,a->grad.p,b->grad.p,n,a->value.n);});return y;}
Node embedding_batch(Node w,const std::vector<unsigned>&tokens,unsigned d){if(!d||tokens.empty()||w->value.n%d)throw std::runtime_error("batch embedding shape");Vec ids;for(unsigned id:tokens){if(id>=w->value.n/d||id>16777216u)throw std::runtime_error("batch embedding token");ids.push_back(float(id));}auto index=std::make_shared<Device>(ids);unsigned slots=unsigned(tokens.size());auto y=std::make_shared<GradNode>(size_t(d)*slots);batch_embed<<<(y->value.n+127)/128,128>>>(w->value.p,index->p,y->value.p,d,slots);reverse.push_back([=](){batch_embed_back<<<(d+127)/128,128>>>(y->grad.p,index->p,w->grad.p,d,slots);});return y;}
Node select_batch(Node a,Node b,std::shared_ptr<Device>mask){if(!mask||!mask->n||a->value.n!=b->value.n||a->value.n%mask->n)throw std::runtime_error("batch select shape");auto y=std::make_shared<GradNode>(a->value.n);size_t n=a->value.n/mask->n;batch_select_forward<<<(a->value.n+127)/128,128>>>(a->value.p,b->value.p,mask->p,y->value.p,n,a->value.n);reverse.push_back([=](){batch_select_backward<<<(a->value.n+127)/128,128>>>(y->grad.p,mask->p,a->grad.p,b->grad.p,n,a->value.n);});return y;}
Node norm_batch(Node x,Node g,unsigned slots){if(!slots||!g->value.n||x->value.n!=g->value.n*slots)throw std::runtime_error("batch norm shape");unsigned n=unsigned(g->value.n);auto y=std::make_shared<GradNode>(x->value.n);auto partial=std::make_shared<Device>(x->value.n);batch_rms_forward<<<slots,256>>>(x->value.p,g->value.p,y->value.p,n);reverse.push_back([=](){batch_rms_backward<<<slots,256>>>(x->value.p,g->value.p,y->grad.p,x->grad.p,partial->p,n);batch_bias_grad<<<(n+127)/128,128>>>(partial->p,g->grad.p,n,slots);});return y;}
Node bias_batch(Node x,Node b,unsigned slots){if(!slots||!b->value.n||x->value.n!=b->value.n*slots)throw std::runtime_error("batch bias shape");auto y=std::make_shared<GradNode>(x->value.n);unsigned n=unsigned(b->value.n);batch_bias_forward<<<(x->value.n+127)/128,128>>>(x->value.p,b->value.p,y->value.p,n,slots);reverse.push_back([=](){ds_add<<<(x->value.n+127)/128,128>>>(x->grad.p,y->grad.p,x->value.n);batch_bias_grad<<<(n+127)/128,128>>>(y->grad.p,b->grad.p,n,slots);});return y;}
Node linear_batch(Node w,Node x,unsigned rows,unsigned cols,unsigned slots){if(!slots||!rows||!cols||x->value.n!=size_t(slots)*cols||w->value.n!=size_t(rows)*cols)throw std::runtime_error("batch linear shape");auto y=std::make_shared<GradNode>(size_t(rows)*slots);
#ifdef TAO_BATCH_CUBLAS
auto engine=blas;engine->forward(w->value.p,x->value.p,y->value.p,rows,cols,slots);
#elif !defined(TAO_BASELINE_GEMM)
if(!launch_bmv_tiled(w->value.p,x->value.p,y->value.p,int(rows),int(cols),int(slots)))
    ds_batched_matvec<<<dim3((rows+3)/4,slots),128>>>(w->value.p,x->value.p,y->value.p,rows,cols,slots);
#else
ds_batched_matvec<<<dim3((rows+3)/4,slots),128>>>(w->value.p,x->value.p,y->value.p,rows,cols,slots);
#endif
check(cudaGetLastError());reverse.push_back([=](){
#ifdef TAO_BATCH_CUBLAS
engine->backward(w->value.p,x->value.p,y->grad.p,x->grad.p,w->grad.p,rows,cols,slots);
#else
#ifndef TAO_BASELINE_GEMM
if(!launch_dx_tiled(w->value.p,y->grad.p,x->grad.p,int(rows),int(cols),int(slots)))
#endif
ds_batch_dx<<<dim3((cols+31)/32,slots),dim3(32,8)>>>(w->value.p,y->grad.p,x->grad.p,rows,cols);ds_batch_dw<<<(size_t(rows)*cols+127)/128,128>>>(x->value.p,y->grad.p,w->grad.p,rows,cols,slots);
#endif
});return y;}
};
}
