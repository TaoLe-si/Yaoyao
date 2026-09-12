#pragma once
#include "cpu_row_parallel_executor.hpp"
#include "cpu_row_parallel_pool.hpp"
#include "dual_state_cpu.hpp"
#include "cpu_pipeline_rows.hpp"
#include "cpu_fast_activation.hpp"
#include "cpu_compact_bundle.hpp"
#include "session_hot.hpp"
#include <map>
#include <array>
#include <utility>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <limits>
#ifdef TAO_CPU_AVX2
#include "cpu_dot_avx2.hpp"
#endif
namespace tao::dual {

#ifdef TAO_PHASE_TIMING
#include <chrono>
// Re-armable scope timer for the S3 phase breakdown. Guarded so production
// builds compile exactly the original statements.
struct SubTimer{double* a;std::chrono::steady_clock::time_point t0;
    explicit SubTimer(double&x):a(&x),t0(std::chrono::steady_clock::now()){}
    ~SubTimer(){*a+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();}};
// 36：表达式计时（区分「分发本身」「add/查表」「norm」各自成本）
#define TAO_T(acc,expr) ([&]()->decltype(expr){SubTimer _t(acc);return (expr);}())
struct ScopedPhase{double* acc=nullptr;std::chrono::steady_clock::time_point t0;
    void flush(){if(acc){*acc+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();acc=nullptr;}}
    void arm(double& a){flush();acc=&a;t0=std::chrono::steady_clock::now();}
    explicit ScopedPhase(double& a){arm(a);}
    ~ScopedPhase(){flush();}
    ScopedPhase(const ScopedPhase&)=delete;ScopedPhase& operator=(const ScopedPhase&)=delete;};
#else
struct SubTimer{explicit SubTimer(double&){} };
#define TAO_T(acc,expr) (expr)
#endif

// Standalone owning CPU candidate; no GPU/training or baseline changes.
// One caller plus N-1 persistent sleeping workers (default 2 threads total,
// identical to the original executor split).
struct GreedyPipelineGroupedModel{
Config c;std::map<std::string,Vec>w;std::map<std::string,PipelineRows>packed;
// 36：子相位累加器。**必须在 group<> 模板定义之前声明** ——
// MSVC 对模板成员函数按两阶段查找，定义点看不到类后面声明的成员。
mutable double ms_grp=0,ms_addop=0,ms_normf=0,ms_pre=0;
// 层维复用（03/13 号文档 R1–R3）：layerSource[l] = 第 l 层实际读取哪一层的权重。
// 默认恒等映射 = 现状，行为与字节完全不变（不变量 I5：默认路径不受影响）。
// 设为 [0,0,0,0,4,4,4,4] 时，层 1–3 与层 5–7 读取的 packed 条目与层 0/4 是**同一个对象**，
// 因此物理地址相同 -> 每 token 触碰的唯一字节从 13.37 MB 降到约 3.34 MB。
// 状态 state[l] 仍是每层独立，只有权重被共享（即 R3，不是 R4）。
mutable std::vector<uint32_t> layerSource;
// R4（03 号文档）：组内复用 Full 层算出的 m 候选/门控向量，省掉 Reuse 层 group<6> 的全部 MAC。
// 由 TAO_LAYER_REUSE_MV=1 开启。仅在 layerSource 非恒等（即 R3 生效）时才可能命中。
mutable bool reuse_mv=false;
// 诊断开关：TAO_LAYER_LIMIT=N 只跑前 N 层，用于量化「层间串行」占总时间的比例。
// 会改变输出（校验和随之变化），只作计时用，不得用于质量结论。
mutable uint32_t layer_limit=0;
// 廉价激活：把状态更新的 sigmoid/tanh 换成三阶 Pade 近似的 AVX2 实现。
mutable bool fast_act_=false;
mutable bool vnni_=false;
// 诊断专用（默认关闭）：把输出头的行数截到 k，用于分离「每行固定开销」与「每 MAC/字节开销」。
// 会改变预测与 checksum，禁止用于生产。
mutable uint32_t head_limit_=0;
 // 推理端重复惩罚：非空时从 token r 的 logit 扣减 head_adj_[r]。
 // 借用指针，调用方保证生存期；nullptr = 关闭（与原实现逐位一致）。
 mutable const float* head_adj_=nullptr;
// 01 号文档 #2（论文 Hierarchical Sparse Indexer 迁移）：两段式候选池词表头。
// 阶段1：G 个簇心打分，取 top-C 簇；阶段2：只算候选簇内的 token 全维点积。
// head2_debug_=1 时同时算完整头，用于统计分歧率（评估聚类代价，不用于生产）。
mutable bool head2_=false,head2_debug_=false;
// 层栈 dispatch 融合（§44.6）：把每层的 s 分支与 m 分支合并成一次 dispatch。
// 需要 mHC 式依赖错位 —— m 分支改读本层更新前的 s，以消除 s→m 的串行依赖。
//
// 【默认已改为关闭】两个理由：
//  1) **违反 I4（训练/推理同算子）**：训练器 dual_state_autograd.cuh:136/146 先算
//     s[l]=update(...)，再让 m 分支读**更新后**的 s[l]（顺序依赖）。而融合在推理期让 m 读
//     **更新前**的 s_old。二者不是同一算子。旧注释称"训练期同域"，但训练器**根本没有**
//     融合路径（grep fuse 无命中），所以该说法不成立。
//  2) **收益已被证伪**：§44.6 实测 0.991×（无加速）。
// 实测（L4 step400，12 个 held-out 名字）：关闭后名字 1→2、颜色 4→5，
// 且出现 `邹晓东，紫色。`（把 held-out 的 `邬晓东` 抄对了一半）。
// 需要旧行为时显式设 TAO_FUSE_SM=1。
mutable bool fuse_sm_=false;
mutable uint32_t head2_G=256,head2_C=4;
mutable std::vector<float> head2_cent_;   // G*d
mutable std::vector<float> head2_cbias_;  // G
mutable std::vector<uint32_t> head2_of_;  // vocab -> cluster
mutable std::vector<uint32_t> head2_start_,head2_members_; // CSR 簇成员
mutable std::vector<size_t> head2_stat_nodiv_,head2_stat_tot_;

static std::vector<uint32_t> identitySource(uint32_t n){std::vector<uint32_t> v(n);for(uint32_t i=0;i<n;++i)v[i]=i;return v;}
// 实验开关：TAO_LAYER_SHARE="0,0,0,0,4,4,4,4"（缺省或格式不符则保持恒等）
static std::vector<uint32_t> parseLayerShare(const char* s,uint32_t n){
    std::vector<uint32_t> v=identitySource(n);
    if(!s||!*s)return v;
    std::vector<uint32_t> got;const char* p=s;
    while(*p){char* e=nullptr;unsigned long x=std::strtoul(p,&e,10);if(e==p)return identitySource(n);
        got.push_back(uint32_t(x));p=e;while(*p==','||*p==' ')++p;}
    if(got.size()!=n)return identitySource(n);
    for(auto x:got)if(x>=n)return identitySource(n);
    return got;}
// 默认层映射：描述**模型自身**层号 -> 权重组的对应，解码时再按 layer_limit 截断。
//   n >= 4 : 两半分组（l < n/2 -> 0，否则 -> n/2），即 8 层的 0,0,0,0,4,4,4,4；
//            这是上一轮固定的部署配置，截到 2 层即 {0,0}。
//   n == 2 : 恒等 {0,1}。新训练从零起时 initialize() 为每层生成**独立**权重，
//            若映射成 {0,0} 就会丢弃第 1 层、把第 0 层跑两遍，违反 I4（训练/推理同算子）。
//            实测 L4 step400：0,0 -> "童小龙，樱桃"（全错），0,1 -> "王小军，紫色"（部分正确）。
//   n < 2  : 恒等。
// 层共享是部署优化，只有在训练期也做权重绑定（R3）时才严格合法；显式给 TAO_LAYER_SHARE 可覆盖。
static std::vector<uint32_t> groupSource(uint32_t n){
    if(n<4u)return identitySource(n);
    std::vector<uint32_t> s(n);
    for(uint32_t i=0;i<n;++i)s[i]=(i<n/2u)?0u:(n/2u);
    return s;}

explicit GreedyPipelineGroupedModel(CompactBundleData&&src):c(src.c),w(std::move(src.vectors)){layerSource=parseLayerShare(std::getenv("TAO_LAYER_SHARE"),c.layers);reuse_mv=(std::getenv("TAO_LAYER_REUSE_MV")!=nullptr);{const char* ll=std::getenv("TAO_LAYER_LIMIT");if(ll&&*ll)layer_limit=uint32_t(std::strtoul(ll,nullptr,10));}for(auto&t:schema(c))if(t.ternary){auto&m=src.matrices.at(t.name);PipelineRows p(Vec(t.cols,0),1,t.cols);p.rows=t.rows;p.q=std::move(m.q);p.scale=std::move(m.scale);packed.emplace(t.name,std::move(p));}
// 必须在 packed 填充【之后】应用，否则遍历空 map（此前的潜伏 bug）
applyDefaultArch();fuse_sm_=envOn("TAO_FUSE_SM",false);if(std::getenv("TAO_LAYER_LIMIT"))layer_limit=uint32_t(std::strtoul(std::getenv("TAO_LAYER_LIMIT"),nullptr,10));dropDuplicatedLayers();if(std::getenv("TAO_WEIGHT_DUMP"))dumpWeights();if(std::getenv("TAO_HEAD_LIMIT"))set_head_limit(uint32_t(std::strtoul(std::getenv("TAO_HEAD_LIMIT"),nullptr,10)));}
explicit GreedyPipelineGroupedModel(const CpuModel&src):c(src.c){layerSource=parseLayerShare(std::getenv("TAO_LAYER_SHARE"),c.layers);reuse_mv=(std::getenv("TAO_LAYER_REUSE_MV")!=nullptr);{const char* ll=std::getenv("TAO_LAYER_LIMIT");if(ll&&*ll)layer_limit=uint32_t(std::strtoul(ll,nullptr,10));}for(auto&t:schema(c)){if(t.ternary)packed.emplace(t.name,PipelineRows(src.w.at(t.name),t.rows,t.cols));else w.emplace(t.name,src.w.at(t.name));}
// 必须在 packed 填充【之后】应用，否则遍历空 map（此前的潜伏 bug）
applyDefaultArch();fuse_sm_=envOn("TAO_FUSE_SM",false);if(std::getenv("TAO_LAYER_LIMIT"))layer_limit=uint32_t(std::strtoul(std::getenv("TAO_LAYER_LIMIT"),nullptr,10));dropDuplicatedLayers();if(std::getenv("TAO_WEIGHT_DUMP"))dumpWeights();if(std::getenv("TAO_HEAD_LIMIT"))set_head_limit(uint32_t(std::strtoul(std::getenv("TAO_HEAD_LIMIT"),nullptr,10)));}
#ifdef TAO_CPU_AVX2
bool fast=true;
// Opt-in experimental backend; existing callers retain the single-row kernel.
// Only the original single-row dot arithmetic is used.

#endif
size_t weight_bytes()const{size_t n=0;for(auto&kv:w)n+=kv.second.size()*sizeof(float);for(auto&kv:packed)n+=kv.second.q.size()+kv.second.scale.size()*sizeof(float);return n;}
size_t memory_size()const{
#ifdef TAO_DELTA_MEM
return size_t(c.m)*c.dk;
#else
return c.m;
#endif
}
std::vector<LayerState> initial()const{
    std::vector<LayerState> st;st.reserve(c.layers);
    for(uint32_t i=0;i<c.layers;++i)st.emplace_back(c.s,memory_size());
    return st;}
// 浮点（非三值）投影的内积；仅 mem.beta 这类 d 维向量使用，成本可忽略。
float dot_float(const std::string&name,const Vec&x)const{const auto&a=w.at(name);if(a.size()!=x.size())throw std::runtime_error("float projection shape");float acc=0;for(size_t j=0;j<x.size();++j)acc+=a[j]*x[j];return acc;}
Vec linear(const std::string&name,const Vec&x,uint32_t rows)const{const auto&p=packed.at(name);if(p.rows!=rows||p.cols!=x.size())throw std::runtime_error("matrix shape");Vec y(rows);
#ifdef TAO_CPU_AVX2
if(fast){{PipelineRows&mp=const_cast<PipelineRows&>(p);if(mp.vnni_)mp.quantize_input(x.data());else mp.qc_x_=nullptr;}dispatch_rows(rows,p.cols,[&](size_t begin,size_t end){p.gemv_rows(begin,end,x.data(),y.data());});return y;}
#endif
for(uint32_t r=0;r<rows;++r)for(size_t j=0;j<x.size();++j)y[r]+=(float(p.q[r*x.size()+j])*p.scale[r])*x[j];return y;}
static void add(Vec&a,const Vec&b){if(a.size()!=b.size())throw std::runtime_error("vector shape");
#ifdef TAO_CPU_AVX2
size_t i=0;for(;i+8<=a.size();i+=8)_mm256_storeu_ps(a.data()+i,_mm256_add_ps(_mm256_loadu_ps(a.data()+i),_mm256_loadu_ps(b.data()+i)));
for(;i<a.size();++i)a[i]+=b[i];
#else
for(size_t i=0;i<a.size();++i)a[i]+=b[i];
#endif
}
Vec norm(const Vec&x,const std::string&name)const{Vec y;norm_into(x,name,y);return y;}
static float sigmoid(float x){if(x>=0)return 1/(1+std::exp(-x));float e=std::exp(x);return e/(1+e);}
void norm_into(const Vec&x,const std::string&name,Vec&y)const{
    const auto&g=w.at(name);if(g.size()!=x.size())throw std::runtime_error("norm shape");
    float sum=0;for(float z:x)sum+=z*z;const float inv=1/std::sqrt(sum/x.size()+1e-5f);
    y.resize(x.size());
#ifdef TAO_CPU_AVX2
    size_t j=0;const __m256 vinv=_mm256_set1_ps(inv);
    for(;j+8<=x.size();j+=8){
        __m256 gg=_mm256_mul_ps(_mm256_loadu_ps(g.data()+j),vinv);
        _mm256_storeu_ps(y.data()+j,_mm256_mul_ps(_mm256_loadu_ps(x.data()+j),gg));
    }
    for(;j<x.size();++j)y[j]=x[j]*inv*g[j];
#else
    for(size_t j=0;j<x.size();++j)y[j]=x[j]*inv*g[j];
#endif
}
void load_token_emb(uint32_t token,Vec&x)const{
    const auto&emb=packed.at("embedding");x.resize(c.d);
    const int8_t* p=emb.q.data()+size_t(token)*c.d;
    const float sc=emb.scale[token]
#ifdef TAO_INPUT_SCALE
        *std::sqrt(float(c.d))
#endif
        ;
#ifdef TAO_CPU_AVX2
    const __m256 vs=_mm256_set1_ps(sc);size_t j=0;
    for(;j+8<=c.d;j+=8){
        __m256 q=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p+j))));
        _mm256_storeu_ps(x.data()+j,_mm256_mul_ps(q,vs));
    }
    for(;j<c.d;++j)x[j]=float(p[j])*sc;
#else
    for(size_t j=0;j<c.d;++j)x[j]=float(p[j])*sc;
#endif
}
template<class X> void linear_into(const std::string&name,const X&x,uint32_t rows,Vec&y)const{
    const auto&p=packed.at(name);if(p.rows!=rows||p.cols!=x.size())throw std::runtime_error("matrix shape");
    y.resize(rows);
#ifdef TAO_CPU_AVX2
    if(fast){{PipelineRows&mp=const_cast<PipelineRows&>(p);if(mp.vnni_)mp.quantize_input(x.data());else mp.qc_x_=nullptr;}
        dispatch_rows(rows,p.cols,[&](size_t begin,size_t end){p.gemv_rows(begin,end,x.data(),y.data());});return;}
#endif
    for(uint32_t r=0;r<rows;++r){float z=0;for(size_t j=0;j<x.size();++j)z+=(float(p.q[r*x.size()+j])*p.scale[r])*x[j];y[r]=z;}
}
struct RecurBuf{Vec x,xn,v,o,r,nrm,hn;std::array<Vec,4> g4;std::array<Vec,2> g2;};
mutable RecurBuf rb_;
void ensure_rb()const{
    if(rb_.x.size()==c.d)return;
    rb_.x.assign(c.d,0);rb_.xn.assign(c.d,0);rb_.v.assign(c.m,0);rb_.o.assign(c.m,0);
    rb_.r.assign(c.d,0);rb_.nrm.assign(c.d,0);rb_.hn.assign(c.d,0);
    for(auto&z:rb_.g4)z.assign(c.s,0);
#ifdef TAO_DELTA_MEM
    rb_.g2[0].assign(c.dk,0);rb_.g2[1].assign(c.dk,0);
#endif
}
private:
struct InX{const float*p;size_t n;const float*data()const{return p;}size_t size()const{return n;}const float&operator[](size_t i)const{return p[i];}};
static InX inx(const Vec&v){return {v.data(),v.size()};}
static InX inx(const FSpan&v){return {v.data(),v.size()};}

// A group is one synchronous dispatch, not one dispatch per projection.
// Cost coordinates concatenate rows*cols, then snap the half-cost boundary
// upward to a whole row. No dot is split: each side differs from half-cost
// by less than one row cost (candidate/gate groups split exactly in half).
// Inputs/outputs are borrowed until run returns, including exception paths.
template<size_t N> std::array<Vec,N> group(const std::array<std::string,N>& names,
                                         const std::array<InX,N>& inputs,
                                         uint32_t rows)const {
    std::array<Vec,N> out;
    std::array<const PipelineRows*,N> matrices{};
    std::array<size_t,N+1> offsets{};
    {SubTimer _tp(ms_pre);
    for(size_t i=0;i<N;++i){
        const auto& p=packed.at(names[i]);
        // 01 号文档 #6（mHC 迁移）：允许组内各矩阵 rows 不同，使 s(128) 与 m(512)
        // 分支能合并进同一次 dispatch。分派本就按 MAC 区间切分并用各自的 p.cols 还原行号。
        if(p.cols!=inputs[i].size()||p.cols==0)
            throw std::runtime_error("matrix shape");
        if(p.rows>(std::numeric_limits<size_t>::max()-offsets[i])/p.cols)
            throw std::overflow_error("group cost overflow");
        matrices[i]=&p;out[i].resize(p.rows);
        offsets[i+1]=offsets[i]+p.rows*p.cols;
    }
    }   // 关闭 ms_pre 的 SubTimer scope（必须紧跟在查表/resize 循环之后）
#ifdef TAO_CPU_AVX2
    if(fast){
        // 32: 量化必须在 dispatch 之前完成 —— 否则 8 个工作线程会并发
        // 对同一个 PipelineRows 做 xu_.resize()，导致堆损坏 (0xC0000374)。
        for(size_t i=0;i<N;++i){
            PipelineRows&mp=const_cast<PipelineRows&>(*matrices[i]);
            if(mp.vnni_)mp.quantize_input(inputs[i].data());
            else mp.qc_x_=nullptr;
        }
    dispatch_rows(offsets[N],1,[&](size_t begin,size_t end){
            for(size_t i=0;i<N;++i){
                if(end<=offsets[i]||begin>=offsets[i+1])continue;
                const auto& p=*matrices[i];
                const size_t lo=begin>offsets[i]?begin-offsets[i]:0;
                const size_t hi=end<offsets[i+1]?end-offsets[i]:offsets[i+1]-offsets[i];
                // Adjacent cost intervals share the same ceil(boundary/cols):
                // caller [0,k), worker [k,rows). Thus no missing/duplicate rows.
                const size_t first=lo/p.cols+(lo%p.cols!=0);
                const size_t last=hi/p.cols+(hi%p.cols!=0);
                p.gemv_rows(first,last,inputs[i].data(),out[i].data());
            }
        });
        return out;
    }
#endif
    for(size_t i=0;i<N;++i){
        const auto& p=*matrices[i];const auto& x=inputs[i];
        for(uint32_t r=0;r<p.rows;++r)for(size_t j=0;j<x.size();++j)
            out[i][r]+=(float(p.q[r*x.size()+j])*p.scale[r])*x[j];
    }
    return out;
}

template<size_t N> void group_into(const std::array<std::string,N>& names,
                                  const std::array<InX,N>& inputs,
                                  std::array<Vec,N>& out)const {
    std::array<const PipelineRows*,N> matrices{};
    std::array<size_t,N+1> offsets{};
    {SubTimer _tp(ms_pre);
    for(size_t i=0;i<N;++i){
        const auto& p=packed.at(names[i]);
        if(p.cols!=inputs[i].size()||p.cols==0)
            throw std::runtime_error("matrix shape");
        if(p.rows>(std::numeric_limits<size_t>::max()-offsets[i])/p.cols)
            throw std::overflow_error("group cost overflow");
        matrices[i]=&p;
        if(out[i].size()!=p.rows)out[i].assign(p.rows,0.f);
        offsets[i+1]=offsets[i]+p.rows*p.cols;
    }
    }
#ifdef TAO_CPU_AVX2
    if(fast){
        for(size_t i=0;i<N;++i){
            PipelineRows&mp=const_cast<PipelineRows&>(*matrices[i]);
            if(mp.vnni_)mp.quantize_input(inputs[i].data());
            else mp.qc_x_=nullptr;
        }
        dispatch_rows(offsets[N],1,[&](size_t begin,size_t end){
            for(size_t i=0;i<N;++i){
                if(end<=offsets[i]||begin>=offsets[i+1])continue;
                const auto& p=*matrices[i];
                const size_t lo=begin>offsets[i]?begin-offsets[i]:0;
                const size_t hi=end<offsets[i+1]?end-offsets[i]:offsets[i+1]-offsets[i];
                const size_t first=lo/p.cols+(lo%p.cols!=0);
                const size_t last=hi/p.cols+(hi%p.cols!=0);
                p.gemv_rows(first,last,inputs[i].data(),out[i].data());
            }
        });
        return;
    }
#endif
    for(size_t i=0;i<N;++i){
        const auto& p=*matrices[i];const auto& x=inputs[i];
        Vec& y=out[i];
        for(uint32_t r=0;r<p.rows;++r){float z=0;for(size_t j=0;j<x.size();++j)
            z+=(float(p.q[r*x.size()+j])*p.scale[r])*x[j];y[r]=z;}
    }
}

Vec recurrent(uint32_t token,std::vector<LayerState>&state)const{if(token>=c.vocab||state.size()!=c.layers)throw std::invalid_argument("token/state");for(const auto&z:state)if(z.s.size()!=c.s||z.m.size()!=memory_size())throw std::invalid_argument("state shape");
prefetch_session_l2(state);ensure_rb();load_token_emb(token,rb_.x);Vec& x=rb_.x;
std::vector<Vec> cachedV,cachedG;if(reuse_mv&&layerSource.size()==c.layers){cachedV.resize(c.layers);cachedG.resize(c.layers);}const uint32_t active_layers=(layer_limit&&layer_limit<c.layers)?layer_limit:c.layers;for(uint32_t l=0;l<active_layers;++l){auto p="layer."+std::to_string(layerSource[l])+".";
#ifdef TAO_PHASE_TIMING
ScopedPhase ph(ms_norm);
#endif
{TAO_T(ms_normf,(norm_into(x,p+"input.norm",rb_.xn),0));}Vec& xn=rb_.xn;auto&s=state[l].s;auto&m=state[l].m;
#ifdef TAO_PHASE_TIMING
ph.arm(ms_s);
#endif
// mHC 式依赖错位（01 号文档 #6）：m 分支改读本层更新前的 s，
// 从而消除 s→m 的串行依赖，使 s(rows=128) 与 m(rows=512) 合并进同一次 dispatch。
// 融合时每层 dispatch 数 3 -> 2，且单次 dispatch 的 MAC 量显著变大（§44.4 的粒度问题）。
// 【TAO_DELTA_MEM 下禁止融合】融合组引用 m.candidate.*/m.gate.*，而 dual-state-4 的
// schema 用 mem.key/query/value 取代了它们（见 dual_state_config.hpp）。若照常融合，
// group<10> 会对不存在的键调用 packed.at()，抛 "invalid map<K, T> key"（实测：默认崩溃，
// TAO_FUSE_SM=0 正常）。这是融合（§44.6）与 H2R 记忆组合后从未被测出的潜伏缺陷。
#ifdef TAO_DELTA_MEM
const bool can_fuse=false;
#else
const bool can_fuse=fuse_sm_&&!(reuse_mv&&!cachedV.empty()&&layerSource[l]!=l);
#endif
Vec s_old;
std::array<Vec,4> st;std::array<Vec,6> mt6;bool have_mt6=false;
if(can_fuse){s_old.assign(s.data(),s.data()+s.size());auto st10=TAO_T(ms_grp,group<10>(
    {p+"s.candidate.x",p+"s.candidate.s",p+"s.gate.x",p+"s.gate.s",
     p+"m.candidate.x",p+"m.candidate.s",p+"m.candidate.m",
     p+"m.gate.x",p+"m.gate.s",p+"m.gate.m"},
    {inx(xn),inx(s_old),inx(xn),inx(s_old),inx(xn),inx(s_old),inx(m),inx(xn),inx(s_old),inx(m)},c.s));
    for(size_t i=0;i<4;++i)st[i]=std::move(st10[i]);
    for(size_t i=0;i<6;++i)mt6[i]=std::move(st10[4+i]);have_mt6=true;
    {TAO_T(ms_addop,(add(st[0],st[1]),add(st[0],w.at(p+"s.candidate.bias")),add(st[2],st[3]),add(st[2],w.at(p+"s.gate.bias")),0));}
    if(fast_act_)fast_gated_update(s.data(),st[0].data(),st[2].data(),s.size());
    else for(size_t j=0;j<s.size();++j)s[j]+=sigmoid(st[2][j])*(std::tanh(st[0][j])-s[j]);}
else{
    {TAO_T(ms_grp,(group_into<4>({p+"s.candidate.x",p+"s.candidate.s",p+"s.gate.x",p+"s.gate.s"},{inx(xn),inx(s),inx(xn),inx(s)},rb_.g4),0));}
    {TAO_T(ms_addop,(add(rb_.g4[0],rb_.g4[1]),add(rb_.g4[0],w.at(p+"s.candidate.bias")),add(rb_.g4[2],rb_.g4[3]),add(rb_.g4[2],w.at(p+"s.gate.bias")),0));}
    if(fast_act_)fast_gated_update(s.data(),rb_.g4[0].data(),rb_.g4[2].data(),s.size());
    else for(size_t j=0;j<s.size();++j)s[j]+=sigmoid(rb_.g4[2][j])*(std::tanh(rb_.g4[0][j])-s[j]);}
#ifdef TAO_PHASE_TIMING
ph.arm(ms_m);
#endif
#ifdef TAO_DELTA_MEM
const uint32_t dk=c.dk,dv=c.m;
{TAO_T(ms_grp,(group_into<2>({p+"mem.key",p+"mem.query"},{inx(xn),inx(xn)},rb_.g2),0));}
Vec& k=rb_.g2[0];Vec& q=rb_.g2[1];linear_into(p+"mem.value",xn,dv,rb_.v);Vec& v=rb_.v;
float kn=0;for(float z:k)kn+=z*z;kn=1.0f/(std::sqrt(kn)+1e-6f);
#ifdef TAO_CPU_AVX2
{size_t jj=0;const __m256 vkn=_mm256_set1_ps(kn);
for(;jj+8<=k.size();jj+=8)_mm256_storeu_ps(k.data()+jj,_mm256_mul_ps(_mm256_loadu_ps(k.data()+jj),vkn));
for(;jj<k.size();++jj)k[jj]*=kn;}
#else
for(float&z:k)z*=kn;
#endif
const float beta=sigmoid(dot_float(p+"mem.beta",xn)+w.at(p+"mem.beta.bias")[0]);
Vec& o=rb_.o;
float kq=0;for(uint32_t j=0;j<dk;++j)kq+=k[j]*q[j];
dispatch_rows(dv,dk,[&](size_t begin,size_t end){for(size_t i=begin;i<end;++i){
    if(i+2<end){const char* nxt=reinterpret_cast<const char*>(m.data()+(i+2)*size_t(dk));
        _mm_prefetch(nxt,_MM_HINT_T0);_mm_prefetch(nxt+64,_MM_HINT_T0);
        _mm_prefetch(nxt+128,_MM_HINT_T0);_mm_prefetch(nxt+192,_MM_HINT_T0);}
    float* row=m.data()+i*dk; float acc=0, rd=0; uint32_t j=0;
#ifdef TAO_CPU_AVX2
    __m256 vacc=_mm256_setzero_ps(), vrd=_mm256_setzero_ps();
    for(;j+8u<=dk;j+=8u){
        __m256 mr=_mm256_loadu_ps(row+j);
        vacc=_mm256_add_ps(vacc,_mm256_mul_ps(mr,_mm256_loadu_ps(k.data()+j)));
        vrd =_mm256_add_ps(vrd ,_mm256_mul_ps(mr,_mm256_loadu_ps(q.data()+j)));
    }
    alignas(32) float ta[8], tr[8];
    _mm256_store_ps(ta,vacc); _mm256_store_ps(tr,vrd);
    for(int t=0;t<8;++t){acc+=ta[t];rd+=tr[t];}
#endif
    for(;j<dk;++j){const float mv=row[j]; acc+=mv*k[j]; rd+=mv*q[j];}
    const float g=beta*(v[i]-acc);
    o[i]=rd+g*kq;
    j=0;
#ifdef TAO_CPU_AVX2
    const __m256 vg=_mm256_set1_ps(g);
    for(;j+8u<=dk;j+=8u){
        __m256 mr=_mm256_loadu_ps(row+j);
        _mm256_storeu_ps(row+j,_mm256_add_ps(mr,_mm256_mul_ps(vg,_mm256_loadu_ps(k.data()+j))));
    }
#endif
    for(;j<dk;++j)row[j]+=g*k[j];
}});
#ifdef TAO_PHASE_TIMING
ph.arm(ms_read);
#endif
linear_into(p+"read.s",s,c.d,rb_.r);add(rb_.r,o);norm_into(rb_.r,p+"read.norm",rb_.nrm);add(x,rb_.nrm);
if(l+1<active_layers)prefetch_layer_l2(state[l+1]);
#else
Vec v,g;
if(reuse_mv&&!cachedV.empty()&&layerSource[l]!=l){v=cachedV[layerSource[l]];g=cachedG[layerSource[l]];}
else if(have_mt6){v=std::move(mt6[0]);g=std::move(mt6[3]);
{TAO_T(ms_addop,(add(v,mt6[1]),add(v,mt6[2]),add(v,w.at(p+"m.candidate.bias")),add(g,mt6[4]),add(g,mt6[5]),add(g,w.at(p+"m.gate.bias")),0));}
if(reuse_mv&&!cachedV.empty()){cachedV[l]=v;cachedG[l]=g;}}
else{auto mt=TAO_T(ms_grp,group<6>({p+"m.candidate.x",p+"m.candidate.s",p+"m.candidate.m",p+"m.gate.x",p+"m.gate.s",p+"m.gate.m"},{inx(xn),inx(s),inx(m),inx(xn),inx(s),inx(m)},c.m));v=std::move(mt[0]);g=std::move(mt[3]);
{TAO_T(ms_addop,(add(v,mt[1]),add(v,mt[2]),add(v,w.at(p+"m.candidate.bias")),add(g,mt[4]),add(g,mt[5]),add(g,w.at(p+"m.gate.bias")),0));}if(reuse_mv&&!cachedV.empty()){cachedV[l]=v;cachedG[l]=g;}}if(fast_act_)fast_gated_update(m.data(),v.data(),g.data(),m.size());
else for(size_t j=0;j<m.size();++j)m[j]+=sigmoid(g[j])*(std::tanh(v[j])-m[j]);
#ifdef TAO_PHASE_TIMING
ph.arm(ms_read);
#endif
auto rt=TAO_T(ms_grp,group<2>({p+"read.s",p+"read.m"},{inx(s),inx(m)},c.d));auto r=std::move(rt[0]);
{TAO_T(ms_addop,(add(r,rt[1]),0));}
{TAO_T(ms_normf,(add(x,norm(r,p+"read.norm")),0));}
#endif
#ifdef TAO_PHASE_TIMING
ph.flush();
#endif
#ifndef TAO_NO_FFN
auto f=linear(p+"ff.up",norm(x,p+"ff.norm"),c.e);for(float&z:f)z*=sigmoid(z);add(x,linear(p+"ff.down",f,c.d));
#endif
}
return x;
}

public:
    void advance(uint32_t token,std::vector<LayerState>& state)const {
        (void)recurrent(token,state);
    }
    Vec step(uint32_t token,std::vector<LayerState>& state)const {
        auto x=recurrent(token,state);
        auto logits=linear("embedding",norm(x,"final.norm"),c.vocab);
        add(logits,w.at("vocab.bias"));return logits;
    }
// Optional generation-only API: does not construct or promise a logits vector.
    // State advances before head validation, as with step followed by a scan.
    uint32_t greedy_step(uint32_t token,std::vector<LayerState>& state)const {
        const uint32_t t=greedy_head(recurrent(token,state));
        prefetch_session_l2(state);
        return t;
    }
    uint32_t greedy_head(const Vec& hidden)const {
        return greedy_head_observe(hidden,[](size_t,float){});
    }
    // Diagnostic observer sees every scalar, including excluded/nonfinite rows.
    // It may run on two threads: write disjoint row slots; do not throw or recurse.
    // Production instantiation has an empty observer and no full logits allocation.
    template<class Observe> uint32_t greedy_head_observe(const Vec& hidden,Observe observe)const {
#ifdef TAO_PHASE_TIMING
        ScopedPhase ph(ms_head);
#endif
        ensure_rb();
        norm_into(hidden,"final.norm",rb_.hn);
        const Vec& x=rb_.hn;
        const auto& p=packed.at("embedding");const auto& bias=w.at("vocab.bias");
        if(p.rows!=c.vocab||p.cols!=x.size()||bias.size()!=p.rows||p.rows==0)
            throw std::runtime_error("greedy head shape");
        // 诊断：截断行数（默认 head_limit_==0 即不截断，行为与原来逐位一致）
        const size_t H=(head_limit_&&head_limit_<p.rows)?size_t(head_limit_):size_t(p.rows);
        struct alignas(64) Partial {
            float value=0;uint32_t token=0;bool found=false,nonfinite=false;
        };
        // Fixed-size and deliberately not value-initialized: only the first
        // thread_count() slots are ever written or read, so only those reset.
        // 32: 贪心头直接调 p.dot，不经过 group/linear，必须在此预量化，
        // 否则工作线程会并发 resize 或读到空 xu_（0xC0000374 / 0xC0000005）。
        {PipelineRows&mp=const_cast<PipelineRows&>(p);if(mp.vnni_)mp.quantize_input(x.data());else mp.qc_x_=nullptr;}
        const size_t active_chunks=size_t(cpu_threads());
        std::array<Partial,CpuRowParallelPool::max_threads> partial;
        for(size_t k=0;k<active_chunks;++k)partial[k]=Partial{};
        auto rows=[&](size_t chunk,size_t begin,size_t end){
            Partial local;
            auto take=[&](size_t r,float value){
                value+=bias[r];
                if(head_adj_)value-=head_adj_[r];
                observe(r,value);
                if(!std::isfinite(value)){local.nonfinite=true;return;}
                if(r==256||r==257||r==258)return;
                if(!local.found||value>local.value||(value==local.value&&r<local.token)){
                    local.value=value;local.token=uint32_t(r);local.found=true;
                }
            };
#ifdef TAO_CPU_AVX2
            if(fast){
                size_t r=begin;
#ifdef TAO_AVX512_KERNEL
                float blk8[8];
                if(p.vnni_&&p.xu_.size()==p.cols){
                    for(;r+8<=end;r+=8){
                        p.vnni8(r,blk8);
                        take(r,blk8[0]);take(r+1,blk8[1]);take(r+2,blk8[2]);take(r+3,blk8[3]);
                        take(r+4,blk8[4]);take(r+5,blk8[5]);take(r+6,blk8[6]);take(r+7,blk8[7]);
                    }
                }
#endif
                float blk[4];
                for(;r+4<=end;r+=4){
                    if(p.vnni_&&p.xu_.size()==p.cols)p.vnni4(r,blk);
                    else p.float4(r,x.data(),blk);
                    take(r,blk[0]);take(r+1,blk[1]);take(r+2,blk[2]);take(r+3,blk[3]);
                }
                for(;r<end;++r)take(r,p.dot(r,x.data()));
            }else
#endif
            for(size_t r=begin;r<end;++r){
                float value=0;
                for(size_t j=0;j<x.size();++j)
                    value+=(float(p.q[r*x.size()+j])*p.scale[r])*x[j];
                take(r,value);
            }
            // The pool calls exactly one task per chunk with a unique chunk
            // index; serial fallback writes chunk 0 only and every other slot
            // stays value-initialized. No assumption about callback arrival
            // order, and a unique destination slot per task.
            partial[chunk]=local;
        };
#ifdef TAO_CPU_AVX2
        if(fast)dispatch(H,p.cols,rows);else
#endif
        rows(0,0,H);
        Partial best;
        for(size_t k=0;k<active_chunks;++k){const auto& part=partial[k];
            if(part.nonfinite)throw std::runtime_error("nonfinite");
            if(part.found&&(!best.found||part.value>best.value||
                (part.value==best.value&&part.token<best.token)))best=part;
        }
        if(!best.found)throw std::runtime_error("no eligible token");
        return best.token;
    }
// Thread count. Numerics are invariant: every row is produced by the same
// per-row kernel call, so any thread count is bitwise identical.
// Two threads keep the original CpuRowParallelExecutor path unchanged; the
// generalized pool is only entered above two.
// n==1 selects a genuine single-thread path. The legacy executor is ALWAYS two
// threads, so before this n==1 and n==2 were silently the same configuration
// (see 02-cpu-decode-throughput.md section 19).
void set_cpu_threads(unsigned n)const{
    if(n<=1u){pool.set_threads(1u);use_legacy=true;single_thread_=true;}
    else if(n==2u){pool.set_threads(1u);use_legacy=true;single_thread_=false;}
    else{pool.set_threads(n);use_legacy=false;single_thread_=false;}
}
unsigned cpu_threads()const{return single_thread_?1u:(use_legacy?2u:pool.thread_count());}
// In-process configuration switching, so A/B comparisons never need two
// processes (cross-process throughput was shown to be unreliable).
void set_layer_share(const std::vector<uint32_t>& s)const{layerSource=s;}
void set_reuse_mv(bool v)const{reuse_mv=v;}
void set_layer_limit(uint32_t n)const{layer_limit=n;}
void set_fast_act(bool v)const{fast_act_=v;}
// 输出头 int8 VNNI（激活量化）。关 TAO_VNNI=0 则回到逐行 float 点积（与旧 checksum 一致）。
void set_vnni(bool v)const{
    if(v==vnni_)return;
    vnni_=v;
    // 分档：>=1e6 MAC（输出头）；层内小矩阵走 4 行打包的 float 点积。
    for(auto&kv:packed){PipelineRows&p=const_cast<PipelineRows&>(kv.second);
        p.vnni_=v&&size_t(p.rows)*size_t(p.cols)>=1000000u;
        if(p.vnni_&&p.rowsum_.empty())p.build_rowsum();}
}
void set_head_limit(uint32_t v)const{head_limit_=v;}
void set_head_adjustment(const float* adj)const{head_adj_=adj;}
unsigned vocab()const{return c.vocab;}
// 权重足迹转储：回答「层共享到底省了多少内存、哪些矩阵还存了多份」
// 真正去重存储：layerSource[l]!=l 的层，其 layer.l.* 权重无人访问（第 197 行已把前缀
// 重映射到源层），故可直接删除，把常驻足迹从 8 层降到「源层数」层。
// std::map 是节点式的，erase 不移动其余元素，对不可移动的 Vec/PipelineRows 安全。
void dropDuplicatedLayers(){
    if(layerSource.size()!=c.layers)return;
    size_t erased=0;
    for(uint32_t l=0;l<c.layers;++l){
        if(layerSource[l]==l)continue;
        const std::string pre="layer."+std::to_string(l)+".";
        for(auto it=packed.begin();it!=packed.end();)if(it->first.rfind(pre,0)==0){it=packed.erase(it);++erased;}else ++it;
        for(auto it=w.begin();it!=w.end();)if(it->first.rfind(pre,0)==0){it=w.erase(it);++erased;}else ++it;
    }
    if(std::getenv("TAO_WEIGHT_DUMP"))std::printf("WD dropped_entries=%zu\n",erased);
}
void dumpWeights()const{
    size_t tot=0,uniq=0;std::map<std::string,size_t> byshape;
    std::printf("WD n_matrices=%zu\n",packed.size());
    for(const auto&kv:packed){const auto&r=kv.second;
        const size_t b=size_t(r.rows)*size_t(r.cols)+size_t(r.rows)*4u;
        tot+=b;std::printf("WD %-30s rows=%6u cols=%6u bytes=%9zu\n",kv.first.c_str(),r.rows,r.cols,b);}
    std::printf("WD TOTAL bytes=%zu (%.2f MB)\n",tot,double(tot)/1048576.0);
}
void set_head2(bool on,uint32_t G,uint32_t C,bool dbg){head2_=on;if(G)head2_G=G;if(C)head2_C=C;head2_debug_=dbg;if(on&&head2_of_.empty())buildHead2();}
// 对 embedding 行做平衡 k-means（行是 三值*scale，用 float 表示）。一次性，结果缓存。
void buildHead2()const{
    const auto&p=packed.at("embedding");
    const uint32_t V=p.rows,D=p.cols,G=head2_G;
    std::vector<float> X(size_t(V)*D);
    for(uint32_t r=0;r<V;++r){const float a=p.scale[r];const int8_t* q=p.q.data()+size_t(r)*D;
        for(uint32_t j=0;j<D;++j)X[size_t(r)*D+j]=float(q[j])*a;}
    head2_cent_.assign(size_t(G)*D,0.0f);head2_of_.assign(V,0);
    // 初始化：均匀抽样 G 行作为初始簇心
    for(uint32_t g=0;g<G;++g){const uint32_t r=(uint64_t(g)*V)/G;
        for(uint32_t j=0;j<D;++j)head2_cent_[size_t(g)*D+j]=X[size_t(r)*D+j];}
    for(int it=0;it<3;++it){
        std::vector<double> sum(size_t(G)*D,0.0);std::vector<uint32_t> cnt(G,0);
        for(uint32_t r=0;r<V;++r){
            const float* xr=X.data()+size_t(r)*D;size_t best=0;float bd=3.4e38f;
            for(uint32_t g=0;g<G;++g){const float* c=head2_cent_.data()+size_t(g)*D;
                float s=0;for(uint32_t j=0;j<D;++j)s+=xr[j]*c[j];
                if(s>bd){bd=s;best=g;}}
            head2_of_[r]=uint32_t(best);cnt[best]++;
            double* sm=sum.data()+best*D;for(uint32_t j=0;j<D;++j)sm[j]+=xr[j];
        }
        for(uint32_t g=0;g<G;++g){if(!cnt[g])continue;float* c=head2_cent_.data()+size_t(g)*D;
            const double* sm=sum.data()+size_t(g)*D;
            for(uint32_t j=0;j<D;++j)c[j]=float(sm[j]/cnt[g]);}
    }
    // CSR
    head2_start_.assign(G+1,0);for(uint32_t r=0;r<V;++r)head2_start_[head2_of_[r]+1]++;
    for(uint32_t g=0;g<G;++g)head2_start_[g+1]+=head2_start_[g];
    head2_members_.assign(V,0);{std::vector<uint32_t> cur(head2_start_.begin(),head2_start_.end()-1);
        for(uint32_t r=0;r<V;++r)head2_members_[cur[head2_of_[r]]++]=r;}
    // 簇 bias = 成员 bias 均值
    head2_cbias_.assign(G,0.0f);{const auto&b=w.at("vocab.bias");
        for(uint32_t g=0;g<G;++g){double s=0;for(uint32_t k=head2_start_[g];k<head2_start_[g+1];++k)s+=b[head2_members_[k]];
            head2_cbias_[g]=float(s/double(head2_start_[g+1]-head2_start_[g]));}}
    head2_stat_nodiv_.assign(64,0);head2_stat_tot_.assign(64,0);
}
// 固化架构：R3（2 组权重共享）+ R4（v/g 复用）+ fast_act + 分档 VNNI。
// 单项覆盖：TAO_LAYER_SHARE / TAO_LAYER_REUSE_MV / TAO_FAST_ACT=0 / TAO_VNNI=0 / TAO_FUSE_SM=0。
//
// 【R4 统一 · doc 24】fast_act 保持默认 **true**（CPU 解码快 49.5×，是解码路径的既定选择）。
// 此前的问题不是"解码器开了 fast_act"，而是**训练器用的是精确 expf/tanhf** ——
// 训练与推理算子不一致，直接违反 R4。
// 修正方向：让训练器也使用同一 Padé 近似（见 dual_state_cuda_resident.cuh 的 ds_act_*），
// 两侧共用同一式子，从而既保住解码速度，又恢复 R4。
// TAO_FAST_ACT=0 仍可回退到精确激活（对照实验用），但此时训练也必须用 -DTAO_TRAIN_EXACT_ACT。
static bool envOn(const char*n,bool d){const char*v=std::getenv(n);return v?(*v!='0'):d;}
void applyDefaultArch(){
    if(!std::getenv("TAO_LAYER_SHARE"))layerSource=groupSource(c.layers);
    if(!std::getenv("TAO_LAYER_REUSE_MV"))reuse_mv=true;
    set_fast_act(envOn("TAO_FAST_ACT",true));
    set_vnni(envOn("TAO_VNNI",true));
    // 只让词表头（Vd≈8.4M MAC）进线程池。s 组与 mem.value 恰好卡在 262144，
    // 唤醒 8 线程的同步成本高于那点 GEMV。提高阈值 bitwise 一致。
    set_row_parallel_minimum(size_t(1)<<20);
    // 解码旧 8 层 checkpoint 时默认只跑 2 层槽位（与固化深度一致）。
    if(!std::getenv("TAO_LAYER_LIMIT")&&c.layers>2u)layer_limit=2u;
}
// Verification hook for the discipline in section 19.6: an experiment must be
// able to print the configuration that actually ran.
std::string config_dump()const{
    std::string s="threads="+std::to_string(cpu_threads());
    s+=single_thread_?" single":"";
    s+=use_legacy?" legacy":" pool";
    s+=reuse_mv?" reuse_mv":"";
    s+=fast_act_?" fast_act":"";
    s+=vnni_?" vnni=on":" vnni=off";
    s+=" layers="+std::to_string(layer_limit?layer_limit:c.layers)+"/"+std::to_string(c.layers);
    s+=" share=";
    for(size_t i=0;i<layerSource.size();++i)s+=(i?",":"")+std::to_string(layerSource[i]);
    return s;
}
// Diagnostic: minimum rows*cols cost before a projection is dispatched to the
// pool. Rows are never split, so any value is bitwise identical.
void set_row_parallel_minimum(size_t v)const{pool.set_minimum_dispatch_cost(v);}
size_t row_parallel_minimum()const{return pool.minimum_dispatch_cost();}
// Row-range dispatch for callbacks that do not care which chunk they are.
// On the two-thread path this forwards the caller's callable verbatim, so the
// original CpuRowParallelExecutor sees exactly the call it saw before.
template<class F> void dispatch_rows(size_t rows,size_t cols,F fn)const{
    if(single_thread_){fn(size_t(0),rows);return;}
    if(use_legacy)legacy.run(rows,cols,fn);
    else pool.run(rows,cols,[&](size_t,size_t b,size_t e){fn(b,e);});
}
// Chunk-aware dispatch. Chunk identity is positional, exactly as in the
// original two-way split: the caller holds [0,split) and is chunk 0, the worker
// holds [split,rows) and is chunk 1, and the serial fallback [0,rows) is chunk
// 0. Callbacks that write per-chunk slots (the greedy head) rely on this.
template<class F> void dispatch(size_t rows,size_t cols,F fn)const{
    if(single_thread_){fn(size_t(0),size_t(0),rows);return;}
    if(use_legacy)legacy.run(rows,cols,[&](size_t b,size_t e){fn(b==0?size_t(0):size_t(1),b,e);});
    else pool.run(rows,cols,fn);
}
#ifdef TAO_PHASE_TIMING
// Phase accumulators, in seconds, per-layer totals. Diagnostic only.
mutable double ms_norm=0,ms_s=0,ms_m=0,ms_read=0,ms_head=0;
void reset_phases()const{ms_norm=ms_s=ms_m=ms_read=ms_head=0;ms_grp=ms_addop=ms_normf=0;ms_pre=0;}
#endif
// Last members: workers shut down before owned weights are destroyed.
mutable CpuRowParallelExecutor legacy;
mutable CpuRowParallelPool pool{1u};
mutable bool use_legacy=true;
    mutable bool single_thread_=false;
};
}
