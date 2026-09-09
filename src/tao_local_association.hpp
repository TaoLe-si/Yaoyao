#pragma once
#include "tao_ternary_embedding.hpp"
#include <optional>
#ifdef __CUDACC__
#define LSA_HD __host__ __device__
#else
#define LSA_HD
#endif
namespace tao { namespace local {
enum class Action:int8_t {Skip=-1,Share=0,Add=1};
// Action code is NOT a numeric multiplier. Inputs validated by host boundary.
LSA_HD inline double action_element(double original,double value,int action,double alpha,double beta){return action==-1?original:action==0?(1-alpha)*original+alpha*value:original+beta*value;}
inline void finite(const std::vector<float>&x){for(float v:x)if(!std::isfinite(v))throw std::invalid_argument("nonfinite feature");}
inline std::vector<float> apply(const std::vector<float>&original,const std::vector<float>&value,Action action,bool has_candidate,float alpha=.25f,float beta=.25f){
 if(!std::isfinite(alpha)||alpha<0||alpha>1||!std::isfinite(beta)||beta<0||beta>1)throw std::invalid_argument("exchange strengths");
 if(int(action)<-1||int(action)>1)throw std::invalid_argument("action code");finite(original);
 if(!has_candidate||action==Action::Skip)return original;
 if(original.size()!=value.size())throw std::invalid_argument("content dimension mismatch");finite(value);std::vector<float>out(original.size());
 for(size_t d=0;d<out.size();++d){double v=action_element(original[d],value[d],int(action),alpha,beta);if(!std::isfinite(v)||std::abs(v)>std::numeric_limits<float>::max())throw std::overflow_error("exchange overflow");out[d]=float(v);}return out;
}
// A shared small ternary projection, represented by output rows x input dimensions.
inline std::vector<float> project(const TernaryEmbedding&p,const std::vector<float>&input){if(p.dims()!=input.size())throw std::invalid_argument("projection dimensions");finite(input);std::vector<float>out(p.rows());for(uint32_t r=0;r<p.rows();++r){const uint8_t*row=p.packed().data()+size_t(r)*p.stride();double sum=0;for(size_t d=0;d<input.size();++d)sum+=ternary::at_unchecked(row,d)*double(input[d]);sum*=p.scales()[r];if(!std::isfinite(sum)||std::abs(sum)>std::numeric_limits<float>::max())throw std::overflow_error("projection overflow");out[r]=float(sum);}return out;}
// Prototype causal feature: own embedding || mean of preceding local embeddings.
// No cumulative chain states, future tokens, or persistent history mutation.
inline std::vector<float> feature(const TernaryEmbedding&e,const std::vector<uint32_t>&ids,size_t position,size_t context){if(position>=ids.size())throw std::out_of_range("feature position");auto out=e.lookup(ids[position]);std::vector<double>mean(e.dims());size_t count=std::min(context,position);for(size_t j=position-count;j<position;++j){auto v=e.lookup(ids[j]);for(size_t d=0;d<v.size();++d)mean[d]+=v[d];}for(double v:mean)out.push_back(count?float(v/count):0);return out;}
struct Selection {std::optional<size_t> index;double score=-2;};
inline double cosine(const std::vector<float>&a,const std::vector<float>&b){if(a.size()!=b.size())throw std::invalid_argument("match dimensions");finite(a);finite(b);double dot=0,aa=0,bb=0;for(size_t d=0;d<a.size();++d){dot+=double(a[d])*b[d];aa+=double(a[d])*a[d];bb+=double(b[d])*b[d];}if(aa==0||bb==0)return -2;return std::max(-1.,std::min(1.,dot/(std::sqrt(aa)*std::sqrt(bb))));}
inline Selection select(const TernaryEmbedding&e,const TernaryEmbedding&p,const std::vector<uint32_t>&ids,size_t current,size_t horizon,size_t context){auto q=project(p,feature(e,ids,current,context));Selection best;size_t count=std::min(horizon,current);for(size_t d=1;d<=count;++d){size_t j=current-d;double s=cosine(q,project(p,feature(e,ids,j,context)));if(s>best.score)best={j,s};}return best;}
// Three logits order:Skip,Share,Add. A learned head supplies these scores later.
inline Action choose(const std::array<float,3>&scores,bool candidate){for(float s:scores)if(!std::isfinite(s))throw std::invalid_argument("action logits");if(!candidate)return Action::Skip;int best=0;for(int i=1;i<3;++i)if(scores[i]>scores[best])best=i;return Action(best-1);}
// Trainable parameter container is shared ternary projection; optimizer not included.
// Head input:current content || selected content. Outputs Skip,Share,Add logits.
inline std::array<float,3> action_scores(const TernaryEmbedding&head,const std::vector<float>&current,const std::vector<float>&selected){if(head.rows()!=3||current.size()!=selected.size())throw std::invalid_argument("action head shape");auto x=current;x.insert(x.end(),selected.begin(),selected.end());auto z=project(head,x);return {z[0],z[1],z[2]};}
struct Result {Selection selected;Action action;std::vector<float>output;};
inline Result forward(const TernaryEmbedding&e,const TernaryEmbedding&p,const std::vector<uint32_t>&ids,size_t current,size_t horizon,size_t context,const std::vector<float>&original,const std::array<float,3>&scores,bool enabled=true){
 if(!enabled)return {{},Action::Skip,apply(original,{},Action::Skip,false)};
 auto selection=select(e,p,ids,current,horizon,context);auto action=choose(scores,selection.index.has_value());auto value=selection.index&&action!=Action::Skip?e.lookup(ids[*selection.index]):std::vector<float>{};
 return {selection,action,apply(original,value,action,selection.index.has_value())};
}
}}
#undef LSA_HD
