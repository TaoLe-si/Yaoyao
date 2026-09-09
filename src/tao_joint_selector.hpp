#pragma once
#include "tao_local_association.hpp"
namespace tao { namespace joint {
using Vector=std::vector<double>;
struct Input {Vector original;std::vector<Vector> values;Vector logits;double temperature=1,alpha=.25,beta=.25;};
inline void finite(const Vector&v){for(double x:v)if(!std::isfinite(x))throw std::invalid_argument("selector nonfinite");}
inline void validate(const Input&x){if(x.original.empty()||x.values.size()>1024||x.logits.size()!=1+2*x.values.size())throw std::invalid_argument("selector shape");finite(x.original);finite(x.logits);for(const auto&v:x.values){if(v.size()!=x.original.size())throw std::invalid_argument("value shape");finite(v);}if(!std::isfinite(x.temperature)||x.temperature<=0||!std::isfinite(x.alpha)||x.alpha<0||x.alpha>1||!std::isfinite(x.beta)||x.beta<0||x.beta>1)throw std::invalid_argument("selector hyperparameters");}
// Alternatives:0=uniqueSkip;1+2j=Share(j);2+2j=Add(j).
inline double branch(const Input&x,size_t k,size_t d){return k==0?x.original[d]:local::action_element(x.original[d],x.values[(k-1)/2][d],int((k-1)%2),x.alpha,x.beta);}
// Dedicated inference: no exp, probabilities or soft branch mixture.
// Strict wrapper validates all input values; lazy variant only validates fetched content.
struct HardOutput {Vector value;size_t choice;};
template<class Fetch>inline HardOutput hard_lazy(const Vector&original,const Vector&logits,size_t candidates,Fetch fetch,double alpha=.25,double beta=.25){
 if(original.empty()||candidates>1024||logits.size()!=1+2*candidates)throw std::invalid_argument("hard selector shape");finite(original);finite(logits);
 if(!std::isfinite(alpha)||alpha<0||alpha>1||!std::isfinite(beta)||beta<0||beta>1)throw std::invalid_argument("hard strengths");
 size_t choice=0;for(size_t k=1;k<logits.size();++k)if(logits[k]>logits[choice])choice=k;
 if(choice==0)return {original,0};
 auto value=fetch((choice-1)/2);if(value.size()!=original.size())throw std::invalid_argument("selected content shape");finite(value);
 Vector out(original.size());for(size_t d=0;d<out.size();++d)out[d]=local::action_element(original[d],value[d],int((choice-1)%2),alpha,beta);finite(out);return {std::move(out),choice};
}
inline HardOutput hard_forward(const Input&x){validate(x);return hard_lazy(x.original,x.logits,x.values.size(),[&](size_t j)->const Vector&{return x.values[j];},x.alpha,x.beta);}
struct Output {Vector probability,soft,hard;size_t choice;};
inline Output forward(const Input&x){validate(x);Output y;size_t n=x.logits.size(),D=x.original.size();y.probability.resize(n);y.soft.assign(D,0);y.choice=0;for(size_t k=1;k<n;++k)if(x.logits[k]>x.logits[y.choice])y.choice=k;double mx=x.logits[y.choice],sum=0;for(size_t k=0;k<n;++k){y.probability[k]=std::exp((x.logits[k]-mx)/x.temperature);sum+=y.probability[k];}for(auto&p:y.probability)p/=sum;y.hard.resize(D);for(size_t d=0;d<D;++d){y.hard[d]=branch(x,y.choice,d);for(size_t k=0;k<n;++k)y.soft[d]+=y.probability[k]*branch(x,k,d);}finite(y.soft);finite(y.hard);return y;}
struct Gradient {Vector logits,original;std::vector<Vector>values;double alpha=0,beta=0;};
// Exact derivative of SOFT output given upstream gradient. Not derivative of argmax.
inline Gradient backward(const Input&x,const Vector&upstream){auto y=forward(x);if(upstream.size()!=x.original.size())throw std::invalid_argument("upstream shape");finite(upstream);Gradient g;g.logits.resize(x.logits.size());g.original.assign(x.original.size(),0);g.values.assign(x.values.size(),Vector(x.original.size(),0));for(size_t k=0;k<x.logits.size();++k){double influence=0;for(size_t d=0;d<upstream.size();++d)influence+=upstream[d]*(branch(x,k,d)-y.soft[d]);g.logits[k]=y.probability[k]*influence/x.temperature;}for(size_t d=0;d<upstream.size();++d){g.original[d]=y.probability[0]*upstream[d];for(size_t j=0;j<x.values.size();++j){double ps=y.probability[1+2*j],pa=y.probability[2+2*j];g.original[d]+=(ps*(1-x.alpha)+pa)*upstream[d];g.values[j][d]=(ps*x.alpha+pa*x.beta)*upstream[d];g.alpha+=ps*(x.values[j][d]-x.original[d])*upstream[d];g.beta+=pa*x.values[j][d]*upstream[d];}}finite(g.logits);finite(g.original);for(auto&v:g.values)finite(v);if(!std::isfinite(g.alpha)||!std::isfinite(g.beta))throw std::overflow_error("gradient overflow");return g;}
// Shared action head conversion:Skip(current,zero),Share/Add(current,candidate)+match.
// Does not call legacy argmax first: ALL candidate logits participate in training.
inline Vector scores(const TernaryEmbedding&head,const std::vector<float>&current,const std::vector<std::vector<float>>&values,const Vector&match){if(values.size()!=match.size())throw std::invalid_argument("matching shape");finite(match);auto skip=local::action_scores(head,current,std::vector<float>(current.size(),0));Vector z{skip[0]};for(size_t j=0;j<values.size();++j){auto a=local::action_scores(head,current,values[j]);z.push_back(match[j]+a[1]);z.push_back(match[j]+a[2]);}finite(z);return z;}
// Explicit surrogate for q=nearest ternary(w/scale); NOT mathematical quantizer derivative.
inline double clipped_ste(double master,double scale,double gradient){if(!std::isfinite(master)||!std::isfinite(scale)||scale<=0||!std::isfinite(gradient))throw std::invalid_argument("STE input");return std::abs(master)<=scale?gradient:0;}
}}
