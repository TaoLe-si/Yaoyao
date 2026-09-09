#pragma once
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef __CUDACC__
#define TAO_HD __host__ __device__
#else
#define TAO_HD
#endif
namespace tao { namespace ternary {
// Shared arithmetic primitives; callers validate external buffers once.
TAO_HD constexpr bool valid(int x){return x>=-1 && x<=1;}
TAO_HD constexpr int8_t mod3(int x){int r=x%3;return int8_t(r>1?r-3:r< -1?r+3:r);}
TAO_HD constexpr int8_t add(int8_t a,int8_t b){return mod3(int(a)+int(b));}
TAO_HD constexpr int8_t subtract(int8_t a,int8_t b){return mod3(int(a)-int(b));}
// Storage v1: 00=0,01=+1,10=-1,11=INVALID; low bits first.
TAO_HD constexpr uint8_t encode_unchecked(int8_t x){return x==0?0:x==1?1:2;}
TAO_HD constexpr int8_t decode_unchecked(uint8_t code){return code==0?0:code==1?1:-1;}
TAO_HD inline int8_t at_unchecked(const uint8_t* p,size_t i){return decode_unchecked(uint8_t((p[i/4]>>(2*(i%4)))&3));}
inline size_t packed_size(size_t count){return count/4+size_t(count%4!=0);}
inline void validate(const std::vector<uint8_t>& p,size_t count){
 if(p.size()!=packed_size(count))throw std::invalid_argument("ternary packed length");
 for(size_t i=0;i<count;++i)if(((p[i/4]>>(2*(i%4)))&3)==3)throw std::invalid_argument("reserved ternary code");
 if(count%4 && (p.back()>>(2*(count%4)))!=0)throw std::invalid_argument("noncanonical ternary padding");
}
inline std::vector<uint8_t> pack(const std::vector<int8_t>& values){
 std::vector<uint8_t> out(packed_size(values.size()),0);
 for(size_t i=0;i<values.size();++i){if(!valid(values[i]))throw std::invalid_argument("invalid trit");out[i/4]|=uint8_t(encode_unchecked(values[i])<<(2*(i%4)));}return out;
}
inline std::vector<int8_t> unpack(const std::vector<uint8_t>& p,size_t count){validate(p,count);std::vector<int8_t> out(count);for(size_t i=0;i<count;++i)out[i]=at_unchecked(p.data(),i);return out;}
inline void validate_scale(float scale){if(!std::isfinite(scale)||scale<=0)throw std::invalid_argument("scale must be finite positive");}
// Integer dot is NOT mod3; int64 accumulator avoids practical row overflow.
inline int64_t dot(const std::vector<uint8_t>& a,const std::vector<uint8_t>& b,size_t count){
 validate(a,count);validate(b,count);if(count>size_t(std::numeric_limits<int64_t>::max()))throw std::overflow_error("dot length");
 int64_t sum=0;for(size_t i=0;i<count;++i)sum+=int(at_unchecked(a.data(),i))*int(at_unchecked(b.data(),i));return sum;
}
inline double dot_float(const std::vector<uint8_t>& weights,const std::vector<float>& input,float scale){
 validate(weights,input.size());validate_scale(scale);double sum=0;for(size_t i=0;i<input.size();++i){if(!std::isfinite(input[i]))throw std::invalid_argument("nonfinite activation");sum+=double(at_unchecked(weights.data(),i))*input[i];}double out=sum*scale;if(!std::isfinite(out))throw std::overflow_error("dot overflow");return out;
}
struct QuantizedRow {size_t count;float scale;std::vector<uint8_t> codes;};
// Explicit caller-selected row scale; nearest code, half-scale ties toward zero.
// Lossy diagnostic converter, NOT an in-place checkpoint migration or QAT optimizer.
inline QuantizedRow quantize(const std::vector<float>& values,float scale){
 validate_scale(scale);std::vector<int8_t> q; q.reserve(values.size());
 for(float x:values){if(!std::isfinite(x))throw std::invalid_argument("nonfinite weight");q.push_back(double(x)>double(scale)/2?1:double(x)<-double(scale)/2?-1:0);}
 return {values.size(),scale,pack(q)};
}
inline std::vector<float> dequantize(const QuantizedRow& row){validate_scale(row.scale);auto q=unpack(row.codes,row.count);std::vector<float> out(q.size());for(size_t i=0;i<q.size();++i)out[i]=q[i]*row.scale;return out;}
}}
#undef TAO_HD
