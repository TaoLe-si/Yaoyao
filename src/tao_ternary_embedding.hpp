#pragma once
#include "tao_ternary.hpp"
#include <array>
#include <cstring>
#include <algorithm>
namespace tao {
// TTE1 independent container. No implicit conversion of legacy Q1 checkpoints.
class TernaryEmbedding {
 uint32_t rows_,dims_;
 std::array<uint8_t,32> tokenizer_;
 std::vector<float> scales_;
 std::vector<uint8_t> packed_;
 static uint32_t crc(const uint8_t* p,size_t n){uint32_t c=~0u;for(size_t i=0;i<n;++i){c^=p[i];for(int b=0;b<8;++b)c=(c>>1)^(0xedb88320u&uint32_t(-int(c&1)));}return ~c;}
 static size_t storage(uint32_t rows,uint32_t dims){if(!rows||!dims||rows>1048576||dims>65536)throw std::invalid_argument("embedding dimensions");uint64_t n=uint64_t(rows)*ternary::packed_size(dims);if(n>uint64_t(1)<<30)throw std::invalid_argument("embedding allocation limit");return size_t(n);}
public:
 TernaryEmbedding(uint32_t rows,uint32_t dims,std::array<uint8_t,32> tokenizer):rows_(rows),dims_(dims),tokenizer_(tokenizer){size_t n=storage(rows,dims);scales_.assign(rows,1);packed_.assign(n,0);}
 uint32_t rows()const{return rows_;}uint32_t dims()const{return dims_;}size_t stride()const{return ternary::packed_size(dims_);}
 const std::vector<uint8_t>& packed()const{return packed_;}const std::vector<float>& scales()const{return scales_;}
 void set_row(uint32_t id,const std::vector<int8_t>& values,float scale){if(id>=rows_||values.size()!=dims_)throw std::invalid_argument("embedding row");ternary::validate_scale(scale);auto p=ternary::pack(values);std::copy(p.begin(),p.end(),packed_.begin()+size_t(id)*stride());scales_[id]=scale;}
 std::vector<float> lookup(uint32_t id)const{if(id>=rows_)throw std::out_of_range("token ID");std::vector<float> out(dims_);const auto*p=packed_.data()+size_t(id)*stride();for(size_t d=0;d<dims_;++d)out[d]=scales_[id]*ternary::at_unchecked(p,d);return out;}
 std::vector<int8_t> codes(uint32_t id)const{if(id>=rows_)throw std::out_of_range("token ID");std::vector<int8_t> out(dims_);const auto*p=packed_.data()+size_t(id)*stride();for(size_t d=0;d<dims_;++d)out[d]=ternary::at_unchecked(p,d);return out;}
 std::vector<uint8_t> bytes()const{
 static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559,"IEEE754 FP32 required");
 std::vector<uint8_t>b;b.reserve(56+4*size_t(rows_)+packed_.size());auto put=[&](uint32_t v){for(int k=0;k<4;++k)b.push_back(uint8_t(v>>(8*k)));};
 put(0x31455454);put(1);put(rows_);put(dims_);put(uint32_t(stride()));b.insert(b.end(),tokenizer_.begin(),tokenizer_.end());
 for(float s:scales_){uint32_t bits;std::memcpy(&bits,&s,4);put(bits);}b.insert(b.end(),packed_.begin(),packed_.end());put(crc(b.data(),b.size()));return b;
 }
 static TernaryEmbedding load(const std::vector<uint8_t>& b,const std::array<uint8_t,32>& expected_tokenizer){
 if(b.size()<56)throw std::invalid_argument("truncated embedding");
 auto get=[&](size_t o){return uint32_t(b[o])|(uint32_t(b[o+1])<<8)|(uint32_t(b[o+2])<<16)|(uint32_t(b[o+3])<<24);};
 if(get(0)!=0x31455454||get(4)!=1)throw std::invalid_argument("embedding format version");
 uint32_t rows=get(8),dims=get(12);size_t size=storage(rows,dims);if(get(16)!=ternary::packed_size(dims)||b.size()!=56+4*size_t(rows)+size)throw std::invalid_argument("embedding exact length/stride");
 if(!std::equal(expected_tokenizer.begin(),expected_tokenizer.end(),b.begin()+20))throw std::invalid_argument("tokenizer SHA256 mismatch");
 if(get(b.size()-4)!=crc(b.data(),b.size()-4))throw std::invalid_argument("embedding CRC mismatch");
 TernaryEmbedding out(rows,dims,expected_tokenizer);for(size_t r=0;r<rows;++r){uint32_t bits=get(52+4*r);float s;std::memcpy(&s,&bits,4);ternary::validate_scale(s);out.scales_[r]=s;}
 std::copy(b.begin()+52+4*size_t(rows),b.end()-4,out.packed_.begin());
 for(size_t r=0;r<rows;++r){std::vector<uint8_t> row(out.packed_.begin()+r*out.stride(),out.packed_.begin()+(r+1)*out.stride());ternary::validate(row,dims);}return out;
 }
};
}
