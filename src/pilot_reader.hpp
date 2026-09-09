#pragma once
#include "language_data_contract.hpp"
#include <istream>
#include <cstdint>
namespace tao::data {
inline unsigned getbyte(std::istream&in){int c=in.get();if(c<0)throw std::runtime_error("truncated pilot");return unsigned(c);}
inline std::vector<std::vector<Token>> read_pilot(std::istream&in){char magic[4];in.read(magic,4);if(!in||std::string(magic,4)!="TLP1")throw std::runtime_error("magic");std::vector<std::vector<Token>> docs;
while(in.peek()!=std::char_traits<char>::eof()){uint32_t n=0;for(int i=0;i<4;++i)n|=getbyte(in)<<(8*i);if(n<4||n>1000000)throw std::runtime_error("length");std::vector<Token>t;for(uint32_t i=0;i<n;++i){unsigned lo=getbyte(in),hi=getbyte(in),mask=getbyte(in);int id=int(lo|(hi<<8));if(id>EOS||mask>1)throw std::runtime_error("token/mask");t.push_back({id,bool(mask)});}if(t.front().id!=BOS||t.front().loss)throw std::runtime_error("BOS");size_t p=1;bool assistant=false;int turns=0;while(p<t.size()&&t[p].id!=EOS){if((t[p].id!=USER&&t[p].id!=ASSISTANT)||t[p].loss)throw std::runtime_error("role");assistant=t[p++].id==ASSISTANT;while(p<t.size()&&t[p].id<256){if(t[p++].loss!=assistant)throw std::runtime_error("body mask");}if(p>=t.size()||t[p].id!=TURN_END||t[p].loss!=assistant)throw std::runtime_error("turn end");++p;++turns;}if(!turns||p+1!=t.size()||t[p].id!=EOS||t[p].loss!=assistant)throw std::runtime_error("EOS");docs.push_back(std::move(t));}if(in.bad())throw std::runtime_error("IO");return docs;}
}
