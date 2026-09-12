#pragma once
#include <vector>
#include <string>
#include <map>
#include <stdexcept>
#include <cstdint>
#include "language_data_contract.hpp"
namespace tao::text {
struct ByteBpe {
std::vector<std::pair<uint32_t,uint32_t>>merges;
// validate() used to rebuild a 16,123-node std::map on EVERY encode()/decode()
// call, which dominated export time (~8 ms per message). The result is a pure
// function of `merges`, so cache it. fit() resets the cache.
mutable bool validated_=false;
void validate()const{if(validated_)return;if(merges.size()>16121)throw std::runtime_error("vocab");std::map<std::pair<uint32_t,uint32_t>,bool>seen;for(size_t i=0;i<merges.size();++i){auto [a,b]=merges[i];auto valid=[&](uint32_t t){return t<256||(t>=uint32_t(tao::data::FIRST_MERGE)&&t<uint32_t(tao::data::FIRST_MERGE)+i);};if(!valid(a)||!valid(b)||!seen.emplace(merges[i],true).second)throw std::runtime_error("merge");}validated_=true;}
// 合并循环原本在每轮 r 里新建一个 std::vector（每文档 16,121 次分配）。16 线程并行时
// 这些分配在堆锁上互相竞争，实测 16 线程只跑到 2.2 核。改为 thread_local 双缓冲后
// 容量跨轮复用 —— 算法逐位不变（同样的扫描顺序、同样的合并、同样的 FIRST_MERGE+r），
// 只是不再反复申请释放内存。
std::vector<uint32_t> encode(const std::string&s)const{validate();static thread_local std::vector<uint32_t>ids,next;ids.clear();next.clear();ids.reserve(s.size());for(unsigned char c:s)ids.push_back(c);for(size_t r=0;r<merges.size();++r){next.clear();for(size_t j=0;j<ids.size();){if(j+1<ids.size()&&ids[j]==merges[r].first&&ids[j+1]==merges[r].second){next.push_back(uint32_t(tao::data::FIRST_MERGE+r));j+=2;}else next.push_back(ids[j++]);}ids.swap(next);}return ids;}
std::string decode(const std::vector<uint32_t>&ids)const{validate();std::string out;std::vector<uint32_t>stack;for(auto t:ids){stack.push_back(t);while(!stack.empty()){auto x=stack.back();stack.pop_back();if(x<256)out.push_back(char(x));else{if(x<uint32_t(tao::data::FIRST_MERGE)||x-uint32_t(tao::data::FIRST_MERGE)>=merges.size())throw std::runtime_error("not text token");auto p=merges[x-uint32_t(tao::data::FIRST_MERGE)];stack.push_back(p.second);stack.push_back(p.first);}}}return out;}
// Reference fit only; caller supplies training message bodies, never crosses messages.
void fit(const std::vector<std::string>&messages,size_t count){if(count>16121||!merges.empty())throw std::runtime_error("fit contract");validated_=false;std::vector<std::vector<uint32_t>>corpus;for(auto&s:messages)corpus.push_back(encode(s));for(size_t round=0;round<count;++round){std::map<std::pair<uint32_t,uint32_t>,uint64_t>freq;for(auto&doc:corpus)for(size_t j=1;j<doc.size();++j)++freq[{doc[j-1],doc[j]}];if(freq.empty())break;auto best=freq.begin();for(auto it=freq.begin();it!=freq.end();++it)if(it->second>best->second)best=it;auto pair=best->first;uint32_t id=uint32_t(261+merges.size());merges.push_back(pair);for(auto&doc:corpus){std::vector<uint32_t>next;for(size_t j=0;j<doc.size();){if(j+1<doc.size()&&doc[j]==pair.first&&doc[j+1]==pair.second){next.push_back(id);j+=2;}else next.push_back(doc[j++]);}doc.swap(next);}}}
};
}
