#pragma once
#include <vector>
#include <string>
#include <map>
#include <stdexcept>
#include <cstdint>
#include <atomic>
namespace tao::text {
struct ByteBpe {
std::vector<std::pair<uint32_t,uint32_t>>merges;
// validate() used to rebuild a 16,123-node std::map on EVERY encode()/decode()
// call, which dominated export time (~8 ms per message). The result is a pure
// function of `merges`, so cache it. fit() resets the cache.
//
// encode() used to allocate a new std::vector on every merge pass (16,123
// heap ops per message). Parallel export then serialized on the CRT heap
// lock, so TAO_CORPUS_THREADS=8 still showed ~1 core. Presence-skip + a
// reused ping-pong buffer makes encode allocation-free after warmup and
// bitwise-identical to the ordered left-to-right passes.
mutable std::atomic<bool> validated_{false};
ByteBpe()=default;
ByteBpe(const ByteBpe& o):merges(o.merges),validated_(o.validated_.load(std::memory_order_relaxed)){}
ByteBpe(ByteBpe&& o)noexcept:merges(std::move(o.merges)),validated_(o.validated_.load(std::memory_order_relaxed)){}
ByteBpe& operator=(const ByteBpe& o){
 if(this!=&o){merges=o.merges;validated_.store(o.validated_.load(std::memory_order_relaxed),std::memory_order_relaxed);}
 return *this;
}
ByteBpe& operator=(ByteBpe&& o)noexcept{
 if(this!=&o){merges=std::move(o.merges);validated_.store(o.validated_.load(std::memory_order_relaxed),std::memory_order_relaxed);}
 return *this;
}
void invalidate()const{validated_.store(false,std::memory_order_relaxed);}
void validate()const{
 if(validated_.load(std::memory_order_acquire))return;
 if(merges.size()>16123)throw std::runtime_error("vocab");
 std::map<std::pair<uint32_t,uint32_t>,bool>seen;
 for(size_t i=0;i<merges.size();++i){
  auto [a,b]=merges[i];
  auto valid=[&](uint32_t t){return t<256||(t>=261&&t<261+i);};
  if(!valid(a)||!valid(b)||!seen.emplace(merges[i],true).second)throw std::runtime_error("merge");
 }
 validated_.store(true,std::memory_order_release);
}
std::vector<uint32_t> encode_ref(const std::string&s)const{
 validate();
 std::vector<uint32_t>ids;
 ids.reserve(s.size());
 for(unsigned char c:s)ids.push_back(c);
 for(size_t r=0;r<merges.size();++r){
  std::vector<uint32_t>next;
  next.reserve(ids.size());
  for(size_t j=0;j<ids.size();){
   if(j+1<ids.size()&&ids[j]==merges[r].first&&ids[j+1]==merges[r].second){
    next.push_back(uint32_t(261+r));j+=2;
   }else next.push_back(ids[j++]);
  }
  ids.swap(next);
 }
 return ids;
}
std::vector<uint32_t> encode(const std::string&s)const{
 validate();
 std::vector<uint32_t>ids;
 ids.reserve(s.size());
 for(unsigned char c:s)ids.push_back(uint32_t(c));
 if(ids.size()<2||merges.empty())return ids;
 thread_local std::vector<uint32_t> buf;
 thread_local std::vector<uint8_t> has;
 buf.reserve(ids.size());
 has.assign(261+merges.size(),0);
 for(uint32_t t:ids)if(t<has.size())has[t]=1;
 for(size_t r=0;r<merges.size()&&ids.size()>=2;++r){
  const uint32_t a=merges[r].first,b=merges[r].second;
  if(a>=has.size()||b>=has.size()||!has[a]||!has[b])continue;
  buf.clear();
  bool hit=false;
  for(size_t j=0;j<ids.size();){
   if(j+1<ids.size()&&ids[j]==a&&ids[j+1]==b){
    buf.push_back(uint32_t(261+r));j+=2;hit=true;
   }else buf.push_back(ids[j++]);
  }
  if(hit){ids.swap(buf);if(261+r<has.size())has[261+r]=1;}
 }
 return ids;
}
std::string decode(const std::vector<uint32_t>&ids)const{
 validate();
 std::string out;
 std::vector<uint32_t>stack;
 for(auto t:ids){
  stack.push_back(t);
  while(!stack.empty()){
   auto x=stack.back();stack.pop_back();
   if(x<256)out.push_back(char(x));
   else{
    if(x<261||x-261>=merges.size())throw std::runtime_error("not text token");
    auto p=merges[x-261];
    stack.push_back(p.second);stack.push_back(p.first);
   }
  }
 }
 return out;
}
// Reference fit only; caller supplies training message bodies, never crosses messages.
void fit(const std::vector<std::string>&messages,size_t count){
 if(count>16123||!merges.empty())throw std::runtime_error("fit contract");
 invalidate();
 std::vector<std::vector<uint32_t>>corpus;
 for(auto&s:messages)corpus.push_back(encode(s));
 for(size_t round=0;round<count;++round){
  std::map<std::pair<uint32_t,uint32_t>,uint64_t>freq;
  for(auto&doc:corpus)for(size_t j=1;j<doc.size();++j)++freq[{doc[j-1],doc[j]}];
  if(freq.empty())break;
  auto best=freq.begin();
  for(auto it=freq.begin();it!=freq.end();++it)if(it->second>best->second)best=it;
  auto pair=best->first;
  uint32_t id=uint32_t(261+merges.size());
  merges.push_back(pair);
  invalidate();
  for(auto&doc:corpus){
   std::vector<uint32_t>next;
   for(size_t j=0;j<doc.size();){
    if(j+1<doc.size()&&doc[j]==pair.first&&doc[j+1]==pair.second){next.push_back(id);j+=2;}
    else next.push_back(doc[j++]);
   }
   doc.swap(next);
  }
 }
}
};
}
