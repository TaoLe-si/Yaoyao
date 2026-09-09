#pragma once
#include "tao_ternary.hpp"
#include <array>
#include <optional>
#include <algorithm>
namespace tao {
// Symbolic semantic IDs encoded injectively, NOT learned token embeddings.
// uint32 needs21 base3 digits (3^21 > 2^32); mapped digits0,1,2 -> trits0,+1,-1.
using RelationCode=std::array<uint8_t,6>;
inline RelationCode relation_code(uint32_t id){std::vector<int8_t>q(21);uint64_t n=id;for(auto&v:q){int d=int(n%3);n/=3;v=d==2?-1:int8_t(d);}auto packed=ternary::pack(q);RelationCode out{};std::copy(packed.begin(),packed.end(),out.begin());return out;}
inline uint32_t relation_id(const RelationCode&code){std::vector<uint8_t>p(code.begin(),code.end());auto q=ternary::unpack(p,21);uint64_t n=0;for(int i=20;i>=0;--i)n=n*3+(q[i]<0?2:q[i]);if(n>UINT32_MAX)throw std::invalid_argument("semantic ID overflow");return uint32_t(n);}
enum class MemoryDomain:uint8_t {Fact,Constraint};
enum class MemoryWrite {Inserted,Updated,Evicted,Ignored,Protected,Full};
struct RelationKey {MemoryDomain domain;RelationCode entity,relation;bool operator==(const RelationKey&b)const{return domain==b.domain&&entity==b.entity&&relation==b.relation;}};
class RelationMemory {
 struct Slot {RelationKey key;RelationCode value;};
 // Oldest WRITE at front; reads never refresh retention priority.
 std::vector<Slot> slots_;size_t capacity_;
 static RelationKey key(MemoryDomain d,uint32_t e,uint32_t r){return {d,relation_code(e),relation_code(r)};}
 MemoryWrite write(MemoryDomain domain,uint32_t entity,uint32_t relation,uint32_t value){
 auto k=key(domain,entity,relation);Slot incoming{k,relation_code(value)};
 auto found=std::find_if(slots_.begin(),slots_.end(),[&](const Slot&s){return s.key==k;});
 if(found!=slots_.end()){slots_.erase(found);slots_.push_back(incoming);return MemoryWrite::Updated;}
 auto result=MemoryWrite::Inserted;if(slots_.size()==capacity_){auto victim=std::find_if(slots_.begin(),slots_.end(),[](const Slot&s){return s.key.domain==MemoryDomain::Fact;});if(victim==slots_.end())return MemoryWrite::Full;slots_.erase(victim);result=MemoryWrite::Evicted;}
 slots_.push_back(incoming);return result;
 }
public:
 explicit RelationMemory(size_t capacity):capacity_(capacity){if(!capacity||capacity>1048576)throw std::invalid_argument("slot capacity");slots_.reserve(capacity);}
 size_t size()const{return slots_.size();}size_t capacity()const{return capacity_;}
 // Explicit task controller only. Authority is API separation, not security authentication.
 MemoryWrite set_constraint(uint32_t task,uint32_t property,uint32_t value){return write(MemoryDomain::Constraint,task,property,value);}
 MemoryWrite observe(uint32_t entity,uint32_t relation,uint32_t value,bool relevant=true,MemoryDomain domain=MemoryDomain::Fact){if(domain!=MemoryDomain::Fact)return MemoryWrite::Protected;if(!relevant)return MemoryWrite::Ignored;return write(domain,entity,relation,value);}
 std::optional<uint32_t> query(uint32_t entity,uint32_t relation,MemoryDomain domain=MemoryDomain::Fact)const{auto k=key(domain,entity,relation);for(const auto&s:slots_)if(s.key==k)return relation_id(s.value);return std::nullopt;}
 bool retract(uint32_t entity,uint32_t relation){auto k=key(MemoryDomain::Fact,entity,relation);auto f=std::find_if(slots_.begin(),slots_.end(),[&](const Slot&s){return s.key==k;});if(f==slots_.end())return false;slots_.erase(f);return true;}
 void reset(){slots_.clear();}
};
}
