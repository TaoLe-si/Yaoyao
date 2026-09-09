#include "tao_relation_memory.hpp"
#include <cstdio>
#include <map>
#include <random>
void check(bool b){if(!b)throw std::runtime_error("test failure");}
int main(){try{using namespace tao;
for(uint32_t id:{0u,1u,1023u,1024u,8191u,UINT32_MAX})check(relation_id(relation_code(id))==id);std::mt19937 rng(42);for(int i=0;i<10000;++i){uint32_t id=rng();check(relation_id(relation_code(id))==id);}
auto rejects=[](auto f){bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}check(caught);};rejects([]{RelationMemory bad(0);});rejects([]{auto c=relation_code(0);c[0]=3;relation_id(c);});rejects([]{auto c=relation_code(0);c[5]=4;relation_id(c);});rejects([]{RelationCode c; c.fill(0xaa);c[5]=2;relation_id(c);});
RelationMemory m(4);m.set_constraint(0,1,100);m.observe(10,1,20);m.observe(11,1,21);check(m.observe(10,1,22)==MemoryWrite::Updated);check(m.query(10,1)==22&&m.query(11,1)==21);
for(int i=0;i<10000;++i)check(m.observe(rng(),rng(),rng(),false)==MemoryWrite::Ignored);check(m.size()==3&&m.query(10,1)==22);
check(m.observe(0,1,999,true,MemoryDomain::Constraint)==MemoryWrite::Protected);check(m.query(0,1,MemoryDomain::Constraint)==100);
m.observe(12,1,30);check(m.observe(13,1,31)==MemoryWrite::Evicted);check(!m.query(11,1)&&m.query(10,1)==22);check(m.retract(10,1)&&!m.query(10,1));m.set_constraint(0,1,101);check(m.query(0,1,MemoryDomain::Constraint)==101);
RelationMemory full(1);full.set_constraint(0,1,3);check(full.observe(10,1,5)==MemoryWrite::Full);check(!full.query(10,1)&&full.size()==1);full.reset();check(full.size()==0);
printf("PASS directed updates,entity isolation,10000 ignored distractors,constraint protection,eviction,unknown,retract,reset\n");
// Independent map oracle with monotonic write age, not vector order.
using Key=std::pair<uint32_t,uint32_t>;struct Ref{uint32_t v;uint64_t age;};std::map<Key,Ref> ref;RelationMemory random(8);random.set_constraint(999,1,17);uint64_t age=0;
for(int step=0;step<50000;++step){uint32_t e=rng()%16,r=rng()%4,v=rng();Key k{e,r};int op=rng()%5;if(op<3){bool relevant=op!=2;auto got=random.observe(e,r,v,relevant);if(!relevant)check(got==MemoryWrite::Ignored);else{bool existed=ref.count(k)!=0;bool evicted=false;if(!existed&&ref.size()==7){auto victim=ref.begin();for(auto it=ref.begin();it!=ref.end();++it)if(it->second.age<victim->second.age)victim=it;ref.erase(victim);evicted=true;}ref[k]={v,++age};check(got==(existed?MemoryWrite::Updated:evicted?MemoryWrite::Evicted:MemoryWrite::Inserted));}}else if(op==3){check(random.retract(e,r)==bool(ref.erase(k)));}
for(uint32_t a=0;a<16;++a)for(uint32_t b=0;b<4;++b){auto x=random.query(a,b);auto it=ref.find({a,b});check(bool(x)==(it!=ref.end()));if(x)check(*x==it->second.v);}check(random.query(999,1,MemoryDomain::Constraint)==17);check(random.size()==ref.size()+1);}
printf("PASS 50000 randomized events against independent age-map oracle;3200000 fact queries;protected constraint retained\n");
// Human-readable semantic scenario; no NLP parser is claimed.
RelationMemory demo(8);demo.observe(1,1,10);demo.observe(1,1,11);demo.observe(1,1,12);demo.observe(2,1,10);check(demo.query(1,1)==12&&demo.query(2,1)==10);printf("DEMO semantic IDs: red_ball.location=cabinet;blue_ball.location=Xiaoming;missing_entity=unknown\n");return 0;}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
