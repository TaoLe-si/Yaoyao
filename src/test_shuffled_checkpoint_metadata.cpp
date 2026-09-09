#include "shuffled_checkpoint_metadata.hpp"
#include "shuffled_epoch_cursor.hpp"
#include <iostream>
#include <tuple>
using namespace tao::data;
static void require(bool ok) {if(!ok)throw std::runtime_error("test assertion");}
template<class F> static void rejects(F f) {
    bool caught=false; try {f();} catch(const std::exception&) {caught=true;} require(caught);
}
static PilotCursor fixture(size_t slots=3) {
    std::vector<std::vector<Token>> docs;
    for(size_t n:{size_t(3),size_t(11),size_t(5),size_t(8),size_t(4)}) {
        std::vector<Token> d(n,Token{42,true}); d.front()={BOS,false}; d.back()={TURN_END,true};
        docs.push_back(d);
    }
    return PilotCursor(std::move(docs),slots,false);
}
// CPU model of ONLY the existing loader's cursor checks. Not an SCP codec or
// a test of actual disk/tensor loading (that header has CUDA dependencies).
static void loader_cursor_checks(const ShuffledCheckpointMetadata& m,const PilotCursor& data) {
    if(m.next>data.docs.size())throw std::runtime_error("next");
    for(const auto& c:m.cursor) {
        if(c.doc==std::numeric_limits<size_t>::max()) {
            if(c.target!=1)throw std::runtime_error("empty cursor");
        } else if(c.doc>=m.next || c.doc>=data.docs.size() || c.target<1 || c.target>data.docs[c.doc].size())
            throw std::runtime_error("cursor");
    }
}
static void unequal_lengths() {
    auto c=fixture(); const std::vector<size_t> order{1,4,0,3,2};
    ShuffledCheckpointMapping map(c,order);
    c.next=2;c.slots[0]={1,10};c.slots[1]={4,4};
    auto encoded=map.before_save(c.slots,c.next);
    require(encoded.cursor[0].doc==0 && encoded.cursor[1].doc==1 && encoded.next==2);
    require(c.slots[0].doc==1); // no mutation of captured/live metadata
    rejects([&]{loader_cursor_checks(encoded,c);}); // length 3 != length 11
    auto physical=map.load_template(c); loader_cursor_checks(encoded,physical);
    for(size_t r=0;r<order.size();++r)require(physical.docs[r].size()==c.docs[order[r]].size());
    require(physical.next==0 && physical.slots[0].doc==std::numeric_limits<size_t>::max());
    auto decoded=map.after_load(encoded.cursor,encoded.next,c);
    require(decoded.cursor[0].doc==1 && decoded.cursor[0].target==10);
    auto bad=encoded;bad.cursor[0].target=12;rejects([&]{map.after_load(bad.cursor,bad.next,c);});
    bad=encoded;bad.cursor[1]=bad.cursor[0];rejects([&]{map.after_load(bad.cursor,bad.next,c);});
    bad=encoded;bad.cursor[2].target=2;rejects([&]{map.after_load(bad.cursor,bad.next,c);});
    bad=encoded;bad.cursor[0].doc=5;rejects([&]{map.after_load(bad.cursor,bad.next,c);});
    bad=encoded;bad.cursor[1].doc=2;rejects([&]{map.after_load(bad.cursor,bad.next,c);});
    bad=encoded;bad.cursor[0].target=0;rejects([&]{map.after_load(bad.cursor,bad.next,c);});
    rejects([&]{map.after_load(encoded.cursor,6,c);});
    auto mismatch=c;mismatch.slots[0].target=9;
    rejects([&]{map.after_load(encoded.cursor,encoded.next,mismatch);});
    mismatch=c;mismatch.next=3;rejects([&]{map.after_load(encoded.cursor,encoded.next,mismatch);});
    mismatch=c;std::swap(mismatch.slots[0],mismatch.slots[1]);
    rejects([&]{map.after_load(encoded.cursor,encoded.next,mismatch);});
    rejects([&]{ShuffledCheckpointMapping x(c,{1,1,0,3,2});});
    rejects([&]{ShuffledCheckpointMapping x(c,{1,5,0,3,2});});
    rejects([&]{ShuffledCheckpointMapping x(c,{1});});
    bad=encoded;bad.cursor.pop_back();rejects([&]{map.after_load(bad.cursor,bad.next,c);});
    mismatch=c;mismatch.docs[0].push_back({42,true});rejects([&]{map.load_template(mismatch);});
    // Canonical doc 4 is valid despite doc >= next: rank is 1.
    require(c.slots[1].doc>=c.next);
    auto invalid=c;invalid.slots[1]={2,2};rejects([&]{map.before_save(invalid.slots,invalid.next);});
    // Identity order retains existing non-shuffled representation.
    auto legacy=fixture();legacy.next=1;legacy.slots[0]={0,3};
    ShuffledCheckpointMapping identity(legacy,{0,1,2,3,4});
    auto same=identity.before_save(legacy.slots,legacy.next);loader_cursor_checks(same,legacy);
    identity.after_load(same.cursor,same.next,legacy);
}
using Target=std::tuple<size_t,size_t,size_t,int,bool,bool>; // slot,doc,target,token,loss,reset
static void tick(ShuffledEpochCursor& e,size_t tickno,std::vector<Target>& trace) {
    const size_t slot=tickno%e.cursor().slots.size(); Work w{};
    if(e.take(slot,1+tickno%4,w))for(size_t t=w.begin;t<w.end;++t) {
        const auto tok=e.cursor().docs[w.doc][t];
        trace.emplace_back(w.slot,w.doc,t,tok.id,tok.loss,w.reset && t==w.begin);
    }
}
static void resume_boundary(ShuffledEpochCursor& e,uint64_t seed) {
    const auto canonical=e.cursor();
    ShuffledCheckpointMapping map(canonical,e.order());
    const auto disk=map.before_save(canonical.slots,canonical.next);
    auto physical=map.load_template(canonical);loader_cursor_checks(disk,physical);
    // Deliberately unverified label: tests exercise consistency, not security.
    const std::string cp="SYNTHETIC-NOT-A-DIGEST",id="SYNTHETIC-CORPUS";
    auto restored=ShuffledEpochCursor::restore(canonical,e.save(cp),seed,id,cp);
    const auto decoded=map.after_load(disk.cursor,disk.next,restored.cursor());
    require(decoded.next==canonical.next);
    for(size_t s=0;s<canonical.slots.size();++s)
        require(decoded.cursor[s].doc==canonical.slots[s].doc && decoded.cursor[s].target==canonical.slots[s].target);
    e=std::move(restored);
}
static void interrupted_traces() {
    for(uint64_t seed:{UINT64_C(1),UINT64_C(17),UINT64_C(991)})for(size_t slots:{size_t(1),size_t(3),size_t(7)}) {
        auto c=fixture(slots);c.next=c.docs.size(); // adopt an exhausted legacy epoch
        ShuffledEpochCursor initial(c,seed,"SYNTHETIC-CORPUS");initial.begin_next_epoch(true);
        auto reference=initial;std::vector<Target> expected;size_t ticks=0;
        while(!reference.exhausted()){require(ticks<1000);tick(reference,ticks++,expected);}
        // Interrupt at every call boundary, including fresh epoch and fully exhausted.
        for(size_t cut=0;cut<=ticks;++cut) {
            auto resumed=initial;std::vector<Target> actual;
            for(size_t t=0;t<ticks;++t) {if(t==cut)resume_boundary(resumed,seed);tick(resumed,t,actual);}
            if(cut==ticks)resume_boundary(resumed,seed);
            require(actual==expected && resumed.exhausted());
            // Compare the following epoch too, using the restored seed/epoch/order.
            auto nextref=reference;nextref.begin_next_epoch(true);resumed.begin_next_epoch(true);
            std::vector<Target> a,b;size_t t=0;
            while(!nextref.exhausted()) {require(t<1000);tick(nextref,t,a);resume_boundary(resumed,seed);tick(resumed,t++,b);}
            require(a==b && resumed.exhausted() && nextref.order()==resumed.order());
        }
    }
}
int main() {
    try {unequal_lengths();interrupted_traces();std::cout<<"shuffled checkpoint metadata PASS\n";}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
