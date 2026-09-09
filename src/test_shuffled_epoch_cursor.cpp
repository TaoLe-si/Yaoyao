// CPU-only, no training, no CUDA. C++17. Assertions remain active under NDEBUG.
#include "shuffled_epoch_cursor.hpp"
#include "shuffled_epoch_batch_plan.hpp"
#include <iostream>
#include <tuple>
#include <fstream>
#include "bpe_pilot_reader.hpp"
using namespace tao::data;
static void check(bool ok){if(!ok)throw std::runtime_error("test failed");}
template<class F> static void rejects(F f){bool yes=false;try{f();}catch(const std::exception&){yes=true;}check(yes);}
static PilotCursor fixture(){
    std::vector<std::vector<Token>> docs;
    for(int d=0;d<9;++d){std::vector<Token> t{{BOS,false},{USER,false}};
        for(int i=0;i<d+2;++i)t.push_back({30+d*10+i,false});
        t.push_back({TURN_END,false});t.push_back({ASSISTANT,false});
        for(int i=0;i<d+1;++i)t.push_back({140+d*10+i,true});
        t.push_back({TURN_END,true});docs.push_back(t);}
    return PilotCursor(docs,3,false);
}
using Event=std::tuple<size_t,size_t,size_t,size_t,bool,int,int,bool>;
static std::vector<Event> drain(ShuffledEpochCursor& c){
    std::vector<Event> events; bool any=true;
    while(any){any=false;for(size_t s=0;s<3;++s){Work w;if(!c.take(s,3,w))continue;any=true;
        const auto& doc=c.cursor().docs[w.doc];
        for(size_t j=w.begin;j<w.end;++j)events.emplace_back(w.slot,w.doc,j,w.end,w.reset,doc[j-1].id,doc[j].id,doc[j].loss);
    }}return events;
}
static void full_epoch(ShuffledEpochCursor& c){
    auto events=drain(c);auto original=fixture();
    std::vector<std::vector<int>> seen;std::vector<int> resets(9);
    for(auto& d:original.docs)seen.emplace_back(d.size(),0);
    for(auto e:events){auto [s,d,j,end,reset,input,target,loss]=e;
        check(d<9 && j>0 && j<original.docs[d].size());++seen[d][j];
        check(input==original.docs[d][j-1].id && target==original.docs[d][j].id && loss==original.docs[d][j].loss);
        if(reset && j==1)++resets[d];
        (void)s;(void)end;
    }
    for(size_t d=0;d<9;++d){check(resets[d]==1);for(size_t j=1;j<seen[d].size();++j)check(seen[d][j]==1);}
    check(c.exhausted());
}
struct Exposure {size_t steps=0,positions=0,supervised=0,docs=0,min_supervised=std::numeric_limits<size_t>::max(),max_supervised=0;};
template<class C> static Exposure exposure(C& c,const std::vector<std::vector<Token>>& docs){
    Exposure e;for(;;){size_t positions=0,supervised=0;
        for(size_t r=0;r<8;++r)for(size_t slot=0;slot<4;++slot){Work w;if(!c.take(slot,256,w))continue;
            e.docs+=w.reset;positions+=w.end-w.begin;
            for(size_t j=w.begin;j<w.end;++j)supervised+=docs[w.doc][j].loss;
        }
        if(!positions)break;++e.steps;e.positions+=positions;e.supervised+=supervised;
        e.min_supervised=std::min(e.min_supervised,supervised);e.max_supervised=std::max(e.max_supervised,supervised);
    }return e;
}
static void print_exposure(const char* label,const Exposure& e){
    std::cout<<label<<" steps="<<e.steps<<" positions="<<e.positions<<" supervised="<<e.supervised
        <<" docs="<<e.docs<<" min_step_supervised="<<e.min_supervised<<" max_step_supervised="<<e.max_supervised
        <<" padded_occupancy="<<(e.steps?double(e.positions)/(e.steps*4*8*256):0)<<'\n';
}
static void compare_exposures(std::vector<std::vector<Token>> docs){
    PilotCursor original(docs,4,false);auto fixed=exposure(original,docs);print_exposure("original",fixed);
    ShuffledEpochCursor permuted(PilotCursor(docs,4,false),123,"cpu-exposure-fixture-not-a-checkpoint");
    auto inherited=exposure(permuted,docs);check(inherited.steps==fixed.steps && inherited.positions==fixed.positions && inherited.supervised==fixed.supervised);
    for(size_t epoch=1;epoch<=4;++epoch){permuted.begin_next_epoch(true);auto shuffled=exposure(permuted,docs);
        std::cout<<"epoch="<<epoch<<' ';print_exposure("permutation",shuffled);
        check(shuffled.positions==fixed.positions && shuffled.supervised==fixed.supervised && shuffled.docs==fixed.docs);
    }
}
int main(int argc,char** argv){try{
    // Optional read-only real-corpus exposure audit: test.exe corpus.bin tokenizer_sha256
    // Reader checks embedded tokenizer identity, NOT cryptographic corpus provenance.
    if(argc!=1 && argc!=3)throw std::runtime_error("usage: test [bpe_corpus.bin tokenizer_sha256]");
    // Adapter is field-for-field identical to the original fixed-order planner.
    auto batch_original=fixture();ShuffledEpochCursor batch_helper(batch_original,123,"batch-fixture");
    for(;;){auto a=take_batch(batch_original,3),b=take_batch(batch_helper,3);
        check(a.slots==b.slots && a.timesteps==b.timesteps && a.positions==b.positions && a.supervised==b.supervised && a.items.size()==b.items.size());
        for(size_t i=0;i<a.items.size();++i){const auto& x=a.items[i];const auto& y=b.items[i];
            check(x.input==y.input && x.target==y.target && x.active==y.active && x.reset==y.reset && x.loss==y.loss);}
        if(!a.positions)break;
    }
    compare_exposures(fixture().docs);
    if(argc==3){std::ifstream input(argv[1],std::ios::binary);if(!input)throw std::runtime_error("corpus open");
        compare_exposures(read_bpe_pilot(input,argv[2]));}

    const std::string id="fixture-verified-corpus-and-tokenizer-v1", cp="fixture-checkpoint-exact-identity";
    // A mid-epoch legacy cursor is adopted without changing any active doc.
    auto legacy=fixture();Work a,b;check(legacy.take(0,2,a));check(legacy.take(1,3,a));
    ShuffledEpochCursor adopted(legacy,123,id);rejects([&]{adopted.begin_next_epoch(true);});
    for(size_t s=0;s<3;++s){check(legacy.take(s,2,a)==adopted.take(s,2,b));
        check(a.slot==b.slot && a.doc==b.doc && a.begin==b.begin && a.end==b.end && a.reset==b.reset);}
    auto text=adopted.save(cp);auto resumed=ShuffledEpochCursor::restore(fixture(),text,123,id,cp);
    check(drain(adopted)==drain(resumed));
    rejects([&]{ShuffledEpochCursor::restore(fixture(),text,124,id,cp);});
    rejects([&]{ShuffledEpochCursor::restore(fixture(),text,123,id+"bad",cp);});
    rejects([&]{ShuffledEpochCursor::restore(fixture(),text,123,id,cp+"bad");});
    rejects([&]{ShuffledEpochCursor::restore(fixture(),text+"junk",123,id,cp);});
    rejects([&]{ShuffledEpochCursor::restore(fixture(),text.substr(0,text.size()/2),123,id,cp);});
    ShuffledEpochCursor c(fixture(),123,id);full_epoch(c);
    for(int epoch=0;epoch<4;++epoch){
        c.begin_next_epoch(true);check(c.epoch()==uint64_t(epoch+1));
        auto twin=ShuffledEpochCursor::restore(fixture(),c.save(cp),123,id,cp);
        check(c.order()==twin.order());full_epoch(twin);
        // Every interruption point in first 30 round-robin scheduling calls:
        for(size_t stop=0;stop<30;++stop){
            auto left=ShuffledEpochCursor::restore(fixture(),c.save(cp),123,id,cp);
            for(size_t k=0;k<stop;++k)left.take(k%3,2,a);
            auto right=ShuffledEpochCursor::restore(fixture(),left.save(cp),123,id,cp);
            check(drain(left)==drain(right));
        }
        full_epoch(c);
    }
    c.begin_next_epoch(false);for(size_t i=0;i<c.order().size();++i)check(c.order()[i]==i);full_epoch(c);
    // next==N is insufficient: an allocated document may still be active.
    auto pending=fixture();for(size_t k=0;k<pending.docs.size();++k)pending.take(0,1000,a);
    pending.slots[0].target=1;ShuffledEpochCursor active(pending,123,id);
    check(!active.exhausted());rejects([&]{active.begin_next_epoch(true);});
    // Corrupt duplicate permutation must fail closed.
    ShuffledEpochCursor fresh(fixture(),123,id);auto bad=fresh.save(cp);
    auto p=bad.find('\n')+1;bad.replace(p,3,"0 0");
    rejects([&]{ShuffledEpochCursor::restore(fixture(),bad,123,id,cp);});
    std::cout<<"PASS shuffled epoch CPU cursor tests\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
