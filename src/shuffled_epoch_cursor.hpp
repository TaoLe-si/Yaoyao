#pragma once
#include "pilot_slot_cursor.hpp"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <cstdint>

namespace tao::data {
// Experimental CPU helper. Does not modify PilotCursor or any trainer.
// docs remain in canonical order; next is an allocation POSITION in order.
class ShuffledEpochCursor {
    PilotCursor cursor_;
    uint64_t seed_, epoch_=0;
    bool shuffled_=false;
    std::string identity_;
    std::vector<size_t> order_;
    static uint64_t draw(uint64_t& s) {
        uint64_t z=(s+=UINT64_C(0x9e3779b97f4a7c15));
        z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
        z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);
        return z^(z>>31);
    }
    void validate() const {
        const size_t n=cursor_.docs.size(), none=std::numeric_limits<size_t>::max();
        if(order_.size()!=n || cursor_.next>n) throw std::runtime_error("epoch size/next");
        std::vector<size_t> rank(n,none); std::vector<bool> active(n,false);
        for(size_t i=0;i<n;++i) {
            if(order_[i]>=n || rank[order_[i]]!=none) throw std::runtime_error("epoch permutation");
            rank[order_[i]]=i;
        }
        if(!shuffled_) for(size_t i=0;i<n;++i)
            if(order_[i]!=i) throw std::runtime_error("legacy order");
        for(const auto& c:cursor_.slots) {
            if(c.doc==none) { if(c.target!=1) throw std::runtime_error("empty target"); continue; }
            if(c.doc>=n || rank[c.doc]>=cursor_.next || active[c.doc] || c.target<1 || c.target>cursor_.docs[c.doc].size())
                throw std::runtime_error("epoch slot");
            active[c.doc]=true;
        }
    }
public:
    // Adopt the checkpoint's current epoch exactly, including active slot targets.
    // identity must be a VERIFIED corpus/tokenizer/preprocessing identity, not a path.
    ShuffledEpochCursor(PilotCursor current,uint64_t seed,std::string identity)
        :cursor_(std::move(current)),seed_(seed),identity_(std::move(identity)) {
        if(identity_.empty()) throw std::invalid_argument("identity");
        order_.resize(cursor_.docs.size()); std::iota(order_.begin(),order_.end(),size_t(0)); validate();
    }
    const PilotCursor& cursor() const {return cursor_;}
    const std::vector<size_t>& order() const {return order_;}
    uint64_t epoch() const {return epoch_;}
    bool exhausted() const {
        if(cursor_.next!=cursor_.docs.size()) return false;
        for(const auto& c:cursor_.slots) if(c.doc!=std::numeric_limits<size_t>::max() && c.target<cursor_.docs[c.doc].size()) return false;
        return true;
    }
    // Caller may invoke only AFTER all issued Work has successfully committed.
    // Never called automatically by take(), and never allowed mid-epoch.
    void begin_next_epoch(bool shuffle) {
        if(!exhausted() || cursor_.docs.empty()) throw std::runtime_error("epoch not exhausted");
        if(epoch_==UINT64_MAX) throw std::overflow_error("epoch overflow");
        ++epoch_; shuffled_=shuffle;
        std::iota(order_.begin(),order_.end(),size_t(0));
        uint64_t state=seed_^ (epoch_*UINT64_C(0xd1342543de82ef95));
        if(shuffle) for(size_t i=order_.size();i>1;--i) {
            const uint64_t bound=static_cast<uint64_t>(i), threshold=(uint64_t(0)-bound)%bound;
            uint64_t r; do {r=draw(state);} while(r<threshold);
            std::swap(order_[i-1],order_[static_cast<size_t>(r%bound)]);
        }
        cursor_.next=0; for(auto& c:cursor_.slots)c=Cursor{};
    }
    bool take(size_t slot,size_t width,Work& out) {
        if(slot>=cursor_.slots.size() || !width) throw std::invalid_argument("cursor");
        auto& c=cursor_.slots[slot]; bool reset=false;
        if(c.doc==std::numeric_limits<size_t>::max() || c.target>=cursor_.docs[c.doc].size()) {
            if(cursor_.next==order_.size()) return false;
            c={order_[cursor_.next++],1}; reset=true;
        }
        const size_t begin=c.target,end=begin+std::min(width,cursor_.docs[c.doc].size()-begin);
        out={slot,c.doc,begin,end,reset}; c.target=end; return true;
    }
    // Write into the same committed checkpoint transaction. checkpoint_id must
    // identify exact checkpoint bytes (e.g. verified SHA256), not merely step.
    std::string save(const std::string& checkpoint_id) const {
        validate(); if(checkpoint_id.empty()) throw std::invalid_argument("checkpoint identity");
        std::ostringstream o; o<<"TAO_SHUFFLED_EPOCH_V1 "<<std::quoted(identity_)<<' '<<std::quoted(checkpoint_id)<<' '
            <<seed_<<' '<<epoch_<<' '<<shuffled_<<' '<<order_.size()<<' '<<cursor_.slots.size()<<' '<<cursor_.next<<'\n';
        for(auto d:order_)o<<d<<' '; o<<'\n';
        for(auto c:cursor_.slots)o<<c.doc<<' '<<c.target<<'\n'; return o.str();
    }
    // canonical supplies verified canonical docs, never already-shuffled docs.
    // Slot recurrent states must come from the same verified checkpoint_id.
    static ShuffledEpochCursor restore(PilotCursor canonical,const std::string& text,
            uint64_t expected_seed,const std::string& identity,const std::string& checkpoint_id) {
        std::istringstream in(text); std::string magic,id,cp; uint64_t seed,epoch; int shuffled;
        size_t n,slots,next;
        if(!(in>>magic>>std::quoted(id)>>std::quoted(cp)>>seed>>epoch>>shuffled>>n>>slots>>next)
            || magic!="TAO_SHUFFLED_EPOCH_V1" || id!=identity || cp!=checkpoint_id || cp.empty()
            || seed!=expected_seed || (shuffled!=0 && shuffled!=1) || n!=canonical.docs.size() || slots!=canonical.slots.size())
            throw std::runtime_error("epoch identity/header");
        // Constructor validates the supplied canonical checkpoint's legacy cursor.
        // Use fresh canonical slots here; sidecar is authoritative for scheduling.
        canonical.next=0; for(auto& c:canonical.slots)c=Cursor{};
        ShuffledEpochCursor out(std::move(canonical),seed,identity);
        out.epoch_=epoch;out.shuffled_=shuffled!=0;out.cursor_.next=next;
        for(auto& d:out.order_)if(!(in>>d))throw std::runtime_error("epoch truncated order");
        for(auto& c:out.cursor_.slots)if(!(in>>c.doc>>c.target))throw std::runtime_error("epoch truncated slots");
        in>>std::ws; if(!in.eof())throw std::runtime_error("epoch trailing data");
        out.validate(); return out;
    }
};
}
