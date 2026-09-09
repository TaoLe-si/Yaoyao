#pragma once
#include "pilot_slot_cursor.hpp"

namespace tao::data {
// Experimental CPU-only adapter. No file IO, hashing, CUDA, or tensor mutation.
// Metadata is positional by SLOT: never permute recurrent state/tensor slots.
struct ShuffledCheckpointMetadata {
    std::vector<Cursor> cursor;
    size_t next;
};

class ShuffledCheckpointMapping {
    std::vector<size_t> order_, rank_, lengths_;
    size_t slots_;
    static constexpr size_t none = std::numeric_limits<size_t>::max();
    void validate(const ShuffledCheckpointMetadata& m, bool physical) const {
        if(m.next>order_.size() || m.cursor.size()!=slots_)
            throw std::runtime_error("shuffle checkpoint shape/next");
        std::vector<bool> used(order_.size(),false);
        for(const auto& c:m.cursor) {
            if(c.doc==none) {
                if(c.target!=1) throw std::runtime_error("shuffle empty target");
                continue;
            }
            if(c.doc>=order_.size()) throw std::runtime_error("shuffle doc range");
            const size_t canonical=physical?order_[c.doc]:c.doc;
            if(rank_[canonical]>=m.next || used[canonical] || c.target<1 || c.target>lengths_[canonical])
                throw std::runtime_error("shuffle checkpoint cursor");
            used[canonical]=true;
        }
    }
public:
    ShuffledCheckpointMapping(const PilotCursor& canonical, std::vector<size_t> order)
        :order_(std::move(order)),rank_(canonical.docs.size(),none),slots_(canonical.slots.size()) {
        if(order_.size()!=rank_.size() || !slots_) throw std::runtime_error("shuffle mapping shape");
        for(size_t r=0;r<order_.size();++r) {
            if(order_[r]>=rank_.size() || rank_[order_[r]]!=none)
                throw std::runtime_error("shuffle mapping permutation");
            rank_[order_[r]]=r;
        }
        for(const auto& d:canonical.docs) lengths_.push_back(d.size());
    }
    // Apply returned cursor/next to a COPY of captured SlotSnapshot before save_slot_file.
    ShuffledCheckpointMetadata before_save(const std::vector<Cursor>& cursor,size_t next) const {
        ShuffledCheckpointMetadata m{cursor,next}; validate(m,false);
        for(auto& c:m.cursor) if(c.doc!=none) c.doc=rank_[c.doc];
        validate(m,true); return m;
    }
    // Pass this physical-order data template to EXISTING load_slot_file. That loader
    // validates target against data.docs[rank]. Canonical docs would be incorrect.
    // Copy the already-normalized cursor: do NOT invoke the legacy EOS constructor.
    PilotCursor load_template(const PilotCursor& canonical) const {
        if(canonical.docs.size()!=order_.size() || canonical.slots.size()!=slots_)
            throw std::runtime_error("shuffle template shape");
        for(size_t d=0;d<lengths_.size();++d) if(canonical.docs[d].size()!=lengths_[d])
            throw std::runtime_error("shuffle template lengths");
        auto physical=canonical;
        for(size_t r=0;r<order_.size();++r) physical.docs[r]=canonical.docs[order_[r]];
        physical.next=0; for(auto& c:physical.slots)c=Cursor{};
        return physical;
    }
    // AFTER existing load, BEFORE any restore/GPU mutation: decode and compare
    // every canonical slot and next with restored canonical shuffled sidecar.
    // Sidecar identity strings alone are NOT cryptographic verification.
    ShuffledCheckpointMetadata after_load(const std::vector<Cursor>& cursor,size_t next,
                                         const PilotCursor& canonical_sidecar) const {
        ShuffledCheckpointMetadata m{cursor,next}; validate(m,true);
        for(auto& c:m.cursor) if(c.doc!=none)c.doc=order_[c.doc];
        validate(m,false);
        if(canonical_sidecar.docs.size()!=lengths_.size()) throw std::runtime_error("shuffle sidecar docs");
        for(size_t d=0;d<lengths_.size();++d) if(canonical_sidecar.docs[d].size()!=lengths_[d])
            throw std::runtime_error("shuffle sidecar lengths");
        ShuffledCheckpointMetadata expected{canonical_sidecar.slots,canonical_sidecar.next};
        validate(expected,false);
        if(m.next!=expected.next) throw std::runtime_error("shuffle sidecar next mismatch");
        for(size_t s=0;s<slots_;++s)
            if(m.cursor[s].doc!=expected.cursor[s].doc || m.cursor[s].target!=expected.cursor[s].target)
                throw std::runtime_error("shuffle sidecar cursor mismatch");
        return m;
    }
};
} // namespace tao::data
