#pragma once
#include "shuffled_epoch_cursor.hpp"
#include "batched_slot_plan.hpp"
namespace tao::data {
// Isolated overload: same BatchPlan layout, canonical doc IDs, masks/reset policy.
// Include ONLY in the new experimental trainer. Original PilotCursor overload unchanged.
inline BatchPlan take_batch(ShuffledEpochCursor& cursor,size_t width) {
    if(!width)throw std::invalid_argument("batch width");
    BatchPlan p;p.slots=cursor.cursor().slots.size();
    std::vector<Work> work(p.slots);std::vector<bool> valid(p.slots);
    for(size_t k=0;k<p.slots;++k){valid[k]=cursor.take(k,width,work[k]);
        if(valid[k])p.timesteps=std::max(p.timesteps,work[k].end-work[k].begin);}
    p.items.resize(p.slots*p.timesteps);
    for(size_t k=0;k<p.slots;++k)if(valid[k]){
        const auto& w=work[k];const auto& doc=cursor.cursor().docs[w.doc];
        for(size_t t=0;t<w.end-w.begin;++t){const size_t i=w.begin+t;
            p.items[t*p.slots+k]={unsigned(doc[i-1].id),unsigned(doc[i].id),true,t==0&&w.reset,bool(doc[i].loss),cursor.cursor().doc_weights.empty()?1.f:cursor.cursor().doc_weights[w.doc]};
            ++p.positions;p.supervised+=doc[i].loss;
        }
    }
    return p;
}
}
