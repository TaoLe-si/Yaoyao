#pragma once
#include "pilot_slot_cursor.hpp"
namespace tao::data {
struct BatchPosition {unsigned input=0,target=0;bool active=false,reset=false,loss=false;};
struct BatchPlan {size_t slots=0,timesteps=0,positions=0,supervised=0;std::vector<BatchPosition>items;};
inline BatchPlan take_batch(PilotCursor&cursor,size_t width){BatchPlan p;p.slots=cursor.slots.size();std::vector<Work>work(p.slots);std::vector<bool>valid(p.slots);for(size_t k=0;k<p.slots;++k){valid[k]=cursor.take(k,width,work[k]);if(valid[k])p.timesteps=std::max(p.timesteps,work[k].end-work[k].begin);}p.items.resize(p.slots*p.timesteps);for(size_t k=0;k<p.slots;++k)if(valid[k]){auto&w=work[k];auto&doc=cursor.docs[w.doc];for(size_t t=0;t<w.end-w.begin;++t){size_t i=w.begin+t;p.items[t*p.slots+k]={unsigned(doc[i-1].id),unsigned(doc[i].id),true,t==0&&w.reset,bool(doc[i].loss)};++p.positions;p.supervised+=doc[i].loss;}}return p;}
}
