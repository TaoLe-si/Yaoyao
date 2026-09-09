#pragma once
#include "pilot_reader.hpp"
#include <limits>
namespace tao::data {
struct Cursor { size_t doc=std::numeric_limits<size_t>::max(),target=1; };
struct Work { size_t slot,doc,begin,end;bool reset; };
struct PilotCursor {
std::vector<std::vector<Token>> docs;std::vector<Cursor>slots;size_t next=0;
PilotCursor(std::vector<std::vector<Token>> input,size_t n,bool legacy=true):docs(std::move(input)),slots(n){if(!n)throw std::invalid_argument("slots");for(auto&t:docs){if(legacy){if(t.size()<4||t.back().id!=EOS)throw std::runtime_error("legacy EOS expected");t.pop_back();}else if(t.size()<3||t.front().id!=BOS||t.back().id!=TURN_END)throw std::runtime_error("BPE record boundary");}}
bool take(size_t slot,size_t width,Work&out){if(slot>=slots.size()||!width)throw std::invalid_argument("cursor");auto&c=slots[slot];bool reset=false;if(c.doc==std::numeric_limits<size_t>::max()||c.target>=docs[c.doc].size()){if(next==docs.size())return false;c={next++,1};reset=true;}size_t begin=c.target,end=std::min(docs[c.doc].size(),begin+width);out={slot,c.doc,begin,end,reset};c.target=end;return true;}
};
}
