#pragma once
#include "byte_bpe.hpp"
#include <set>
#include <queue>
namespace tao::text {
inline ByteBpe fit_incremental(const std::vector<std::string>&messages,size_t limit){if(limit>16123)throw std::runtime_error("limit");struct N{uint32_t token;int prev,next;};std::vector<N>nodes;using Pair=std::pair<uint32_t,uint32_t>;std::map<Pair,std::set<int>>occ;for(auto&s:messages){int last=-1;for(unsigned char c:s){int i=int(nodes.size());nodes.push_back({c,last,-1});if(last>=0){nodes[last].next=i;occ[{nodes[last].token,c}].insert(last);}last=i;}}
struct Entry{size_t count;Pair pair;bool operator<(const Entry&b)const{return count!=b.count?count<b.count:pair>b.pair;}};std::priority_queue<Entry>heap;for(auto&kv:occ)heap.push({kv.second.size(),kv.first});ByteBpe b;
for(size_t round=0;round<limit;++round){while(!heap.empty()&&occ[heap.top().pair].size()!=heap.top().count)heap.pop();if(heap.empty()||!heap.top().count)break;auto pair=heap.top().pair;heap.pop();std::vector<int>positions(occ[pair].begin(),occ[pair].end());std::set<Pair>changed;auto remove=[&](int i){if(i<0||nodes[i].next<0)return;Pair p{nodes[i].token,nodes[nodes[i].next].token};occ[p].erase(i);changed.insert(p);};auto add=[&](int i){if(i<0||nodes[i].next<0)return;Pair p{nodes[i].token,nodes[nodes[i].next].token};occ[p].insert(i);changed.insert(p);};uint32_t token=261+uint32_t(b.merges.size());b.merges.push_back(pair);for(int i:positions){int j=nodes[i].next;if(j<0||nodes[i].token!=pair.first||nodes[j].token!=pair.second)continue;int prev=nodes[i].prev,next=nodes[j].next;remove(prev);remove(i);remove(j);nodes[i].token=token;nodes[i].next=next;if(next>=0)nodes[next].prev=i;nodes[j].next=-1;nodes[j].prev=-1;add(prev);add(i);}for(auto p:changed)if(!occ[p].empty())heap.push({occ[p].size(),p});}b.validate();return b;}
}
