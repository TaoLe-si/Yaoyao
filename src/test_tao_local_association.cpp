#include "tao_local_association.hpp"
#include <cstdio>
#include <cstring>
#include <cfloat>
void need(bool b){if(!b)throw std::runtime_error("assertion");}
template<class F>void rejects(F f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}need(caught);}
int main(){try{using namespace tao;using namespace tao::local;
std::vector<float>r{2,-4,0},v{6,4,8};need(apply(r,v,Action::Skip,true)==r);need(apply(r,v,Action::Share,true)==std::vector<float>({3,-2,2}));need(apply(r,v,Action::Add,true)==std::vector<float>({3.5f,-3,2}));need(apply(r,{},Action::Add,false)==r);
std::vector<float>signedzero{-0.f,0.f};auto identity=apply(signedzero,{},Action::Skip,false);need(std::memcmp(identity.data(),signedzero.data(),8)==0);
rejects([&]{apply(r,v,Action(2),true);});rejects([&]{apply(r,{1},Action::Share,true);});rejects([&]{apply(r,v,Action::Add,true,2);});rejects([&]{apply({INFINITY}, {},Action::Skip,false);});rejects([&]{apply({1},{NAN},Action::Add,true);});rejects([&]{apply({FLT_MAX},{FLT_MAX},Action::Add,true,1,1);});
need(choose({0,0,0},true)==Action::Skip);need(choose({0,1,0},true)==Action::Share);need(choose({0,0,1},true)==Action::Add);need(choose({0,0,1},false)==Action::Skip);rejects([]{choose({NAN,0,0},false);});
std::array<uint8_t,32>h{};TernaryEmbedding e(8,3,h),p(3,6,h);e.set_row(0,{1,0,0},1);e.set_row(1,{0,1,0},1);e.set_row(2,{1,0,0},1);for(int i=0;i<3;++i){std::vector<int8_t>w(6);w[i]=1;p.set_row(i,w,1);}
need(choose(action_scores(p,{1,0,0},{0,0,0}),true)==Action::Skip);need(choose(action_scores(p,{0,1,0},{0,0,0}),true)==Action::Share);need(choose(action_scores(p,{0,0,1},{0,0,0}),true)==Action::Add);
std::vector<uint32_t>ids{0,1,2};auto result=forward(e,p,ids,2,2,2,r,{0,1,0});need(result.selected.index==0&&result.action==Action::Share);need(result.output==apply(r,e.lookup(0),Action::Share,true));auto saved=e.bytes();ids.push_back(7);need(feature(e,ids,2,2)==feature(e,{0,1,2},2,2));need(forward(e,p,ids,2,2,2,r,{0,1,0}).output==result.output);need(e.bytes()==saved);need(!select(e,p,ids,0,2,2).index);need(select(e,p,ids,2,1,2).index==1);need(!select(e,p,ids,2,0,2).index);need(forward(e,p,ids,2,2,2,r,{NAN,0,0},false).output==r);
TernaryEmbedding contextProjection(3,6,h);for(int i=0;i<3;++i){std::vector<int8_t>w(6);w[3+i]=1;contextProjection.set_row(i,w,1);}need(project(contextProjection,feature(e,{0,2},1,1))!=project(contextProjection,feature(e,{1,2},1,1)));
printf("PASS three action semantics,bitwise skip,no candidate,nonfinite/overflow checks,causal features,horizon,shared projection,context sensitivity,no writeback\n");return 0;}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
