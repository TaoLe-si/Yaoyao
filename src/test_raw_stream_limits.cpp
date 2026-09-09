#include "raw_stream_commit4.hpp"
#include <fstream>
#include <string>
#include <random>
#include <cstdio>
#include <limits>
int main(){try{for(unsigned seed:{7u,123u,2026u,42u,999u}){std::array<double,1065>w;std::array<double,217>b;std::ifstream f("build/raw_commit4_"+std::to_string(seed)+".weights",std::ios::binary),g("build/raw_boundary_"+std::to_string(seed)+".weights",std::ios::binary);if(!f.read((char*)w.data(),sizeof(w))||!g.read((char*)b.data(),sizeof(b)))throw std::runtime_error("weights");tao::RawStreamCommit4 m(w,b);
int rejected=0;try{m.step(4,0);}catch(const std::invalid_argument&){++rejected;}try{m.step(0,13);}catch(const std::invalid_argument&){++rejected;}auto bad=b;bad[0]=std::numeric_limits<double>::quiet_NaN();try{tao::RawStreamCommit4 invalid(w,bad);}catch(const std::invalid_argument&){++rejected;}if(rejected!=3)throw std::runtime_error("validation");
for(int mode=0;mode<4;++mode){std::mt19937 rng(392171);int correct=0,stable=0;for(int n=0;n<256;++n){int q=rng()%4,target=-1;m.reset();for(int event=0;event<32;++event){int o=rng()%4,who=rng()%8;if(event==31)o=q;if(o==q)target=who;int before=m.memory;int tokens[7],count=0;if(mode==0){tokens[count++]=o;tokens[count++]=4+who;tokens[count++]=12;}if(mode==1){tokens[count++]=o;for(int k=0;k<3;++k)tokens[count++]=12;tokens[count++]=4+who;tokens[count++]=12;}if(mode==2){tokens[count++]=4+who;tokens[count++]=12;tokens[count++]=o;tokens[count++]=12;}if(mode==3){tokens[count++]=o;tokens[count++]=4+who;tokens[count++]=12;tokens[count++]=12;}
for(int k=0;k<count;++k)m.step(q,tokens[k]);if(o!=q&&before!=-1)stable+=m.memory==before;
}correct+=m.memory==target;}printf("seed=%u mode=%d many_updates_correct=%d/256 irrelevant_stable_count=%d\n",seed,mode,correct,stable);}
m.reset();if(m.memory!=-1||m.commits!=0)throw std::runtime_error("reset");}return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
