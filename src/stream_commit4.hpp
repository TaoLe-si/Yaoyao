#pragma once
#include <array>
#include <cmath>
#include <stdexcept>
#include <algorithm>
namespace tao {
// Diagnostic: assumes known object/owner token categories and supervised boundary table.
struct StreamCommit4 {
 std::array<double,1065>w;std::array<int,81> table;
 std::array<int,4> local{12,12,12,12};int window=80,memory=-1;
 size_t commits=0,unknown=0;
 StreamCommit4(const std::array<double,1065>&weights,const std::array<int,81>&t):w(weights),table(t){for(double v:w)if(!std::isfinite(v))throw std::invalid_argument("weight");}
 void reset(){local.fill(12);window=80;memory=-1;commits=unknown=0;}
 static int mapped(int i,bool gate){if(i<4||i==56)return i;int offset=4+13*((i-4)/13),token=i-offset;if(gate&&token>=4&&token<12)return offset+4;if(!gate&&token<4)return offset;return i;}
 int step(int query,int token){if(query<0||query>=4||token<0||token>12)throw std::invalid_argument("token");
 window=(window%27)*3+(token<4?0:token<12?1:2);
 std::rotate(local.begin(),local.begin()+1,local.end());local[3]=token;
 if(table[window]<0){++unknown;return memory;}if(table[window]==0)return memory;
 ++commits;int ix[6]={query,4+local[3],17+local[2],30+local[1],43+local[0],56};
 double h[16];for(int j=0;j<8;++j){double a=0;for(int i:ix)a+=w[j*57+mapped(i,true)];h[j]=std::tanh(a);}
 double gate=w[928];for(int j=0;j<8;++j)gate+=w[912+j]*h[j];
 if(gate>=0){for(int j=8;j<16;++j){double a=0;for(int i:ix)if(i>=4)a+=w[j*57+mapped(i,false)];h[j]=std::tanh(a);}double best=0;int winner=-1;for(int k=1;k<9;++k){double z=w[912+k*17+16];for(int j=8;j<16;++j)z+=w[912+k*17+j]*h[j];if(winner<0||z>best){best=z;winner=k-1;}}memory=winner;}
 local.fill(12);return memory;
 }
};
}
