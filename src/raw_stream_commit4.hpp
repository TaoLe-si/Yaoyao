#pragma once
#include <array>
#include <cmath>
#include <stdexcept>
#include <algorithm>
namespace tao {
// Diagnostic: raw finite-vocabulary tokens; auxiliary-supervised causal boundary MLP.
struct RawStreamCommit4 {
 std::array<double,1065>w;std::array<double,217> boundary;std::array<int,4> history{12,12,12,12};
 std::array<int,4> local{12,12,12,12};int window=80,memory=-1;
 size_t commits=0,unknown=0;
 RawStreamCommit4(const std::array<double,1065>&weights,const std::array<double,217>&t):w(weights),boundary(t){for(double v:w)if(!std::isfinite(v))throw std::invalid_argument("weight");for(double v:boundary)if(!std::isfinite(v))throw std::invalid_argument("boundary weight");}
 void reset(){local.fill(12);history.fill(12);window=80;memory=-1;commits=unknown=0;}
 int step(int query,int token){if(query<0||query>=4||token<0||token>12)throw std::invalid_argument("token");
 std::rotate(history.begin(),history.begin()+1,history.end());history[3]=token;
 std::rotate(local.begin(),local.begin()+1,local.end());local[3]=token;
 double z=boundary[216];for(int j=0;j<4;++j){double v=boundary[j*53+52];for(int k=0;k<4;++k)v+=boundary[j*53+k*13+history[k]];z+=boundary[212+j]*std::tanh(v);}if(z<0)return memory;
 ++commits;int ix[6]={query,4+local[3],17+local[2],30+local[1],43+local[0],56};
 double h[16];for(int j=0;j<8;++j){double a=0;for(int i:ix)a+=w[j*57+i];h[j]=std::tanh(a);}
 double gate=w[928];for(int j=0;j<8;++j)gate+=w[912+j]*h[j];
 if(gate>=0){for(int j=8;j<16;++j){double a=0;for(int i:ix)if(i>=4)a+=w[j*57+i];h[j]=std::tanh(a);}double best=0;int winner=-1;for(int k=1;k<9;++k){double z=w[912+k*17+16];for(int j=8;j<16;++j)z+=w[912+k*17+j]*h[j];if(winner<0||z>best){best=z;winner=k-1;}}memory=winner;}
 local.fill(12);return memory;
 }
};
}
