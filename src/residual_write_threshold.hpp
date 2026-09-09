#pragma once
#include <array>
#include <cmath>
#include <algorithm>
#include <stdexcept>
namespace tao {
template<int E>struct ResidualWriteThreshold {
 static_assert(E==16||E==32);static constexpr int N=1065+E*17+16*(E+1);
 std::array<double,N>w;std::array<int,4>history{12,12,12,12};int memory=0;size_t writes=0;
 explicit ResidualWriteThreshold(const std::array<double,N>&a):w(a){for(double v:w)if(!std::isfinite(v))throw std::invalid_argument("weights");}
 void reset(){history.fill(12);memory=0;writes=0;}
 int step(int q,int t,double threshold=0){if(q<0||q>=4||t<0||t>=13)throw std::invalid_argument("input");std::rotate(history.begin(),history.begin()+1,history.end());history[3]=t;int ix[6]={q,4+t,17+history[2],30+history[1],43+history[0],56};double a[16],u[E],h[16];for(int j=0;j<16;++j){double z=0;for(int i:ix)z+=w[j*57+i];a[j]=std::tanh(z);}for(int j=0;j<E;++j){double z=w[1065+j*17+16];for(int k=0;k<16;++k)z+=w[1065+j*17+k]*a[k];u[j]=std::tanh(z);}for(int k=0;k<16;++k){double z=a[k]+w[1065+E*17+k*(E+1)+E];for(int j=0;j<E;++j)z+=w[1065+E*17+k*(E+1)+j]*u[j];h[k]=std::tanh(z);}double gate=w[928];for(int j=0;j<16;++j)gate+=w[912+j]*h[j];if(gate<threshold)return memory;int best=-1;double max=0;for(int k=1;k<9;++k){double z=w[912+k*17+16];for(int j=0;j<16;++j)z+=w[912+k*17+j]*h[j];if(best<0||z>max){max=z;best=k-1;}}++writes;return memory=best;}
};
}
