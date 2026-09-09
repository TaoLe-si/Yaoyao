#pragma once
#include <array>
#include <cmath>
#include <algorithm>
#include <stdexcept>
struct PackedResidualLut {
std::array<unsigned char,516>bits{};std::array<float,73>scale{},bias{};std::array<int,4>hist{12,12,12,12};int memory=0;
static int offset(int row){return row<16?row*56:row<25?896+(row-16)*16:row<57?1040+(row-25)*16:1552+(row-57)*32;}
int sign(int row,int col)const{int i=offset(row)+col,c=(bits[i/4]>>(2*(i%4)))&3;if(c==3)throw std::runtime_error("code");return c==1?1:c==2?-1:0;}
explicit PackedResidualLut(const std::array<double,2137>&w){auto row=[&](int r,int start,int n){double a=0;for(int j=0;j<n;++j){double v=w[start+j];if(!std::isfinite(v))throw std::runtime_error("weight");if(v!=0){if(a&&a!=std::abs(v))throw std::runtime_error("ternary");a=std::abs(v);}int i=offset(r)+j;bits[i/4]|=(v==0?0:v>0?1:2)<<(2*(i%4));}scale[r]=float(a?a:1);bias[r]=float(w[start+n]);if(!std::isfinite(scale[r])||scale[r]<=0||!std::isfinite(bias[r]))throw std::runtime_error("scale bias");};for(int j=0;j<16;++j)row(j,j*57,56);for(int j=0;j<9;++j)row(16+j,912+j*17,16);for(int j=0;j<32;++j)row(25+j,1065+j*17,16);for(int j=0;j<16;++j)row(57+j,1609+j*33,32);}
float dot(int r,const float*x,int n)const{float sum=0;static constexpr float lut[4]={0,1,-1,0};int base=offset(r)/4;for(int j=0;j<n;j+=4){unsigned c=bits[base+j/4];for(int k=0;k<4;++k){unsigned code=(c>>(2*k))&3;if(code==3)throw std::runtime_error("code");sum+=lut[code]*x[j+k];}}return scale[r]*sum;}
void reset(){hist.fill(12);memory=0;}
int step(int q,int t){if(q<0||q>=4||t<0||t>12)throw std::runtime_error("input");std::rotate(hist.begin(),hist.begin()+1,hist.end());hist[3]=t;int ix[5]={q,4+t,17+hist[2],30+hist[1],43+hist[0]};float a[16],u[32],h[16];for(int j=0;j<16;++j){int sum=0;for(int i:ix)sum+=sign(j,i);a[j]=std::tanh(bias[j]+scale[j]*sum);}for(int j=0;j<32;++j)u[j]=std::tanh(bias[25+j]+dot(25+j,a,16));for(int j=0;j<16;++j)h[j]=std::tanh(a[j]+bias[57+j]+dot(57+j,u,32));if(bias[16]+dot(16,h,16)<0)return memory;float best=0;int win=-1;for(int k=0;k<8;++k){float z=bias[17+k]+dot(17+k,h,16);if(win<0||z>best){win=k;best=z;}}return memory=win;}
};
