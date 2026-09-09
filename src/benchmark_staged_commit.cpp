#include <vector>
#include <random>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <algorithm>
int main(){volatile float sink=0;std::mt19937 rng(301);for(int D:{256,1024,4096}){
std::vector<float>x(5*D),wb(16*D),wg(40*D),wc(32*D),wo(8*D);
for(auto*p:{&x,&wb,&wg,&wc,&wo})for(auto&v:*p)v=float(int(rng()%3)-1)*.01f;
for(int mode=0;mode<3;++mode){double times[7];for(int round=0;round<7;++round){auto start=std::chrono::steady_clock::now();for(int token=0;token<240;++token){x[0]=float(token%7)/7;
float b=0;for(int h=0;h<4;++h){float z=0;for(int i=0;i<4*D;++i)z+=wb[h*4*D+i]*x[i];b+=std::tanh(z);}sink+=b;
bool commit=mode==2||(mode==1&&token%3==0);bool write=mode==2||(mode==1&&token%12==0);
if(commit){float g=0;for(int h=0;h<8;++h){float z=0;for(int i=0;i<5*D;++i)z+=wg[h*5*D+i]*x[i];g+=std::tanh(z);}sink+=g;}
if(write){float c[8];for(int h=0;h<8;++h){float z=0;for(int i=0;i<4*D;++i)z+=wc[h*4*D+i]*x[i];c[h]=std::tanh(z);}for(int d=0;d<D;++d){float z=0;for(int h=0;h<8;++h)z+=wo[d*8+h]*c[h];sink+=z;}}
}times[round]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/240;}std::sort(times,times+7);printf("D=%d mode=%d weights_bytes=%zu median_us=%.6f min=%.6f max=%.6f\n",D,mode,(wb.size()+wg.size()+wc.size()+wo.size())*4,times[3],times[0],times[6]);}}printf("sink=%f\n",float(sink));}
