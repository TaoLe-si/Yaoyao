#include <vector>
#include <random>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <algorithm>
int main(){volatile float sink=0;std::mt19937 rng(33);for(int D:{256,1024,4096})for(int H:{16,32})for(int write:{0,1}){int I=4*D;std::vector<float>w(size_t(I)*H),u(size_t(D)*H),x(I),h(H);for(auto&v:w)v=float(int(rng()%3)-1)*.01f;for(auto&v:u)v=float(int(rng()%3)-1)*.01f;for(auto&v:x)v=float(int(rng()%101)-50)/50;double time[7];for(int round=0;round<7;++round){auto start=std::chrono::steady_clock::now();for(int rep=0;rep<100;++rep){x[0]=float(rep%7)/7;for(int j=0;j<H;++j){float a=0;for(int i=0;i<I;++i)a+=w[size_t(j)*I+i]*x[i];h[j]=std::tanh(a);}float z=0;for(float v:h)z+=v;if(write){for(int d=0;d<D;++d){float a=0;for(int j=0;j<H;++j)a+=u[size_t(d)*H+j]*h[j];sink+=a;}}sink+=z;}time[round]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/100;}std::sort(time,time+7);printf("D=%d H=%d write=%d FP32_resident_weights=%zu median_us=%.6f min=%.6f max=%.6f\n",D,H,write,(w.size()+u.size())*4,time[3],time[0],time[6]);}printf("sink=%f\n",float(sink));}
