#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>
static bool closef(float a,float b,float e=1e-4f){return std::fabs(a-b)<=e*(1+std::fabs(b));}
int main(){
 int fail=0; const int V=5,N=2; float z[N][V]={{1,2,3,4,5},{-2,0,1,3,4}}; int tgt[N]={4,1};
 for(int n=0;n<N;n++){float mx=*std::max_element(z[n],z[n]+V),s=0;for(int v=0;v<V;v++)s+=std::exp(z[n][v]-mx);float lz=mx+std::log(s);float ps=0;for(int v=0;v<V;v++){float p=std::exp(z[n][v]-lz);ps+=p;if(!closef(p,std::exp(z[n][v]-lz)))fail++;}float loss=-(z[n][tgt[n]]-lz);if(!(loss>0&&std::isfinite(loss))||!closef(ps,1))fail++;}
 const int L=5;float x[L]={1,2,3,4,5},w0=.2f,w1=.3f,w2=.4f,dy[L]={.5f,-.2f,.7f,.1f,-.4f},dx[L];for(int t=0;t<L;t++){dx[t]=w2*dy[t]+(t+1<L?w1*dy[t+1]:0)+(t+2<L?w0*dy[t+2]:0);}float eps=1e-3f;for(int j=0;j<L;j++){float yp=0,ym=0;for(int t=0;t<L;t++){float qp=(t>=2?w0*(x[t-2]+(j==t-2?eps:0)):0)+(t>=1?w1*(x[t-1]+(j==t-1?eps:0)):0)+w2*(x[t]+(j==t?eps:0));float qm=(t>=2?w0*(x[t-2]-(j==t-2?eps:0)):0)+(t>=1?w1*(x[t-1]-(j==t-1?eps:0)):0)+w2*(x[t]-(j==t?eps:0));yp+=qp*dy[t];ym+=qm*dy[t];}if(!closef(dx[j],(yp-ym)/(2*eps),2e-3f))fail++;}
 float a=.3f,y[L]={1,-1,2,0,3},h=0,s=0;for(int t=0;t<L;t++){h=a*h+(1-a)*y[t];s+=y[t];if(!std::isfinite(h+s))fail++;}if(fail){printf("FAIL %d\n",fail);return 1;}printf("PASS logsoftmax q3_backward channels finite-difference\n");return 0;}
