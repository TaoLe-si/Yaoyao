#define main old_main
#include "diagnose_sparse_write4.cpp"
#undef main
#include <fstream>
// Freeze learned raw-token relevance/content network. No oracle event inputs.
int predict(const Net&n,const Seq&s,int mode,double*mass){M m{};m.fill(.125);for(size_t t=0;t<s.tokens.size();++t){int ix[6]={s.query,4+s.tokens[t],17+(t?s.tokens[t-1]:12),30+(t>1?s.tokens[t-2]:12),43+(t>2?s.tokens[t-3]:12),56};double h[16],z[9];for(int j=0;j<16;++j){double a=0;for(int i:ix)a+=n.w[j*57+i];h[j]=std::tanh(a);}for(int o=0;o<9;++o){z[o]=n.w[912+o*17+16];for(int j=0;j<16;++j)z[o]+=n.w[912+o*17+j]*h[j];}double p=std::max(0.,std::min(1.,.5+.25*z[0]));double top=*std::max_element(z+1,z+9),sum=0;M v;for(int k=0;k<8;++k)sum+=(v[k]=std::exp(z[k+1]-top));for(auto&x:v)x/=sum;
double retain=1-p,write=p;
if(mode==1)retain=.99; // uniform time decay, same learned input weight
if(mode==2)retain=1-.5*p; // only partially remove obsolete content
if(mode==3){retain=1-p;write=p*p;} // attenuate weak inputs, retention unchanged
if(mode==4){write=p>=.5?p:0;retain=1-write;} // exact rejection of weak input
for(int k=0;k<8;++k)m[k]=retain*m[k]+write*v[k];}
*mass=0;for(auto x:m)*mass+=x;return int(std::max_element(m.begin(),m.end())-m.begin());}
int main(){try{for(unsigned seed:{42u,123u,2026u,7u,999u}){Net n(seed);std::ifstream f("build/init_write4_2_adam_"+std::to_string(seed)+".weights",std::ios::binary);if(!f.read((char*)n.w.data(),sizeof(n.w)))throw std::runtime_error("weights");for(int mode=0;mode<5;++mode)for(int noise:{2,128}){std::mt19937 r(783129+noise);int correct=0,changed=0;double mass=0;for(int i=0;i<128;++i){auto s=make(r,noise);double x;int pred=predict(n,s,mode,&x);correct+=pred==s.target;mass+=x;if(mode==0){int ref;n.run(s,nullptr,&ref);if(ref!=pred)throw std::runtime_error("baseline parity");}auto other=s;other.target=(s.target+1)%8;for(size_t t=other.tokens.size();t>0;t-=3)if(other.tokens[t-3]==s.query){other.tokens[t-2]=4+other.target;break;}changed+=predict(n,other,mode,&x)==other.target;}printf("seed=%u mode=%d noise=%d correct=%d/128 required_update=%d/128 mean_mass=%.6f\n",seed,mode,noise,correct,changed,mass/128);}}return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
