#define main old_main
#include "diagnose_sparse_write4.cpp"
#undef main
#include <fstream>
// Freeze learned raw-token relevance/content network. No oracle event inputs.
int predict(const Net&n,const Seq&s,int mode,double*mass,double bias=0){M m{};m.fill(.125);for(size_t t=0;t<s.tokens.size();++t){int ix[6]={s.query,4+s.tokens[t],17+(t?s.tokens[t-1]:12),30+(t>1?s.tokens[t-2]:12),43+(t>2?s.tokens[t-3]:12),56};double h[16],z[9];for(int j=0;j<16;++j){double a=0;for(int i:ix)a+=n.w[j*57+i];h[j]=std::tanh(a);}for(int o=0;o<9;++o){z[o]=n.w[912+o*17+16];for(int j=0;j<16;++j)z[o]+=n.w[912+o*17+j]*h[j];}double p=std::max(0.,std::min(1.,.5+.25*(z[0]+bias)));double top=*std::max_element(z+1,z+9),sum=0;M v;for(int k=0;k<8;++k)sum+=(v[k]=std::exp(z[k+1]-top));for(auto&x:v)x/=sum;
double retain=1-p,write=p;
if(mode==1)retain=.99; // uniform time decay, same learned input weight
if(mode==2)retain=1-.5*p; // only partially remove obsolete content
if(mode==3){retain=1-p;write=p*p;} // attenuate weak inputs, retention unchanged
if(mode==4){write=p>=.5?p:0;retain=1-write;} // exact rejection of weak input
for(int k=0;k<8;++k)m[k]=retain*m[k]+write*v[k];}
*mass=0;for(auto x:m)*mass+=x;return int(std::max_element(m.begin(),m.end())-m.begin());}
int main(){try{for(double bias:{-2.,-1.,0.,1.,2.})for(int mode:{0,3,4}){int correct=0,updates=0,total=0;double mass=0;for(unsigned seed:{42u,123u,2026u,7u,999u}){Net net(seed);std::ifstream f("build/init_write4_2_adam_"+std::to_string(seed)+".weights",std::ios::binary);if(!f.read((char*)net.w.data(),sizeof(net.w)))throw std::runtime_error("weights");std::mt19937 rng(632871);for(int i=0;i<64;++i){auto s=make(rng,256);double value;correct+=predict(net,s,mode,&value,bias)==s.target;mass+=value;auto changed=s;changed.target=(s.target+1)%8;for(size_t j=changed.tokens.size();j>0;j-=3)if(changed.tokens[j-3]==s.query){changed.tokens[j-2]=4+changed.target;break;}updates+=predict(net,changed,mode,&value,bias)==changed.target;++total;}}printf("bias=%+.1f mode=%d original=%d/%d necessary_update=%d/%d mass=%.6f\n",bias,mode,correct,total,updates,total,mass/total);}return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
