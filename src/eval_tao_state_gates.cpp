#define INVERSE_READER_HEAD_ONLY
#include "test_integrated_inverse_reader.cpp"
#include "reader_training_data.hpp"
#include "tao_state_gates.hpp"
int run(int argc,char** argv){
 if(argc!=10 && argc!=11)return 1;
 M model,parent;std::mt19937 rng(42);model.init(rng,1024);parent.init(rng,1024);
 if(!model.load(argv[1])||!parent.load(argv[2]))throw std::runtime_error("model load");
 tds::validate_transfer(model.q1.trits.data(),parent.q1.trits.data(),model.q1.trits.size(),model.Wbi.data(),parent.Wbi.data(),model.Wbi.size()*4);
 tds::Dependencies dep;dep.model=file_sha256(argv[1]);dep.parent=file_sha256(argv[2]);dep.train=file_sha256(argv[3]);dep.heldout=file_sha256(argv[4]);dep.reader=file_sha256(argv[5]);dep.tcg=file_sha256(argv[6]);dep.q1=tds::memory_sha256(model.q1.trits.data(),model.q1.trits.size());dep.wbi=tds::memory_sha256(model.Wbi.data(),model.Wbi.size()*4);
 auto frozen=tds::load_tcg(argv[6],dep.parent,dep.train,dep.reader);auto a=tds::load_reader(argv[5],dep.train,dep.heldout);
 std::vector<uint8_t> gates(1024*15,0);if(std::string(argv[7])!="identity")gates=tds::Artifact::load(argv[7],dep).codes;tds::validate_union(gates,frozen);
 unsigned sequences=std::stoul(argv[8]);auto batch=read_reader_batch(argv[9],0,sequences);
 auto code=[&](int id){return model.q1.trits.data()+Q1::hash(id,128)*16*256;};
 std::ofstream dump;if(argc==11)dump.open(argv[10],std::ios::binary);
 auto evaluate=[&](bool trained){double loss=0;Node node;
 for(size_t n=0;n<batch.input.size();++n){int t=int(n%64);if(!t)node={};node.push(batch.input[n],code);auto cur=node;std::array<float,256> mixed{};
 for(int k=0;;++k){float sc=trained?tds::distance_scale(gates.data(),batch.input[n],k):1.f;for(int d=0;d<256;++d){float r=cur.trit[d]*sc;if(a[k*256+d]==1)mixed[d]+=r;else if(a[k*256+d]==-1)mixed[d]-=r;}if(!cur.count)break;cur.pop(code);}
 float s[320],h[320];for(int d=0;d<256;++d)s[d]=mixed[d];extract_hash_features(node.hash,s+256);
 for(int j=0;j<320;++j){float g=model.b_sgl_gate[j],u=model.b_sgl_up[j];for(int d=0;d<320;++d){g+=model.W_sgl_gate[j*320+d]*s[d];u+=model.W_sgl_up[j*320+d]*s[d];}h[j]=g/(1+std::exp(-g))*u;}
 std::vector<float> z(1024);int prev=t?batch.input[n-1]:0;for(int v=0;v<1024;++v){float x=model.Wbi[prev*1024+v];for(int j=0;j<320;++j)x+=model.W_sgl_out[v*320+j]*h[j];z[v]=x;}
 cur=node;for(int k=0;cur.count;++k){int id=cur.pop(code);if(k==0)continue;int q=frozen[batch.input[n]*15+k-1];if(q)for(int v=0;v<1024;++v)z[v]+=float(q)/32*model.Wbi[id*1024+v];}
 if(trained&&dump.is_open())dump.write(reinterpret_cast<const char*>(z.data()),z.size()*4);
 double mx=*std::max_element(z.begin(),z.end()),den=0;for(float x:z){if(!std::isfinite(x))throw std::runtime_error("nonfinite");den+=std::exp(double(x)-mx);}loss+=mx+std::log(den)-z[batch.target[n]];
 }return loss/batch.input.size();};
 double baseline=evaluate(false),trained=evaluate(true);printf("CPU targets=%zu baseline=%.12f trained=%.12f delta=%.12f union<=2\n",batch.input.size(),baseline,trained,trained-baseline);return 0;
}
int main(int argc,char** argv){try{return run(argc,argv);}catch(const std::exception& e){fprintf(stderr,"EVAL FAIL %s\n",e.what());return 10;}}
