// Step 12: 夭夭 v0.9 - Word-level vocab + larger pretraining
// Architecture unchanged, vocab upgraded to word-level (1024 most common words)
#define D_H 128
#define NL_H 2
#define Q1_B 128
#define Q1_K 16
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <random>
#include <numeric>
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <unordered_map>
#include <sstream>
#include <iomanip>
#include <map>
#include <fstream>

struct Q1 {
    int B, K, D;
    std::vector<int8_t> trits;
    std::vector<float> adam_m, adam_v;
    int step=0;
    void init(int B_, int K_, int D_, std::mt19937& rng){
        B=B_; K=K_; D=D_;
        trits.assign((size_t)B*K*D, 0);
        adam_m.assign((size_t)B*K*D, 0.0f);
        adam_v.assign((size_t)B*K*D, 0.0f);
        std::uniform_int_distribution<int> ud(-1, 1);
        for(auto& t:trits) t=(int8_t)ud(rng);
    }
    static int hash(int id, int B){
        uint64_t x=(uint32_t)id*2654435761u;
        x=(x>>16)^x;
        return (int)(x%(uint64_t)B);
    }
    struct Aux { int id, h; std::vector<float> weights; const float* query; };
    void forward(int id, const float* query, float* out, Aux& aux) const {
        aux.id=id; aux.h=hash(id, B); aux.query=query;
        aux.weights.assign(K, 0.0f);
        float mx=-1e9f;
        for(int k=0;k<K;++k){
            float s=0;
            for(int d=0;d<D;++d) s+=query[d]*trits[(aux.h*K+k)*D+d];
            aux.weights[k]=s;
            if(s>mx) mx=s;
        }
        for(int k=0;k<K;++k) aux.weights[k]=std::exp(aux.weights[k]-mx);
        float sum=0;
        for(int k=0;k<K;++k) sum+=aux.weights[k];
        for(int k=0;k<K;++k) aux.weights[k]/=sum;
        for(int d=0;d<D;++d){
            float v=0;
            for(int k=0;k<K;++k) v+=aux.weights[k]*trits[(aux.h*K+k)*D+d];
            out[d]=v;
        }
    }
    void backward(const Aux& aux, const float* d_out, std::vector<float>& d_trit, float* d_query) const {
        std::vector<float> d_w(K, 0.0f);
        for(int k=0;k<K;++k){
            float s=0;
            for(int d=0;d<D;++d) s+=d_out[d]*trits[(aux.h*K+k)*D+d];
            d_w[k]=s;
        }
        float dot=0;
        for(int k=0;k<K;++k) dot+=aux.weights[k]*d_w[k];
        std::vector<float> d_s(K);
        for(int k=0;k<K;++k) d_s[k]=aux.weights[k]*(d_w[k]-dot);
        for(int k=0;k<K;++k){
            float* g=&d_trit[(aux.h*K+k)*D];
            float ds=d_s[k];
            float wk=aux.weights[k];
            for(int d=0;d<D;++d) g[d] += d_out[d]*wk + aux.query[d]*ds;
        }
        for(int d=0;d<D;++d){
            float s=0;
            for(int k=0;k<K;++k) s+=trits[(aux.h*K+k)*D+d]*d_s[k];
            d_query[d]+=s;
        }
    }
    void adam_update(const std::vector<float>& grad, float lr, float b1, float b2, float eps){
        step++;
        float bc1=1-std::pow(b1,(float)step), bc2=1-std::pow(b2,(float)step);
        for(size_t i=0;i<trits.size();++i){
            float g=grad[i]; if(g>1)g=1; if(g<-1)g=-1;
            adam_m[i]=b1*adam_m[i]+(1-b1)*g;
            adam_v[i]=b2*adam_v[i]+(1-b2)*g*g;
            float st=lr*(adam_m[i]/bc1)/(std::sqrt(adam_v[i]/bc2)+eps);
            float nv=(float)trits[i]-st;
            if(nv>0.5f) trits[i]=1; else if(nv<-0.5f) trits[i]=-1; else trits[i]=0;
        }
    }
};

// Word-level vocab from text
struct Vocab {
    std::unordered_map<std::string,int> word_to_id;
    std::vector<std::string> id_to_word;
    int pad_id=0, unk_id=1;
    void build(const std::string& text, int max_size){
        // Tokenize: split by whitespace and punctuation
        std::map<std::string,int> freq;
        std::string cur;
        for(char c:text){
            if(c==' '||c=='\n'||c=='\t'||c=='\r'){
                if(!cur.empty()){freq[cur]++; cur.clear();}
            } else if(std::ispunct((unsigned char)c)){
                if(!cur.empty()){freq[cur]++; cur.clear();}
                std::string s(1,c); freq[s]++;
            } else {
                cur+=c;
            }
        }
        if(!cur.empty()) freq[cur]++;
        std::vector<std::pair<std::string,int>> v(freq.begin(), freq.end());
        std::sort(v.begin(), v.end(), [](auto& a, auto& b){return a.second>b.second;});
        id_to_word.clear();
        id_to_word.push_back("<pad>"); id_to_word.push_back("<unk>"); id_to_word.push_back("<eos>");
        word_to_id["<pad>"]=0; word_to_id["<unk>"]=1; word_to_id["<eos>"]=2;
        for(auto& p:v){
            if((int)id_to_word.size()>=max_size) break;
            int id=(int)id_to_word.size();
            word_to_id[p.first]=id;
            id_to_word.push_back(p.first);
        }
    }
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> tokens;
        std::string cur;
        for(char c:text){
            if(c==' '||c=='\n'||c=='\t'||c=='\r'){
                if(!cur.empty()){
                    auto it=word_to_id.find(cur);
                    tokens.push_back(it==word_to_id.end()?unk_id:it->second);
                    cur.clear();
                }
            } else if(std::ispunct((unsigned char)c)){
                if(!cur.empty()){
                    auto it=word_to_id.find(cur);
                    tokens.push_back(it==word_to_id.end()?unk_id:it->second);
                    cur.clear();
                }
                std::string s(1,c);
                auto it=word_to_id.find(s);
                tokens.push_back(it==word_to_id.end()?unk_id:it->second);
            } else {
                cur+=std::tolower((unsigned char)c);
            }
        }
        if(!cur.empty()){
            auto it=word_to_id.find(cur);
            tokens.push_back(it==word_to_id.end()?unk_id:it->second);
        }
        return tokens;
    }
    std::string decode(const std::vector<int>& ids) const {
        std::string s;
        for(size_t i=0;i<ids.size();++i){
            int id=ids[i];
            if(id<(int)id_to_word.size() && id>=2){
                std::string w=id_to_word[id];
                bool is_punct = (w.size()==1) && std::ispunct((unsigned char)w[0]);
                if(i>0 && !is_punct) s+=" ";
                s+=w;
            }
        }
        return s;
    }
};

struct M {
    Q1 q1;
    std::vector<float> Wh, Ws, Wbi;
    std::vector<float> q3w0, q3w1, q3w2;
    std::vector<float> aW, ab;
    std::vector<float> Wh_m, Wh_v, Ws_m, Ws_v, Wbi_m, Wbi_v;
    std::vector<float> q3w0_m,q3w0_v,q3w1_m,q3w1_v,q3w2_m,q3w2_v;
    std::vector<float> aW_m, aW_v, ab_m, ab_v;
    std::vector<float> q1_grad;
    int step=0;
    int V_unit=0;
        void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if(!f){std::fprintf(stderr,"Cannot open for save\n");return;}
        int magic=0x59414F59; f.write((char*)&magic,4);
        int version=1; f.write((char*)&version,4);
        int v=V_unit; f.write((char*)&v,4);
        auto wr=[&](const void* p, size_t n){f.write((char*)p,n);};
        wr(Wh.data(),Wh.size()*4); wr(Wh_m.data(),Wh_m.size()*4); wr(Wh_v.data(),Wh_v.size()*4);
        wr(Ws.data(),Ws.size()*4); wr(Ws_m.data(),Ws_m.size()*4); wr(Ws_v.data(),Ws_v.size()*4);
        wr(Wbi.data(),Wbi.size()*4); wr(Wbi_m.data(),Wbi_m.size()*4); wr(Wbi_v.data(),Wbi_v.size()*4);
        wr(q3w0.data(),q3w0.size()*4); wr(q3w0_m.data(),q3w0_m.size()*4); wr(q3w0_v.data(),q3w0_v.size()*4);
        wr(q3w1.data(),q3w1.size()*4); wr(q3w1_m.data(),q3w1_m.size()*4); wr(q3w1_v.data(),q3w1_v.size()*4);
        wr(q3w2.data(),q3w2.size()*4); wr(q3w2_m.data(),q3w2_m.size()*4); wr(q3w2_v.data(),q3w2_v.size()*4);
        wr(aW.data(),aW.size()*4); wr(aW_m.data(),aW_m.size()*4); wr(aW_v.data(),aW_v.size()*4);
        wr(ab.data(),ab.size()*4); wr(ab_m.data(),ab_m.size()*4); wr(ab_v.data(),ab_v.size()*4);
        wr(q1.trits.data(),q1.trits.size()); wr(q1.adam_m.data(),q1.adam_m.size()*4); wr(q1.adam_v.data(),q1.adam_v.size()*4);
        f.write((char*)&q1.step,4); f.write((char*)&step,4);
        std::printf("Saved model to %s (step=%d)\n", path.c_str(), step);
    }
    bool load(const std::string& path){
        std::ifstream f(path, std::ios::binary);
        if(!f) return false;
        int magic; f.read((char*)&magic,4); if(magic!=0x59414F59) return false;
        int version; f.read((char*)&version,4); if(version!=1) return false;
        int v; f.read((char*)&v,4);
        auto rd=[&](void* p, size_t n){f.read((char*)p,n);};
        rd(Wh.data(),Wh.size()*4); rd(Wh_m.data(),Wh_m.size()*4); rd(Wh_v.data(),Wh_v.size()*4);
        rd(Ws.data(),Ws.size()*4); rd(Ws_m.data(),Ws_m.size()*4); rd(Ws_v.data(),Ws_v.size()*4);
        rd(Wbi.data(),Wbi.size()*4); rd(Wbi_m.data(),Wbi_m.size()*4); rd(Wbi_v.data(),Wbi_v.size()*4);
        rd(q3w0.data(),q3w0.size()*4); rd(q3w0_m.data(),q3w0_m.size()*4); rd(q3w0_v.data(),q3w0_v.size()*4);
        rd(q3w1.data(),q3w1.size()*4); rd(q3w1_m.data(),q3w1_m.size()*4); rd(q3w1_v.data(),q3w1_v.size()*4);
        rd(q3w2.data(),q3w2.size()*4); rd(q3w2_m.data(),q3w2_m.size()*4); rd(q3w2_v.data(),q3w2_v.size()*4);
        rd(aW.data(),aW.size()*4); rd(aW_m.data(),aW_m.size()*4); rd(aW_v.data(),aW_v.size()*4);
        rd(ab.data(),ab.size()*4); rd(ab_m.data(),ab_m.size()*4); rd(ab_v.data(),ab_v.size()*4);
        rd(q1.trits.data(),q1.trits.size()); rd(q1.adam_m.data(),q1.adam_m.size()*4); rd(q1.adam_v.data(),q1.adam_v.size()*4);
        f.read((char*)&q1.step,4); f.read((char*)&step,4);
        std::printf("Loaded model from %s (step=%d)\n", path.c_str(), step);
        return true;
    }

    void init(std::mt19937& rng, int V){
        const int D=D_H, NL=NL_H;
        q1.init(Q1_B, Q1_K, D, rng);
        q1_grad.assign((size_t)Q1_B*Q1_K*D, 0.0f);
        Wh.assign(V*D, 0); Ws.assign(V*D, 0); Wbi.assign(V*V, 0);
        q3w0.assign(NL*D, 0); q3w1.assign(NL*D, 0); q3w2.assign(NL*D, 0);
        aW.assign(NL*D*D, 0); ab.assign(NL*D, 0);
        Wh_m.assign(V*D,0); Wh_v.assign(V*D,0);
        Ws_m.assign(V*D,0); Ws_v.assign(V*D,0);
        Wbi_m.assign(V*V,0); Wbi_v.assign(V*V,0);
        q3w0_m.assign(NL*D,0); q3w0_v.assign(NL*D,0);
        q3w1_m.assign(NL*D,0); q3w1_v.assign(NL*D,0);
        q3w2_m.assign(NL*D,0); q3w2_v.assign(NL*D,0);
        aW_m.assign(NL*D*D,0); aW_v.assign(NL*D*D,0);
        ab_m.assign(NL*D,0); ab_v.assign(NL*D,0);
        std::normal_distribution<float> nde(0,0.3f), ndw(0,0.1f);
        for(auto& x:Wh) x=ndw(rng);
        for(auto& x:Ws) x=ndw(rng);
        std::vector<float> tmp(D);
        for(int l=0;l<NL;++l){
            for(auto& x:tmp) x=ndw(rng)*0.3f;
            for(int d=0;d<D;++d){q3w0[l*D+d]=tmp[d];q3w1[l*D+d]=tmp[d];q3w2[l*D+d]=tmp[d];}
        }
        for(int l=0;l<NL;++l){
            for(int r=0;r<D;++r){
                std::vector<float> row(D); float n=0;
                for(int k=0;k<D;++k){row[k]=((rng()&1)?1.0f:-1.0f); n+=row[k]*row[k];}
                n=std::sqrt(n);
                for(int k=0;k<D;++k) aW[l*D*D+r*D+k]=row[k]/n;
            }
            for(auto& x:ab) x=-1.7f;
        }
    }
};

void forward(M& m, const std::vector<int>& inp, int BATCH, int SEQ, int PAD,
             int D, int NL, int V_unit,
             std::vector<float>& x, std::vector<float>& x_prev,
             std::vector<float>& y, std::vector<float>& alpha,
             std::vector<float>& h, std::vector<float>& s,
             std::vector<float>& hg, std::vector<float>& sg,
             std::vector<float>& logits, std::vector<float>& probs,
             std::vector<float>& xs, std::vector<float>& ys, std::vector<float>& alphas,
             std::vector<float>& hs, std::vector<float>& ss,
             std::vector<float>& hgs, std::vector<float>& sgs,
             std::vector<Q1::Aux>& q1_aux){
    int BL=BATCH*SEQ;
    for(int d=0;d<D;++d) x_prev[d]=0.0f;
    for(int t=0;t<BL;++t){
        int id=inp[t];
        m.q1.forward(id, x_prev.data(), x.data()+t*D, q1_aux[t]);
        for(int d=0;d<D;++d) x_prev[d]=0.0f;  // fixed embedding (no chain)
    }
    for(int l=0;l<NL;++l){
        std::memcpy(xs.data()+l*BL*D, x.data(), BL*D*sizeof(float));
        for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t){int bt=b*SEQ+t;
            for(int d=0;d<D;++d){
                float v=0;
                if(t>=2) v+=m.q3w0[l*D+d]*x[(b*SEQ+t-2)*D+d];
                if(t>=1) v+=m.q3w1[l*D+d]*x[(b*SEQ+t-1)*D+d];
                v+=m.q3w2[l*D+d]*x[bt*D+d];
                if(v>4)v=4; if(v<-4)v=-4;
                y[bt*D+d]=v;
            }
        }
        std::memcpy(ys.data()+l*BL*D, y.data(), BL*D*sizeof(float));
        for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t){int bt=b*SEQ+t;
            for(int d=0;d<D;++d){
                float z=m.ab[l*D+d];
                for(int k=0;k<D;++k) z+=m.aW[l*D*D+d*D+k]*x[bt*D+k];
                float zT=z/2.0f;
                if(zT>20)zT=20; if(zT<-20)zT=-20;
                alpha[bt*D+d]=1.0f/(1.0f+std::exp(-zT));
            }
        }
        std::memcpy(alphas.data()+l*BL*D, alpha.data(), BL*D*sizeof(float));
        for(int b=0;b<BATCH;++b) for(int d=0;d<D;++d){
            float hc=0,sc=0;
            int hoff=b*(SEQ+1)*D+d;
            h[hoff]=0; s[hoff]=0;
            for(int t=0;t<SEQ;++t){int bt=b*SEQ+t; float yt=y[bt*D+d], a=alpha[bt*D+d];
                hc=a*hc+(1-a)*yt; sc+=yt;
                h[hoff+(t+1)*D]=hc; s[hoff+(t+1)*D]=sc;
            }
        }
        std::memcpy(hs.data()+l*BATCH*(SEQ+1)*D, h.data(), BATCH*(SEQ+1)*D*sizeof(float));
        std::memcpy(ss.data()+l*BATCH*(SEQ+1)*D, s.data(), BATCH*(SEQ+1)*D*sizeof(float));
        for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t){int bt=b*SEQ+t;
            float ms_h=0, ms_s=0;
            for(int d=0;d<D;++d){float v=h[(b*(SEQ+1)+t+1)*D+d]; ms_h+=v*v;}
            ms_h=ms_h/(float)D+1e-5f; float r_h=1.0f/std::sqrt(ms_h);
            for(int d=0;d<D;++d) hg[bt*D+d]=h[(b*(SEQ+1)+t+1)*D+d]*r_h;
            for(int d=0;d<D;++d){float v=s[(b*(SEQ+1)+t+1)*D+d]; ms_s+=v*v;}
            ms_s=ms_s/(float)D+1e-5f; float r_s=1.0f/std::sqrt(ms_s);
            for(int d=0;d<D;++d) sg[bt*D+d]=s[(b*(SEQ+1)+t+1)*D+d]*r_s;
        }
        std::memcpy(hgs.data()+l*BL*D, hg.data(), BL*D*sizeof(float));
        std::memcpy(sgs.data()+l*BL*D, sg.data(), BL*D*sizeof(float));
        for(int n=0;n<BL;++n) for(int d=0;d<D;++d) x[n*D+d]=hg[n*D+d]+sg[n*D+d];
    }
    for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t){int bt=b*SEQ+t;
        int prev=(t>0)?inp[(b*SEQ+t-1)]:PAD;
        for(int v=0;v<V_unit;++v){
            float lv=m.Wbi[prev*V_unit+v];
            for(int d=0;d<D;++d) lv+=m.Wh[v*D+d]*hg[bt*D+d]+m.Ws[v*D+d]*sg[bt*D+d];
            logits[bt*V_unit+v]=lv;
        }
    }
    for(int n=0;n<BL;++n){
        float mx=logits[n*V_unit];
        for(int v=1;v<V_unit;++v) if(logits[n*V_unit+v]>mx) mx=logits[n*V_unit+v];
        float sum=0;
        for(int v=0;v<V_unit;++v){ probs[n*V_unit+v]=std::exp(logits[n*V_unit+v]-mx); sum+=probs[n*V_unit+v]; }
        for(int v=0;v<V_unit;++v) probs[n*V_unit+v]/=sum;
    }
}

int main(int argc, char** argv){
    bool server_mode=false;
    bool streaming=false;
    const char* model_path=argv[1];
    if(argc>=2 && std::string(argv[1])=="--server"){server_mode=true;}
    
    if(server_mode){
        // SERVER MODE: reads prompts from stdin, generates, prints tokens
        if(argc<3){std::fprintf(stderr,"Usage: %s --server <model.bin>\n",argv[0]);return 1;}
        model_path=argv[2];
        std::mt19937 server_rng(42);
        std::printf("Loading %s\n",model_path);
        // Load vocab from a fixed file
        std::ifstream tf("D:\\TaoVm\\tinystories_train.txt");
        std::stringstream sstrm; sstrm<<tf.rdbuf();
        std::string text=sstrm.str();
        tf.close();
        Vocab vocab;
        vocab.build(text.substr(0, std::min<size_t>(text.size(), 5000000)), 1024);
        // Read model header
        std::ifstream mf(model_path, std::ios::binary);
        int magic, version, V_unit;
        mf.read((char*)&magic,4); mf.read((char*)&version,4); mf.read((char*)&V_unit,4);
        mf.close();
        M m;
        m.init(server_rng, V_unit); m.V_unit=V_unit;
        if(!m.load(model_path)){std::fprintf(stderr,"Load failed\n");return 1;}
        std::printf("Server ready step=%d V=%d\n",m.step,V_unit);
        std::fflush(stdout);
        
        const int D=m.q1.D, NL=2;
        int SEQ=64, BATCH=1, BL=SEQ*BATCH, PAD=vocab.pad_id;
        std::vector<float> x(BL*D), x_prev(D), y(BL*D), alpha(BL*D);
        std::vector<float> h(BATCH*(SEQ+1)*D), s(BATCH*(SEQ+1)*D);
        std::vector<float> hg(BL*D), sg(BL*D), logits(BL*V_unit), probs(BL*V_unit);
        std::vector<float> xs(NL*BL*D), ys(NL*BL*D), alphas(NL*BL*D);
        std::vector<float> hs(NL*BATCH*(SEQ+1)*D), ss(NL*BATCH*(SEQ+1)*D);
        std::vector<float> hgs(NL*BL*D), sgs(NL*BL*D);
        std::vector<Q1::Aux> q1_aux(BL);
        
        std::string line;
        while(true){
            // Read request: prompt|stream|max_tokens|temperature|top_p
            if(!std::getline(std::cin, line)) break;
            if(line.empty()) continue;
            // Parse request line (pipe-separated)
            std::vector<std::string> parts;
            size_t pos=0;
            while((pos=line.find('|'))!=std::string::npos){
                parts.push_back(line.substr(0,pos));
                line=line.substr(pos+1);
            }
            parts.push_back(line);
            
            std::string prompt=parts[0];
            bool s_stream=parts.size()>1?parts[1]=="1":true;
            int s_max_tokens=parts.size()>2?atoi(parts[2].c_str()):60;
            float s_T=parts.size()>3?atof(parts[3].c_str()):0.9f;
            float s_top_p=parts.size()>4?atof(parts[4].c_str()):0.9f;
            
            std::vector<int> ids=vocab.encode(prompt);
            if((int)ids.size()==0){std::printf("ERROR empty_prompt\n");std::fflush(stdout);continue;}
            std::printf("PROMPT_TOKENS %d\n",(int)ids.size());std::fflush(stdout);
            
            auto t_total_start=std::chrono::steady_clock::now();
            for(int step=0;step<s_max_tokens;++step){
                auto t_step_start=std::chrono::steady_clock::now();
                int L=(int)ids.size();
                std::vector<int> in2(SEQ);
                for(int i=0;i<SEQ;++i){int idx=L-SEQ+i; in2[i]=(idx<0)?PAD:ids[idx];}
                std::vector<int> inBL(BL);
                for(int t=0;t<SEQ;++t) inBL[t]=in2[t];
                forward(m, inBL, BATCH, SEQ, PAD, D, NL, V_unit,
                        x, x_prev, y, alpha, h, s, hg, sg, logits, probs,
                        xs, ys, alphas, hs, ss, hgs, sgs, q1_aux);
                int bt=(BATCH-1)*SEQ+(SEQ-1);
                std::vector<float> adj_logit(V_unit);
                for(int v=0;v<V_unit;++v) adj_logit[v]=logits[bt*V_unit+v]/s_T;
                for(size_t back=0; back<ids.size() && back<6; ++back){
                    int tk = ids[ids.size()-1-back];
                    if(tk>=2 && tk<V_unit){float penalty=3.0f*std::pow(0.65f,(float)back); adj_logit[tk]-=penalty;}
                }
                std::vector<int> idx(V_unit);
                std::iota(idx.begin(),idx.end(),0);
                std::sort(idx.begin(),idx.end(),[&](int a,int b){return adj_logit[a]>adj_logit[b];});
                float mx=adj_logit[idx[0]];
                std::vector<float> probs2(V_unit);
                float sum=0;
                for(int v=0;v<V_unit;++v){probs2[v]=std::exp(adj_logit[v]-mx);sum+=probs2[v];}
                for(int v=0;v<V_unit;++v) probs2[v]/=sum;
                float cum=0; int nuc=V_unit;
                for(int i=0;i<V_unit;++i){cum+=probs2[idx[i]]; if(cum>=s_top_p){nuc=i+1;break;}}
                std::vector<float> nuc_probs(nuc);
                for(int i=0;i<nuc;++i) nuc_probs[i]=probs2[idx[i]];
                float ns=0; for(int i=0;i<nuc;++i) ns+=nuc_probs[i];
                for(int i=0;i<nuc;++i) nuc_probs[i]/=ns;
                std::discrete_distribution<int> dist(nuc_probs.begin(),nuc_probs.end());
                int best=idx[dist(server_rng)];
                ids.push_back(best);
                auto t_now=std::chrono::steady_clock::now();
                double step_ms=std::chrono::duration<double,std::milli>(t_now-t_step_start).count();
                double total_ms=std::chrono::duration<double,std::milli>(t_now-t_total_start).count();
                std::string tok_str=vocab.decode({best});
                for(char& c:tok_str){if(c=='\n')c='/'; if(c=='\r')c='/'; if(c=='\t')c=' ';}
                std::printf("TOKEN %d %d %.2fms %.2fms %s\n", step, best, step_ms, total_ms, tok_str.c_str());
                std::fflush(stdout);
            }
            std::printf("DONE\n");std::fflush(stdout);
        }
        return 0;
    }
    
    if(argc<3){std::fprintf(stderr,"Usage: %s [--server <model.bin>] | <model.bin> <prompt|stream> [max_tokens=60] [temperature=0.9] [top_p=0.9]\n",argv[0]); return 1;}
    std::string arg2=argv[2];
    std::string prompt;
    int arg_offset=2;
    if(arg2=="stream"){streaming=true; arg_offset=3; if(argc<4){std::fprintf(stderr,"Need prompt after stream\n");return 1;} prompt=argv[3];}
    else prompt=arg2;
    int max_tokens=(argc>arg_offset+1)?atoi(argv[arg_offset+1]):60;
    float T=(argc>arg_offset+2)?atof(argv[arg_offset+2]):0.9f;
    float top_p=(argc>arg_offset+3)?atof(argv[arg_offset+3]):0.9f;
    
    std::mt19937 rng(42);
    std::printf("Loading %s\n",model_path);
    
    // Load vocab from a fixed file (or use built-in)
    // For simplicity, encode/decode based on English rules
    // Read model to get V
    std::ifstream mf(model_path, std::ios::binary);
    if(!mf){std::fprintf(stderr,"Cannot open model\n");return 1;}
    int magic, version, V_unit;
    mf.read((char*)&magic,4); mf.read((char*)&version,4); mf.read((char*)&V_unit,4);
    if(magic!=0x59414F59){std::fprintf(stderr,"Bad magic\n");return 1;}
    mf.close();
    
    // Build a simple vocab from a sample text (same as training)
    const char* text_path="D:\\TaoVm\\tinystories_train.txt";
    std::ifstream tf(text_path);
    if(!tf){std::fprintf(stderr,"Cannot open text\n");return 1;}
    std::stringstream sstrm; sstrm<<tf.rdbuf();
    std::string text=sstrm.str();
    tf.close();
    
    Vocab vocab;
    vocab.build(text.substr(0, std::min<size_t>(text.size(), 5000000)), 1024);
    
    M m;
    m.init(rng, V_unit);
    m.V_unit=V_unit;
    if(!m.load(model_path)){std::fprintf(stderr,"Load failed\n");return 1;}
    std::printf("Loaded step=%d V=%d\n",m.step,V_unit);
    
    const int D=m.q1.D, NL=2;
    int SEQ=64, BATCH=1, BL=SEQ*BATCH, PAD=vocab.pad_id;
    std::vector<float> x(BL*D), x_prev(D), y(BL*D), alpha(BL*D);
    std::vector<float> h(BATCH*(SEQ+1)*D), s(BATCH*(SEQ+1)*D);
    std::vector<float> hg(BL*D), sg(BL*D), logits(BL*V_unit), probs(BL*V_unit);
    std::vector<float> xs(NL*BL*D), ys(NL*BL*D), alphas(NL*BL*D);
    std::vector<float> hs(NL*BATCH*(SEQ+1)*D), ss(NL*BATCH*(SEQ+1)*D);
    std::vector<float> hgs(NL*BL*D), sgs(NL*BL*D);
    std::vector<Q1::Aux> q1_aux(BL);
    
    std::vector<int> ids=vocab.encode(prompt);
    if((int)ids.size()==0){std::fprintf(stderr,"Empty prompt after tokenization\n");return 1;}
    
    auto t_total_start=std::chrono::steady_clock::now();
    for(int step=0;step<max_tokens;++step){
        auto t_step_start=std::chrono::steady_clock::now();
        int L=(int)ids.size();
        std::vector<int> in2(SEQ);
        for(int i=0;i<SEQ;++i){int idx=L-SEQ+i; in2[i]=(idx<0)?PAD:ids[idx];}
        std::vector<int> inBL(BL);
        for(int t=0;t<SEQ;++t) inBL[t]=in2[t];
        forward(m, inBL, BATCH, SEQ, PAD, D, NL, V_unit,
                x, x_prev, y, alpha, h, s, hg, sg, logits, probs,
                xs, ys, alphas, hs, ss, hgs, sgs, q1_aux);
        int bt=(BATCH-1)*SEQ+(SEQ-1);
        std::vector<float> adj_logit(V_unit);
        for(int v=0;v<V_unit;++v) adj_logit[v]=logits[bt*V_unit+v]/T;
        for(size_t back=0; back<ids.size() && back<6; ++back){
            int tk = ids[ids.size()-1-back];
            if(tk>=2 && tk<V_unit){float penalty=3.0f*std::pow(0.65f,(float)back); adj_logit[tk]-=penalty;}
        }
        std::vector<int> idx(V_unit);
        std::iota(idx.begin(),idx.end(),0);
        std::sort(idx.begin(),idx.end(),[&](int a,int b){return adj_logit[a]>adj_logit[b];});
        float mx=adj_logit[idx[0]];
        std::vector<float> probs2(V_unit);
        float sum=0;
        for(int v=0;v<V_unit;++v){probs2[v]=std::exp(adj_logit[v]-mx);sum+=probs2[v];}
        for(int v=0;v<V_unit;++v) probs2[v]/=sum;
        float cum=0; int nuc=V_unit;
        for(int i=0;i<V_unit;++i){cum+=probs2[idx[i]]; if(cum>=top_p){nuc=i+1;break;}}
        std::vector<float> nuc_probs(nuc);
        for(int i=0;i<nuc;++i) nuc_probs[i]=probs2[idx[i]];
        float ns=0; for(int i=0;i<nuc;++i) ns+=nuc_probs[i];
        for(int i=0;i<nuc;++i) nuc_probs[i]/=ns;
        std::discrete_distribution<int> dist(nuc_probs.begin(),nuc_probs.end());
        int best=idx[dist(rng)];
        ids.push_back(best);
        auto t_step_end=std::chrono::steady_clock::now();
        double step_ms=std::chrono::duration<double,std::milli>(t_step_end-t_step_start).count();
        if(streaming){
            std::string tok_str=vocab.decode({best});
            // Sanitize token string for line protocol
            for(char& c:tok_str){if(c=='\n')c='/'; if(c=='\r')c='/'; if(c=='\t')c=' ';}
            auto t_now=std::chrono::steady_clock::now();
            double elapsed_total_ms=std::chrono::duration<double,std::milli>(t_now-t_total_start).count();
            std::printf("TOKEN %d %d %.2fms %.2fms %s\n", step, best, step_ms, elapsed_total_ms, tok_str.c_str());
            std::fflush(stdout);
        }
    }
    // Decode generated tokens only (skip prompt tokens)
    std::vector<int> gen_ids(ids.begin() + (int)prompt.size()/2, ids.end());
    // Actually safer: decode ALL and strip the prompt part via token count
    // Encode prompt again to get exact token count
    std::vector<int> prompt_ids=vocab.encode(prompt);
    int pcount=(int)prompt_ids.size();
    std::vector<int> new_part(ids.begin()+pcount, ids.end());
    std::string response=vocab.decode(new_part);
    // Trim leading whitespace
    while(!response.empty() && std::isspace((unsigned char)response[0])) response.erase(response.begin());
    
    if(!streaming){
        std::printf("RESPONSE:\n%s\n",response.c_str());
    } else {
        std::printf("DONE\n");
    }
    return 0;
}
