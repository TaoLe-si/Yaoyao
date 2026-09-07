// Step 12: 夭夭 v0.9 - Word-level vocab + larger pretraining
// Architecture unchanged, vocab upgraded to word-level (1024 most common words)
#define D_H 128
#define NL_H 2
#define Q1_B 128
#define Q1_K 16
#define Q3_K 5   // Conv kernel size (was 3, now 5)
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
    std::vector<float> q3w[Q3_K];
    std::vector<float> aW, ab;
    std::vector<float> Wh_m, Wh_v, Ws_m, Ws_v, Wbi_m, Wbi_v;
    std::vector<float> q3w_m[Q3_K], q3w_v[Q3_K];
    std::vector<float> aW_m, aW_v, ab_m, ab_v;
    std::vector<float> q1_grad;
    int step=0;
    int V_unit=0;
        void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if(!f){std::fprintf(stderr,"Cannot open for save\n");return;}
        int magic=0x59414F59; f.write((char*)&magic,4);
        int version=2; f.write((char*)&version,4);
        int v=V_unit; f.write((char*)&v,4);
        auto wr=[&](const void* p, size_t n){f.write((char*)p,n);};
        wr(Wh.data(),Wh.size()*4); wr(Wh_m.data(),Wh_m.size()*4); wr(Wh_v.data(),Wh_v.size()*4);
        wr(Ws.data(),Ws.size()*4); wr(Ws_m.data(),Ws_m.size()*4); wr(Ws_v.data(),Ws_v.size()*4);
        wr(Wbi.data(),Wbi.size()*4); wr(Wbi_m.data(),Wbi_m.size()*4); wr(Wbi_v.data(),Wbi_v.size()*4);
        for(int kk=0;kk<Q3_K;++kk){wr(q3w[kk].data(),q3w[kk].size()*4); wr(q3w_m[kk].data(),q3w_m[kk].size()*4); wr(q3w_v[kk].data(),q3w_v[kk].size()*4);}
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
        int version; f.read((char*)&version,4); if(version!=1 && version!=2) return false;
        int v; f.read((char*)&v,4);
        auto rd=[&](void* p, size_t n){f.read((char*)p,n);};
        rd(Wh.data(),Wh.size()*4); rd(Wh_m.data(),Wh_m.size()*4); rd(Wh_v.data(),Wh_v.size()*4);
        rd(Ws.data(),Ws.size()*4); rd(Ws_m.data(),Ws_m.size()*4); rd(Ws_v.data(),Ws_v.size()*4);
        rd(Wbi.data(),Wbi.size()*4); rd(Wbi_m.data(),Wbi_m.size()*4); rd(Wbi_v.data(),Wbi_v.size()*4);
        // Read Q3_K weight sets (v1: 3 sets, v2: 5 sets - all stored as Q3_K float arrays)
        // v11 stored q3w with INDEX=k being x[t-k] (k=0 newest, k=2 oldest). v12 uses SAME convention.
        // v11 file order: q3w0 (oldest, x[t-2]), q3w1 (x[t-1]), q3w2 (newest, x[t])
        // v12 array order: q3w[0] newest, q3w[Q3_K-1] oldest
        // So v11 idx i (file slot i) → v12 slot: for i=0 (v11's oldest) → v12's q3w[2]
        int kk_limit=(version==1)?3:Q3_K;
        for(int kk=0;kk<kk_limit;++kk){
            int target_kk = (version==1) ? (kk_limit-1-kk) : kk;
            // Initialize target to 0 if not loaded
            if(version==1 && (target_kk >= Q3_K)){ continue; }
            rd(q3w[target_kk].data(),q3w[target_kk].size()*4);
            rd(q3w_m[target_kk].data(),q3w_m[target_kk].size()*4);
            rd(q3w_v[target_kk].data(),q3w_v[target_kk].size()*4);
        }
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
        for(int kk=0;kk<Q3_K;++kk) q3w[kk].assign(NL*D, 0);
        aW.assign(NL*D*D, 0); ab.assign(NL*D, 0);
        Wh_m.assign(V*D,0); Wh_v.assign(V*D,0);
        Ws_m.assign(V*D,0); Ws_v.assign(V*D,0);
        Wbi_m.assign(V*V,0); Wbi_v.assign(V*V,0);
        for(int kk=0;kk<Q3_K;++kk){q3w_m[kk].assign(NL*D,0); q3w_v[kk].assign(NL*D,0);}
        aW_m.assign(NL*D*D,0); aW_v.assign(NL*D*D,0);
        ab_m.assign(NL*D,0); ab_v.assign(NL*D,0);
        std::normal_distribution<float> nde(0,0.3f), ndw(0,0.1f);
        for(auto& x:Wh) x=ndw(rng);
        for(auto& x:Ws) x=ndw(rng);
        std::vector<float> tmp(D);
        for(int l=0;l<NL;++l){
            for(auto& x:tmp) x=ndw(rng)*0.3f;
            for(int d=0;d<D;++d){float w=tmp[d]; for(int kk=0;kk<Q3_K;++kk) q3w[kk][l*D+d]=w;}
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
                for(int kk=0;kk<Q3_K;++kk){ if(t>=kk) v+=m.q3w[kk][l*D+d]*x[(b*SEQ+t-kk)*D+d]; }
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
    #pragma omp parallel for
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
        if(!(sum>0)) { std::fprintf(stderr,"NaN at softmax n=%d mx=%f sum=%f\n",n,mx,sum); }
        for(int v=0;v<V_unit;++v) probs[n*V_unit+v]/=sum;
    }
}

int main(int argc, char** argv){
    const int D=D_H, NL=NL_H;
    std::mt19937 rng(42);
    
    // Load text
    const char* path=(argc>1)?argv[1]:"D:\\TaoVm\\tinystories_train.txt";
    std::printf("Loading %s ...\n", path);
    std::ifstream f(path, std::ios::binary);
    std::stringstream sstrm; sstrm<<f.rdbuf();
    std::string text=sstrm.str();
    std::printf("Loaded %zu chars\n", text.size());
    
    // Build vocab from sample
    Vocab vocab;
    vocab.build(text.substr(0, std::min<size_t>(text.size(), 5000000)), 1024);
    int V_unit=(int)vocab.id_to_word.size();
    std::printf("Vocab=%d tokens (word-level)\n", V_unit);
    std::printf("Sample tokens: ");
    for(int i=2;i<std::min(12,V_unit);++i) std::printf("'%s' ", vocab.id_to_word[i].c_str());
    std::printf("\n");
    
    // Encode sample (5M chars)
    std::string sample_text=text.substr(0, std::min<size_t>(text.size(), 5000000));
    std::vector<int> tokens=vocab.encode(sample_text);
    std::printf("Encoded %zu tokens\n", tokens.size());
    
    M m; m.init(rng, V_unit); m.V_unit=V_unit;
    const char* loadpath=(argc>2)?argv[2]:nullptr;
    if(loadpath) m.load(loadpath);
    std::printf("Model: V=%d D=%d NL=%d Q1[B=%d,K=%d]\n", V_unit, D, NL, Q1_B, Q1_K);
    
    int SEQ=64, BATCH=16, BL=SEQ*BATCH;
    int N_WIN=(argc>3)?atoi(argv[3]):2000;
    int EPOCHS=(argc>4)?atoi(argv[4]):2;
    float LR=(argc>5)?(float)atof(argv[5]):0.003f; float LR_ALPHA=(argc>6)?(float)atof(argv[6]):0.0003f;
    int PAD=vocab.pad_id;
    std::printf("Config: BATCH=%d SEQ=%d N_WIN=%d EPOCHS=%d LR=%.5f LR_ALPHA=%.5f\n", BATCH, SEQ, N_WIN, EPOCHS, LR, LR_ALPHA);
    
    std::vector<float> x(BL*D), x_prev(D), y(BL*D), alpha(BL*D);
    std::vector<float> h(BATCH*(SEQ+1)*D), s(BATCH*(SEQ+1)*D);
    std::vector<float> hg(BL*D), sg(BL*D), logits(BL*V_unit), probs(BL*V_unit);
    std::vector<float> d_logits(BL*V_unit), d_hg(BL*D), d_sg(BL*D);
    std::vector<float> d_h(BATCH*(SEQ+1)*D), d_s(BATCH*(SEQ+1)*D);
    std::vector<float> d_alpha(BL*D), d_y(BL*D), d_x(BL*D);
    std::vector<float> xs(NL*BL*D), ys(NL*BL*D), alphas(NL*BL*D);
    std::vector<float> hs(NL*BATCH*(SEQ+1)*D), ss(NL*BATCH*(SEQ+1)*D);
    std::vector<float> hgs(NL*BL*D), sgs(NL*BL*D);
    std::vector<float> d_alphas(NL*BL*D), d_ys(NL*BL*D);
    std::vector<float> d_q3w_g[Q3_K]; for(int kk=0;kk<Q3_K;++kk) d_q3w_g[kk].assign(NL*D, 0);
    std::vector<float> d_aW(NL*D*D,0), d_ab(NL*D,0);
    std::vector<float> d_Wh_g(V_unit*D,0), d_Ws_g(V_unit*D,0), d_Wbi_g(V_unit*V_unit,0);
    std::vector<Q1::Aux> q1_aux(BL);
    
    float b1=0.9f, b2=0.999f, eps=1e-8f;
    auto t0=std::chrono::steady_clock::now();
    
    std::vector<int> all_in(N_WIN*BATCH*SEQ), all_tg(N_WIN*BATCH*SEQ);
    for(int epoch=0;epoch<EPOCHS;++epoch){
        std::uniform_int_distribution<int> udist(0, (int)tokens.size()-N_WIN*SEQ-SEQ-1);
        int offset=udist(rng);
        for(int i=0;i<N_WIN;++i) for(int b=0;b<BATCH;++b){
            int start=offset+i*SEQ+b*7;  // shift by 7 for each batch (different story slices)
            if(start+SEQ+1>=(int)tokens.size()) start=offset+i*SEQ;
            for(int j=0;j<SEQ;++j){
                all_in[(i*BATCH+b)*SEQ+j]=tokens[start+j];
                all_tg[(i*BATCH+b)*SEQ+j]=tokens[start+j+1];
            }
        }
        float total=0; int nb=0;
        for(int w=0;w<N_WIN;++w){
            std::vector<int> inpBL(BL), tgtBL(BL);
            for(int n=0;n<BL;++n){inpBL[n]=all_in[w*BL+n]; tgtBL[n]=all_tg[w*BL+n];}
            
            forward(m, inpBL, BATCH, SEQ, PAD, D, NL, V_unit,
                    x, x_prev, y, alpha, h, s, hg, sg, logits, probs,
                    xs, ys, alphas, hs, ss, hgs, sgs, q1_aux);
            
            float loss=0;
            for(int n=0;n<BL;++n){int t=tgtBL[n]; float p=std::max(probs[n*V_unit+t], 1e-9f); if(std::isnan(p)){std::fprintf(stderr,"NaN probs at n=%d tgt=%d p=%f\n",n,t,p); break;} loss+=-std::log(p);}
            loss/=BL; total+=loss; nb++; m.step++;
            
            for(int n=0;n<BL;++n){for(int v=0;v<V_unit;++v) d_logits[n*V_unit+v]=probs[n*V_unit+v]; d_logits[n*V_unit+tgtBL[n]]-=1.0f;}
            // W_bi accumulation: race condition on +=, keep sequential
    for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t){int prev=(t>0)?inpBL[(b*SEQ+t-1)]:PAD; for(int v=0;v<V_unit;++v) d_Wbi_g[prev*V_unit+v]+=d_logits[(b*SEQ+t)*V_unit+v];}
            #pragma omp parallel for
    for(int n=0;n<BL;++n){for(int d=0;d<D;++d){float s1=0,s2=0; for(int v=0;v<V_unit;++v){s1+=d_logits[n*V_unit+v]*m.Wh[v*D+d]; s2+=d_logits[n*V_unit+v]*m.Ws[v*D+d];} d_hg[n*D+d]=s1; d_sg[n*D+d]=s2;}}
            #pragma omp parallel for collapse(2) schedule(static)
    for(int v=0;v<V_unit;++v) for(int d=0;d<D;++d){float s1=0,s2=0; for(int n=0;n<BL;++n){s1+=d_logits[n*V_unit+v]*hg[n*D+d]; s2+=d_logits[n*V_unit+v]*sg[n*D+d];} d_Wh_g[v*D+d]=s1; d_Ws_g[v*D+d]=s2;}
            
            for(int l=NL-1;l>=0;--l){
                std::fill(d_x.begin(), d_x.end(), 0.0f);
                std::fill(d_alpha.begin(), d_alpha.end(), 0.0f);
                // Loop over all batches - batch bb
                for(int bb=0;bb<BATCH;++bb){
                int boff=bb*SEQ;
                std::vector<float> h_seq(SEQ*D), s_seq(SEQ*D);
                for(int tt=0;tt<SEQ;++tt) for(int d=0;d<D;++d){
                    h_seq[tt*D+d]=h[(bb*(SEQ+1)+tt+1)*D+d];
                    s_seq[tt*D+d]=s[(bb*(SEQ+1)+tt+1)*D+d];
                }
                std::vector<float> d_h_seq(SEQ*D,0), d_s_seq(SEQ*D,0);
                for(int tt=0;tt<SEQ;++tt){
                    float ms_h=0, ms_s=0;
                    for(int d=0;d<D;++d){float v=h_seq[tt*D+d]; ms_h+=v*v;}
                    ms_h=ms_h/(float)D+1e-5f; float r_h=std::sqrt(ms_h);
                    for(int d=0;d<D;++d){float v=s_seq[tt*D+d]; ms_s+=v*v;}
                    ms_s=ms_s/(float)D+1e-5f; float r_s=std::sqrt(ms_s);
                    float dot_h=0, dot_s=0;
                    for(int d=0;d<D;++d){dot_h+=d_hg[(boff+tt)*D+d]*h_seq[tt*D+d]; dot_s+=d_sg[(boff+tt)*D+d]*s_seq[tt*D+d];}
                    for(int d=0;d<D;++d){
                        d_h_seq[tt*D+d]=d_hg[(boff+tt)*D+d]/r_h-h_seq[tt*D+d]*dot_h/((float)D*r_h*r_h*r_h);
                        d_s_seq[tt*D+d]=d_sg[(boff+tt)*D+d]/r_s-s_seq[tt*D+d]*dot_s/((float)D*r_s*r_s*r_s);
                    }
                }
                std::vector<float> cum_s((SEQ+1)*D,0);
                for(int tt=SEQ-1;tt>=0;--tt) for(int d=0;d<D;++d) cum_s[tt*D+d]=cum_s[(tt+1)*D+d]+d_s_seq[tt*D+d];
                std::vector<float> d_h_total((SEQ+1)*D,0);
                for(int tt=SEQ-1;tt>=0;--tt){
                    for(int d=0;d<D;++d){
                        d_h_total[(tt+1)*D+d] += d_h_seq[tt*D+d];
                        float a=alphas[l*BL*D+(boff+tt)*D+d];
                        float h_prev=hs[l*BATCH*(SEQ+1)*D+(bb*(SEQ+1)+tt)*D+d];
                        float dh=d_h_total[(tt+1)*D+d];
                        d_ys[l*BL*D+(boff+tt)*D+d]=(1-a)*dh+cum_s[tt*D+d];
                        d_alphas[l*BL*D+(boff+tt)*D+d]=(h_prev-ys[l*BL*D+(boff+tt)*D+d])*dh;
                        d_h_total[tt*D+d] += a*dh;
                    }
                }
                for(int d=0;d<D;++d){
                    for(int tt=0;tt<SEQ;++tt){
                        float dy=d_ys[l*BL*D+(boff+tt)*D+d];
                        for(int kk=0;kk<Q3_K;++kk){ if(tt>=kk){ float ww=m.q3w[kk][l*D+d]; int xidx=(boff+tt-kk)*D+d; d_q3w_g[kk][l*D+d]+=dy*xs[l*BL*D+xidx]; d_x[xidx]+=dy*ww; } }
                    }
                    // (accumulation moved into Q3 backward loop above)
                }
                for(int tt=0;tt<SEQ;++tt){
                    for(int d=0;d<D;++d){
                        float a=alphas[l*BL*D+(boff+tt)*D+d];
                        d_alpha[(boff+tt)*D+d]=d_alphas[l*BL*D+(boff+tt)*D+d]*a*(1-a)/2.0f;
                    }
                }
                for(int d=0;d<D;++d) for(int k=0;k<D;++k){
                    float s=0;
                    for(int tt=0;tt<SEQ;++tt) s+=d_alpha[(boff+tt)*D+d]*xs[l*BL*D+(boff+tt)*D+k];
                    d_aW[l*D*D+d*D+k]+=s;
                }
                for(int d=0;d<D;++d){float s=0; for(int tt=0;tt<SEQ;++tt) s+=d_alpha[(boff+tt)*D+d]; d_ab[l*D+d]+=s;}
                for(int tt=0;tt<SEQ;++tt){
                    for(int d=0;d<D;++d){float dz=d_alpha[(boff+tt)*D+d]; for(int k=0;k<D;++k) d_x[(boff+tt)*D+k]+=dz*m.aW[l*D*D+d*D+k];}
                }
                } // end bb loop
                if(l>0){for(int n=0;n<BL;++n) for(int d=0;d<D;++d) d_hg[n*D+d]=d_sg[n*D+d]=d_x[n*D+d];}
                else{
                    for(int t=0;t<BL;++t){
                        std::vector<float> d_q(D,0);
                        m.q1.backward(q1_aux[t], d_x.data()+t*D, m.q1_grad, d_q.data());
                    }
                }
            }
            
            float bc1=1-std::pow(b1,(float)m.step), bc2=1-std::pow(b2,(float)m.step);
            for(int i=0;i<V_unit*V_unit;++i){float g=d_Wbi_g[i]; if(g>1)g=1; if(g<-1)g=-1; m.Wbi_m[i]=b1*m.Wbi_m[i]+(1-b1)*g; m.Wbi_v[i]=b2*m.Wbi_v[i]+(1-b2)*g*g; float st=LR*(m.Wbi_m[i]/bc1)/(std::sqrt(m.Wbi_v[i]/bc2)+eps); m.Wbi[i]=std::min(std::max(m.Wbi[i]-st,-8.0f),8.0f);}
            for(int i=0;i<V_unit*D;++i){float g=d_Wh_g[i]; if(g>1)g=1; if(g<-1)g=-1; m.Wh_m[i]=b1*m.Wh_m[i]+(1-b1)*g; m.Wh_v[i]=b2*m.Wh_v[i]+(1-b2)*g*g; float st=LR*(m.Wh_m[i]/bc1)/(std::sqrt(m.Wh_v[i]/bc2)+eps); m.Wh[i]=std::min(std::max(m.Wh[i]-st,-1.0f),1.0f);}
            for(int i=0;i<V_unit*D;++i){float g=d_Ws_g[i]; if(g>1)g=1; if(g<-1)g=-1; m.Ws_m[i]=b1*m.Ws_m[i]+(1-b1)*g; m.Ws_v[i]=b2*m.Ws_v[i]+(1-b2)*g*g; float st=LR*(m.Ws_m[i]/bc1)/(std::sqrt(m.Ws_v[i]/bc2)+eps); m.Ws[i]=std::min(std::max(m.Ws[i]-st,-1.0f),1.0f);}
            for(int kk=0;kk<Q3_K;++kk){for(int i=0;i<NL*D;++i){float g=d_q3w_g[kk][i]; if(g>1)g=1; if(g<-1)g=-1; m.q3w_m[kk][i]=b1*m.q3w_m[kk][i]+(1-b1)*g; m.q3w_v[kk][i]=b2*m.q3w_v[kk][i]+(1-b2)*g*g; float st=LR*(m.q3w_m[kk][i]/bc1)/(std::sqrt(m.q3w_v[kk][i]/bc2)+eps); m.q3w[kk][i]=std::min(std::max(m.q3w[kk][i]-st,-2.0f),2.0f);}}
            for(int i=0;i<NL*D*D;++i){float g=d_aW[i]; if(g>1)g=1; if(g<-1)g=-1; m.aW_m[i]=b1*m.aW_m[i]+(1-b1)*g; m.aW_v[i]=b2*m.aW_v[i]+(1-b2)*g*g; float st=LR_ALPHA*(m.aW_m[i]/bc1)/(std::sqrt(m.aW_v[i]/bc2)+eps); m.aW[i]=std::min(std::max(m.aW[i]-st,-4.0f),4.0f);}
            for(int i=0;i<NL*D;++i){float g=d_ab[i]; if(g>1)g=1; if(g<-1)g=-1; m.ab_m[i]=b1*m.ab_m[i]+(1-b1)*g; m.ab_v[i]=b2*m.ab_v[i]+(1-b2)*g*g; float st=LR_ALPHA*(m.ab_m[i]/bc1)/(std::sqrt(m.ab_v[i]/bc2)+eps); m.ab[i]=std::min(std::max(m.ab[i]-st,-8.0f),8.0f);}
            m.q1.adam_update(m.q1_grad, LR, b1, b2, eps);
            
            std::fill(d_Wbi_g.begin(),d_Wbi_g.end(),0.0f);
            std::fill(d_Wh_g.begin(),d_Wh_g.end(),0.0f); std::fill(d_Ws_g.begin(),d_Ws_g.end(),0.0f);
            for(int kk=0;kk<Q3_K;++kk) std::fill(d_q3w_g[kk].begin(),d_q3w_g[kk].end(),0.0f); std::fill(d_aW.begin(),d_aW.end(),0.0f);
            std::fill(d_ab.begin(),d_ab.end(),0.0f);
            std::fill(m.q1_grad.begin(),m.q1_grad.end(),0.0f);
            
            // UNIT TEST: print every 5 windows
            if((w+1)%5==0||w==0){
                float el=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
                std::printf("  ep%d win%d/%d loss=%.4f avg=%.4f (%.1fs)\n", epoch+1, w+1, N_WIN, loss, total/nb, el);
            }
        }
        float avg=total/nb;
        std::printf("Epoch %2d/%d avg_loss=%.4f\n", epoch+1, EPOCHS, avg);
    }
    
    // Generation
    std::printf("\n=== Generation ===\n");
    std::vector<std::string> prompts={"Once upon a time", "The little girl", "He was very", "Lily and", "Today was"};
    for(const auto& prompt:prompts){
        std::vector<int> prompt_ids=vocab.encode(prompt);
        std::printf("Prompt: '%s'\n", prompt.c_str());
        std::vector<int> ids=prompt_ids;
    for(int step=0;step<40;++step){
        int L=(int)ids.size();
        std::vector<int> in2(SEQ);
        for(int i=0;i<SEQ;++i){int idx=L-SEQ+i; in2[i]=(idx<0)?PAD:ids[idx];}
        std::vector<int> inBL(BL);
        for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t) inBL[b*SEQ+t]=in2[t];
        forward(m, inBL, BATCH, SEQ, PAD, D, NL, V_unit,
                x, x_prev, y, alpha, h, s, hg, sg, logits, probs,
                xs, ys, alphas, hs, ss, hgs, sgs, q1_aux);
        int bt=(BATCH-1)*SEQ+(SEQ-1);
        // Gentle repetition penalty + top-p (nucleus) sampling
        float T=0.9f;
        std::vector<float> adj_logit(V_unit);
        for(int v=0;v<V_unit;++v) adj_logit[v]=logits[bt*V_unit+v]/T;
        for(size_t back=0; back<ids.size() && back<6; ++back){
            int tk = ids[ids.size()-1-back];
            if(tk>=2 && tk<V_unit){
                float penalty = 3.0f * std::pow(0.65f, (float)back);
                adj_logit[tk] -= penalty;
            }
        }
        // Top-p (nucleus) sampling
        std::vector<int> idx(V_unit);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
            [&](int a, int b){return adj_logit[a]>adj_logit[b];});
        // Compute softmax probabilities
        float mx=adj_logit[idx[0]];
        std::vector<float> probs(V_unit);
        float sum=0;
        for(int v=0;v<V_unit;++v){ probs[v]=std::exp(adj_logit[v]-mx); sum+=probs[v]; }
        for(int v=0;v<V_unit;++v) probs[v]/=sum;
        // Find nucleus: smallest set with cumulative prob >= p
        float p=0.9f;
        float cum=0;
        int nuc_size=V_unit;
        for(int i=0;i<V_unit;++i){
            cum+=probs[idx[i]];
            if(cum>=p){ nuc_size=i+1; break; }
        }
        // Sample from nucleus
        std::vector<float> nuc_probs(nuc_size);
        for(int i=0;i<nuc_size;++i) nuc_probs[i]=probs[idx[i]];
        // Renormalize
        float nuc_sum=0; for(int i=0;i<nuc_size;++i) nuc_sum+=nuc_probs[i];
        for(int i=0;i<nuc_size;++i) nuc_probs[i]/=nuc_sum;
        std::discrete_distribution<int> dist(nuc_probs.begin(), nuc_probs.end());
        int best=idx[dist(rng)];
        ids.push_back(best);
    }
    std::string gen=vocab.decode(ids);
        std::printf("Generated: \"%s\"\n\n", gen.c_str());
    }
        m.save("D:\\TaoVm\\yaoyao_v09_model.bin");
return 0;
}
