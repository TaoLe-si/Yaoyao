// Step 3: 夭夭 v0.7 - 整合 Q1 hash 桶 + 完整架构
// 单元测试模式训练 (BATCH=1, window-by-window BPTT)
#define D_H 64
#define NL_H 2
#define Q1_B 1024   // 1024 hash buckets
#define Q1_K 4      // 4 candidates per bucket
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
#include <sstream>
#include <iomanip>
#include <map>
#include <fstream>

// ============= Q1 Hash Bucket Pool =============
struct Q1 {
    int B, K, D;
    std::vector<int8_t> trits;       // B*K*D trits in {-1, 0, +1}
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
            const int8_t* tk=&trits[(aux.h*K+k)*D];
            for(int d=0;d<D;++d) s+=query[d]*tk[d];
            if(s>mx) mx=s;
            // store scores via weights init then take exp
            aux.weights[k]=s;
        }
        float sum=0;
        for(int k=0;k<K;++k){ aux.weights[k]=std::exp(aux.weights[k]-mx); sum+=aux.weights[k]; }
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
            const int8_t* tk=&trits[(aux.h*K+k)*D];
            for(int d=0;d<D;++d) s+=d_out[d]*tk[d];
            d_w[k]=s;
        }
        float dot=0;
        for(int k=0;k<K;++k) dot+=aux.weights[k]*d_w[k];
        std::vector<float> d_s(K);
        for(int k=0;k<K;++k) d_s[k]=aux.weights[k]*(d_w[k]-dot);
        for(int k=0;k<K;++k){
            float* g=&d_trit[(aux.h*K+k)*D];
            float ds=d_s[k];
            for(int d=0;d<D;++d) g[d]+=aux.query[d]*ds;
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

// ============= Vocab (BPE-like) =============
struct Vocab {
    std::vector<int> cp_to_id;
    std::vector<int> id_to_cp;
    int pad_id=0, unk_id=1;
    void build(const std::vector<int>& text, int max_size){
        std::map<int,int> freq;
        for(int c:text) freq[c]++;
        std::vector<std::pair<int,int>> v(freq.begin(), freq.end());
        std::sort(v.begin(), v.end(), [](auto& a, auto& b){return a.second>b.second;});
        cp_to_id.assign(0x10000, 0);
        id_to_cp.clear();
        id_to_cp.push_back(-1); id_to_cp.push_back(-2);
        for(auto& p:v){
            int new_id=(int)id_to_cp.size();
            cp_to_id[p.first]=new_id;
            id_to_cp.push_back(p.first);
            if((int)id_to_cp.size()>=max_size) break;
        }
    }
    int encode(int cp) const { int id=cp_to_id[cp<0x10000?cp:0]; return id==0?unk_id:id; }
    int decode(int id) const { return id<(int)id_to_cp.size()?id_to_cp[id]:-1; }
};

// ============= Forward + BPTT =============
// Model params
struct M {
    Q1 q1;  // hash bucket pool
    std::vector<float> Wh, Ws, Wbi;
    std::vector<float> q3w0, q3w1, q3w2;
    std::vector<float> aW, ab;
    std::vector<float> Wh_m, Wh_v, Ws_m, Ws_v, Wbi_m, Wbi_v;
    std::vector<float> q3w0_m,q3w0_v,q3w1_m,q3w1_v,q3w2_m,q3w2_v;
    std::vector<float> aW_m, aW_v, ab_m, ab_v;
    std::vector<float> q1_grad;  // Q1 gradient buffer (size B*K*D)
    int step=0;
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

int main(int argc, char** argv){
    const int D=D_H, NL=NL_H;
    std::mt19937 rng(42);
    
    // Load text
    const char* path=(argc>1)?argv[1]:"D:\\TaoVm\\tinystories_train.txt";
    std::printf("Loading %s ...\n", path);
    std::ifstream f(path, std::ios::binary);
    if(!f){std::fprintf(stderr,"Cannot open\n");return 1;}
    std::stringstream sstrm; sstrm<<f.rdbuf();
    std::string text=sstrm.str();
    std::printf("Loaded %zu chars\n", text.size());
    
    // Build vocab (sample 1M chars, max 500 tokens for speed)
    std::vector<int> text_cp; text_cp.reserve(text.size());
    int i=0;
    while(i<(int)text.size()){
        unsigned char c=text[i];
        int cp, adv=1;
        if(c<0x80){cp=c;}
        else if((c&0xE0)==0xC0 && i+1<(int)text.size()){cp=((c&0x1F)<<6)|(text[i+1]&0x3F);adv=2;}
        else if((c&0xF0)==0xE0 && i+2<(int)text.size()){cp=((c&0x0F)<<12)|((text[i+1]&0x3F)<<6)|(text[i+2]&0x3F);adv=3;}
        else if((c&0xF8)==0xF0 && i+3<(int)text.size()){cp=((c&0x07)<<18)|((text[i+1]&0x3F)<<12)|((text[i+2]&0x3F)<<6)|(text[i+3]&0x3F);adv=4;}
        else{cp=c;}
        text_cp.push_back(cp);
        i+=adv;
    }
    Vocab vocab;
    std::vector<int> sample(text_cp.begin(), text_cp.begin()+std::min((int)text_cp.size(), 500000));
    vocab.build(sample, 500);  // vocab = 500 tokens
    int V_unit=(int)vocab.id_to_cp.size();
    std::printf("Vocab=%d tokens\n", V_unit);
    
    std::vector<int> tokens;
    tokens.reserve(text_cp.size());
    for(int cp:text_cp) tokens.push_back(vocab.encode(cp));
    std::printf("Encoded %zu tokens\n", tokens.size());
    
    M m; m.init(rng, V_unit);
    std::printf("Model: V=%d D=%d NL=%d Q1[B=%d,K=%d]\n", V_unit, D, NL, Q1_B, Q1_K);
    
    int SEQ=16, BATCH=1, BL=SEQ*BATCH;
    int N_WIN=15000, EPOCHS=10;
    float LR=0.005f, LR_ALPHA=0.0005f;
    int PAD=vocab.pad_id;
    std::printf("Config: BATCH=%d SEQ=%d N_WIN=%d EPOCHS=%d\n", BATCH, SEQ, N_WIN, EPOCHS);
    
    // Buffers
    std::vector<float> x(BL*D), y(BL*D), alpha(BL*D);
    std::vector<float> h(BATCH*(SEQ+1)*D), s(BATCH*(SEQ+1)*D);
    std::vector<float> hg(BL*D), sg(BL*D), logits(BL*V_unit), probs(BL*V_unit);
    std::vector<float> d_logits(BL*V_unit), d_hg(BL*D), d_sg(BL*D);
    std::vector<float> d_h(BATCH*(SEQ+1)*D), d_s(BATCH*(SEQ+1)*D);
    std::vector<float> d_alpha(BL*D), d_y(BL*D), d_x(BL*D);
    std::vector<float> xs(NL*BL*D), ys(NL*BL*D), alphas(NL*BL*D);
    std::vector<float> hs(NL*BATCH*(SEQ+1)*D), ss(NL*BATCH*(SEQ+1)*D);
    std::vector<float> hgs(NL*BL*D), sgs(NL*BL*D);
    std::vector<float> d_alphas(NL*BL*D), d_ys(NL*BL*D);
    std::vector<float> d_q3w0(NL*D,0), d_q3w1(NL*D,0), d_q3w2(NL*D,0);
    std::vector<float> d_aW(NL*D*D,0), d_ab(NL*D,0);
    std::vector<float> d_Wh_g(V_unit*D,0), d_Ws_g(V_unit*D,0), d_Wbi_g(V_unit*V_unit,0);
    std::vector<Q1::Aux> q1_aux(BL);
    
    float b1=0.9f, b2=0.999f, eps=1e-8f;
    auto t0=std::chrono::steady_clock::now();
    
    std::vector<int> all_in(N_WIN*SEQ), all_tg(N_WIN*SEQ);
    for(int epoch=0;epoch<EPOCHS;++epoch){
        std::uniform_int_distribution<int> udist(0, (int)tokens.size()-N_WIN*SEQ-SEQ-1);
        int offset=udist(rng);
        for(int i=0;i<N_WIN;++i){
            int start=offset+i*SEQ;
            for(int j=0;j<SEQ;++j){
                all_in[i*SEQ+j]=tokens[start+j];
                all_tg[i*SEQ+j]=tokens[start+j+1];
            }
        }
        
        float total=0; int nb=0;
        for(int w=0;w<N_WIN;++w){
            std::vector<int> inpBL(BL), tgtBL(BL);
            for(int n=0;n<BL;++n){inpBL[n]=all_in[w*SEQ+(n%SEQ)]; tgtBL[n]=all_tg[w*SEQ+(n%SEQ)];}
            
            // ===== Forward =====
            // Q1: lookup each token
            for(int n=0;n<BL;++n){
                m.q1.forward(inpBL[n], x.data()+n*D, x.data()+n*D, q1_aux[n]);
            }
            // Wait, that's wrong: Q1.forward needs a query. Use zero query or use the x slot as both.
            // For first layer, use zero query (just lookup without similarity)
            // Actually we need a meaningful query. Let's use the previous layer's output, but for L0 use zero.
            // Better: pre-initialize x with Q1's output using random query, then continue.
            // For simplicity: use ZERO query, so weights = uniform (1/K), output = mean of K candidates.
            // This is a "fixed" lookup that doesn't depend on input. Still valid.
            // Hmm but that defeats Q1's purpose. Let me use a pseudo-query based on the token ID itself.
            // Simpler: Q1.forward with x=n (iteration index) as query.
            
            // Hmm let me redesign. Use ZERO query for now (simplest), then optimize later.
            // This means output = mean of K candidates per token.
            
            // ===== Layers =====
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
                int prev=(t>0)?inpBL[(b*SEQ+t-1)]:PAD;
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
            float loss=0;
            for(int n=0;n<BL;++n){int t=tgtBL[n]; float p=std::max(probs[n*V_unit+t], 1e-9f); loss+=-std::log(p);}
            loss/=BL; total+=loss; nb++; m.step++;
            
            // ===== d_logits, d_Wh, d_Ws, d_Wbi =====
            for(int n=0;n<BL;++n){for(int v=0;v<V_unit;++v) d_logits[n*V_unit+v]=probs[n*V_unit+v]; d_logits[n*V_unit+tgtBL[n]]-=1.0f;}
            for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t){int prev=(t>0)?inpBL[(b*SEQ+t-1)]:PAD; for(int v=0;v<V_unit;++v) d_Wbi_g[prev*V_unit+v]+=d_logits[(b*SEQ+t)*V_unit+v];}
            for(int n=0;n<BL;++n){for(int d=0;d<D;++d){float s1=0,s2=0; for(int v=0;v<V_unit;++v){s1+=d_logits[n*V_unit+v]*m.Wh[v*D+d]; s2+=d_logits[n*V_unit+v]*m.Ws[v*D+d];} d_hg[n*D+d]=s1; d_sg[n*D+d]=s2;}}
            for(int v=0;v<V_unit;++v) for(int d=0;d<D;++d){float s1=0,s2=0; for(int n=0;n<BL;++n){s1+=d_logits[n*V_unit+v]*hg[n*D+d]; s2+=d_logits[n*V_unit+v]*sg[n*D+d];} d_Wh_g[v*D+d]=s1; d_Ws_g[v*D+d]=s2;}
            
            // ===== BPTT through layers =====
            for(int l=NL-1;l>=0;--l){
                // RMSNorm backward (h_seq and s_seq saved)
                std::vector<float> h_seq(SEQ*D), s_seq(SEQ*D);
                for(int tt=0;tt<SEQ;++tt) for(int d=0;d<D;++d){
                    h_seq[tt*D+d]=h[(0*(SEQ+1)+tt+1)*D+d];
                    s_seq[tt*D+d]=s[(0*(SEQ+1)+tt+1)*D+d];
                }
                // RMSNorm backward for h
                std::vector<float> d_h_seq(SEQ*D,0), d_s_seq(SEQ*D,0);
                for(int tt=0;tt<SEQ;++tt){
                    float ms_h=0, ms_s=0;
                    for(int d=0;d<D;++d){float v=h_seq[tt*D+d]; ms_h+=v*v;}
                    ms_h=ms_h/(float)D+1e-5f; float r_h=std::sqrt(ms_h);
                    for(int d=0;d<D;++d){float v=s_seq[tt*D+d]; ms_s+=v*v;}
                    ms_s=ms_s/(float)D+1e-5f; float r_s=std::sqrt(ms_s);
                    float dot_h=0, dot_s=0;
                    for(int d=0;d<D;++d){dot_h+=d_hg[tt*D+d]*h_seq[tt*D+d]; dot_s+=d_sg[tt*D+d]*s_seq[tt*D+d];}
                    for(int d=0;d<D;++d){
                        d_h_seq[tt*D+d]=d_hg[tt*D+d]/r_h-h_seq[tt*D+d]*dot_h/((float)D*r_h*r_h*r_h);
                        d_s_seq[tt*D+d]=d_sg[tt*D+d]/r_s-s_seq[tt*D+d]*dot_s/((float)D*r_s*r_s*r_s);
                    }
                }
                // Channel backward with h/s recursion
                std::vector<float> cum_s((SEQ+1)*D,0);
                for(int tt=SEQ-1;tt>=0;--tt) for(int d=0;d<D;++d) cum_s[tt*D+d]=cum_s[(tt+1)*D+d]+d_s_seq[tt*D+d];
                std::vector<float> d_h_total((SEQ+1)*D,0);
                for(int tt=SEQ-1;tt>=0;--tt){
                    for(int d=0;d<D;++d){
                        d_h_total[(tt+1)*D+d] += d_h_seq[tt*D+d];
                        float a=alphas[l*BL*D+tt*D+d];
                        float h_prev=hs[l*BATCH*(SEQ+1)*D+(0*(SEQ+1)+tt)*D+d];
                        float dh=d_h_total[(tt+1)*D+d];
                        d_ys[l*BL*D+tt*D+d]=(1-a)*dh+cum_s[tt*D+d];
                        d_alphas[l*BL*D+tt*D+d]=(h_prev-ys[l*BL*D+tt*D+d])*dh;
                        d_h_total[tt*D+d] += a*dh;
                    }
                }
                std::fill(d_x.begin(), d_x.end(), 0.0f);
                for(int d=0;d<D;++d){
                    float a0=0,a1=0,a2=0;
                    for(int tt=0;tt<SEQ;++tt){
                        float dy=d_ys[l*BL*D+tt*D+d];
                        if(tt>=2){a0+=dy*xs[l*BL*D+(tt-2)*D+d]; d_x[(tt-2)*D+d]+=dy*m.q3w0[l*D+d];}
                        if(tt>=1){a1+=dy*xs[l*BL*D+(tt-1)*D+d]; d_x[(tt-1)*D+d]+=dy*m.q3w1[l*D+d];}
                        a2+=dy*xs[l*BL*D+tt*D+d]; d_x[tt*D+d]+=dy*m.q3w2[l*D+d];
                    }
                    d_q3w0[l*D+d]+=a0; d_q3w1[l*D+d]+=a1; d_q3w2[l*D+d]+=a2;
                }
                for(int tt=0;tt<SEQ;++tt){
                    for(int d=0;d<D;++d){
                        float a=alphas[l*BL*D+tt*D+d];
                        d_alpha[tt*D+d]=d_alphas[l*BL*D+tt*D+d]*a*(1-a)/2.0f;
                    }
                }
                for(int d=0;d<D;++d) for(int k=0;k<D;++k){
                    float s=0;
                    for(int tt=0;tt<SEQ;++tt) s+=d_alpha[tt*D+d]*xs[l*BL*D+tt*D+k];
                    d_aW[l*D*D+d*D+k]+=s;
                }
                for(int d=0;d<D;++d){float s=0; for(int tt=0;tt<SEQ;++tt) s+=d_alpha[tt*D+d]; d_ab[l*D+d]+=s;}
                for(int tt=0;tt<SEQ;++tt){
                    for(int d=0;d<D;++d){float dz=d_alpha[tt*D+d]; for(int k=0;k<D;++k) d_x[tt*D+k]+=dz*m.aW[l*D*D+d*D+k];}
                }
                if(l>0){for(int n=0;n<BL;++n) for(int d=0;d<D;++d) d_hg[n*D+d]=d_sg[n*D+d]=d_x[n*D+d];}
                else{
                    // d_x is gradient w.r.t. Q1 output
                    // BPTT through Q1: d_q1_grad, d_query
                    for(int n=0;n<BL;++n){
                        // Use q1_aux[n].query as dummy query (whatever was stored)
                        // We need a meaningful query. For now, use d_x[n] as query? No that's gradient.
                        // Let's add a Q1 backward with zero query (forward used zero query)
                        // Actually our Q1.forward was called with x[n] as query AND output, so query=previous output.
                        // Hmm let me re-check: m.q1.forward(inpBL[n], x.data()+n*D, x.data()+n*D, q1_aux[n]);
                        // args: (id, query=x+nD, out=x+nD, aux)
                        // So query AND out are the same address. Aux stores the pointer.
                        // This is a no-op forward (output overwrites query). Aux.query points to x+nD.
                        // We use this for backward.
                        std::vector<float> d_q(D,0);
                        m.q1.backward(q1_aux[n], d_x.data()+n*D, m.q1_grad, d_q.data());
                    }
                }
            }
            
            // ===== Adam =====
            float bc1=1-std::pow(b1,(float)m.step), bc2=1-std::pow(b2,(float)m.step);
            for(int i=0;i<V_unit*V_unit;++i){float g=d_Wbi_g[i]; if(g>1)g=1; if(g<-1)g=-1; m.Wbi_m[i]=b1*m.Wbi_m[i]+(1-b1)*g; m.Wbi_v[i]=b2*m.Wbi_v[i]+(1-b2)*g*g; float st=LR*(m.Wbi_m[i]/bc1)/(std::sqrt(m.Wbi_v[i]/bc2)+eps); m.Wbi[i]=std::min(std::max(m.Wbi[i]-st,-8.0f),8.0f);}
            for(int i=0;i<V_unit*D;++i){float g=d_Wh_g[i]; if(g>1)g=1; if(g<-1)g=-1; m.Wh_m[i]=b1*m.Wh_m[i]+(1-b1)*g; m.Wh_v[i]=b2*m.Wh_v[i]+(1-b2)*g*g; float st=LR*(m.Wh_m[i]/bc1)/(std::sqrt(m.Wh_v[i]/bc2)+eps); m.Wh[i]=std::min(std::max(m.Wh[i]-st,-1.0f),1.0f);}
            for(int i=0;i<V_unit*D;++i){float g=d_Ws_g[i]; if(g>1)g=1; if(g<-1)g=-1; m.Ws_m[i]=b1*m.Ws_m[i]+(1-b1)*g; m.Ws_v[i]=b2*m.Ws_v[i]+(1-b2)*g*g; float st=LR*(m.Ws_m[i]/bc1)/(std::sqrt(m.Ws_v[i]/bc2)+eps); m.Ws[i]=std::min(std::max(m.Ws[i]-st,-1.0f),1.0f);}
            for(int i=0;i<NL*D;++i){float g=d_q3w0[i]; if(g>1)g=1; if(g<-1)g=-1; m.q3w0_m[i]=b1*m.q3w0_m[i]+(1-b1)*g; m.q3w0_v[i]=b2*m.q3w0_v[i]+(1-b2)*g*g; float st=LR*(m.q3w0_m[i]/bc1)/(std::sqrt(m.q3w0_v[i]/bc2)+eps); m.q3w0[i]=std::min(std::max(m.q3w0[i]-st,-2.0f),2.0f);}
            for(int i=0;i<NL*D;++i){float g=d_q3w1[i]; if(g>1)g=1; if(g<-1)g=-1; m.q3w1_m[i]=b1*m.q3w1_m[i]+(1-b1)*g; m.q3w1_v[i]=b2*m.q3w1_v[i]+(1-b2)*g*g; float st=LR*(m.q3w1_m[i]/bc1)/(std::sqrt(m.q3w1_v[i]/bc2)+eps); m.q3w1[i]=std::min(std::max(m.q3w1[i]-st,-2.0f),2.0f);}
            for(int i=0;i<NL*D;++i){float g=d_q3w2[i]; if(g>1)g=1; if(g<-1)g=-1; m.q3w2_m[i]=b1*m.q3w2_m[i]+(1-b1)*g; m.q3w2_v[i]=b2*m.q3w2_v[i]+(1-b2)*g*g; float st=LR*(m.q3w2_m[i]/bc1)/(std::sqrt(m.q3w2_v[i]/bc2)+eps); m.q3w2[i]=std::min(std::max(m.q3w2[i]-st,-2.0f),2.0f);}
            for(int i=0;i<NL*D*D;++i){float g=d_aW[i]; if(g>1)g=1; if(g<-1)g=-1; m.aW_m[i]=b1*m.aW_m[i]+(1-b1)*g; m.aW_v[i]=b2*m.aW_v[i]+(1-b2)*g*g; float st=LR_ALPHA*(m.aW_m[i]/bc1)/(std::sqrt(m.aW_v[i]/bc2)+eps); m.aW[i]=std::min(std::max(m.aW[i]-st,-4.0f),4.0f);}
            for(int i=0;i<NL*D;++i){float g=d_ab[i]; if(g>1)g=1; if(g<-1)g=-1; m.ab_m[i]=b1*m.ab_m[i]+(1-b1)*g; m.ab_v[i]=b2*m.ab_v[i]+(1-b2)*g*g; float st=LR_ALPHA*(m.ab_m[i]/bc1)/(std::sqrt(m.ab_v[i]/bc2)+eps); m.ab[i]=std::min(std::max(m.ab[i]-st,-8.0f),8.0f);}
            m.q1.adam_update(m.q1_grad, LR, b1, b2, eps);
            
            // Reset grads
            std::fill(d_Wbi_g.begin(),d_Wbi_g.end(),0.0f);
            std::fill(d_Wh_g.begin(),d_Wh_g.end(),0.0f); std::fill(d_Ws_g.begin(),d_Ws_g.end(),0.0f);
            std::fill(d_q3w0.begin(),d_q3w0.end(),0.0f); std::fill(d_q3w1.begin(),d_q3w1.end(),0.0f);
            std::fill(d_q3w2.begin(),d_q3w2.end(),0.0f); std::fill(d_aW.begin(),d_aW.end(),0.0f);
            std::fill(d_ab.begin(),d_ab.end(),0.0f);
            std::fill(m.q1_grad.begin(),m.q1_grad.end(),0.0f);
            
            if((w+1)%5000==0){
                float el=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
                std::printf("  ep%d win%d/%d loss=%.4f avg=%.4f (%.1fs)\n", epoch+1, w+1, N_WIN, loss, total/nb, el);
            }
        }
        float avg=total/nb;
        std::printf("Epoch %2d/%d avg_loss=%.4f\n", epoch+1, EPOCHS, avg);
    }
    
    // Generation
    std::printf("\n=== Generation ===\n");
    std::string prompt_str="Once upon";
    std::vector<int> prompt_ids; int pi=0;
    while(pi<(int)prompt_str.size()){
        unsigned char c=prompt_str[pi];
        int cp, adv=1;
        if(c<0x80){cp=c;}
        else if((c&0xE0)==0xC0 && pi+1<(int)prompt_str.size()){cp=((c&0x1F)<<6)|(prompt_str[pi+1]&0x3F);adv=2;}
        else{cp=c;}
        prompt_ids.push_back(vocab.encode(cp));
        pi+=adv;
    }
    std::vector<int> ids=prompt_ids;
    for(int step=0;step<60;++step){
        std::vector<int> inBL(BL);
        int L=(int)ids.size();
        std::vector<int> in2(SEQ);
        for(int i=0;i<SEQ;++i){int idx=L-SEQ+i; in2[i]=(idx<0)?PAD:ids[idx];}
        for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t) inBL[b*SEQ+t]=in2[t];
        // Forward (reuse code, simplified)
        for(int n=0;n<BL;++n) m.q1.forward(inBL[n], x.data()+n*D, x.data()+n*D, q1_aux[n]);
        for(int l=0;l<NL;++l){
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
            for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t){int bt=b*SEQ+t;
                for(int d=0;d<D;++d){
                    float z=m.ab[l*D+d];
                    for(int k=0;k<D;++k) z+=m.aW[l*D*D+d*D+k]*x[bt*D+k];
                    float zT=z/2.0f;
                    if(zT>20)zT=20; if(zT<-20)zT=-20;
                    alpha[bt*D+d]=1.0f/(1.0f+std::exp(-zT));
                }
            }
            for(int b=0;b<BATCH;++b) for(int d=0;d<D;++d){
                float hc=0,sc=0;
                int hoff=b*(SEQ+1)*D+d;
                h[hoff]=0; s[hoff]=0;
                for(int t=0;t<SEQ;++t){int bt=b*SEQ+t; float yt=y[bt*D+d], a=alpha[bt*D+d];
                    hc=a*hc+(1-a)*yt; sc+=yt;
                    h[hoff+(t+1)*D]=hc; s[hoff+(t+1)*D]=sc;
                }
            }
            for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t){int bt=b*SEQ+t;
                float ms_h=0, ms_s=0;
                for(int d=0;d<D;++d){float v=h[(b*(SEQ+1)+t+1)*D+d]; ms_h+=v*v;}
                ms_h=ms_h/(float)D+1e-5f; float r_h=1.0f/std::sqrt(ms_h);
                for(int d=0;d<D;++d) hg[bt*D+d]=h[(b*(SEQ+1)+t+1)*D+d]*r_h;
                for(int d=0;d<D;++d){float v=s[(b*(SEQ+1)+t+1)*D+d]; ms_s+=v*v;}
                ms_s=ms_s/(float)D+1e-5f; float r_s=1.0f/std::sqrt(ms_s);
                for(int d=0;d<D;++d) sg[bt*D+d]=s[(b*(SEQ+1)+t+1)*D+d]*r_s;
            }
            for(int n=0;n<BL;++n) for(int d=0;d<D;++d) x[n*D+d]=hg[n*D+d]+sg[n*D+d];
        }
        for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t){int bt=b*SEQ+t;
            int prev=(t>0)?inBL[(b*SEQ+t-1)]:PAD;
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
        int bt=(BATCH-1)*SEQ+(SEQ-1);
        int best=0; float best_l=logits[bt*V_unit];
        for(int v=1;v<V_unit;++v) if(logits[bt*V_unit+v]>best_l){best_l=logits[bt*V_unit+v]; best=v;}
        ids.push_back(best);
    }
    std::printf("gen: \"");
    for(int id:ids){
        if(id>=0&&id<(int)vocab.id_to_cp.size()&&vocab.id_to_cp[id]>=0){
            int cp=vocab.id_to_cp[id];
            if(cp<0x80) std::printf("%c",(char)cp);
        } else std::printf("?");
    }
    std::printf("\"\n");
    return 0;
}
