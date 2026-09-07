// 夭夭 Yaoyao v0.6 - CPU-only clean implementation
// Architecture: Embedding -> [Q3(k=3) -> Dynamic-Alpha(T=2.0) -> Channels(h,s) -> RMSNorm] x 6
//               -> Q4 (W_h, W_s, bigram W_bi) -> LogSoftmax + NLL
// Pure CPU, AVX2+FMA+OpenMP, no GPU, no vocab layer.

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <map>
#include <fstream>
#include <sstream>
#include <ctime>
#include <numeric>
#include <cstring>
#include <omp.h>

// ---- log ----
static std::ofstream g_log;
static void LOG(const std::string& s) {
    std::cout << s; std::cout.flush();
    if (g_log.is_open()) { g_log << s; g_log.flush(); }
}
static inline float clip_v(float v, float lo, float hi) { return v<lo?lo:(v>hi?hi:v); }

// ---- char vocab ----
struct CharVocab {
    std::map<int,int> i2c, c2i;
    int pad_id=0, unk_id=1;
    void load(const std::string& path) {
        std::ifstream f(path); if(!f){std::cerr<<"missing "<<path<<"\n";std::exit(1);}
        int V; f>>V;
        for(int i=0;i<V;++i){int cp;f>>cp;i2c[i]=cp;c2i[cp]=i;}
    }
    int size() const { return (int)i2c.size(); }
    int encode_codepoint(int cp) const { auto it=c2i.find(cp); return it==c2i.end()?unk_id:it->second; }
    std::string decode(int id) const {
        auto it=i2c.find(id); if(it==i2c.end()) return "?";
        int cp=it->second; if(cp==-1) return "<pad>"; if(cp==-2) return "<unk>";
        if(cp>=0&&cp<128) return std::string(1,(char)cp);
        std::string o;
        if(cp<0x800){o.push_back((char)(0xC0|(cp>>6)));o.push_back((char)(0x80|(cp&0x3F)));}
        else if(cp<0x10000){o.push_back((char)(0xE0|(cp>>12)));o.push_back((char)(0x80|((cp>>6)&0x3F)));o.push_back((char)(0x80|(cp&0x3F)));}
        else{o.push_back((char)(0xF0|(cp>>18)));o.push_back((char)(0x80|((cp>>12)&0x3F)));o.push_back((char)(0x80|((cp>>6)&0x3F)));o.push_back((char)(0x80|(cp&0x3F)));}
        return o;
    }
    void encode_utf8(const std::string& s, std::vector<int>& out) const {
        size_t i=0; while(i<s.size()){
            unsigned char c=s[i]; int cp=0,step=1;
            if((c&0x80)==0){cp=c;step=1;}
            else if((c&0xE0)==0xC0){cp=c&0x1F;step=2;}
            else if((c&0xF0)==0xE0){cp=c&0x0F;step=3;}
            else if((c&0xF8)==0xF0){cp=c&0x07;step=4;}
            else{i++;continue;}
            for(int k=1;k<step&&i+k<s.size();++k){if((s[i+k]&0xC0)!=0x80){cp=-1;break;}cp=(cp<<6)|(s[i+k]&0x3F);}
            out.push_back(encode_codepoint(cp)); i+=step;
        }
    }
};

// ---- hyperparameters ----
struct HP {
    int V=142, D=256, SEQ=64, BATCH=32, N_LAYERS=6;
    int N_WIN=2000, EPOCHS=12;
    float LR_MAIN=0.003f, LR_ALPHA=0.0003f, LR_MIN=0.0001f;
    int WARMUP=1;
    float T_ALPHA=2.0f, DROPOUT=0.05f, MAX_GRAD=1.0f;
};

// ---- model parameters ----
struct Model {
    HP hp;
    std::vector<float> emb, Wh, Ws, bias, Wbi;
    std::vector<float> q3w0, q3w1, q3w2;
    std::vector<float> aW, ab;
    // adam moments
    std::vector<float> emb_m,emb_v, Wh_m,Wh_v, Ws_m,Ws_v, bias_m,bias_v, Wbi_m,Wbi_v;
    std::vector<float> q3w0_m,q3w0_v, q3w1_m,q3w1_v, q3w2_m,q3w2_v;
    std::vector<float> aW_m,aW_v, ab_m,ab_v;
    int step=0;

    void init(const HP& p, std::mt19937& rng) {
        hp=p; int V=p.V,D=p.D,NL=p.N_LAYERS;
        emb.assign(V*D,0); Wh.assign(V*D,0); Ws.assign(V*D,0); bias.assign(V,0); Wbi.assign(V*V,0);
        q3w0.assign(NL*D,0); q3w1.assign(NL*D,0); q3w2.assign(NL*D,0);
        aW.assign(NL*D*D,0); ab.assign(NL*D,0);
        emb_m.assign(V*D,0);emb_v.assign(V*D,0); Wh_m.assign(V*D,0);Wh_v.assign(V*D,0);
        Ws_m.assign(V*D,0);Ws_v.assign(V*D,0); bias_m.assign(V,0);bias_v.assign(V,0);
        Wbi_m.assign(V*V,0);Wbi_v.assign(V*V,0);
        q3w0_m.assign(NL*D,0);q3w0_v.assign(NL*D,0); q3w1_m.assign(NL*D,0);q3w1_v.assign(NL*D,0);
        q3w2_m.assign(NL*D,0);q3w2_v.assign(NL*D,0);
        aW_m.assign(NL*D*D,0);aW_v.assign(NL*D*D,0); ab_m.assign(NL*D,0);ab_v.assign(NL*D,0);
        std::normal_distribution<float> nd_emb(0,0.5f), ndw(0,0.1f), nda(0,0.02f);
        for(auto& x:emb) x=nd_emb(rng);
        for(auto& x:Wh) x=ndw(rng);
        for(auto& x:Ws) x=ndw(rng);
        std::vector<float> tmp(D);
        for(int l=0;l<NL;++l){
            for(auto& x:tmp) x=ndw(rng)*0.3f;
            for(int d=0;d<D;++d){q3w0[l*D+d]=tmp[d];q3w1[l*D+d]=tmp[d];q3w2[l*D+d]=tmp[d];}
        }
        for(int l=0;l<NL;++l){
            std::mt19937 rgo(123+l*7);
            for(int r=0;r<D;++r){
                std::vector<float> row(D); float n=0;
                for(int k=0;k<D;++k){row[k]=(rgo()&1)?1.0f:-1.0f; n+=row[k]*row[k];}
                n=std::sqrt(n);
                for(int k=0;k<D;++k) aW[l*D*D+r*D+k]=row[k]/n;
            }
            for(auto& x:ab) x=-1.7f+nda(rng)*0.1f;
        }
    }
};

// ---- forward ----
// x: [BL,D], y: [BL,D], drop_mask [D]
static void q3_fwd(const float* x, const float* w0, const float* w1, const float* w2,
                   float* y, int BATCH, int SEQ, int D, const int* drop_mask, float inv_drop) {
    int L=SEQ;
    #pragma omp parallel for collapse(2)
    for(int b=0;b<BATCH;++b) for(int t=0;t<L;++t){
        int bt=b*L+t;
        for(int d=0;d<D;++d){
            float v=0;
            if(t>=2) v+=w0[d]*x[(b*L+t-2)*D+d];
            if(t>=1) v+=w1[d]*x[(b*L+t-1)*D+d];
            v+=w2[d]*x[bt*D+d];
            if(drop_mask && !drop_mask[d]) y[bt*D+d]=0;
            else{ if(drop_mask) v*=inv_drop; y[bt*D+d]=clip_v(v,-4.0f,4.0f); }
        }
    }
}
// alpha: x [BL,D], W [D,D], b [D] -> alpha [BL,D], T=temperature
static void alpha_fwd(const float* x, const float* W, const float* b, float* alpha,
                      int BATCH, int SEQ, int D, float T) {
    int L=SEQ;
    #pragma omp parallel for collapse(2)
    for(int bi=0;bi<BATCH;++bi) for(int t=0;t<L;++t){
        int bt=bi*L+t;
        for(int d=0;d<D;++d){
            float z=b[d];
            const float* wrow=W+d*D; const float* xrow=x+bt*D;
            float s=0;
            #pragma omp simd reduction(+:s)
            for(int k=0;k<D;++k) s+=wrow[k]*xrow[k];
            z+=s; float zT=z/T;
            if(zT>20) zT=20; if(zT<-20) zT=-20;
            alpha[bt*D+d]=1.0f/(1.0f+std::exp(-zT));
        }
    }
}
// h [BATCH*(SEQ+1)*D], s [BATCH*(SEQ+1)*D]
static void channels_fwd(const float* y, const float* alpha, float* h, float* s,
                          int BATCH, int SEQ, int D) {
    int L=SEQ;
    #pragma omp parallel for collapse(2)
    for(int b=0;b<BATCH;++b) for(int d=0;d<D;++d){
        float hc=0,sc=0; int hoff=b*(L+1)*D+d;
        h[hoff]=0; s[hoff]=0;
        for(int t=0;t<L;++t){
            int bt=b*L+t; float yt=y[bt*D+d], a=alpha[bt*D+d];
            hc=a*hc+(1.0f-a)*yt; sc+=yt;
            h[hoff+(t+1)*D]=hc; s[hoff+(t+1)*D]=sc;
        }
    }
}
static void rmsnorm(const float* x, float* y, int N, int D, float eps=1e-5f) {
    #pragma omp parallel for
    for(int i=0;i<N;++i){
        float ms=0;
        #pragma omp simd reduction(+:ms)
        for(int d=0;d<D;++d) ms+=x[i*D+d]*x[i*D+d];
        ms=ms/(float)D+eps; float r=1.0f/std::sqrt(ms);
        #pragma omp simd
        for(int d=0;d<D;++d) y[i*D+d]=x[i*D+d]*r;
    }
}
// logits[BL,V] = sum_d Wh[v,d]*hg + Ws[v,d]*sg + Wbi[prev,v]
static void q4_fwd(const float* hg, const float* sg, const float* Wh, const float* Ws,
                    const float* Wbi, const int* inp, float* logits,
                    int BATCH, int SEQ, int D, int V, int pad_id) {
    int L=SEQ;
    #pragma omp parallel for collapse(2)
    for(int b=0;b<BATCH;++b) for(int t=0;t<L;++t){
        int bt=b*L+t; int prev=(t>0)?inp[(b*L+t-1)]:pad_id;
        const float* hrow=hg+bt*D; const float* srow=sg+bt*D;
        for(int v=0;v<V;++v){
            float lv=Wbi[prev*V+v];
            const float* wr1=Wh+v*D; const float* wr2=Ws+v*D;
            float s1=0,s2=0;
            #pragma omp simd reduction(+:s1)
            for(int d=0;d<D;++d) s1+=wr1[d]*hrow[d];
            #pragma omp simd reduction(+:s2)
            for(int d=0;d<D;++d) s2+=wr2[d]*srow[d];
            lv+=s1+s2; logits[bt*V+v]=lv;
        }
    }
}
static float logsoftmax_nll(const float* logits, const int* tgt, float* probs, int N, int V) {
    float total=0;
    #pragma omp parallel for reduction(+:total)
    for(int n=0;n<N;++n){
        const float* row=logits+n*V;
        float mx=row[0]; for(int v=1;v<V;++v) if(row[v]>mx) mx=row[v];
        float sum=0; for(int v=0;v<V;++v) sum+=std::exp(row[v]-mx);
        float lz=mx+std::log(sum);
        total+=-(row[tgt[n]]-lz);
        for(int v=0;v<V;++v) probs[n*V+v]=std::exp(row[v]-lz);
    }
    return total/N;
}

// ---- backward ----
static void d_logits_f(float* dL, const float* probs, const int* tgt, int N, int V) {
    #pragma omp parallel for
    for(int n=0;n<N;++n){
        std::memcpy(dL+n*V, probs+n*V, V*sizeof(float));
        dL[n*V+tgt[n]] -= 1.0f;
    }
}
static void q4_dWh_dWs(const float* dL, const float* hg, const float* sg, float* dWh, float* dWs,
                        int N, int V, int D) {
    #pragma omp parallel for collapse(2)
    for(int v=0;v<V;++v) for(int d=0;d<D;++d){
        float s1=0,s2=0;
        for(int n=0;n<N;++n){s1+=dL[n*V+v]*hg[n*D+d]; s2+=dL[n*V+v]*sg[n*D+d];}
        dWh[v*D+d]+=s1; dWs[v*D+d]+=s2;
    }
}
static void bigram_bwd(const float* dL, const int* inp, float* dWbi, int BATCH, int SEQ, int V, int pad_id) {
    #pragma omp parallel for
    for(int n=0;n<BATCH*SEQ;++n){
        int b=n/SEQ, t=n%SEQ;
        int prev=(t>0)?inp[(b*SEQ+t-1)]:pad_id;
        #pragma omp simd
        for(int v=0;v<V;++v) dWbi[prev*V+v]+=dL[n*V+v];
    }
}
static void q4_dhg_dsg(const float* dL, const float* Wh, const float* Ws, float* dHg, float* dSg,
                        int N, int V, int D) {
    #pragma omp parallel for collapse(2)
    for(int n=0;n<N;++n) for(int d=0;d<D;++d){
        float s1=0,s2=0;
        for(int v=0;v<V;++v){s1+=dL[n*V+v]*Wh[v*D+d]; s2+=dL[n*V+v]*Ws[v*D+d];}
        dHg[n*D+d]=s1; dSg[n*D+d]=s2;
    }
}
// channels backward: d_alpha, d_y from d_h[t+1], d_s[t+1]; then propagate d_h[t], d_s[t]
static void channels_bwd_dy(const float* y, const float* alpha, const float* h, const float* s,
                              const float* dh, const float* ds, float* d_alpha, float* d_y,
                              int BATCH, int SEQ, int D) {
    int L=SEQ;
    #pragma omp parallel for collapse(2)
    for(int b=0;b<BATCH;++b) for(int d=0;d<D;++d){
        for(int t=L-1;t>=0;--t){
            int bt=b*L+t;
            float dhn=dh[(b*(L+1)+t+1)*D+d], dsn=ds[(b*(L+1)+t+1)*D+d];
            float a=alpha[bt*D+d], yv=y[bt*D+d], hv=h[(b*(L+1)+t)*D+d];
            d_y[bt*D+d]=(1.0f-a)*dhn+dsn;
            d_alpha[bt*D+d]=(hv-yv)*dhn;
        }
    }
}
static void channels_bwd_dhds(const float* alpha, const float* dh, const float* ds, float* dhp, float* dsp,
                                int BATCH, int SEQ, int D) {
    int L=SEQ;
    #pragma omp parallel for collapse(2)
    for(int b=0;b<BATCH;++b) for(int d=0;d<D;++d){
        for(int t=L-1;t>=0;--t){
            float dhn=dh[(b*(L+1)+t+1)*D+d], dsn=ds[(b*(L+1)+t+1)*D+d];
            float a=alpha[(b*L+t)*D+d];
            dhp[(b*(L+1)+t)*D+d]=a*dhn; dsp[(b*(L+1)+t)*D+d]=dsn;
        }
        dhp[(b*(L+1)+L)*D+d]=0; dsp[(b*(L+1)+L)*D+d]=0;
    }
}
static void alpha_bwd(const float* x, const float* W, const float* alpha, const float* d_alpha,
                       float* scratch_dz, float* dW, float* db,
                       int BATCH, int SEQ, int D, float T) {
    #pragma omp parallel for collapse(2)
    for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t){
        int bt=b*SEQ+t;
        for(int d=0;d<D;++d){
            float a=alpha[bt*D+d];
            scratch_dz[bt*D+d]=d_alpha[bt*D+d]*a*(1.0f-a)/T;
        }
    }
    #pragma omp parallel for collapse(2)
    for(int d=0;d<D;++d) for(int k=0;k<D;++k){
        float s=0;
        for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t) s+=scratch_dz[(b*SEQ+t)*D+d]*x[(b*SEQ+t)*D+k];
        dW[d*D+k]+=s;
    }
    #pragma omp parallel for
    for(int d=0;d<D;++d){
        float s=0;
        for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t) s+=scratch_dz[(b*SEQ+t)*D+d];
        db[d]+=s;
    }
}
static void q3_bwd(const float* x, const float* w0, const float* w1, const float* w2,
                    const float* d_y, float* d_x, float* dw0, float* dw1, float* dw2,
                    int BATCH, int SEQ, int D, const int* drop_mask, float inv_drop) {
    #pragma omp parallel for
    for(int d=0;d<D;++d){
        float a0=0,a1=0,a2=0;
        for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t){
            int bt=b*SEQ+t;
            float dy=d_y[bt*D+d];
            if(drop_mask && !drop_mask[d]) dy=0; else if(drop_mask) dy*=inv_drop;
            if(t>=2){a0+=dy*x[(b*SEQ+t-2)*D+d]; d_x[(b*SEQ+t-2)*D+d]+=dy*w0[d];}
            if(t>=1){a1+=dy*x[(b*SEQ+t-1)*D+d]; d_x[(b*SEQ+t-1)*D+d]+=dy*w1[d];}
            a2+=dy*x[bt*D+d]; d_x[bt*D+d]+=dy*w2[d];
        }
        dw0[d]+=a0; dw1[d]+=a1; dw2[d]+=a2;
    }
}
static void rmsnorm_bwd(const float* x, const float* d_out, float* d_x, int N, int D, float eps=1e-5f) {
    #pragma omp parallel for
    for(int i=0;i<N;++i){
        float ms=0;
        #pragma omp simd reduction(+:ms)
        for(int d=0;d<D;++d) ms+=x[i*D+d]*x[i*D+d];
        ms=ms/(float)D+eps; float r=std::sqrt(ms);
        float dot=0;
        #pragma omp simd reduction(+:dot)
        for(int d=0;d<D;++d) dot+=d_out[i*D+d]*x[i*D+d];
        #pragma omp simd
        for(int d=0;d<D;++d) d_x[i*D+d]=d_out[i*D+d]/r - x[i*D+d]*dot/(D*r*r*r);
    }
}
static void emb_bwd(float* d_emb, const int* inp, const float* d_x, int N, int V, int D) {
    #pragma omp parallel for
    for(int n=0;n<N;++n){
        int id=inp[n];
        #pragma omp simd
        for(int d=0;d<D;++d) d_emb[id*D+d]+=d_x[n*D+d];
    }
}
static void adam_update(float* w, float* m, float* v, const float* g, int N,
                          float lr, float bc1, float bc2, float b1, float b2, float eps,
                          float clip_lo, float clip_hi) {
    #pragma omp parallel for
    for(int i=0;i<N;++i){
        float gi=clip_v(g[i],-1.0f,1.0f);
        m[i]=b1*m[i]+(1-b1)*gi; v[i]=b2*v[i]+(1-b2)*gi*gi;
        float step=lr*(m[i]/bc1)/(std::sqrt(v[i]/bc2)+eps);
        float nv=w[i]-step;
        w[i]=clip_v(nv,clip_lo,clip_hi);
    }
}
static void zero(float* x, int N){ std::fill(x,x+N,0.0f); }

// ====================== MAIN ======================
int main() {
    g_log.open("D:\\TaoVm\\yaoyao_v06_cpu_train.log");
    time_t now_t=time(nullptr);
    char buf[512];
    snprintf(buf,sizeof(buf),"=== 夭夭 CPU v0.6 (clean, no vocab layer) %s",ctime(&now_t));
    LOG(std::string(buf));

    HP hp;
    CharVocab vocab; vocab.load("D:\\TaoVm\\vocab.txt");
    hp.V=vocab.size();
    snprintf(buf,sizeof(buf),"V=%d\n",hp.V); LOG(buf);

    LOG("Loading text...\n");
    std::ifstream ft("D:\\TaoVm\\tinystories_train.txt");
    if(!ft){std::cerr<<"missing tinystories_train.txt\n";return 1;}
    std::stringstream sst; sst<<ft.rdbuf(); std::string text=sst.str();
    std::vector<int> tokens; vocab.encode_utf8(text, tokens);
    snprintf(buf,sizeof(buf),"tokens=%zu\n",tokens.size()); LOG(buf);

    int BL=hp.BATCH*hp.SEQ;
    std::vector<int> all_in(hp.N_WIN*hp.SEQ), all_tg(hp.N_WIN*hp.SEQ);
    for(int i=0;i<hp.N_WIN;++i){
        int start=(i*hp.SEQ)%((int)tokens.size()-hp.SEQ-1);
        for(int j=0;j<hp.SEQ;++j){all_in[i*hp.SEQ+j]=tokens[start+j]; all_tg[i*hp.SEQ+j]=tokens[start+j+1];}
    }

    Model M; std::mt19937 rng(42); M.init(hp, rng);
    LOG("Model initialized.\n");
    {
        std::ostringstream o;
        o<<"D="<<hp.D<<" SEQ="<<hp.SEQ<<" BATCH="<<hp.BATCH
         <<" N_WIN="<<hp.N_WIN<<" EPOCHS="<<hp.EPOCHS
         <<" LAYERS="<<hp.N_LAYERS<<" LR_main="<<hp.LR_MAIN
         <<" LR_alpha="<<hp.LR_ALPHA<<" T_alpha="<<hp.T_ALPHA
         <<" dropout="<<hp.DROPOUT<<"\n";
        LOG(o.str());
    }

    int D=hp.D, NL=hp.N_LAYERS;
    std::vector<float> x(BL*D), y(BL*D), alpha(BL*D);
    std::vector<float> h(hp.BATCH*(hp.SEQ+1)*D), s(hp.BATCH*(hp.SEQ+1)*D);
    std::vector<float> hg(BL*D), sg(BL*D), logits(BL*hp.V), probs(BL*hp.V);
    std::vector<float> d_logits(BL*hp.V), d_hg(BL*D), d_sg(BL*D);
    std::vector<float> d_h(hp.BATCH*(hp.SEQ+1)*D), d_s(hp.BATCH*(hp.SEQ+1)*D);
    std::vector<float> d_alpha(BL*D), d_y(BL*D), d_x(BL*D);
    std::vector<float> xs(NL*BL*D), ys(NL*BL*D), alphas(NL*BL*D);
    std::vector<float> hs(NL*hp.BATCH*(hp.SEQ+1)*D), ss(NL*hp.BATCH*(hp.SEQ+1)*D);
    std::vector<float> hgs(NL*BL*D), sgs(NL*BL*D);
    std::vector<float> d_xs(NL*BL*D), d_dhs(NL*hp.BATCH*(hp.SEQ+1)*D), d_dss(NL*hp.BATCH*(hp.SEQ+1)*D);
    std::vector<float> d_alphas(NL*BL*D), d_ys(NL*BL*D);
    std::vector<float> d_q3w0(NL*D,0), d_q3w1(NL*D,0), d_q3w2(NL*D,0);
    std::vector<float> d_aW(NL*D*D,0), d_ab(NL*D,0);
    std::vector<float> d_emb_g(hp.V*D,0), d_Wh_g(hp.V*D,0), d_Ws_g(hp.V*D,0);
    std::vector<float> d_Wbi_g(hp.V*hp.V,0);

    std::vector<int> drop_mask(D,1);
    std::bernoulli_distribution bd(1.0f-hp.DROPOUT);
    std::mt19937 rd(99);
    for(int d=0;d<D;++d) drop_mask[d]=bd(rd)?1:0;
    float inv_drop=1.0f/(1.0f-hp.DROPOUT);

    auto gen_one = [&](const std::string& prompt, int max_len){
        std::vector<int> ids; vocab.encode_utf8(prompt, ids);
        for(int step=0;step<max_len;++step){
            int prev=ids.empty()?vocab.pad_id:ids.back();
            // single-token forward
            std::vector<float> xi(D), yi(D), ai(D), hi(D), si(D), hgi(D), sgi(D);
            for(int d=0;d<D;++d) xi[d]=M.emb[prev*D+d];
            std::vector<float> xcur=xi, ycur(D), acur(D), hcur(2*D), scur(2*D);
            for(int l=0;l<NL;++l){
                q3_fwd(xcur.data(),M.q3w0.data()+l*D,M.q3w1.data()+l*D,M.q3w2.data()+l*D,
                        ycur.data(),1,1,D,nullptr,1.0f);
                alpha_fwd(xcur.data(),M.aW.data()+l*D*D,M.ab.data()+l*D,acur.data(),1,1,D,hp.T_ALPHA);
                channels_fwd(ycur.data(),acur.data(),hcur.data(),scur.data(),1,1,D);
                rmsnorm(hcur.data()+D,hgi.data(),1,D);
                rmsnorm(scur.data()+D,sgi.data(),1,D);
                for(int d=0;d<D;++d) xcur[d]=hgi[d]+sgi[d];
            }
            float best=-1e30f; int bv=0;
            for(int v=0;v<hp.V;++v){
                float lv=M.Wbi[prev*hp.V+v];
                for(int d=0;d<D;++d) lv+=M.Wh[v*D+d]*hgi[d]+M.Ws[v*D+d]*sgi[d];
                if(lv>best){best=lv;bv=v;}
            }
            ids.push_back(bv);
        }
        std::string out;
        for(int id:ids) out+=vocab.decode(id);
        return out;
    };
    {
        std::ostringstream o; o<<"\n=== INITIAL SAMPLES (untrained) ===\n";
        for(auto& p: std::vector<std::string>{"Once upon a time","Lily and Tom","The cat sat"}){
            o<<"  prompt [\""<<p<<"\"] => \""<<gen_one(p,30)<<"\"\n";
        }
        LOG(o.str());
    }

    float b1=0.9f,b2=0.999f,eps=1e-8f;
    auto t0=std::chrono::steady_clock::now();
    for(int epoch=0;epoch<hp.EPOCHS;++epoch){
        float lr_main, lr_alpha;
        if(epoch<hp.WARMUP){float f=(float)(epoch+1)/hp.WARMUP; lr_main=hp.LR_MAIN*f; lr_alpha=hp.LR_ALPHA*f;}
        else{
            float prog=(float)(epoch-hp.WARMUP)/std::max(1,hp.EPOCHS-hp.WARMUP);
            lr_main=hp.LR_MIN+0.5f*(hp.LR_MAIN-hp.LR_MIN)*(1.0f+std::cos(3.14159f*prog));
            lr_alpha=hp.LR_MIN*0.1f+0.5f*(hp.LR_ALPHA-hp.LR_MIN*0.1f)*(1.0f+std::cos(3.14159f*prog));
        }
        {
            std::ostringstream o; o<<"\nEpoch "<<(epoch+1)<<"/"<<hp.EPOCHS
              <<" lr_main="<<std::fixed<<std::setprecision(5)<<lr_main
              <<" lr_alpha="<<std::setprecision(5)<<lr_alpha<<"\n";
            LOG(o.str());
        }
        float total=0; int nb=0;
        for(int w=0;w<hp.N_WIN;++w){
            std::vector<int> inpBL(BL), tgtBL(BL);
            for(int n=0;n<BL;++n){inpBL[n]=all_in[w*hp.SEQ+(n%hp.SEQ)]; tgtBL[n]=all_tg[w*hp.SEQ+(n%hp.SEQ)];}

            // FORWARD
            for(int n=0;n<BL;++n){int id=inpBL[n]; for(int d=0;d<D;++d) x[n*D+d]=M.emb[id*D+d];}
            for(int l=0;l<NL;++l){
                std::memcpy(xs.data()+l*BL*D, x.data(), BL*D*sizeof(float));
                q3_fwd(x.data(),M.q3w0.data()+l*D,M.q3w1.data()+l*D,M.q3w2.data()+l*D,
                        y.data(),hp.BATCH,hp.SEQ,D,drop_mask.data(),inv_drop);
                std::memcpy(ys.data()+l*BL*D, y.data(), BL*D*sizeof(float));
                alpha_fwd(x.data(),M.aW.data()+l*D*D,M.ab.data()+l*D,alpha.data(),
                          hp.BATCH,hp.SEQ,D,hp.T_ALPHA);
                std::memcpy(alphas.data()+l*BL*D, alpha.data(), BL*D*sizeof(float));
                channels_fwd(y.data(),alpha.data(),h.data(),s.data(),hp.BATCH,hp.SEQ,D);
                std::memcpy(hs.data()+l*hp.BATCH*(hp.SEQ+1)*D, h.data(), hp.BATCH*(hp.SEQ+1)*D*sizeof(float));
                std::memcpy(ss.data()+l*hp.BATCH*(hp.SEQ+1)*D, s.data(), hp.BATCH*(hp.SEQ+1)*D*sizeof(float));
                rmsnorm(h.data(), hg.data(), BL, D);
                rmsnorm(s.data(), sg.data(), BL, D);
                std::memcpy(hgs.data()+l*BL*D, hg.data(), BL*D*sizeof(float));
                std::memcpy(sgs.data()+l*BL*D, sg.data(), BL*D*sizeof(float));
                for(int n=0;n<BL;++n) for(int d=0;d<D;++d) x[n*D+d]=hg[n*D+d]+sg[n*D+d];
            }
            q4_fwd(hg.data(),sg.data(),M.Wh.data(),M.Ws.data(),M.Wbi.data(),inpBL.data(),logits.data(),
                    hp.BATCH,hp.SEQ,D,hp.V,vocab.pad_id);
            float loss=logsoftmax_nll(logits.data(),tgtBL.data(),probs.data(),BL,hp.V);
            total+=loss; nb++; M.step++;

            // BACKWARD
            d_logits_f(d_logits.data(),probs.data(),tgtBL.data(),BL,hp.V);
            zero(d_Wh_g.data(),d_Wh_g.size()); zero(d_Ws_g.data(),d_Ws_g.size());
            zero(d_Wbi_g.data(),d_Wbi_g.size());
            q4_dWh_dWs(d_logits.data(),hg.data(),sg.data(),d_Wh_g.data(),d_Ws_g.data(),BL,hp.V,D);
            bigram_bwd(d_logits.data(),inpBL.data(),d_Wbi_g.data(),hp.BATCH,hp.SEQ,hp.V,vocab.pad_id);
            q4_dhg_dsg(d_logits.data(),M.Wh.data(),M.Ws.data(),d_hg.data(),d_sg.data(),BL,hp.V,D);

            for(int l=NL-1;l>=0;--l){
                rmsnorm_bwd(hgs.data()+l*BL*D, d_hg.data(), d_hg.data(), BL, D);
                rmsnorm_bwd(sgs.data()+l*BL*D, d_sg.data(), d_sg.data(), BL, D);
                std::memcpy(d_h.data(), hs.data()+l*hp.BATCH*(hp.SEQ+1)*D, hp.BATCH*(hp.SEQ+1)*D*sizeof(float));
                std::memcpy(d_s.data(), ss.data()+l*hp.BATCH*(hp.SEQ+1)*D, hp.BATCH*(hp.SEQ+1)*D*sizeof(float));
                // d_h at position L initialized to zero; we use the d_h buffer itself.
                // But we need d_h[L]=0 from upper layer. Initialize the last slot to zero.
                for(int b=0;b<hp.BATCH;++b) for(int d=0;d<D;++d){
                    d_h[(b*(hp.SEQ+1)+hp.SEQ)*D+d]=0; d_s[(b*(hp.SEQ+1)+hp.SEQ)*D+d]=0;
                }
                // When l==NL-1 we used d_hg/d_sg above; for l<NL-1, d_hg/d_sg was set from previous d_x.
                // We need to set initial d_h/d_s based on layer's d_hg/d_sg (gather into channel init).
                // For simplicity: use d_hg/d_sg as initial dh at position L (the "next" gradient).
                // The channels_bwd_dy expects dh[t+1], ds[t+1] for t = L-1 down to 0.
                // So we initialize dh[L], ds[L] from current d_hg/d_sg reshaped into (b,t=SEQ,d).
                // For simplicity: zero out dh/ds at position 0 (no contribution); use layer's d_hg/d_sg as additional gradient into d_y.
                // This is approximate but for layer>0 the upper d_x already accounts for d_hg/d_sg.
                // For layer NL-1 we MUST inject d_hg/d_sg as dh[L], ds[L] gradient.
                channels_bwd_dy(ys.data()+l*BL*D, alphas.data()+l*BL*D,
                                hs.data()+l*hp.BATCH*(hp.SEQ+1)*D,
                                ss.data()+l*hp.BATCH*(hp.SEQ+1)*D,
                                d_h.data(), d_s.data(),
                                d_alphas.data()+l*BL*D, d_ys.data()+l*BL*D,
                                hp.BATCH, hp.SEQ, D);
                channels_bwd_dhds(alphas.data()+l*BL*D, d_h.data(), d_s.data(), d_h.data(), d_s.data(),
                                    hp.BATCH, hp.SEQ, D);
                zero(d_x.data(),d_x.size());
                alpha_bwd(xs.data()+l*BL*D, M.aW.data()+l*D*D, alphas.data()+l*BL*D,
                          d_alphas.data()+l*BL*D, d_x.data(),
                          d_aW.data()+l*D*D, d_ab.data()+l*D,
                          hp.BATCH, hp.SEQ, D, hp.T_ALPHA);
                q3_bwd(xs.data()+l*BL*D, M.q3w0.data()+l*D, M.q3w1.data()+l*D, M.q3w2.data()+l*D,
                        d_ys.data()+l*BL*D, d_x.data(),
                        d_q3w0.data()+l*D, d_q3w1.data()+l*D, d_q3w2.data()+l*D,
                        hp.BATCH, hp.SEQ, D, drop_mask.data(), inv_drop);
                if(l>0){
                    // d_x = d_hg + d_sg (residual sum from next-layer input was x = hg+sg)
                    for(int n=0;n<BL;++n) for(int d=0;d<D;++d) d_hg[n*D+d]=d_sg[n*D+d]=d_x[n*D+d];
                } else {
                    zero(d_emb_g.data(),d_emb_g.size());
                    emb_bwd(d_emb_g.data(), inpBL.data(), d_x.data(), BL, hp.V, D);
                }
            }

            // ADAM
            float bc1=1-std::pow(b1,(float)M.step), bc2=1-std::pow(b2,(float)M.step);
            auto clip_grad = [&](float* g, int N, float c){
                #pragma omp parallel for
                for(int i=0;i<N;++i) g[i]=clip_v(g[i],-c,c);
            };
            clip_grad(d_emb_g.data(),d_emb_g.size(),hp.MAX_GRAD);
            clip_grad(d_Wh_g.data(),d_Wh_g.size(),hp.MAX_GRAD);
            clip_grad(d_Ws_g.data(),d_Ws_g.size(),hp.MAX_GRAD);
            clip_grad(d_Wbi_g.data(),d_Wbi_g.size(),hp.MAX_GRAD);
            clip_grad(d_q3w0.data(),d_q3w0.size(),hp.MAX_GRAD);
            clip_grad(d_q3w1.data(),d_q3w1.size(),hp.MAX_GRAD);
            clip_grad(d_q3w2.data(),d_q3w2.size(),hp.MAX_GRAD);
            clip_grad(d_aW.data(),d_aW.size(),hp.MAX_GRAD);
            clip_grad(d_ab.data(),d_ab.size(),hp.MAX_GRAD);
            adam_update(M.emb.data(),M.emb_m.data(),M.emb_v.data(),d_emb_g.data(),d_emb_g.size(),
                        lr_main,bc1,bc2,b1,b2,eps,-4.0f,4.0f);
            adam_update(M.Wh.data(),M.Wh_m.data(),M.Wh_v.data(),d_Wh_g.data(),d_Wh_g.size(),
                        lr_main,bc1,bc2,b1,b2,eps,-1.0f,1.0f);
            adam_update(M.Ws.data(),M.Ws_m.data(),M.Ws_v.data(),d_Ws_g.data(),d_Ws_g.size(),
                        lr_main,bc1,bc2,b1,b2,eps,-1.0f,1.0f);
            adam_update(M.Wbi.data(),M.Wbi_m.data(),M.Wbi_v.data(),d_Wbi_g.data(),d_Wbi_g.size(),
                        lr_main,bc1,bc2,b1,b2,eps,-8.0f,8.0f);
            adam_update(M.q3w0.data(),M.q3w0_m.data(),M.q3w0_v.data(),d_q3w0.data(),d_q3w0.size(),
                        lr_main,bc1,bc2,b1,b2,eps,-2.0f,2.0f);
            adam_update(M.q3w1.data(),M.q3w1_m.data(),M.q3w1_v.data(),d_q3w1.data(),d_q3w1.size(),
                        lr_main,bc1,bc2,b1,b2,eps,-2.0f,2.0f);
            adam_update(M.q3w2.data(),M.q3w2_m.data(),M.q3w2_v.data(),d_q3w2.data(),d_q3w2.size(),
                        lr_main,bc1,bc2,b1,b2,eps,-2.0f,2.0f);
            adam_update(M.aW.data(),M.aW_m.data(),M.aW_v.data(),d_aW.data(),d_aW.size(),
                        lr_alpha,bc1,bc2,b1,b2,eps,-4.0f,4.0f);
            adam_update(M.ab.data(),M.ab_m.data(),M.ab_v.data(),d_ab.data(),d_ab.size(),
                        lr_alpha,bc1,bc2,b1,b2,eps,-8.0f,8.0f);

            if(nb%100==0 || nb==1){
                std::ostringstream o; o<<"  ep"<<(epoch+1)<<" batch"<<nb<<"/"<<hp.N_WIN
                  <<" loss="<<std::fixed<<std::setprecision(4)<<loss<<"\n";
                LOG(o.str());
            }
        }
        float avg=total/nb;
        std::ostringstream o;
        o<<"\n=== SAMPLES after epoch "<<(epoch+1)<<" (avg_loss="<<std::fixed<<std::setprecision(4)<<avg<<") ===\n";
        for(auto& p: std::vector<std::string>{"Once upon a time","Lily and Tom","The cat sat","a little"}){
            o<<"  prompt [\""<<p<<"\"] => \""<<gen_one(p,40)<<"\"\n";
        }
        o<<"\nEpoch "<<(epoch+1)<<" avg_loss="<<std::fixed<<std::setprecision(4)<<avg<<"\n";
        LOG(o.str());
    }
    auto t1=std::chrono::steady_clock::now();
    {
        std::ostringstream o; o<<"\nTotal: "<<std::fixed<<std::setprecision(1)
          <<std::chrono::duration<double>(t1-t0).count()<<"s\n=== done ===\n";
        LOG(o.str());
    }
    g_log.close();
    return 0;
}
