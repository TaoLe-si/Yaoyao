// 夭夭 Agent 训练测试 - 多 batch size 对比
// 验证 BPTT 在 BATCH=1, 2, 4, 8 下都稳定学习
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

// === 共享类型 ===
struct M {
    static const int D=32, NL=2;
    std::vector<float> emb, Wh, Ws, Wbi;
    std::vector<float> q3w0, q3w1, q3w2;
    std::vector<float> aW, ab;
    std::vector<float> emb_m,emb_v, Wh_m,Wh_v, Ws_m,Ws_v, Wbi_m,Wbi_v;
    std::vector<float> q3w0_m,q3w0_v,q3w1_m,q3w1_v,q3w2_m,q3w2_v;
    std::vector<float> aW_m,aW_v, ab_m,ab_v;
    int step=0;

    void init(std::mt19937& rng, int V) {
        emb.assign(V*D,0); Wh.assign(V*D,0); Ws.assign(V*D,0); Wbi.assign(V*V,0);
        q3w0.assign(NL*D,0); q3w1.assign(NL*D,0); q3w2.assign(NL*D,0);
        aW.assign(NL*D*D,0); ab.assign(NL*D,0);
        emb_m.assign(V*D,0);emb_v.assign(V*D,0);
        Wh_m.assign(V*D,0);Wh_v.assign(V*D,0);
        Ws_m.assign(V*D,0);Ws_v.assign(V*D,0);
        Wbi_m.assign(V*V,0);Wbi_v.assign(V*V,0);
        q3w0_m.assign(NL*D,0);q3w0_v.assign(NL*D,0);
        q3w1_m.assign(NL*D,0);q3w1_v.assign(NL*D,0);
        q3w2_m.assign(NL*D,0);q3w2_v.assign(NL*D,0);
        aW_m.assign(NL*D*D,0);aW_v.assign(NL*D*D,0);
        ab_m.assign(NL*D,0);ab_v.assign(NL*D,0);
        std::normal_distribution<float> nde(0,0.3f), ndw(0,0.1f);
        for(auto& x:emb) x=nde(rng);
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

static int char_id[256];
static std::string id_chars;
static int V_unit;
static void init_vocab() {
    for(int i=0;i<256;++i) char_id[i]=0;
    for(int i=0;i<V_unit;++i) char_id[(unsigned char)id_chars[i]] = i;
}
static int encode_char(unsigned char c) { return char_id[c]; }
static unsigned char decode_id(int id) { return id_chars[id]; }

static void forward_inline(M& M, const std::vector<int>& inp, int BATCH, int SEQ, int PAD,
                            std::vector<float>& x, std::vector<float>& h, std::vector<float>& s,
                            std::vector<float>& y, std::vector<float>& alpha,
                            std::vector<float>& hg, std::vector<float>& sg,
                            std::vector<float>& logits, std::vector<float>& probs,
                            std::vector<float>& xs, std::vector<float>& ys, std::vector<float>& alphas,
                            std::vector<float>& hs, std::vector<float>& ss,
                            std::vector<float>& hgs, std::vector<float>& sgs) {
    int D=32, NL=2;
    int BL=BATCH*SEQ;
    for(int n=0;n<BL;++n){int id=inp[n]; for(int d=0;d<D;++d) x[n*D+d]=M.emb[id*D+d];}
    for(int l=0;l<NL;++l){
        std::memcpy(xs.data()+l*BL*D, x.data(), BL*D*sizeof(float));
        for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t){
            int bt=b*SEQ+t;
            for(int d=0;d<D;++d){
                float v=0;
                if(t>=2) v+=M.q3w0[l*D+d]*x[(b*SEQ+t-2)*D+d];
                if(t>=1) v+=M.q3w1[l*D+d]*x[(b*SEQ+t-1)*D+d];
                v+=M.q3w2[l*D+d]*x[bt*D+d];
                if(v>4)v=4; if(v<-4)v=-4;
                y[bt*D+d]=v;
            }
        }
        std::memcpy(ys.data()+l*BL*D, y.data(), BL*D*sizeof(float));
        for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t){
            int bt=b*SEQ+t;
            for(int d=0;d<D;++d){
                float z=M.ab[l*D+d];
                for(int k=0;k<D;++k) z+=M.aW[l*D*D+d*D+k]*x[bt*D+k];
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
            for(int t=0;t<SEQ;++t){
                int bt=b*SEQ+t; float yt=y[bt*D+d], a=alpha[bt*D+d];
                hc=a*hc+(1-a)*yt; sc+=yt;
                h[hoff+(t+1)*D]=hc; s[hoff+(t+1)*D]=sc;
            }
        }
        std::memcpy(hs.data()+l*BATCH*(SEQ+1)*D, h.data(), BATCH*(SEQ+1)*D*sizeof(float));
        std::memcpy(ss.data()+l*BATCH*(SEQ+1)*D, s.data(), BATCH*(SEQ+1)*D*sizeof(float));
        for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t){
            int bt=b*SEQ+t;
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
    for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t){
        int bt=b*SEQ+t;
        int prev=(t>0)?inp[(b*SEQ+t-1)]:PAD;
        for(int v=0;v<V_unit;++v){
            float lv=M.Wbi[prev*V_unit+v];
            for(int d=0;d<D;++d) lv+=M.Wh[v*D+d]*hg[bt*D+d]+M.Ws[v*D+d]*sg[bt*D+d];
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

float run_batch_size(int BATCH) {
    std::mt19937 rng(42);
    M m; m.init(rng, 17);
    id_chars="the cat sat on ma "; V_unit=16;
    init_vocab();
    std::string pattern="the cat sat on a ";
    std::string text;
    for(int i=0;i<200;++i) text+=pattern;
    std::vector<int> tokens;
    for(char c:text) tokens.push_back(encode_char((unsigned char)c));
    int N_tokens=tokens.size();
    int SEQ=8, BL=SEQ*BATCH;
    int N_WIN=200, EPOCHS=15;
    float LR=0.01f, LR_ALPHA=0.001f;
    int PAD=0;
    std::vector<int> all_in(N_WIN*SEQ), all_tg(N_WIN*SEQ);
    for(int i=0;i<N_WIN;++i){
        int start=(i*SEQ)%(N_tokens-SEQ-1);
        for(int j=0;j<SEQ;++j){
            all_in[i*SEQ+j]=tokens[start+j];
            all_tg[i*SEQ+j]=tokens[start+j+1];
        }
    }
    std::vector<float> x(BL*32), y(BL*32), alpha(BL*32);
    std::vector<float> h(BATCH*(SEQ+1)*32), s(BATCH*(SEQ+1)*32);
    std::vector<float> hg(BL*32), sg(BL*32), logits(BL*V_unit), probs(BL*V_unit);
    std::vector<float> d_logits(BL*V_unit), d_hg(BL*32), d_sg(BL*32);
    std::vector<float> d_h(BATCH*(SEQ+1)*32), d_s(BATCH*(SEQ+1)*32);
    std::vector<float> d_alpha(BL*32), d_y(BL*32), d_x(BL*32);
    std::vector<float> xs(2*BL*32), ys(2*BL*32), alphas(2*BL*32);
    std::vector<float> hs(2*BATCH*(SEQ+1)*32), ss(2*BATCH*(SEQ+1)*32);
    std::vector<float> hgs(2*BL*32), sgs(2*BL*32);
    std::vector<float> d_alphas(2*BL*32), d_ys(2*BL*32);
    std::vector<float> d_q3w0(2*32,0), d_q3w1(2*32,0), d_q3w2(2*32,0);
    std::vector<float> d_aW(2*32*32,0), d_ab(2*32,0);
    std::vector<float> d_emb_g(V_unit*32,0), d_Wh_g(V_unit*32,0), d_Ws_g(V_unit*32,0), d_Wbi_g(V_unit*V_unit,0);
    float b1=0.9f,b2=0.999f,eps=1e-8f;
    std::printf("\n=== BATCH=%d (BL=%d tokens/batch) ===\n", BATCH, BL);
    for(int epoch=0;epoch<EPOCHS;++epoch){
        float total=0; int nb=0;
        for(int w=0;w<N_WIN;++w){
            std::vector<int> inpBL(BL), tgtBL(BL);
            for(int n=0;n<BL;++n){inpBL[n]=all_in[w*SEQ+(n%SEQ)]; tgtBL[n]=all_tg[w*SEQ+(n%SEQ)];}
            forward_inline(m, inpBL, BATCH, SEQ, PAD, x, h, s, y, alpha, hg, sg, logits, probs, xs, ys, alphas, hs, ss, hgs, sgs);
            float loss=0;
            for(int n=0;n<BL;++n){int t=tgtBL[n]; float p=std::max(probs[n*V_unit+t], 1e-9f); loss+=-std::log(p);}
            loss/=BL; total+=loss; nb++; m.step++;
            for(int n=0;n<BL;++n){for(int v=0;v<V_unit;++v) d_logits[n*V_unit+v]=probs[n*V_unit+v]; d_logits[n*V_unit+tgtBL[n]]-=1.0f;}
            for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t){int prev=(t>0)?inpBL[(b*SEQ+t-1)]:PAD; for(int v=0;v<V_unit;++v) d_Wbi_g[prev*V_unit+v]+=d_logits[(b*SEQ+t)*V_unit+v];}
            for(int n=0;n<BL;++n){for(int d=0;d<32;++d){float s1=0,s2=0; for(int v=0;v<V_unit;++v){s1+=d_logits[n*V_unit+v]*m.Wh[v*32+d]; s2+=d_logits[n*V_unit+v]*m.Ws[v*32+d];} d_hg[n*32+d]=s1; d_sg[n*32+d]=s2;}}
            for(int v=0;v<V_unit;++v) for(int d=0;d<32;++d){float s1=0,s2=0; for(int n=0;n<BL;++n){s1+=d_logits[n*V_unit+v]*hg[n*32+d]; s2+=d_logits[n*V_unit+v]*sg[n*32+d];} d_Wh_g[v*32+d]=s1; d_Ws_g[v*32+d]=s2;}
            // BPTT
            for(int l=2-1;l>=0;--l){
                {
                    for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t){int bt=b*SEQ+t;
                        float ms_h=0, ms_s=0;
                        for(int d=0;d<32;++d){float v=hgs[l*BL*32+bt*32+d]; ms_h+=v*v;}
                        ms_h=ms_h/32.0f+1e-5f; float r_h=std::sqrt(ms_h);
                        for(int d=0;d<32;++d){float v=sgs[l*BL*32+bt*32+d]; ms_s+=v*v;}
                        ms_s=ms_s/32.0f+1e-5f; float r_s=std::sqrt(ms_s);
                        float dot_h=0, dot_s=0;
                        for(int d=0;d<32;++d){dot_h+=d_hg[bt*32+d]*hgs[l*BL*32+bt*32+d]; dot_s+=d_sg[bt*32+d]*sgs[l*BL*32+bt*32+d];}
                        for(int d=0;d<32;++d){d_h[(b*(SEQ+1)+t+1)*32+d]=d_hg[bt*32+d]/r_h-hgs[l*BL*32+bt*32+d]*dot_h/(32.0f*r_h*r_h*r_h); d_s[(b*(SEQ+1)+t+1)*32+d]=d_sg[bt*32+d]/r_s-sgs[l*BL*32+bt*32+d]*dot_s/(32.0f*r_s*r_s*r_s);}
                    }
                }
                for(int b=0;b<BATCH;++b) for(int d=0;d<32;++d){
                    for(int t=SEQ-1;t>=0;--t){int bt=b*SEQ+t;
                        float dhn=d_h[(b*(SEQ+1)+t+1)*32+d], dsn=d_s[(b*(SEQ+1)+t+1)*32+d];
                        float a=alphas[l*BL*32+bt*32+d], yv=ys[l*BL*32+bt*32+d], hv=hs[l*BATCH*(SEQ+1)*32+(b*(SEQ+1)+t)*32+d];
                        d_ys[l*BL*32+bt*32+d]=(1.0f-a)*dhn+dsn;
                        d_alphas[l*BL*32+bt*32+d]=(hv-yv)*dhn;
                        d_h[(b*(SEQ+1)+t)*32+d]=a*dhn;
                        d_s[(b*(SEQ+1)+t)*32+d]=dsn;
                    }
                }
                std::fill(d_x.begin(), d_x.end(), 0.0f);
                for(int d=0;d<32;++d){
                    float a0=0,a1=0,a2=0;
                    for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t){int bt=b*SEQ+t;
                        float dy=d_ys[l*BL*32+bt*32+d];
                        if(t>=2){a0+=dy*xs[l*BL*32+(b*SEQ+t-2)*32+d]; d_x[(b*SEQ+t-2)*32+d]+=dy*m.q3w0[l*32+d];}
                        if(t>=1){a1+=dy*xs[l*BL*32+(b*SEQ+t-1)*32+d]; d_x[(b*SEQ+t-1)*32+d]+=dy*m.q3w1[l*32+d];}
                        a2+=dy*xs[l*BL*32+bt*32+d]; d_x[bt*32+d]+=dy*m.q3w2[l*32+d];
                    }
                    d_q3w0[l*32+d]+=a0; d_q3w1[l*32+d]+=a1; d_q3w2[l*32+d]+=a2;
                }
                for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t){int bt=b*SEQ+t;
                    for(int d=0;d<32;++d){
                        float a=alphas[l*BL*32+bt*32+d];
                        float dz=d_alphas[l*BL*32+bt*32+d]*a*(1.0f-a)/2.0f;
                        d_alpha[bt*32+d]=dz;
                    }
                }
                for(int d=0;d<32;++d) for(int k=0;k<32;++k){
                    float s=0;
                    for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t) s+=d_alpha[(b*SEQ+t)*32+d]*xs[l*BL*32+(b*SEQ+t)*32+k];
                    d_aW[l*32*32+d*32+k]+=s;
                }
                for(int d=0;d<32;++d){float s=0; for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t) s+=d_alpha[(b*SEQ+t)*32+d]; d_ab[l*32+d]+=s;}
                for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t){int bt=b*SEQ+t;
                    for(int d=0;d<32;++d){float dz=d_alpha[bt*32+d]; for(int k=0;k<32;++k) d_x[bt*32+k]+=dz*m.aW[l*32*32+d*32+k];}
                }
                if(l>0){for(int n=0;n<BL;++n) for(int d=0;d<32;++d) d_hg[n*32+d]=d_sg[n*32+d]=d_x[n*32+d];}
                else{for(int n=0;n<BL;++n){int id=inpBL[n]; for(int d=0;d<32;++d) d_emb_g[id*32+d]+=d_x[n*32+d];}}
            }
            // Adam
            float bc1=1-std::pow(b1,(float)m.step), bc2=1-std::pow(b2,(float)m.step);
            for(int i=0;i<V_unit*V_unit;++i){float g=d_Wbi_g[i]; if(g>1)g=1; if(g<-1)g=-1; m.Wbi_m[i]=b1*m.Wbi_m[i]+(1-b1)*g; m.Wbi_v[i]=b2*m.Wbi_v[i]+(1-b2)*g*g; float step=LR*(m.Wbi_m[i]/bc1)/(std::sqrt(m.Wbi_v[i]/bc2)+eps); m.Wbi[i]=std::min(std::max(m.Wbi[i]-step,-8.0f),8.0f);}
            for(int i=0;i<V_unit*32;++i){float g=d_emb_g[i]; if(g>1)g=1; if(g<-1)g=-1; m.emb_m[i]=b1*m.emb_m[i]+(1-b1)*g; m.emb_v[i]=b2*m.emb_v[i]+(1-b2)*g*g; float step=LR*(m.emb_m[i]/bc1)/(std::sqrt(m.emb_v[i]/bc2)+eps); m.emb[i]=std::min(std::max(m.emb[i]-step,-4.0f),4.0f);}
            for(int i=0;i<V_unit*32;++i){float g=d_Wh_g[i]; if(g>1)g=1; if(g<-1)g=-1; m.Wh_m[i]=b1*m.Wh_m[i]+(1-b1)*g; m.Wh_v[i]=b2*m.Wh_v[i]+(1-b2)*g*g; float step=LR*(m.Wh_m[i]/bc1)/(std::sqrt(m.Wh_v[i]/bc2)+eps); m.Wh[i]=std::min(std::max(m.Wh[i]-step,-1.0f),1.0f);}
            for(int i=0;i<V_unit*32;++i){float g=d_Ws_g[i]; if(g>1)g=1; if(g<-1)g=-1; m.Ws_m[i]=b1*m.Ws_m[i]+(1-b1)*g; m.Ws_v[i]=b2*m.Ws_v[i]+(1-b2)*g*g; float step=LR*(m.Ws_m[i]/bc1)/(std::sqrt(m.Ws_v[i]/bc2)+eps); m.Ws[i]=std::min(std::max(m.Ws[i]-step,-1.0f),1.0f);}
            for(int i=0;i<2*32;++i){float g=d_q3w0[i]; if(g>1)g=1; if(g<-1)g=-1; m.q3w0_m[i]=b1*m.q3w0_m[i]+(1-b1)*g; m.q3w0_v[i]=b2*m.q3w0_v[i]+(1-b2)*g*g; float step=LR*(m.q3w0_m[i]/bc1)/(std::sqrt(m.q3w0_v[i]/bc2)+eps); m.q3w0[i]=std::min(std::max(m.q3w0[i]-step,-2.0f),2.0f);}
            for(int i=0;i<2*32;++i){float g=d_q3w1[i]; if(g>1)g=1; if(g<-1)g=-1; m.q3w1_m[i]=b1*m.q3w1_m[i]+(1-b1)*g; m.q3w1_v[i]=b2*m.q3w1_v[i]+(1-b2)*g*g; float step=LR*(m.q3w1_m[i]/bc1)/(std::sqrt(m.q3w1_v[i]/bc2)+eps); m.q3w1[i]=std::min(std::max(m.q3w1[i]-step,-2.0f),2.0f);}
            for(int i=0;i<2*32;++i){float g=d_q3w2[i]; if(g>1)g=1; if(g<-1)g=-1; m.q3w2_m[i]=b1*m.q3w2_m[i]+(1-b1)*g; m.q3w2_v[i]=b2*m.q3w2_v[i]+(1-b2)*g*g; float step=LR*(m.q3w2_m[i]/bc1)/(std::sqrt(m.q3w2_v[i]/bc2)+eps); m.q3w2[i]=std::min(std::max(m.q3w2[i]-step,-2.0f),2.0f);}
            for(int i=0;i<2*32*32;++i){float g=d_aW[i]; if(g>1)g=1; if(g<-1)g=-1; m.aW_m[i]=b1*m.aW_m[i]+(1-b1)*g; m.aW_v[i]=b2*m.aW_v[i]+(1-b2)*g*g; float step=LR_ALPHA*(m.aW_m[i]/bc1)/(std::sqrt(m.aW_v[i]/bc2)+eps); m.aW[i]=std::min(std::max(m.aW[i]-step,-4.0f),4.0f);}
            for(int i=0;i<2*32;++i){float g=d_ab[i]; if(g>1)g=1; if(g<-1)g=-1; m.ab_m[i]=b1*m.ab_m[i]+(1-b1)*g; m.ab_v[i]=b2*m.ab_v[i]+(1-b2)*g*g; float step=LR_ALPHA*(m.ab_m[i]/bc1)/(std::sqrt(m.ab_v[i]/bc2)+eps); m.ab[i]=std::min(std::max(m.ab[i]-step,-8.0f),8.0f);}
            std::fill(d_Wbi_g.begin(),d_Wbi_g.end(),0.0f); std::fill(d_emb_g.begin(),d_emb_g.end(),0.0f);
            std::fill(d_Wh_g.begin(),d_Wh_g.end(),0.0f); std::fill(d_Ws_g.begin(),d_Ws_g.end(),0.0f);
            std::fill(d_q3w0.begin(),d_q3w0.end(),0.0f); std::fill(d_q3w1.begin(),d_q3w1.end(),0.0f);
            std::fill(d_q3w2.begin(),d_q3w2.end(),0.0f); std::fill(d_aW.begin(),d_aW.end(),0.0f);
            std::fill(d_ab.begin(),d_ab.end(),0.0f);
        }
        float avg=total/nb;
        std::printf("  ep%2d/%d avg_loss=%.4f\n",epoch+1,EPOCHS,avg);
    }
    // Generation
    std::vector<int> ids={encode_char('t'),encode_char('h'),encode_char('e')};
    for(int step=0;step<40;++step){
        std::vector<int> inBL(BL);
        int L=(int)ids.size();
        std::vector<int> in2(SEQ);
        for(int i=0;i<SEQ;++i){int idx=L-SEQ+i; in2[i]=(idx<0)?PAD:ids[idx];}
        for(int b=0;b<BATCH;++b) for(int t=0;t<SEQ;++t) inBL[b*SEQ+t]=in2[t];
        forward_inline(m, inBL, BATCH, SEQ, PAD, x, h, s, y, alpha, hg, sg, logits, probs, xs, ys, alphas, hs, ss, hgs, sgs);
        int bt=(BATCH-1)*SEQ+(SEQ-1);
        int best=0; float best_l=logits[bt*V_unit];
        for(int v=1;v<V_unit;++v) if(logits[bt*V_unit+v]>best_l){best_l=logits[bt*V_unit+v]; best=v;}
        ids.push_back(best);
    }
    std::printf("  gen: \"");
    for(int id:ids) std::printf("%c",decode_id(id));
    std::printf("\"\n");
    return 0;
}

int main(){
    run_batch_size(1);
    run_batch_size(2);
    run_batch_size(4);
    run_batch_size(8);
    return 0;
}
