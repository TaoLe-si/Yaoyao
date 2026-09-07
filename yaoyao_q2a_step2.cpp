// Step 2: SEQ 扩展到 64 + Q2-A 通道数学验证 (clean)
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <chrono>

#define TEST(cond, msg) do { \
    if(!(cond)){ std::printf("FAIL: %s (line %d)\n", msg, __LINE__); fails++; } \
    else { std::printf("PASS: %s\n", msg); passes++; } \
} while(0)

struct Channels {
    int D;
    std::vector<float> h, s;
    void forward(const std::vector<float>& y, const std::vector<float>& alpha, int SEQ){
        D = (int)y.size() / SEQ;
        h.assign((SEQ+1)*D, 0.0f);
        s.assign((SEQ+1)*D, 0.0f);
        for(int t=0;t<SEQ;++t){
            for(int d=0;d<D;++d){
                float yt=y[t*D+d], a=alpha[t*D+d];
                h[(t+1)*D+d] = a*h[t*D+d] + (1-a)*yt;
                s[(t+1)*D+d] = s[t*D+d] + yt;
            }
        }
    }
    void rms(const std::vector<float>& in, std::vector<float>& out, int SEQ){
        D = (int)in.size() / SEQ;
        out.assign(SEQ*D, 0.0f);
        for(int t=0;t<SEQ;++t){
            float ms=0;
            for(int d=0;d<D;++d){float v=in[t*D+d]; ms+=v*v;}
            ms = ms/D + 1e-5f;
            float r = 1.0f/std::sqrt(ms);
            for(int d=0;d<D;++d) out[t*D+d] = in[t*D+d]*r;
        }
    }
    void rms_backward(const std::vector<float>& in, const std::vector<float>& d_out,
                      std::vector<float>& d_in, int SEQ){
        D = (int)in.size() / SEQ;
        d_in.assign(SEQ*D, 0.0f);
        for(int t=0;t<SEQ;++t){
            float ms=0;
            for(int d=0;d<D;++d){float v=in[t*D+d]; ms+=v*v;}
            ms = ms/D + 1e-5f;
            float r = 1.0f/std::sqrt(ms);
            float dot=0;
            for(int d=0;d<D;++d) dot += d_out[t*D+d]*in[t*D+d];
            for(int d=0;d<D;++d){
                // Correct formula: d_in[d] = d_out[d]*r - in[d]*r^3*dot/D
                d_in[t*D+d] = d_out[t*D+d]*r - in[t*D+d]*r*r*r*dot/D;
            }
        }
    }
};

int main(){
    int passes=0, fails=0;
    std::mt19937 rng(42);

    // ===== Test 1: h 通道 forward 手算 (SEQ=4, D=2) =====
    std::printf("=== Test 1: h 通道 forward 手算 ===\n");
    {
        Channels ch;
        std::vector<float> y = {1,2, 5,6, 9,10, 13,14};
        std::vector<float> alpha = {0.5f,0.5f, 0.5f,0.5f, 0.5f,0.5f, 0.5f,0.5f};
        ch.forward(y, alpha, 4);
        // h_0=[0,0], h_1=[0.5,1], h_2=[2.75,3.5], h_3=[5.875,6.75], h_4=[9.4375,10.375]
        float exp_h[5][2] = {{0,0},{0.5f,1.0f},{2.75f,3.5f},{5.875f,6.75f},{9.4375f,10.375f}};
        for(int t=0;t<5;++t) for(int d=0;d<2;++d)
            TEST(std::abs(ch.h[t*2+d]-exp_h[t][d])<1e-5f, "h channel manual");
    }

    // ===== Test 2: s 通道 forward 手算 =====
    std::printf("\n=== Test 2: s 通道 forward 手算 ===\n");
    {
        Channels ch;
        std::vector<float> y = {1,2, 5,6, 9,10, 13,14};
        std::vector<float> alpha = {0.5f,0.5f, 0.5f,0.5f, 0.5f,0.5f, 0.5f,0.5f};
        ch.forward(y, alpha, 4);
        // s_0=[0,0], s_1=[1,2], s_2=[6,8], s_3=[15,18], s_4=[28,32]
        float exp_s[5][2] = {{0,0},{1,2},{6,8},{15,18},{28,32}};
        for(int t=0;t<5;++t) for(int d=0;d<2;++d)
            TEST(std::abs(ch.s[t*2+d]-exp_s[t][d])<1e-5f, "s channel manual");
    }

    // ===== Test 3: h 通道 BPTT 数值梯度 =====
    std::printf("\n=== Test 3: h 通道 BPTT 数值梯度 ===\n");
    {
        int SEQ=4, D=3;
        std::normal_distribution<float> nd(0,1);
        std::vector<float> y2(SEQ*D), alpha2(SEQ*D);
        for(auto& v:y2) v=nd(rng);
        for(auto& v:alpha2) v=std::abs(nd(rng))*0.4f+0.3f;
        Channels ch2; ch2.forward(y2, alpha2, SEQ);
        std::vector<float> d_h_next(D);
        for(auto& v:d_h_next) v=nd(rng);
        // BPTT step at t=SEQ-1: d_y[t] = (1-α[t])*d_h_next + d_s_next(=0)
        std::vector<float> d_y(SEQ*D,0), d_alpha(SEQ*D,0);
        std::vector<float> d_h_curr(D,0);
        for(int d=0;d<D;++d){
            float a=alpha2[(SEQ-1)*D+d];
            d_y[(SEQ-1)*D+d] = (1-a)*d_h_next[d];
            d_alpha[(SEQ-1)*D+d] = (ch2.h[(SEQ-1)*D+d] - y2[(SEQ-1)*D+d])*d_h_next[d];
        }
        float eps=1e-3f;
        for(int d=0;d<D;++d){
            float orig=y2[(SEQ-1)*D+d];
            y2[(SEQ-1)*D+d] = orig+eps;
            Channels ch_p; ch_p.forward(y2, alpha2, SEQ);
            float L_p=0; for(int dd=0;dd<D;++dd) L_p+=d_h_next[dd]*ch_p.h[SEQ*D+dd];
            y2[(SEQ-1)*D+d] = orig-eps;
            Channels ch_n; ch_n.forward(y2, alpha2, SEQ);
            float L_n=0; for(int dd=0;dd<D;++dd) L_n+=d_h_next[dd]*ch_n.h[SEQ*D+dd];
            y2[(SEQ-1)*D+d] = orig;
            float num = (L_p-L_n)/(2*eps);
            TEST(std::abs(d_y[(SEQ-1)*D+d]-num)<1e-3f, "h BPTT d_y matches numerical");
        }
    }

    // ===== Test 4: s 通道 BPTT 数值梯度 (cumulative) =====
    std::printf("\n=== Test 4: s 通道 BPTT 数值梯度 (cumulative) ===\n");
    {
        int SEQ=4, D=3;
        std::normal_distribution<float> nd(0,1);
        std::vector<float> y2(SEQ*D), alpha2(SEQ*D);
        for(auto& v:y2) v=nd(rng);
        for(auto& v:alpha2) v=0.5f;
        std::vector<float> d_s_full(SEQ*D);
        for(auto& v:d_s_full) v=nd(rng);
        // y[t] affects s[t+1], s[t+2], ... so d_loss/d_y[t] (s part) = Σ_{t'≥t} d_s_full[t'+1][d]
        // Wait, d_s_full[t+1] is gradient w.r.t. s[t+1] (which is sum of y[0..t])
        // And s[t+1] only depends on y[0..t] (cumulative sum). So d_loss/d_y[t] = Σ_{t'≥t} d_s_full[t'+1]
        // = Σ_{t'=t}^{SEQ-1} d_s_full[t'*D+d]
        // But the test uses index t*D for d_s_full, so d_s_full[t+1] = d_s_full[(t+1)*D+d]
        // d_loss/d_y[t][d] = Σ_{t'=t+1}^{SEQ} d_s_full[t'*D+d]
        // Hmm let me re-derive.
        // d_loss = Σ_{t'} d_s_full[t'+1] * s[t'+1]
        // d_y[t][d] = Σ_{t'} d_s_full[t'+1] * d_s[t'+1]/d_y[t][d]
        // d_s[t'+1]/d_y[t][d] = 1 if t' ≥ t (because s[t'+1] = sum of y[0..t'])
        // So d_y[t][d] = Σ_{t' ≥ t} d_s_full[(t'+1)*D+d]
        // = Σ_{tt ≥ t+1} d_s_full[tt*D+d] for tt in 1..SEQ (i.e., tt = t'+1)
        // So range: tt in [t+1, SEQ], i.e., sum over (t+1)..SEQ.
        // Wait, t ranges 0..SEQ-1. tt = t'+1 ranges 1..SEQ. For y[t], d_s[tt] depends on y[t] if tt > t, i.e., tt in t+1..SEQ.
        // d_y[t][d] = Σ_{tt=t+1}^{SEQ} d_s_full[tt*D+d]
        // But d_s_full is only indexed 0..SEQ-1 (size SEQ*D). So we need tt in t+1..SEQ-1, missing tt=SEQ.
        // d_s_full at index SEQ is out of range. Let me redefine.
        // Actually let me redefine d_s_full to be size (SEQ+1)*D so we can include the gradient at s[SEQ].
        // For simplicity, use size (SEQ+1)*D with d_s_full[0..SEQ*D] = gradients at s[1..SEQ+1]... wait s[SEQ+1] doesn't exist (max is s[SEQ]).
        // Hmm, the issue is what is d_s_full[SEQ*D+d] = gradient at s[SEQ+1] (which doesn't exist).
        // 
        // Actually let me re-think. s is indexed 0..SEQ (size (SEQ+1)*D). s[0]=0 (initial).
        // y[t] affects s[t+1]. s[t+1] = Σ_{i=0}^{t} y[i].
        // So d_y[t] = Σ_{t' ≥ t+1} d_loss/d_s[t']
        // d_loss/d_s[t'] is the gradient w.r.t. s[t'] from the rest of the network.
        // If we define d_s_full[t'] = d_loss/d_s[t'] for t' in 1..SEQ, then
        // d_y[t][d] = Σ_{t' = t+1}^{SEQ} d_s_full[t'][d]
        //          = Σ_{t' = t+1}^{SEQ} d_s_full[t'*D+d]
        // 
        // But the test has d_s_full size SEQ*D (not (SEQ+1)*D). So d_s_full[t'*D+d] for t' in 0..SEQ-1, missing t'=SEQ.
        // For the test, let me just use Σ over tt in t..SEQ-1 (which is d_s_full[(tt+1)*D+d] in original notation).
        // Actually let me redefine: d_s_full[t] = gradient at s[t+1]. Then size SEQ*D.
        // d_y[t][d] = Σ_{tt ≥ t} d_s_full[tt*D+d]
        float eps=1e-3f;
        for(int t=SEQ-1;t>=0;--t){
            for(int d=0;d<D;++d){
                float orig=y2[t*D+d];
                y2[t*D+d] = orig+eps;
                Channels ch_p; ch_p.forward(y2, alpha2, SEQ);
                float L_p=0; for(int tt=t;tt<SEQ;++tt) for(int dd=0;dd<D;++dd) L_p+=d_s_full[tt*D+dd]*ch_p.s[(tt+1)*D+dd];
                y2[t*D+d] = orig-eps;
                Channels ch_n; ch_n.forward(y2, alpha2, SEQ);
                float L_n=0; for(int tt=t;tt<SEQ;++tt) for(int dd=0;dd<D;++dd) L_n+=d_s_full[tt*D+dd]*ch_n.s[(tt+1)*D+dd];
                y2[t*D+d] = orig;
                float num = (L_p-L_n)/(2*eps);
                float expected = 0;
                for(int tt=t; tt<SEQ; ++tt) expected += d_s_full[tt*D+d];
                TEST(std::abs(num-expected)<1e-3f, "s BPTT cumulative sum");
            }
        }
    }

    // ===== Test 5: RMSNorm forward 数学 =====
    std::printf("\n=== Test 5: RMSNorm forward 数学 ===\n");
    {
        Channels ch;
        std::vector<float> in = {3,4,0,0};
        std::vector<float> out;
        ch.rms(in, out, 1);
        TEST(std::abs(out[0]-1.2f)<1e-4f, "RMSNorm[0]");
        TEST(std::abs(out[1]-1.6f)<1e-4f, "RMSNorm[1)");
    }

    // ===== Test 6: RMSNorm BPTT 数值梯度 =====
    std::printf("\n=== Test 6: RMSNorm BPTT 数值梯度 ===\n");
    {
        Channels ch;
        std::vector<float> in = {3,4,0,0};
        std::vector<float> d_out = {1,-1,0.5f,0.2f};
        std::vector<float> d_in;
        ch.rms_backward(in, d_out, d_in, 1);
        float eps=1e-3f;
        for(int d=0;d<4;++d){
            float orig=in[d];
            in[d]=orig+eps;
            std::vector<float> op; ch.rms(in,op,1);
            float Lp=0; for(int dd=0;dd<4;++dd) Lp+=d_out[dd]*op[dd];
            in[d]=orig-eps;
            std::vector<float> on; ch.rms(in,on,1);
            float Ln=0; for(int dd=0;dd<4;++dd) Ln+=d_out[dd]*on[dd];
            in[d]=orig;
            float num=(Lp-Ln)/(2*eps);
            TEST(std::abs(d_in[d]-num)<1e-3f, "RMSNorm d_in");
        }
    }

    // ===== Test 7: 端到端 SEQ=64 完整 BPTT =====
    std::printf("\n"); std::printf("\n=== Test 7: ===\n");
            {
        int SEQ=64, D=8;
        std::normal_distribution<float> nd(0,1);
        std::vector<float> y(SEQ*D), alpha(SEQ*D);
        for(auto& v:y) v=nd(rng);
        for(auto& v:alpha) v=std::abs(nd(rng))*0.4f+0.3f;
        std::vector<float> coef(D);
        for(auto& v:coef) v=nd(rng)*0.5f;
        // Forward
        Channels ch; ch.forward(y, alpha, SEQ);
        std::vector<float> h_seq(SEQ*D), s_seq(SEQ*D);
        for(int t=0;t<SEQ;++t) for(int d=0;d<D;++d){
            h_seq[t*D+d] = ch.h[(t+1)*D+d];
            s_seq[t*D+d] = ch.s[(t+1)*D+d];
        }
        std::vector<float> hg, sg;
        ch.rms(h_seq, hg, SEQ);
        ch.rms(s_seq, sg, SEQ);
        // Backward
        std::vector<float> d_hg(SEQ*D), d_sg(SEQ*D);
        for(int t=0;t<SEQ;++t) for(int d=0;d<D;++d){ d_hg[t*D+d]=coef[d]; d_sg[t*D+d]=coef[d]; }
        std::vector<float> d_h_seq, d_s_seq;
        ch.rms_backward(h_seq, d_hg, d_h_seq, SEQ);
        ch.rms_backward(s_seq, d_sg, d_s_seq, SEQ);
        // d_y[t] = (1-α[t]) * d_loss/d_h[t+1] + Σ_{t'≥t} d_s_seq[t']
        // d_loss/d_h[t+1] = d_h_seq[t] + α[t+1] * d_loss/d_h[t+2]
        std::vector<float> cum_s((SEQ+1)*D, 0);
        for(int t=SEQ-1;t>=0;--t) for(int d=0;d<D;++d) cum_s[t*D+d] = cum_s[(t+1)*D+d] + d_s_seq[t*D+d];
        std::vector<float> d_h_total((SEQ+1)*D, 0.0f);
        std::vector<float> d_y(SEQ*D,0), d_alpha(SEQ*D,0);
        for(int t=SEQ-1;t>=0;--t){
            for(int d=0;d<D;++d){
                d_h_total[(t+1)*D+d] += d_h_seq[t*D+d];
                float a = alpha[t*D+d];
                float h_prev = ch.h[t*D+d];
                float dh = d_h_total[(t+1)*D+d];
                d_y[t*D+d] += (1-a)*dh + cum_s[t*D+d];
                d_alpha[t*D+d] += (h_prev - y[t*D+d])*dh;
                d_h_total[t*D+d] += a * dh;
            }
        }
        
        // Numerical check d_y[0]
        float eps=1e-3f;
        for(int d=0;d<D;++d){
            float orig=y[d];
            y[d]=orig+eps;
            Channels ch_p; ch_p.forward(y, alpha, SEQ);
            std::vector<float> hp(SEQ*D), sp(SEQ*D);
            for(int tt=0;tt<SEQ;++tt) for(int dd=0;dd<D;++dd){
                hp[tt*D+dd]=ch_p.h[(tt+1)*D+dd]; sp[tt*D+dd]=ch_p.s[(tt+1)*D+dd];
            }
            std::vector<float> hgp, sgp;
            ch_p.rms(hp, hgp, SEQ); ch_p.rms(sp, sgp, SEQ);
            float Lp=0; for(int tt=0;tt<SEQ;++tt) for(int dd=0;dd<D;++dd) Lp+=(hgp[tt*D+dd]+sgp[tt*D+dd])*coef[dd];
            y[d]=orig-eps;
            Channels ch_n; ch_n.forward(y, alpha, SEQ);
            for(int tt=0;tt<SEQ;++tt) for(int dd=0;dd<D;++dd){
                hp[tt*D+dd]=ch_n.h[(tt+1)*D+dd]; sp[tt*D+dd]=ch_n.s[(tt+1)*D+dd];
            }
            ch_n.rms(hp, hgp, SEQ); ch_n.rms(sp, sgp, SEQ);
            float Ln=0; for(int tt=0;tt<SEQ;++tt) for(int dd=0;dd<D;++dd) Ln+=(hgp[tt*D+dd]+sgp[tt*D+dd])*coef[dd];
            y[d]=orig;
            float num=(Lp-Ln)/(2*eps);
            float rel = std::abs(num)>0.1f ? std::abs(d_y[d]-num)/std::abs(num) : std::abs(d_y[d]-num);
            std::printf("    d=%d: d_y=%.4f num=%.4f rel=%.4f\n", d, d_y[d], num, rel);
            TEST(rel<0.10f, "long-seq BPTT d_y[0]");
        }
    }

    // ===== Test 8: 性能基准 (SEQ=64) =====
    std::printf("\n=== Test 8: 性能基准 (SEQ=64) ===\n");
    {
        Channels ch;
        std::normal_distribution<float> nd(0,1);
        std::vector<float> y(64*8), a(64*8);
        for(auto& v:y) v=nd(rng);
        for(auto& v:a) v=0.5f;
        for(int i=0;i<100;++i) ch.forward(y,a,64);
        auto t0=std::chrono::steady_clock::now();
        for(int i=0;i<10000;++i) ch.forward(y,a,64);
        auto t1=std::chrono::steady_clock::now();
        double sec=std::chrono::duration<double>(t1-t0).count();
        std::printf("  10000 fwds in %.3fs (%.0f/s)\n", sec, 10000/sec);
    }

    std::printf("\n========== SUMMARY: %d passed, %d failed ==========\n", passes, fails);
    return fails>0?1:0;
}
