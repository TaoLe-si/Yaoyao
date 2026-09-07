// Step 6: Q1 numerical gradient verification with REAL query (not zero)
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <cstring>
#include <cstdio>
#include <algorithm>

#define TEST(cond, msg) do { \
    if(!(cond)){ std::printf("FAIL: %s (line %d)\n", msg, __LINE__); fails++; } \
    else { std::printf("PASS: %s\n", msg); passes++; } \
} while(0)

struct Q1 {
    int B, K, D;
    std::vector<int8_t> trits;
    void init(int B_, int K_, int D_, std::mt19937& rng){
        B=B_; K=K_; D=D_;
        trits.assign((size_t)B*K*D, 0);
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
    // Q1 backward - full gradient (direct + indirect)
    // d_trits[(h*K+k)*D+d] += d_out[d]*weights[k] + query[d]*d_scores[k]
    // d_query[d] += sum_k trits[(h*K+k)*D+d] * d_scores[k]
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
};

int main(){
    int passes=0, fails=0;
    std::mt19937 rng(42);

    std::printf("=== Test 1: Q1 forward with real query ===\n");
    {
        int B=8, K=3, D=4;
        Q1 q; q.init(B, K, D, rng);
        std::normal_distribution<float> nd(0,1);
        std::vector<float> query(D);
        for(auto& v:query) v=nd(rng);
        std::vector<float> out(D, 0);
        Q1::Aux aux;
        q.forward(42, query.data(), out.data(), aux);
        int h=Q1::hash(42, B);
        std::vector<float> scores(K);
        for(int k=0;k<K;++k){
            float s=0;
            for(int d=0;d<D;++d) s+=query[d]*q.trits[(h*K+k)*D+d];
            scores[k]=s;
        }
        float mx=*std::max_element(scores.begin(), scores.end());
        std::vector<float> w(K);
        float sum=0;
        for(int k=0;k<K;++k){w[k]=std::exp(scores[k]-mx); sum+=w[k];}
        for(int k=0;k<K;++k) w[k]/=sum;
        std::vector<float> expected(D, 0);
        for(int d=0;d<D;++d) for(int k=0;k<K;++k) expected[d]+=w[k]*q.trits[(h*K+k)*D+d];
        for(int d=0;d<D;++d) TEST(std::abs(out[d]-expected[d])<1e-5f, "Q1 out manual");
    }

    std::printf("\n=== Test 2: Q1 backward - d_query numerical ===\n");
    {
        int B=8, K=3, D=4;
        Q1 q; q.init(B, K, D, rng);
        std::normal_distribution<float> nd(0,1);
        std::vector<float> query(D);
        for(auto& v:query) v=nd(rng);
        std::vector<float> d_out(D);
        for(auto& v:d_out) v=nd(rng);
        std::vector<float> out(D);
        Q1::Aux aux;
        q.forward(42, query.data(), out.data(), aux);
        std::vector<float> d_query(D, 0);
        std::vector<float> d_trit(q.trits.size(), 0);
        q.backward(aux, d_out.data(), d_trit, d_query.data());
        float eps=1e-3f;
        for(int d=0;d<D;++d){
            float orig=query[d];
            query[d]=orig+eps;
            std::vector<float> op(D); q.forward(42, query.data(), op.data(), aux);
            float Lp=0; for(int dd=0;dd<D;++dd) Lp+=d_out[dd]*op[dd];
            query[d]=orig-eps;
            std::vector<float> on(D); q.forward(42, query.data(), on.data(), aux);
            float Ln=0; for(int dd=0;dd<D;++dd) Ln+=d_out[dd]*on[dd];
            query[d]=orig;
            float num=(Lp-Ln)/(2*eps);
            TEST(std::abs(d_query[d]-num)<1e-2f, "Q1 d_query matches numerical");
        }
    }

    std::printf("\n=== Test 3: Q1 backward - d_trit numerical ===\n");
    {
        int B=8, K=3, D=4;
        Q1 q; q.init(B, K, D, rng);
        std::normal_distribution<float> nd(0,1);
        std::vector<float> query(D);
        for(auto& v:query) v=nd(rng);
        std::vector<float> d_out(D);
        for(auto& v:d_out) v=nd(rng);
        Q1::Aux aux;
        std::vector<float> d_trit(q.trits.size(), 0);
        std::vector<float> out(D); q.forward(42, query.data(), out.data(), aux);
        std::vector<float> d_query(D, 0);
        q.backward(aux, d_out.data(), d_trit, d_query.data());
        float eps=1e-3f;
        int h=aux.h;
        for(int k=0;k<K;++k) for(int d=0;d<D;++d){
            int8_t orig=q.trits[(h*K+k)*D+d];
            q.trits[(h*K+k)*D+d]=(int8_t)(orig+1);
            std::vector<float> op(D); q.forward(42, query.data(), op.data(), aux);
            float Lp=0; for(int dd=0;dd<D;++dd) Lp+=d_out[dd]*op[dd];
            q.trits[(h*K+k)*D+d]=(int8_t)(orig-1);
            std::vector<float> on(D); q.forward(42, query.data(), on.data(), aux);
            float Ln=0; for(int dd=0;dd<D;++dd) Ln+=d_out[dd]*on[dd];
            q.trits[(h*K+k)*D+d]=orig;
            float num=(Lp-Ln)/(2.0f);
            TEST(std::abs(d_trit[(h*K+k)*D+d]-num)<8e-2f, "Q1 d_trit matches numerical");
        }
    }

    std::printf("\n=== Test 4: Q1 chain rule verification ===\n");
    {
        int B=16, K=4, D=8;
        Q1 q; q.init(B, K, D, rng);
        std::normal_distribution<float> nd(0,1);
        std::vector<float> query(D);
        for(auto& v:query) v=nd(rng);
        std::vector<float> d_out(D);
        for(auto& v:d_out) v=nd(rng);
        Q1::Aux aux;
        std::vector<float> out(D); q.forward(42, query.data(), out.data(), aux);
        std::vector<float> d_trit(q.trits.size(), 0);
        std::vector<float> d_query(D, 0);
        q.backward(aux, d_out.data(), d_trit, d_query.data());
        std::vector<float> dq(D);
        for(auto& v:dq) v=nd(rng)*0.01f;
        float analytic=0; for(int d=0;d<D;++d) analytic+=d_query[d]*dq[d];
        float eps=1e-4f;
        std::vector<float> q_orig=query;
        for(int d=0;d<D;++d) query[d]=q_orig[d]+eps*dq[d];
        std::vector<float> op(D); q.forward(42, query.data(), op.data(), aux);
        float Lp=0; for(int dd=0;dd<D;++dd) Lp+=d_out[dd]*op[dd];
        for(int d=0;d<D;++d) query[d]=q_orig[d]-eps*dq[d];
        std::vector<float> on(D); q.forward(42, query.data(), on.data(), aux);
        float Ln=0; for(int dd=0;dd<D;++dd) Ln+=d_out[dd]*on[dd];
        query=q_orig;
        float num=(Lp-Ln)/(2*eps);
        TEST(std::abs(analytic-num)<5e-2f, "Q1 chain rule d_query");
    }

    std::printf("\n=== Test 5: Q1 with zero query -> fixed clustering ===\n");
    {
        int B=8, K=3, D=4;
        Q1 q; q.init(B, K, D, rng);
        std::vector<float> zero_query(D, 0.0f);
        std::vector<float> out1(D), out2(D);
        Q1::Aux aux1, aux2;
        std::vector<int> tokens;
        int h=Q1::hash(42, 8);
        for(int i=0;i<100;++i) if(Q1::hash(i,8)==h && i!=42) tokens.push_back(i);
        if(!tokens.empty()){
            q.forward(42, zero_query.data(), out1.data(), aux1);
            q.forward(tokens[0], zero_query.data(), out2.data(), aux2);
            bool equal=true;
            for(int d=0;d<D;++d) if(std::abs(out1[d]-out2[d])>1e-5f) equal=false;
            TEST(equal, "same hash with zero query -> same output");
        } else TEST(false, "found collision");
    }

    std::printf("\n=== Test 6: Q1 real query differentiates same-hash tokens ===\n");
    {
        int B=8, K=4, D=8;
        Q1 q; q.init(B, K, D, rng);
        std::vector<float> q1(D, 0.0f), q2(D, 0.0f);
        q1[0]=1.0f; q2[1]=1.0f;
        std::vector<float> o1(D), o2(D);
        Q1::Aux a1, a2;
        std::vector<int> tokens;
        for(int i=0;i<100;++i) for(int j=i+1;j<100;++j)
            if(Q1::hash(i,8)==Q1::hash(j,8)){tokens.push_back(i); tokens.push_back(j); goto found;}
        found:
        if(tokens.size()>=2){
            q.forward(tokens[0], q1.data(), o1.data(), a1);
            q.forward(tokens[1], q2.data(), o2.data(), a2);
            float diff=0; for(int d=0;d<D;++d) diff+=std::abs(o1[d]-o2[d]);
            TEST(diff>0.01f, "different queries -> different outputs for same hash");
        } else TEST(false, "found collision");
    }

    std::printf("\n========== SUMMARY: %d passed, %d failed ==========\n", passes, fails);
    return fails>0?1:0;
}
