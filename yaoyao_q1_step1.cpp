// Step 1: Q1 Hash Bucket Pool - 独立实现 + 完整单元测试
// 三值设计: trit ∈ {-1, 0, +1}, 存储为 int8 (4 trits/byte 优化未来再加)
// 数学:
//   forward:  h = hash(id)
//             bucket = trits[h*K .. h*K+K-1]   (K 个 D 维候选)
//             scores[k] = <query, bucket[k]>
//             weights[k] = softmax(scores)[k]
//             output[d] = Σ_k weights[k] * trit[h,k,d]
//   backward: d_trit[h,k,d] += query[d] * d_scores[k]
//             d_query[d]    += Σ_k trit[h,k,d] * d_scores[k]
//             d_scores[k]   = weights[k] * (d_weights[k] - Σ_j weights[j]*d_weights[j])
//             d_weights[k]  = Σ_d d_output[d] * trit[h,k,d]
//   adam:     m = b1*m + (1-b1)*g; v = b2*v + (1-b2)*g²
//             step = lr * (m/(1-b1^t)) / (sqrt(v/(1-b2^t)) + eps)
//             new_trit = clip(trit - sign(step), {-1, 0, +1})  [三值约束]
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <cassert>
#include <chrono>

struct Q1 {
    int B, K, D;
    std::vector<int8_t> trits;       // B*K*D, values ∈ {-1,0,+1}
    std::vector<float> adam_m, adam_v;
    int step=0;

    void init(int B_, int K_, int D_, std::mt19937& rng){
        B=B_; K=K_; D=D_;
        trits.assign((size_t)B*K*D, 0);
        adam_m.assign((size_t)B*K*D, 0.0f);
        adam_v.assign((size_t)B*K*D, 0.0f);
        std::uniform_int_distribution<int> ud(-1, 1);
        for(auto& t:trits) t = (int8_t)ud(rng);
    }

    static int hash(int id, int B){
        // Knuth multiplicative hash, ensures uniform distribution
        uint64_t x = (uint32_t)id;
        x = x * 2654435761u;
        x = (x >> 16) ^ x;  // extra mixing
        return (int)(x % (uint64_t)B);
    }

    // Forward: lookup id, return D-dim output. Also store aux for BPTT.
    struct Aux { int id; int h; std::vector<float> weights; std::vector<float> scores; const float* query; };
    void forward(int id, const float* query, float* out, Aux& aux) const {
        aux.id = id;
        aux.h = hash(id, B);
        aux.query = query;
        aux.weights.assign(K, 0.0f);
        aux.scores.assign(K, 0.0f);
        float mx = -1e9f;
        for(int k=0;k<K;++k){
            float s=0;
            const int8_t* tk = &trits[(aux.h*K+k)*D];
            for(int d=0;d<D;++d) s += query[d]*tk[d];
            aux.scores[k] = s;
            if(s>mx) mx=s;
        }
        float sum=0;
        for(int k=0;k<K;++k){ aux.weights[k] = std::exp(aux.scores[k]-mx); sum += aux.weights[k]; }
        for(int k=0;k<K;++k) aux.weights[k] /= sum;
        for(int d=0;d<D;++d){
            float v=0;
            for(int k=0;k<K;++k) v += aux.weights[k]*trits[(aux.h*K+k)*D+d];
            out[d] = v;
        }
    }

    // Backward: given d_output[d] and aux, accumulate d_trit and d_query
    void backward(const Aux& aux, const float* d_out, std::vector<float>& d_trit, float* d_query) const {
        // d_weights[k] = Σ_d d_out[d] * trit[h,k,d]
        std::vector<float> d_weights(K, 0.0f);
        for(int k=0;k<K;++k){
            float s=0;
            const int8_t* tk = &trits[(aux.h*K+k)*D];
            for(int d=0;d<D;++d) s += d_out[d]*tk[d];
            d_weights[k] = s;
        }
        // d_scores[k] = weights[k] * (d_weights[k] - Σ_j weights[j]*d_weights[j])
        float dot = 0;
        for(int k=0;k<K;++k) dot += aux.weights[k]*d_weights[k];
        std::vector<float> d_scores(K);
        for(int k=0;k<K;++k) d_scores[k] = aux.weights[k]*(d_weights[k]-dot);
        // d_trit[h,k,d] += query[d] * d_scores[k]
        for(int k=0;k<K;++k){
            float* tk_g = &d_trit[(aux.h*K+k)*D];
            float ds = d_scores[k];
            for(int d=0;d<D;++d) tk_g[d] += aux.query[d]*ds;
        }
        // d_query[d] += Σ_k trit[h,k,d] * d_scores[k]
        for(int d=0;d<D;++d){
            float s=0;
            for(int k=0;k<K;++k) s += trits[(aux.h*K+k)*D+d]*d_scores[k];
            d_query[d] += s;
        }
    }

    // Adam update with three-value constraint: new value in {-1, 0, +1}
    void adam_update(const std::vector<float>& grad, float lr, float b1, float b2, float eps){
        step++;
        float bc1 = 1-std::pow(b1, (float)step);
        float bc2 = 1-std::pow(b2, (float)step);
        for(size_t i=0;i<trits.size();++i){
            float g = grad[i];
            if(g>1.0f) g=1.0f; if(g<-1.0f) g=-1.0f;  // soft clip
            adam_m[i] = b1*adam_m[i] + (1-b1)*g;
            adam_v[i] = b2*adam_v[i] + (1-b2)*g*g;
            float step_amt = lr*(adam_m[i]/bc1)/(std::sqrt(adam_v[i]/bc2)+eps);
            float new_val = (float)trits[i] - step_amt;
            // hard clip to {-1, 0, +1}
            if(new_val > 0.5f) trits[i] = 1;
            else if(new_val < -0.5f) trits[i] = -1;
            else trits[i] = 0;
        }
    }
};

// ============== 单元测试 ==============

#define TEST(cond, msg) do { \
    if(!(cond)){ std::printf("FAIL: %s (line %d)\n", msg, __LINE__); fails++; } \
    else { std::printf("PASS: %s\n", msg); passes++; } \
} while(0)

int main(){
    int passes=0, fails=0;
    std::mt19937 rng(42);

    std::printf("=== Test 1: 构造与初始化 ===\n");
    Q1 q1; q1.init(1024, 8, 32, rng);
    TEST(q1.B==1024 && q1.K==8 && q1.D==32, "init dimensions");
    TEST(q1.trits.size() == 1024*8*32, "init size = B*K*D");
    int n_neg=0, n_zero=0, n_pos=0;
    for(auto t:q1.trits){ if(t==-1) n_neg++; else if(t==0) n_zero++; else n_pos++; }
    std::printf("  init distribution: -1=%d, 0=%d, +1=%d\n", n_neg, n_zero, n_pos);
    TEST(n_neg>0 && n_zero>0 && n_pos>0, "init has all 3 trit values");

    std::printf("\n=== Test 2: Hash 函数分布 ===\n");
    // Test 1: same id → same hash
    int h1 = Q1::hash(12345, 1024);
    int h2 = Q1::hash(12345, 1024);
    TEST(h1==h2, "hash deterministic");
    // Test 2: distribution uniform over [0, B)
    std::vector<int> cnt(1024, 0);
    for(int id=0;id<100000;++id) cnt[Q1::hash(id, 1024)]++;
    int min_cnt = *std::min_element(cnt.begin(), cnt.end());
    int max_cnt = *std::max_element(cnt.begin(), cnt.end());
    float mean = 100000.0f/1024.0f;
    std::printf("  distribution: min=%d max=%d mean=%.1f\n", min_cnt, max_cnt, mean);
    TEST(min_cnt > mean*0.5f, "hash roughly uniform (min > 50% of mean)");
    TEST(max_cnt < mean*2.0f, "hash roughly uniform (max < 200% of mean)");

    std::printf("\n=== Test 3: Forward 数学验证 ===\n");
    // 单个查询, 已知 query, 手动计算预期输出
    Q1 q2; q2.init(4, 3, 4, rng);  // B=4, K=3, D=4
    int id = 5; int h = Q1::hash(id, 4);
    std::printf("  id=%d -> bucket h=%d\n", id, h);
    // 手动构造 query
    float query[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4] = {0};
    Q1::Aux aux;
    q2.forward(id, query, out, aux);
    // 手动算 scores[k] = Σ query[d] * trits[h,k,d]
    std::printf("  weights: ");
    for(int k=0;k<3;++k) std::printf("%.3f ", aux.weights[k]);
    std::printf("\n  out: ");
    for(int d=0;d<4;++d) std::printf("%.3f ", out[d]);
    std::printf("\n");
    // 验证 weights 和为 1
    float wsum = 0;
    for(int k=0;k<3;++k) wsum += aux.weights[k];
    TEST(std::abs(wsum-1.0f) < 1e-5f, "weights sum to 1");
    // 验证 output 是 weights 加权组合
    float expected[4] = {0,0,0,0};
    for(int d=0;d<4;++d){
        for(int k=0;k<3;++k) expected[d] += aux.weights[k] * q2.trits[(h*3+k)*4+d];
        TEST(std::abs(out[d]-expected[d]) < 1e-5f, "output = weighted sum");
    }

    std::printf("\n=== Test 4: 三值约束保持 (Adam 后 trits ∈ {-1,0,+1}) ===\n");
    Q1 q3; q3.init(16, 4, 8, rng);
    std::vector<float> grad(q3.trits.size(), 0.0f);
    std::uniform_real_distribution<float> ur(-2.0f, 2.0f);
    for(int t=0;t<100;++t){
        // simulate gradient updates
        for(auto& g:grad) g = ur(rng);
        q3.adam_update(grad, 0.1f, 0.9f, 0.999f, 1e-8f);
    }
    bool all_valid = true;
    for(auto trit : q3.trits) if(trit!=-1 && trit!=0 && trit!=1){ all_valid=false; break; }
    TEST(all_valid, "all trits stay in {-1, 0, +1} after 100 Adam steps");

    std::printf("\n=== Test 5: Backward 数学验证 (numerical check) ===\n");
    // 用 numerical gradient 验证 backward 公式
    Q1 q4; q4.init(8, 4, 4, rng);
    int test_id = 42;
    float test_query[4] = {0.5f, -0.3f, 0.8f, 0.1f};
    float test_out[4];
    Q1::Aux aux4;
    q4.forward(test_id, test_query, test_out, aux4);
    // 任意 d_output
    float d_out[4] = {1.0f, -1.0f, 0.5f, 0.2f};
    // 算 backward
    std::vector<float> d_trit(q4.trits.size(), 0.0f);
    float d_query[4] = {0};
    q4.backward(aux4, d_out, d_trit, d_query);
    // Numerical check: d_query[d] ≈ (L(q+eps) - L(q-eps)) / (2*eps)
    // 其中 L = Σ_out d_out[o] * out[o]
    float eps = 1e-3f;
    for(int d=0;d<4;++d){
        test_query[d] += eps;
        float out_p[4]; Q1::Aux aux_p;
        q4.forward(test_id, test_query, out_p, aux_p);
        float L_p=0; for(int o=0;o<4;++o) L_p += d_out[o]*out_p[o];
        test_query[d] -= 2*eps;
        float out_n[4]; Q1::Aux aux_n;
        q4.forward(test_id, test_query, out_n, aux_n);
        float L_n=0; for(int o=0;o<4;++o) L_n += d_out[o]*out_n[o];
        test_query[d] += eps;
        float numerical = (L_p - L_n) / (2*eps);
        TEST(std::abs(d_query[d]-numerical) < 1e-2f, "d_query matches numerical");
    }

    std::printf("\n=== Test 6: 端到端学习 (人造可分任务) ===\n");
    // 任务: id 的 hash 桶的 第一个 candidate (h*K+0) 的 trit[0] 决定 label
    // 训练一个 Q1 学会预测这个
    Q1 q5; q5.init(32, 4, 8, rng);
    std::vector<int> train_ids;
    std::vector<float> train_labels;
    std::mt19937 rng6(123);
    for(int i=0;i<200;++i){
        int id = rng6()%1000;
        train_ids.push_back(id);
        // label = sign of trit[h*K + 0, dim=0]
        int h = Q1::hash(id, 32);
        train_labels.push_back((float)q5.trits[(h*4+0)*8+0]);
    }
    // 简单的监督: 训练 query 让 output[0] 接近 label
    // 用 0 向量作为 query (这样 output[0] = Σ_k weights[k]*trit[h,k,0])
    // 学会让 weights 偏向 label 接近的 candidate
    float lr=0.05f;
    for(int epoch=0;epoch<50;++epoch){
        float total_loss=0;
        std::vector<float> d_trit_acc(q5.trits.size(), 0.0f);
        for(size_t i=0;i<train_ids.size();++i){
            float query[8] = {0};
            float out[8]; Q1::Aux aux;
            q5.forward(train_ids[i], query, out, aux);
            float d_out[8] = {0};
            d_out[0] = 2.0f*(out[0]-train_labels[i]);  // MSE gradient
            float d_q[8] = {0};
            std::vector<float> d_trit(q5.trits.size(), 0.0f);
            q5.backward(aux, d_out, d_trit, d_q);
            for(size_t j=0;j<d_trit.size();++j) d_trit_acc[j] += d_trit[j];
        }
        q5.adam_update(d_trit_acc, lr, 0.9f, 0.999f, 1e-8f);
        if(epoch%10==0){
            // eval
            float eval_loss=0;
            for(size_t i=0;i<train_ids.size();++i){
                float query[8] = {0};
                float out[8]; Q1::Aux aux;
                q5.forward(train_ids[i], query, out, aux);
                float diff = out[0]-train_labels[i];
                eval_loss += diff*diff;
            }
            std::printf("  epoch %d: eval_loss=%.4f\n", epoch, eval_loss/train_ids.size());
            total_loss = eval_loss/train_ids.size();
        }
    }

    std::printf("\n=== Test 7: CPU 性能基准 ===\n");
    Q1 q6; q6.init(1024, 8, 128, rng);  // 真实尺寸
    std::vector<float> query7(128);
    std::vector<float> output(128);
    Q1::Aux aux7;
    // Warmup
    for(int i=0;i<100;++i) q6.forward(i%50000, query7.data(), output.data(), aux7);
    auto t0=std::chrono::steady_clock::now();
    int N=100000;
    for(int i=0;i<N;++i) q6.forward(i%50000, query7.data(), output.data(), aux7);
    auto t1=std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1-t0).count();
    std::printf("  %d forwards in %.3fs (%.0f lookups/sec)\n", N, sec, N/sec);

    std::printf("\n========== SUMMARY: %d passed, %d failed ==========\n", passes, fails);
    return fails > 0 ? 1 : 0;
}
