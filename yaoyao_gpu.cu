// 夭夭 GPU v0.6 FIXED - Full BPTT + Separate LR + Temperature
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>
#include <ctime>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); fflush(stdout); exit(1);} } while(0)
#define BK(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("cuBLAS err %s:%d %d\n",__FILE__,__LINE__,(int)s); fflush(stdout); exit(1);} } while(0)
#define SYNC() do { cudaDeviceSynchronize(); cudaError_t _e = cudaGetLastError(); if(_e != cudaSuccess){printf("CUDA sync err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); fflush(stdout); exit(1);} } while(0)

static FILE* g_log = nullptr;
static inline void L(const char* s) { fputs(s, stdout); fflush(stdout); if (g_log) { fputs(s, g_log); fflush(g_log); } }

struct Vocab {
    std::map<int,int> c2i;
    std::map<int,int> i2c;
    int pad=0;
    void load(const std::string& p) {
        std::ifstream f(p); int V; f >> V;
        for (int i=0;i<V;++i) { int cp; f >> cp; i2c[i]=cp; if (cp >= 0) c2i[cp]=i; }
    }
    int size() const { return (int)i2c.size(); }
    void encode_utf8(const std::string& s, std::vector<int>& ids) const {
        ids.clear();
        for (size_t i=0; i<s.size();) {
            const unsigned char* p=(const unsigned char*)s.data(); unsigned char b=p[i]; int cp=0, n=1;
            if (b<0x80) cp=b;
            else if ((b&0xE0)==0xC0 && i+1<s.size()) { cp=((b&31)<<6)|(p[i+1]&63); n=2; }
            else if ((b&0xF0)==0xE0 && i+2<s.size()) { cp=((b&15)<<12)|((p[i+1]&63)<<6)|(p[i+2]&63); n=3; }
            else if ((b&0xF8)==0xF0 && i+3<s.size()) { cp=((b&7)<<18)|((p[i+1]&63)<<12)|((p[i+2]&63)<<6)|(p[i+3]&63); n=4; }
            else cp=0xFFFD;
            i += n; auto it=c2i.find(cp); ids.push_back(it==c2i.end()?pad:it->second);
        }
    }
    std::string dec(int id) const {
        auto it=i2c.find(id); if (it==i2c.end() || it->second<0) return "";
        int cp=it->second; std::string o;
        if(cp<0x80)o.push_back((char)cp); else if(cp<0x800){o.push_back((char)(0xC0|(cp>>6)));o.push_back((char)(0x80|(cp&63)));}
        else if(cp<0x10000){o.push_back((char)(0xE0|(cp>>12)));o.push_back((char)(0x80|((cp>>6)&63)));o.push_back((char)(0x80|(cp&63)));}
        else{o.push_back((char)(0xF0|(cp>>18)));o.push_back((char)(0x80|((cp>>12)&63)));o.push_back((char)(0x80|((cp>>6)&63)));o.push_back((char)(0x80|(cp&63)));} return o;
    }
};

// Alpha forward with temperature T
__global__ void k_alpha_fwd(const float* xs, const float* W_alpha, const float* b_alpha,
                             float* alpha_t, int BL, int D, float T) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int bt = blockIdx.y;
    if (d >= D) return;
    float z = b_alpha[d];
    for (int k = 0; k < D; ++k) z += W_alpha[d * D + k] * xs[bt * D + k];
    float zT = z / T;
    if (zT > 20.0f) zT = 20.0f;
    if (zT < -20.0f) zT = -20.0f;
    alpha_t[bt * D + d] = 1.0f / (1.0f + __expf(-zT));
}

// Alpha backward: dz = dalpha * alpha * (1 - alpha) / T (because alpha = sigmoid(z/T))
__global__ void k_alpha_bwd(const float* xs, const float* W_alpha, const float* alpha_t,
                              const float* dalpha, float* dx_extra,
                              float* dW_alpha, float* db_alpha,
                              int BL, int D, float T) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int bt = blockIdx.y;
    if (d >= D) return;
    float a = alpha_t[bt * D + d];
    float dz = dalpha[bt * D + d] * a * (1.0f - a) / T;
    atomicAdd(&db_alpha[d], dz);
    for (int k = 0; k < D; ++k) {
        atomicAdd(&dx_extra[bt * D + k], W_alpha[d * D + k] * dz);
        atomicAdd(&dW_alpha[d * D + k], dz * xs[bt * D + k]);
    }
}

__global__ void k_row_l2norm(float* W, int rows, int D, float eps){
    int v=blockIdx.x*blockDim.x+threadIdx.x; if(v>=rows) return;
    float s2=0; for(int d=0;d<D;d++){float x=W[v*D+d]; s2+=x*x;}
    float r=rsqrtf(s2+eps);
    for(int d=0;d<D;d++) W[v*D+d]*=r;
}
__global__ void k_zero(float* x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] = 0;
}

__global__ void k_embed(const float* emb, const int* inp, float* x, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int bt = blockIdx.y;
    if (d >= D) return;
    x[bt * D + d] = emb[inp[bt] * D + d];
}

__global__ void k_q3_fwd(const float* x, const float* w0, const float* w1, const float* w2,
                          const char* drop_mask, float* y, float inv_drop, int L, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    int b = blockIdx.z;
    if (d >= D) return;
    int bt = b * L + t;
    float v = 0;
    if (t >= 2) v += w0[d] * x[(b*L + t - 2) * D + d];
    if (t >= 1) v += w1[d] * x[(b*L + t - 1) * D + d];
    v += w2[d] * x[bt * D + d];
    if (drop_mask && !drop_mask[d]) y[bt * D + d] = 0;
    else {
        if (drop_mask) v *= inv_drop;
        if (v > 4.0f) v = 4.0f; if (v < -4.0f) v = -4.0f;
        y[bt * D + d] = v;
    }
}

__global__ void k_channels_fwd(const float* y, const float* alpha_t, float* h, float* s, int L, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (d >= D) return;
    float hc = 0, sc = 0;
    int off = b * (L + 1) * D + d;
    h[off] = 0; s[off] = 0;
    for (int t = 0; t < L; ++t) {
        int bt = b * L + t;
        float yt = y[bt * D + d];
        float a = alpha_t[bt * D + d];
        hc = a * hc + (1.0f - a) * yt;
        sc += yt;
        h[off + (t+1) * D] = hc;
        s[off + (t+1) * D] = sc;
    }
}

__global__ void k_rmsnorm(const float* x, float* y, int BL, int D, float eps){
    int bt=blockIdx.y; float ms=0; for(int d=0;d<D;d++) ms+=x[bt*D+d]*x[bt*D+d]; ms=ms/(float)D+eps;
    float r=rsqrtf(ms);
    for(int d=threadIdx.x;d<D;d+=blockDim.x) y[bt*D+d]=x[bt*D+d]*r;
}
// Q4 ternary STE forward: cond accum {-1,0,+1} * hg + {-1,0,+1} * sg, scaled by scale_h, scale_s
__global__ void k_q4_ternary(const float* Wh, const float* Ws, const float* hg, const float* sg,
                              float scale_h, float scale_s,
                              float* logits, int BL, int V, int D, float thr) {
    int v = blockIdx.x;
    int n = blockIdx.y;
    if (v >= V || n >= BL) return;
    float s = 0.f;
    int off = v * D;
    const float* row_h = Wh + off;
    const float* row_s = Ws + off;
    const float* xh = hg + n * D;
    const float* xs = sg + n * D;
    float lh = 0.f, ls = 0.f;
    for (int d = 0; d < D; ++d) {
        float wh = row_h[d];
        float ws = row_s[d];
        float xhg = xh[d];
        float xsg = xs[d];
        if (wh >  thr) lh += xhg; else if (wh < -thr) lh -= xhg;
        if (ws >  thr) ls += xsg; else if (ws < -thr) ls -= xsg;
    }
    logits[n * V + v] = scale_h * lh + scale_s * ls;
}
// Backward for ternary STE: d_Wh, d_Ws gradients from full float Wh/Ws (no sign pass-through)
__global__ void k_q4_dWh_dWs(const float* hg, const float* sg, const float* d_logits,
                               float* d_Wh, float* d_Ws, int BL, int V, int D) {
    int v = blockIdx.x;
    int n = blockIdx.y;
    if (v >= V || n >= BL) return;
    float g = d_logits[n * V + v];
    float gh = g * hg[n * D];
    float gs = g * sg[n * D];
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        atomicAdd(&d_Wh[v * D + d], gh);
        atomicAdd(&d_Ws[v * D + d], gs);
    }
}
// dhg/dsg recompute from float Wh/Ws (STE so float weights still used)
__global__ void k_q4_dhg_dsg(const float* Wh, const float* Ws, const float* d_logits,
                               float* d_hg, float* d_sg, int BL, int V, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y;
    if (d >= D || n >= BL) return;
    float sh = 0.f, ss = 0.f;
    for (int v = 0; v < V; ++v) {
        float g = d_logits[n * V + v];
        sh += g * Wh[v * D + d];
        ss += g * Ws[v * D + d];
    }
    d_hg[n * D + d] = sh;
    d_sg[n * D + d] = ss;
}

__global__ void k_gather(const float* h, const float* s, float* hg, float* sg, int L, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    int t = blockIdx.z;
    if (d >= D || t >= L) return;
    int src = b * (L+1) * D + (t+1) * D + d;
    hg[(b * L + t) * D + d] = h[src];
    sg[(b * L + t) * D + d] = s[src];
}

__global__ void k_add_bias(float* logits, const float* bias, int N, int V) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (v >= V || n >= N) return;
    logits[n * V + v] += bias[v];
}

// Bigram vocabulary memory: logits[t][v] += W_bi[prev_token[t]][v]
// prev_token[t] = input[t-1] (t=0 uses pad token). W_bi is V x V.
__global__ void k_bigram_fwd(const int* inp, const float* W_bi, float* logits,
                              int BL, int V, int pad_tok) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (v >= V || t >= BL) return;
    int prev = (t > 0) ? inp[t - 1] : pad_tok;
    if (prev < 0 || prev >= V) prev = 0;
    logits[t * V + v] += W_bi[prev * V + v];
}

__global__ void k_bigram_bwd(const int* inp, const float* d_logits, float* d_W_bi,
                              int BL, int V, int pad_tok) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (v >= V || t >= BL) return;
    int prev = (t > 0) ? inp[t - 1] : pad_tok;
    if (prev < 0 || prev >= V) prev = 0;
    atomicAdd(&d_W_bi[prev * V + v], d_logits[t * V + v]);
}

// Numerically stable log-softmax + NLL. The gradient remains p - onehot.
__global__ void k_logsoftmax_nll(const float* logits, float* probs, const int* tgt, float* loss, int N, int V) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float* row = logits + n * V;
    float* prow = probs + n * V;
    float mx = -CUDART_INF_F;
    for (int v = 0; v < V; ++v) mx = fmaxf(mx, row[v]);
    float sum = 0.0f;
    for (int v = 0; v < V; ++v) sum += __expf(row[v] - mx);
    float log_z = mx + logf(fmaxf(sum, 1e-30f));
    int t = tgt[n];
    loss[n] = -(row[t] - log_z);
    for (int v = 0; v < V; ++v) prow[v] = __expf(row[v] - log_z);
}

__global__ void k_d_logits(const float* probs, float* d_logits, const int* tgt, int N, int V) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y;
    if (v >= V || n >= N) return;
    int t = tgt[n];
    d_logits[n * V + v] = probs[n * V + v] - (v == t ? 1.0f : 0.0f);
}

__global__ void k_scatter(const float* gather, float* h, int L, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    int t = blockIdx.z;
    if (d >= D || t >= L) return;
    h[b * (L+1) * D + (t+1) * D + d] += gather[(b * L + t) * D + d];
}

// Channels backward: given dh[t+1], ds[t+1], compute dy[t] (gradient to Q3 output)
__global__ void k_channels_bwd_dy(const float* alpha_t, const float* dh, const float* ds,
                                    float* dy_out, int L, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (d >= D) return;
    float dh_next = 0, ds_next = 0;
    for (int t = L - 1; t >= 0; --t) {
        int bt = b * L + t;
        ds_next += ds[b * (L+1) * D + (t+1) * D + d];
        float dhv = dh[b * (L+1) * D + (t+1) * D + d];
        float a = alpha_t[bt * D + d];
        dy_out[bt * D + d] = ds_next + dh_next * (1.0f - a);
        dh_next = dh_next * a + dhv;
    }
}

// Channels INPUT-TO-STATE backward: given dy[t] (gradient to channels INPUT), compute dh[t+1], ds[t+1]
// dh[t+1] += (1 - alpha[t]) * dy[t]
// ds[t+1] += dy[t]
__global__ void k_dy_to_dhds(const float* dy, const float* alpha_t, float* dh, float* ds, int L, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (d >= D) return;
    for (int t = 0; t < L; ++t) {
        int bt = b * L + t;
        float dy_t = dy[bt * D + d];
        float a = alpha_t[bt * D + d];
        atomicAdd(&dh[b * (L+1) * D + (t+1) * D + d], (1.0f - a) * dy_t);
        atomicAdd(&ds[b * (L+1) * D + (t+1) * D + d], dy_t);
    }
}

// dalpha = dh[t+1] * (h[t] - y[t])
// h_prev layout: BATCH*(L+1)*D, indexed as b*(L+1)*D + t*D + d
// y_curr  layout: BATCH*L*D,    indexed as b*L*D + t*D + d (= bt*D + d)
__global__ void k_dalpha(const float* dh, const float* h_prev, const float* y_curr, float* dalpha,
                          int L, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (d >= D) return;
    for (int t = 0; t < L; ++t) {
        int bt = b * L + t;
        dalpha[bt * D + d] = dh[b * (L+1) * D + (t+1) * D + d]
                            * (h_prev[b * (L+1) * D + t * D + d] - y_curr[bt * D + d]);
    }
}

// Q3 backward: dy[t,d] -> dx[t,d] + dw0/1/2
// y[t] = w0*x[t-2] + w1*x[t-1] + w2*x[t]
// dx[t] = w2*dy[t] + w1*dy[t+1] + w0*dy[t+2]
__global__ void k_q3_bwd(const float* xs, const float* dy, const float* w0, const float* w1, const float* w2,
                          float* dx, float* dw0, float* dw1, float* dw2,
                          int L, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (d >= D) return;
    float w0v = w0[d], w1v = w1[d], w2v = w2[d];
    float dw0v = 0, dw1v = 0, dw2v = 0;
    for (int t = 0; t < L; ++t) {
        int bt = b * L + t;
        float dy_t = dy[bt * D + d];
        float dxv = w2v * dy_t;
        if (t + 1 < L) dxv += w1v * dy[(bt+1) * D + d];
        if (t + 2 < L) dxv += w0v * dy[(bt+2) * D + d];
        dx[bt * D + d] = dxv;
        dw2v += xs[bt * D + d] * dy_t;
        if (t >= 1) dw1v += xs[(bt-1) * D + d] * dy_t;
        if (t >= 2) dw0v += xs[(bt-2) * D + d] * dy_t;
    }
    dw0[d] += dw0v;
    dw1[d] += dw1v;
    dw2[d] += dw2v;
}

__global__ void k_emb_grad(float* demb, const int* inp, const float* dy, int N, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y;
    if (d >= D || n >= N) return;
    atomicAdd(&demb[inp[n] * D + d], dy[n * D + d]);
}

__global__ void k_adam(float* p, float* m, float* v, const float* g,
                       float lr, float bc1, float bc2, float b1, float b2, float eps, int N, float pmin, float pmax) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float gi = isfinite(g[i]) ? g[i] : 0.0f;
    m[i] = b1 * m[i] + (1.0f - b1) * gi;
    v[i] = b2 * v[i] + (1.0f - b2) * gi * gi;
    float denom = sqrtf(fmaxf(v[i] / fmaxf(bc2, 1e-12f), 0.0f)) + eps;
    float step = lr * (m[i] / fmaxf(bc1, 1e-12f)) / denom;
    float pi = p[i] - step;
    if (!isfinite(pi)) pi = 0.0f;
    p[i] = fminf(fmaxf(pi, pmin), pmax);
}

__global__ void k_clip(float* g, float max_norm, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (g[i] > max_norm) g[i] = max_norm;
    if (g[i] < -max_norm) g[i] = -max_norm;
}

__global__ void k_sum_rows(const float* mat, float* out, int N, int V) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;
    float s = 0;
    for (int n = 0; n < N; ++n) s += mat[n * V + v];
    out[v] += s;
}

__global__ void k_dropout_mask(char* mask, int N, float p_drop, unsigned seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    unsigned int h = seed ^ (i * 0x9E3779B9u);
    h = (h ^ (h >> 16)) * 0x85ebca6bu;
    h = (h ^ (h >> 13)) * 0xc2b2ae35u;
    h = h ^ (h >> 16);
    float r = (h & 0xFFFFFF) / float(0xFFFFFF);
    mask[i] = (r > p_drop) ? 1 : 0;
}

__global__ void k_channels_inference(const float* y, const float* alpha_t, float* h, float* s, int L, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float hc = 0, sc = 0;
    h[d] = 0; s[d] = 0;
    for (int t = 0; t < L; ++t) {
        float yt = y[t * D + d];
        float a = alpha_t[t * D + d];
        hc = a * hc + (1.0f - a) * yt;
        sc += yt;
        h[(t+1) * D + d] = hc;
        s[(t+1) * D + d] = sc;
    }
}

int main() {
    g_log = fopen("D:\\TaoVm\\yaoyao_gpu_train.log", "w");
    time_t now_t = time(nullptr);
    char buf[512];
    snprintf(buf, sizeof(buf), "=== 夭夭 GPU v0.6 FIXED (full-BPTT, T=2.0, alpha-LR=0.0005) log started %s", ctime(&now_t)); L(buf);

    Vocab vocab;
    vocab.load("D:\\TaoVm\\vocab.txt");
    int V = vocab.size();
    snprintf(buf, sizeof(buf), "V=%d\n", V); L(buf);

    cublasHandle_t handle;
    BK(cublasCreate(&handle));
    float one = 1.0f, zero = 0.0f;

    int D = 256, SEQ = 64, BATCH = 32, N_WIN = 30000, EPOCHS = 12, N_LAYERS = 6;
    float LR_MAIN = 0.005f, LR_ALPHA = 0.0005f, LR_MIN = 0.0001f, WARMUP_EPOCHS = 1;
    float T_ALPHA = 2.0f;  // temperature for sigmoid
    float dropout = 0.1f, max_grad_norm = 1.0f;
    snprintf(buf, sizeof(buf), "D=%d SEQ=%d BATCH=%d N_WIN=%d EPOCHS=%d LAYERS=%d LR_main=%.4f LR_alpha=%.4f T=%.1f dropout=%.2f [v0.6 full-BPTT]\n",
             D, SEQ, BATCH, N_WIN, EPOCHS, N_LAYERS, LR_MAIN, LR_ALPHA, T_ALPHA, dropout); L(buf);

    L("Loading text...\n");
    std::ifstream ft("D:\\TaoVm\\tinystories_train.txt");
    std::stringstream ss; ss << ft.rdbuf();
    std::string text = ss.str();
    std::vector<int> tokens;
    vocab.encode_utf8(text, tokens);
    snprintf(buf, sizeof(buf), "tokens=%zu\n", tokens.size()); L(buf);

    std::vector<int> all_in(N_WIN * SEQ), all_tg(N_WIN * SEQ);
    for (int i = 0; i < N_WIN; ++i) {
        int start = (i * SEQ) % (tokens.size() - SEQ - 1);
        for (int j = 0; j < SEQ; ++j) {
            all_in[i*SEQ+j] = tokens[start+j];
            all_tg[i*SEQ+j] = tokens[start+j+1];
        }
    }

    int N_emb = V * D, N_W = V * D, N_b = V;
    int N_alpha_w = N_LAYERS * D * D, N_alpha_b = N_LAYERS * D;
    float *d_emb, *d_emb_m, *d_emb_v, *d_emb_g;
    float *d_Wh, *d_Wh_m, *d_Wh_v, *d_Wh_g;
    float *d_Ws, *d_Ws_m, *d_Ws_v, *d_Ws_g;
    float *d_bias, *d_bias_m, *d_bias_v, *d_bias_g;
    float *d_W_bi, *d_W_bi_m, *d_W_bi_v, *d_W_bi_g;
    float *d_alpha_W, *d_alpha_W_m, *d_alpha_W_v, *d_alpha_W_g;
    float *d_alpha_b, *d_alpha_b_m, *d_alpha_b_v, *d_alpha_b_g;
    float *d_q3w0, *d_q3w1, *d_q3w2;
    float *d_q3w0_m, *d_q3w1_m, *d_q3w2_m;
    float *d_q3w0_v, *d_q3w1_v, *d_q3w2_v;
    float *d_q3w0_g, *d_q3w1_g, *d_q3w2_g;
    CK(cudaMalloc(&d_emb, N_emb*4)); CK(cudaMalloc(&d_emb_m, N_emb*4)); CK(cudaMalloc(&d_emb_v, N_emb*4)); CK(cudaMalloc(&d_emb_g, N_emb*4));
    CK(cudaMalloc(&d_Wh, N_W*4)); CK(cudaMalloc(&d_Wh_m, N_W*4)); CK(cudaMalloc(&d_Wh_v, N_W*4)); CK(cudaMalloc(&d_Wh_g, N_W*4));
    CK(cudaMalloc(&d_Ws, N_W*4)); CK(cudaMalloc(&d_Ws_m, N_W*4)); CK(cudaMalloc(&d_Ws_v, N_W*4)); CK(cudaMalloc(&d_Ws_g, N_W*4));
    CK(cudaMalloc(&d_bias, N_b*4)); CK(cudaMalloc(&d_bias_m, N_b*4)); CK(cudaMalloc(&d_bias_v, N_b*4)); CK(cudaMalloc(&d_bias_g, N_b*4));
    CK(cudaMalloc(&d_W_bi, N_b*V*4)); CK(cudaMalloc(&d_W_bi_m, N_b*V*4)); CK(cudaMalloc(&d_W_bi_v, N_b*V*4)); CK(cudaMalloc(&d_W_bi_g, N_b*V*4));
    CK(cudaMalloc(&d_alpha_W, N_alpha_w*4)); CK(cudaMalloc(&d_alpha_W_m, N_alpha_w*4)); CK(cudaMalloc(&d_alpha_W_v, N_alpha_w*4)); CK(cudaMalloc(&d_alpha_W_g, N_alpha_w*4));
    CK(cudaMalloc(&d_alpha_b, N_alpha_b*4)); CK(cudaMalloc(&d_alpha_b_m, N_alpha_b*4)); CK(cudaMalloc(&d_alpha_b_v, N_alpha_b*4)); CK(cudaMalloc(&d_alpha_b_g, N_alpha_b*4));
    CK(cudaMalloc(&d_q3w0, N_LAYERS*D*4)); CK(cudaMalloc(&d_q3w1, N_LAYERS*D*4)); CK(cudaMalloc(&d_q3w2, N_LAYERS*D*4));
    CK(cudaMalloc(&d_q3w0_m, N_LAYERS*D*4)); CK(cudaMalloc(&d_q3w1_m, N_LAYERS*D*4)); CK(cudaMalloc(&d_q3w2_m, N_LAYERS*D*4));
    CK(cudaMalloc(&d_q3w0_v, N_LAYERS*D*4)); CK(cudaMalloc(&d_q3w1_v, N_LAYERS*D*4)); CK(cudaMalloc(&d_q3w2_v, N_LAYERS*D*4));
    CK(cudaMalloc(&d_q3w0_g, N_LAYERS*D*4)); CK(cudaMalloc(&d_q3w1_g, N_LAYERS*D*4)); CK(cudaMalloc(&d_q3w2_g, N_LAYERS*D*4));
    CK(cudaMemset(d_emb_m, 0, N_emb*4)); CK(cudaMemset(d_emb_v, 0, N_emb*4));
    CK(cudaMemset(d_Wh_m, 0, N_W*4)); CK(cudaMemset(d_Wh_v, 0, N_W*4));
    CK(cudaMemset(d_Ws_m, 0, N_W*4)); CK(cudaMemset(d_Ws_v, 0, N_W*4));
    CK(cudaMemset(d_bias_m, 0, N_b*4)); CK(cudaMemset(d_bias_v, 0, N_b*4));
    CK(cudaMemset(d_W_bi_m, 0, N_b*V*4)); CK(cudaMemset(d_W_bi_v, 0, N_b*V*4));
    CK(cudaMemset(d_alpha_W_m, 0, N_alpha_w*4)); CK(cudaMemset(d_alpha_W_v, 0, N_alpha_w*4));
    CK(cudaMemset(d_alpha_b_m, 0, N_alpha_b*4)); CK(cudaMemset(d_alpha_b_v, 0, N_alpha_b*4));
    CK(cudaMemset(d_q3w0_m, 0, N_LAYERS*D*4)); CK(cudaMemset(d_q3w1_m, 0, N_LAYERS*D*4)); CK(cudaMemset(d_q3w2_m, 0, N_LAYERS*D*4));
    CK(cudaMemset(d_q3w0_v, 0, N_LAYERS*D*4)); CK(cudaMemset(d_q3w1_v, 0, N_LAYERS*D*4)); CK(cudaMemset(d_q3w2_v, 0, N_LAYERS*D*4));

    std::vector<float> h_w(D), h_alpha_w(D * D), h_alpha_b(D);
    std::mt19937 rng(42);
    std::normal_distribution<float> nd_emb(0, 0.5f), ndw(0, 0.1f), nda_w(0, 0.02f);
    std::vector<float> h_emb(N_emb);
    for (auto& x : h_emb) x = nd_emb(rng);
    CK(cudaMemcpy(d_emb, h_emb.data(), N_emb*4, cudaMemcpyHostToDevice));
    std::vector<float> h_W(N_W);
    for (auto& x : h_W) x = ndw(rng);
    CK(cudaMemcpy(d_Wh, h_W.data(), N_W*4, cudaMemcpyHostToDevice));
    for (auto& x : h_W) x = ndw(rng);
    CK(cudaMemcpy(d_Ws, h_W.data(), N_W*4, cudaMemcpyHostToDevice));
    std::vector<float> h_bias(N_b, 0.0f);
    CK(cudaMemcpy(d_bias, h_bias.data(), N_b*4, cudaMemcpyHostToDevice));
    std::vector<float> h_W_bi(N_b * V, 0.0f);
    CK(cudaMemcpy(d_W_bi, h_W_bi.data(), N_b*V*4, cudaMemcpyHostToDevice));
    float scale_h = 1.0f / sqrtf((float)D);
    float scale_s = 1.0f / sqrtf((float)D);
    std::vector<float> h_q3w0(N_LAYERS * D), h_q3w1(N_LAYERS * D), h_q3w2(N_LAYERS * D);
    for (int l = 0; l < N_LAYERS; ++l) {
        for (auto& x : h_w) x = ndw(rng) * 0.3f;
        for (int d = 0; d < D; ++d) {
            h_q3w0[l*D + d] = h_w[d];
            h_q3w1[l*D + d] = h_w[d];
            h_q3w2[l*D + d] = h_w[d];
        }
    }
    CK(cudaMemcpy(d_q3w0, h_q3w0.data(), N_LAYERS*D*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_q3w1, h_q3w1.data(), N_LAYERS*D*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_q3w2, h_q3w2.data(), N_LAYERS*D*4, cudaMemcpyHostToDevice));
    // Alpha: b_alpha such that sigmoid(b_alpha / T) ~ 0.3 → b_alpha = T * logit(0.3) ≈ 2 * (-0.847) = -1.69
    for (int l = 0; l < N_LAYERS; ++l) {
        std::mt19937 rgo(123 + l * 7);
        for (int rr = 0; rr < D; ++rr) {
            std::vector<float> row(D); float n=0; for(int k=0;k<D;k++){row[k]=((rgo()&1)?1.0f:-1.0f);n+=row[k]*row[k];} n=sqrtf(n);
            for(int k=0;k<D;k++) row[k]/=n; for(int k=0;k<D;k++) h_alpha_w[rr*D+k]=row[k];
        }
        CK(cudaMemcpy(d_alpha_W + l*D*D, h_alpha_w.data(), D*D*4, cudaMemcpyHostToDevice));
        for (auto& x : h_alpha_b) x = -1.7f + nda_w(rng) * 0.1f;  // ≈ sigmoid(-1.7/2)=sigmoid(-0.85)≈0.3
        CK(cudaMemcpy(d_alpha_b + l*D, h_alpha_b.data(), D*4, cudaMemcpyHostToDevice));
    }

    int BL = BATCH * SEQ;
    int *d_inp, *d_tgt;
    float *d_x, *d_y, *d_h, *d_s, *d_hg, *d_sg, *d_logits, *d_probs;
    float *d_dh, *d_ds, *d_dx, *d_dy, *d_alpha_t, *d_dalpha;
    float *d_dx_extra;
    float *d_dhg, *d_dsg;
    float *d_loss;
    char *d_drop_mask;
    // Per-layer storage
    std::vector<float*> d_xs(N_LAYERS), d_ys(N_LAYERS), d_hs(N_LAYERS), d_ss(N_LAYERS), d_alpha_ts(N_LAYERS);
    CK(cudaMalloc(&d_inp, BL*4)); CK(cudaMalloc(&d_tgt, BL*4));
    CK(cudaMalloc(&d_x, BL*D*4)); CK(cudaMalloc(&d_y, BL*D*4));
    CK(cudaMalloc(&d_h, BATCH*(SEQ+1)*D*4)); CK(cudaMalloc(&d_s, BATCH*(SEQ+1)*D*4));
    CK(cudaMalloc(&d_hg, BL*D*4)); CK(cudaMalloc(&d_sg, BL*D*4));
    CK(cudaMalloc(&d_logits, BL*V*4)); CK(cudaMalloc(&d_probs, BL*V*4));
    CK(cudaMalloc(&d_dh, BATCH*(SEQ+1)*D*4)); CK(cudaMalloc(&d_ds, BATCH*(SEQ+1)*D*4));
    CK(cudaMalloc(&d_dx, BL*D*4)); CK(cudaMalloc(&d_dy, BL*D*4));
    CK(cudaMalloc(&d_dx_extra, BL*D*4));
    CK(cudaMalloc(&d_dhg, BL*D*4)); CK(cudaMalloc(&d_dsg, BL*D*4));
    CK(cudaMalloc(&d_alpha_t, BL*D*4)); CK(cudaMalloc(&d_dalpha, BL*D*4));
    CK(cudaMalloc(&d_loss, BL*4));
    CK(cudaMalloc(&d_drop_mask, D * sizeof(char)));

    for (int l = 0; l < N_LAYERS; ++l) {
        CK(cudaMalloc(&d_xs[l], BL*D*4));
        CK(cudaMalloc(&d_ys[l], BL*D*4));
        CK(cudaMalloc(&d_hs[l], BATCH*(SEQ+1)*D*4));
        CK(cudaMalloc(&d_ss[l], BATCH*(SEQ+1)*D*4));
        CK(cudaMalloc(&d_alpha_ts[l], BL*D*4));
    }

    L("Init done. Training...\n");
    auto t0 = std::chrono::steady_clock::now();
    int adam_t = 0;
    float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float inv_dropout = 1.0f / (1.0f - dropout);

    L("\n=== INITIAL SAMPLES (untrained) ===\n");
    for (const std::string& prompt : std::vector<std::string>{"Once upon a time"}) {
        snprintf(buf, sizeof(buf), "  prompt [\"%s\"] => (no inference yet, just prompt)\n", prompt.c_str());
        L(buf);
    }

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float lr;
        if (epoch < WARMUP_EPOCHS) {
            lr = LR_MIN + (LR_MAIN - LR_MIN) * (float)(epoch + 1) / WARMUP_EPOCHS;
        } else {
            float t = (float)(epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS);
            lr = LR_MIN + 0.5f * (LR_MAIN - LR_MIN) * (1.0f + cosf(3.14159f * t));
        }
        std::vector<int> idx(N_WIN);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), std::mt19937(epoch + 1));
        float total = 0;
        int n_batches = 0;
        auto ep0 = std::chrono::steady_clock::now();
        snprintf(buf, sizeof(buf), "\nEpoch %d/%d lr=%.5f (alpha_lr=%.5f)\n", epoch+1, EPOCHS, lr, LR_ALPHA); L(buf);
        int total_batches = (N_WIN + BATCH - 1) / BATCH;

        std::vector<float> h_loss(BL);
        for (int i = 0; i < N_WIN; i += BATCH) {
            int actual_bs = std::min(BATCH, N_WIN - i);
            std::vector<int> b_in(BL), b_tg(BL);
            for (int j = 0; j < actual_bs; ++j) {
                int w = idx[i + j];
                for (int k = 0; k < SEQ; ++k) {
                    b_in[j*SEQ+k] = all_in[w*SEQ+k];
                    b_tg[j*SEQ+k] = all_tg[w*SEQ+k];
                }
            }
            CK(cudaMemcpy(d_inp, b_in.data(), BL*4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_tgt, b_tg.data(), BL*4, cudaMemcpyHostToDevice));

            k_zero<<<(N_emb+255)/256, 256>>>(d_emb_g, N_emb);
            k_zero<<<(N_W+255)/256, 256>>>(d_Wh_g, N_W);
            k_zero<<<(N_W+255)/256, 256>>>(d_Ws_g, N_W);
            k_zero<<<(N_b+255)/256, 256>>>(d_bias_g, N_b);
            k_zero<<<(N_b*V+255)/256, 256>>>(d_W_bi_g, N_b*V);
            k_zero<<<(N_alpha_w+255)/256, 256>>>(d_alpha_W_g, N_alpha_w);
            k_zero<<<(N_alpha_b+255)/256, 256>>>(d_alpha_b_g, N_alpha_b);
            k_zero<<<(N_LAYERS*D+255)/256, 256>>>(d_q3w0_g, N_LAYERS*D);
            k_zero<<<(N_LAYERS*D+255)/256, 256>>>(d_q3w1_g, N_LAYERS*D);
            k_zero<<<(N_LAYERS*D+255)/256, 256>>>(d_q3w2_g, N_LAYERS*D);

            // Forward: save per-layer state
            k_embed<<<dim3((D+255)/256, BL), 256>>>(d_emb, d_inp, d_x, D);
            for (int l = 0; l < N_LAYERS; ++l) {
                CK(cudaMemcpy(d_xs[l], d_x, BL*D*4, cudaMemcpyDeviceToDevice));
                k_dropout_mask<<<(D+255)/256, 256>>>(d_drop_mask, D, dropout, (unsigned)(epoch*1000+n_batches*10+l+1));
                k_q3_fwd<<<dim3((D+255)/256, BATCH, SEQ), 256>>>(d_x, d_q3w0 + l*D, d_q3w1 + l*D, d_q3w2 + l*D,
                                                                  d_drop_mask, d_y, inv_dropout, SEQ, D);
                CK(cudaMemcpy(d_ys[l], d_y, BL*D*4, cudaMemcpyDeviceToDevice));
                k_alpha_fwd<<<dim3((D+255)/256, BL), 256>>>(d_x, d_alpha_W + l*D*D, d_alpha_b + l*D,
                                                            d_alpha_ts[l], BL, D, T_ALPHA);
                k_channels_fwd<<<dim3((D+255)/256, BATCH), 256>>>(d_y, d_alpha_ts[l], d_h, d_s, SEQ, D);
                CK(cudaMemcpy(d_hs[l], d_h, BATCH*(SEQ+1)*D*4, cudaMemcpyDeviceToDevice));
                CK(cudaMemcpy(d_ss[l], d_s, BATCH*(SEQ+1)*D*4, cudaMemcpyDeviceToDevice));
                CK(cudaMemcpy(d_x, d_y, BL*D*4, cudaMemcpyDeviceToDevice));
            }
            k_gather<<<dim3((D+255)/256, BATCH, SEQ), 256>>>(d_h, d_s, d_hg, d_sg, SEQ, D);
            k_rmsnorm<<<BATCH*SEQ, 256>>>(d_hg, d_hg, BL, D, 1e-5f);
            k_rmsnorm<<<BATCH*SEQ, 256>>>(d_sg, d_sg, BL, D, 1e-5f);
            BK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, V, BL, D, &one, d_Wh, D, d_hg, D, &zero, d_logits, V));
            BK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, V, BL, D, &one, d_Ws, D, d_sg, D, &one, d_logits, V));
            k_bigram_fwd<<<dim3((V+255)/256, BL), 256>>>(d_inp, d_W_bi, d_logits, BL, V, vocab.pad);
            // bias disabled: not added to logits
            k_logsoftmax_nll<<<(BL+255)/256, 256>>>(d_logits, d_probs, d_tgt, d_loss, BL, V);

            // Backward: d_logits
            k_d_logits<<<dim3((V+255)/256, BL), 256>>>(d_probs, d_logits, d_tgt, BL, V);
            k_q4_dWh_dWs<<<dim3(V, BL), 256>>>(d_hg, d_sg, d_logits, d_Wh_g, d_Ws_g, BL, V, D);
            k_bigram_bwd<<<dim3((V+255)/256, BL), 256>>>(d_inp, d_logits, d_W_bi_g, BL, V, vocab.pad);
            // bias disabled: no bias gradient
            k_q4_dhg_dsg<<<dim3((D+255)/256, BL), 256>>>(d_Wh, d_Ws, d_logits, d_dhg, d_dsg, BL, V, D);


            // Full BPTT through all layers
            for (int l = N_LAYERS - 1; l >= 0; --l) {
                // Initialize d_dh, d_ds for this layer
                k_zero<<<(BATCH*(SEQ+1)*D+255)/256, 256>>>(d_dh, BATCH*(SEQ+1)*D);
                k_zero<<<(BATCH*(SEQ+1)*D+255)/256, 256>>>(d_ds, BATCH*(SEQ+1)*D);
                if (l == N_LAYERS - 1) {
                    // Initialize from d_dhg, d_dsg
                    k_scatter<<<dim3((D+255)/256, BATCH, SEQ), 256>>>(d_dhg, d_dh, SEQ, D);
                    k_scatter<<<dim3((D+255)/256, BATCH, SEQ), 256>>>(d_dsg, d_ds, SEQ, D);
                } else {
                    // Initialize from layer l+1's Q3 bwd output (d_dx)
                    // d_dx is gradient to layer l+1's Q3 INPUT = layer l's Q3 OUTPUT = layer l's channels INPUT
                    // So convert via k_dy_to_dhds using layer l's alpha
                    k_dy_to_dhds<<<dim3((D+255)/256, BATCH), 256>>>(d_dx, d_alpha_ts[l], d_dh, d_ds, SEQ, D);
                }
                // DEEPSEEK FIX: clip d_dh/d_ds to cut positive feedback loop
                k_clip<<<(BATCH*(SEQ+1)*D+255)/256, 256>>>(d_dh, max_grad_norm, BATCH*(SEQ+1)*D);
                k_clip<<<(BATCH*(SEQ+1)*D+255)/256, 256>>>(d_ds, max_grad_norm, BATCH*(SEQ+1)*D);
                // Channels bwd -> d_dy
                k_channels_bwd_dy<<<dim3((D+255)/256, BATCH), 256>>>(d_alpha_ts[l], d_dh, d_ds, d_dy, SEQ, D);
                // dalpha from d_dh[l], d_hs[l], d_ys[l]
                k_dalpha<<<dim3((D+255)/256, BATCH), 256>>>(d_dh, d_hs[l], d_ys[l], d_dalpha, SEQ, D);
                // alpha backward
                k_alpha_bwd<<<dim3((D+255)/256, BL), 256>>>(d_xs[l], d_alpha_W + l*D*D, d_alpha_ts[l], d_dalpha,
                                                             d_dx_extra, d_alpha_W_g + l*D*D, d_alpha_b_g + l*D,
                                                             BL, D, T_ALPHA);
                // Q3 backward
                k_q3_bwd<<<dim3((D+255)/256, BATCH), 256>>>(d_xs[l], d_dy, d_q3w0 + l*D, d_q3w1 + l*D, d_q3w2 + l*D,
                                                            d_dx, d_q3w0_g + l*D, d_q3w1_g + l*D, d_q3w2_g + l*D,
                                                            SEQ, D);
                // Add dx_extra to d_dx
                cublasSaxpy(handle, BL*D, &one, d_dx_extra, 1, d_dx, 1);
                k_zero<<<(BL*D+255)/256, 256>>>(d_dx_extra, BL*D);
                // DEEPSEEK FIX: clip d_dx to prevent cross-layer amplification
                k_clip<<<(BL*D+255)/256, 256>>>(d_dx, max_grad_norm, BL*D);
                if (l == 0) {
                    // Embed grad
                    k_emb_grad<<<dim3((D+255)/256, BL), 256>>>(d_emb_g, d_inp, d_dx, BL, D);
                }
            }

            // Clip
            k_clip<<<(N_emb+255)/256, 256>>>(d_emb_g, max_grad_norm, N_emb);
            k_clip<<<(N_W+255)/256, 256>>>(d_Wh_g, max_grad_norm, N_W);
            k_clip<<<(N_W+255)/256, 256>>>(d_Ws_g, max_grad_norm, N_W);
            k_clip<<<(N_b+255)/256, 256>>>(d_bias_g, max_grad_norm, N_b);
            k_clip<<<(N_b*V+255)/256, 256>>>(d_W_bi_g, max_grad_norm, N_b*V);
            k_clip<<<(N_alpha_w+255)/256, 256>>>(d_alpha_W_g, max_grad_norm, N_alpha_w);
            k_clip<<<(N_alpha_b+255)/256, 256>>>(d_alpha_b_g, max_grad_norm, N_alpha_b);
            for (int l = 0; l < N_LAYERS; ++l) {
                k_clip<<<(D+255)/256, 256>>>(d_q3w0_g + l*D, max_grad_norm, D);
                k_clip<<<(D+255)/256, 256>>>(d_q3w1_g + l*D, max_grad_norm, D);
                k_clip<<<(D+255)/256, 256>>>(d_q3w2_g + l*D, max_grad_norm, D);
            }

            // Adam with separate LR for alpha
            adam_t++;
            float bc1 = 1.0f - std::pow(b1, (float)adam_t);
            float bc2 = 1.0f - std::pow(b2, (float)adam_t);
            // Main params
            k_adam<<<(N_emb+255)/256, 256>>>(d_emb, d_emb_m, d_emb_v, d_emb_g, lr, bc1, bc2, b1, b2, eps, N_emb, -4.0f, 4.0f);
            k_adam<<<(N_W+255)/256, 256>>>(d_Wh, d_Wh_m, d_Wh_v, d_Wh_g, lr, bc1, bc2, b1, b2, eps, N_W, -4.0f, 4.0f);
            k_adam<<<(N_W+255)/256, 256>>>(d_Ws, d_Ws_m, d_Ws_v, d_Ws_g, lr, bc1, bc2, b1, b2, eps, N_W, -4.0f, 4.0f);
            // ternary clipping keeps weights inside quantization range
            {
                float* tmp=nullptr; CK(cudaMalloc(&tmp, V*D*4));
                CK(cudaMemcpy(tmp, d_Wh, V*D*4, cudaMemcpyDeviceToDevice));
                float* tp=nullptr;
            }            k_clip<<<(V*D+255)/256,256>>>(d_Wh, 1.0f, V*D);
            k_clip<<<(V*D+255)/256,256>>>(d_Ws, 1.0f, V*D);
            // bias disabled (kept zero, no update)
            // Alpha with smaller LR
            k_adam<<<(N_alpha_w+255)/256, 256>>>(d_alpha_W, d_alpha_W_m, d_alpha_W_v, d_alpha_W_g, LR_ALPHA, bc1, bc2, b1, b2, eps, N_alpha_w, -4.0f, 4.0f);
            k_adam<<<(N_alpha_b+255)/256, 256>>>(d_alpha_b, d_alpha_b_m, d_alpha_b_v, d_alpha_b_g, LR_ALPHA, bc1, bc2, b1, b2, eps, N_alpha_b, -8.0f, 8.0f);
            for (int l = 0; l < N_LAYERS; ++l) {
                k_adam<<<(D+255)/256, 256>>>(d_q3w0 + l*D, d_q3w0_m + l*D, d_q3w0_v + l*D, d_q3w0_g + l*D, lr, bc1, bc2, b1, b2, eps, D, -2.0f, 2.0f);
                k_adam<<<(D+255)/256, 256>>>(d_q3w1 + l*D, d_q3w1_m + l*D, d_q3w1_v + l*D, d_q3w1_g + l*D, lr, bc1, bc2, b1, b2, eps, D, -2.0f, 2.0f);
                k_adam<<<(D+255)/256, 256>>>(d_q3w2 + l*D, d_q3w2_m + l*D, d_q3w2_v + l*D, d_q3w2_g + l*D, lr, bc1, bc2, b1, b2, eps, D, -2.0f, 2.0f);
            }

            // Debug: alpha stats every 200 batches (read from LAST layer's per-layer buffer)
            if (n_batches % 200 == 0) {
                std::vector<float> h_alpha_check(BL * D);
                CK(cudaMemcpy(h_alpha_check.data(), d_alpha_ts[N_LAYERS-1], BL*D*4, cudaMemcpyDeviceToHost));
                float a_min = 1e9f, a_max = -1e9f, a_sum = 0;
                for (int j = 0; j < BL * D; ++j) {
                    float v = h_alpha_check[j];
                    if (v < a_min) a_min = v;
                    if (v > a_max) a_max = v;
                    a_sum += v;
                    if (!std::isfinite(v)) { a_min = -99; break; }
                }
                snprintf(buf, sizeof(buf), "\n  [debug ep%d batch%d alpha[%d] range: min=%.4f max=%.4f mean=%.4f]\n",
                       epoch+1, n_batches, N_LAYERS-1, a_min, a_max, a_sum/(BL*D));
                L(buf);
            }
            SYNC();

            CK(cudaMemcpy(h_loss.data(), d_loss, BL*4, cudaMemcpyDeviceToHost));
            float bl = 0;
            for (int j = 0; j < BL; ++j) bl += h_loss[j];
            total += bl / BL;
            n_batches++;

            if (n_batches % 100 == 0 || n_batches == 1) {
                auto now = std::chrono::steady_clock::now();
                double el = std::chrono::duration<double>(now - t0).count();
                snprintf(buf, sizeof(buf), "  ep%d batch%d/%d loss=%.4f t=%.1fs\r",
                       epoch+1, n_batches, total_batches, total/n_batches, el);
                L(buf);
            }
        }
        auto ep1 = std::chrono::steady_clock::now();
        double ep_sec = std::chrono::duration<double>(ep1 - ep0).count();
        float avg_loss = total / n_batches;
        snprintf(buf, sizeof(buf), "\nEpoch %d avg_loss=%.4f time=%.1fs\n", epoch+1, avg_loss, ep_sec);
        L(buf);

        // Generation samples
        L("  --- Generation (epoch ");
        snprintf(buf, sizeof(buf), "%d", epoch+1); L(buf);
        L(") ---\n");
        for (const std::string& prompt : std::vector<std::string>{"Once upon a time", "Lily and Tom", "The cat sat"}) {
            std::vector<int> gen_ids;
            vocab.encode_utf8(prompt, gen_ids);
            for (int step = 0; step < 40 && gen_ids.size() < 100; ++step) {
                int L_gen = (int)gen_ids.size();
                float *h_xg, *h_yg, *h_hg, *h_sg, *h_alphag;
                CK(cudaMalloc(&h_xg, L_gen * D * 4));
                CK(cudaMalloc(&h_yg, L_gen * D * 4));
                CK(cudaMalloc(&h_hg, (L_gen + 1) * D * 4));
                CK(cudaMalloc(&h_sg, (L_gen + 1) * D * 4));
                CK(cudaMalloc(&h_alphag, L_gen * D * 4));
                int *h_inpg;
                CK(cudaMalloc(&h_inpg, L_gen * 4));
                CK(cudaMemcpy(h_inpg, gen_ids.data(), L_gen * 4, cudaMemcpyHostToDevice));
                k_embed<<<dim3((D+255)/256, L_gen), 256>>>(d_emb, h_inpg, h_xg, D);
                for (int l = 0; l < N_LAYERS; ++l) {
                    char* fake_mask = nullptr;
                    k_q3_fwd<<<dim3((D+255)/256, 1, L_gen), 256>>>(h_xg, d_q3w0 + l*D, d_q3w1 + l*D, d_q3w2 + l*D,
                                                                       fake_mask, h_yg, 1.0f, L_gen, D);
                    k_alpha_fwd<<<dim3((D+255)/256, L_gen), 256>>>(h_xg, d_alpha_W + l*D*D, d_alpha_b + l*D,
                                                                        h_alphag, L_gen, D, T_ALPHA);
                    k_channels_inference<<<dim3((D+255)/256, 1), 256>>>(h_yg, h_alphag, h_hg, h_sg, L_gen, D);
                    CK(cudaMemcpy(h_xg, h_yg, L_gen*D*4, cudaMemcpyDeviceToDevice));
                }
                std::vector<float> last_h(D), last_s(D);
                CK(cudaMemcpy(last_h.data(), h_hg + L_gen * D, D*4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(last_s.data(), h_sg + L_gen * D, D*4, cudaMemcpyDeviceToHost));
                std::vector<float> logits(V);
                std::vector<float> h_W_hv(V * D), h_W_sv(V * D), h_biasv(V), h_W_bi_v(V);
                CK(cudaMemcpy(h_W_hv.data(), d_Wh, V*D*4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(h_W_sv.data(), d_Ws, V*D*4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(h_biasv.data(), d_bias, V*4, cudaMemcpyDeviceToHost));
                {
                    int prev = gen_ids.size() > 1 ? gen_ids[gen_ids.size()-1] : (gen_ids.size() == 1 ? gen_ids[0] : vocab.pad);
                    if (prev < 0 || prev >= V) prev = 0;
                    CK(cudaMemcpy(h_W_bi_v.data(), d_W_bi + prev * V, V*4, cudaMemcpyDeviceToHost));
                }
                for (int v = 0; v < V; ++v) {
                    float lv = h_biasv[v] + h_W_bi_v[v];
                    for (int d = 0; d < D; ++d) {
                        lv += h_W_hv[v*D+d] * last_h[d];
                        lv += h_W_sv[v*D+d] * last_s[d];
                    }
                    logits[v] = lv;
                }
                int best = 0; float best_l = logits[0];
                for (int v = 1; v < V; ++v) if (logits[v] > best_l) { best_l = logits[v]; best = v; }
                // Debug: if logits are extreme or non-finite, log warning
                if (step < 3 && epoch == 0) {
                    float log_min = logits[0], log_max = logits[0];
                    int nan_count = 0;
                    for (int v = 0; v < V; ++v) {
                        if (!std::isfinite(logits[v])) nan_count++;
                        if (logits[v] < log_min) log_min = logits[v];
                        if (logits[v] > log_max) log_max = logits[v];
                    }
                    snprintf(buf, sizeof(buf), "       [gen step %d best=%d logit=%.2f range=[%.2f,%.2f] nan=%d]\n",
                           step, best, best_l, log_min, log_max, nan_count);
                    L(buf);
                }
                gen_ids.push_back(best);
                cudaFree(h_xg); cudaFree(h_yg); cudaFree(h_hg); cudaFree(h_sg);
                cudaFree(h_alphag); cudaFree(h_inpg);
            }
            std::string out;
            for (int id : gen_ids) out += vocab.dec(id);
            snprintf(buf, sizeof(buf), "    greedy [\"%s\"] => \"%s\"\n", prompt.c_str(), out.c_str());
            L(buf);
        }
    }
    L("\n=== done ===\n");
    if (g_log) { fclose(g_log); g_log = nullptr; }
    return 0;
}
