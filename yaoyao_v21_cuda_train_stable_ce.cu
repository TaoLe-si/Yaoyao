#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <unordered_map>
#include <string>
#include <math.h>
#include <stdint.h>
#include <fstream>
#include <vector>
#include <random>

typedef int8_t trit; typedef uint32_t hash_t;
const int D = 256, V = 1024, V_unit = 1024;
const int B = 128, K = 16, NL = 2;
const int HASH_FEATURES = 64;
const int HIDDEN = D + HASH_FEATURES;  // 320 when D=256
const int BATCH = 16, SEQ = 64, BL = BATCH * SEQ;
const int PAD = 0;  // previous-token padding for t==0

// ---------- CUDA / cuBLAS error checks (train aborts on any failure) ----------
#define cudaCheck(call) do { \
    cudaError_t e__ = (call); \
    if (e__ != cudaSuccess) { \
        fprintf(stderr, "[CUDA FAIL] %s:%d %s err=%s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e__)); \
        std::exit(2); /* fail immediately; never reach save */ \
    } \
} while (0)

#define cublasCheck(call) do { \
    cublasStatus_t s__ = (call); \
    if (s__ != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "[CUBLAS FAIL] %s:%d %s status=%d\n", __FILE__, __LINE__, #call, (int)s__); \
        std::exit(2); /* fail immediately; never reach save */ \
    } \
} while (0)

#define kernelCheck() do { \
    cudaError_t e__ = cudaGetLastError(); \
    if (e__ != cudaSuccess) { \
        fprintf(stderr, "[KERNEL FAIL] %s:%d err=%s\n", __FILE__, __LINE__, cudaGetErrorString(e__)); \
        std::exit(2); /* fail immediately; never reach save */ \
    } \
} while (0)

volatile int g_train_abort = 0;

// [V21-Phase6-GPU] 因果单链 hash + 64 维映射 (TBC 映射, 待父代理确认)
// 单链递推: h_t = h_{t-1} * 33 + token_t,  seed h_0 = 5381 (即 t=0 时 prev=5381)
//   注: 不带 +7 legacy 偏移, 与上轮父代理指定公式字面一致
//   hash_t = uint32_t, 算术 mod 2^32 (与 INV33_MOD_2_32 一致)
// 映射 (TBC): f_i = ((h_t >> (4*(i%8))) & 0xFu) / 15.0f - 0.5f,  i = 0..63
//   - 采用 8 个 4-bit nibble 循环填 64 维;  上轮父代理的临时方案, 此处仅作为待核对配置占位
//   - 用户最终映射若改, 只改 nibble_cycle_feature 即可, 计算图其余部分不动
//   - 映射与维度均待父代理确认 (HASH_FEATURES=64 也待确认, 当前为保持向后兼容暂不调整)
// 输入: inp[BL] int tokens; 输出: hf_out[BL, HASH_FEATURES];  同时把 h_t 末态写到 h_last[BATCH] (debug)
__device__ inline float nibble_cycle_feature(hash_t h, int i) {
    int nibble_idx = i & 7;          // i % 8
    int shift = 4 * nibble_idx;
    hash_t nib = (h >> shift) & 0xFu;
    return ((float)nib) / 15.0f - 0.5f;
}

__global__ void hash_extract_kernel(const int* inp, hash_t* h_last, float* hf_out, int Bd, int Sd) {
    int b = blockIdx.x; if (b >= Bd) return;
    hash_t h = 5381u;
    for (int t = 0; t < Sd; ++t) {
        int id = inp[b*Sd + t];
        // [V21-Phase6-GPU] 因果: 每一步都基于上一步的 h; t=0 也直接累加 token (与公式一致)
        h = h * 33u + (hash_t)id;
        int base = (b*Sd + t) * HASH_FEATURES;
        for (int i = 0; i < HASH_FEATURES; ++i) {
            hf_out[base + i] = nibble_cycle_feature(h, i);
        }
    }
    if (h_last) h_last[b] = h;
}

__global__ void trit_acc_kernel(const trit* q1, const int* inp, float* trit_out, int Bd, int Sd, int Dd) {
    int b = blockIdx.x; if (b >= Bd) return;
    for (int d = 0; d < Dd; ++d) trit_out[b*Sd*Dd + d] = 0;
    for (int t = 0; t < Sd; ++t) {
        int id = inp[b*Sd + t];
        uint32_t xh = (uint32_t)id * 2654435761u;
        xh = (xh >> 16) ^ xh;
        int bucket = (int)(xh % B);
        for (int d = 0; d < Dd; ++d) {
            float prev = (t > 0) ? trit_out[b*Sd*Dd + (t-1)*Dd + d] : 0.0f;
            float emb = (float)q1[(bucket*K + 0)*Dd + d];
            int sum = (int)prev + (int)emb;
            int r = sum % 3; if (r > 1) r -= 3; if (r < -1) r += 3;
            trit_out[b*Sd*Dd + t*Dd + d] = (float)r;
        }
    }
}

__global__ void concat_kernel(float* state, const float* trit_f, const float* hash_f, int BL_d) {
    int bt = blockIdx.x; if (bt >= BL_d) return;
    for (int d = 0; d < D; ++d) state[bt*HIDDEN + d] = trit_f[bt*D + d];
    for (int f = 0; f < HASH_FEATURES; ++f) state[bt*HIDDEN + D + f] = hash_f[bt*HASH_FEATURES + f];
}

__global__ void silu_mul_kernel(float* gate, const float* up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i], u = up[i];
    float sig = 1.0f / (1.0f + expf(-g));
    gate[i] = g * sig * u;
}



__global__ void add_wbi_kernel(float* logits, const float* Wbi, const int* inp, int PAD, int BL_d, int Vd) {
    int bt = blockIdx.x; if (bt >= BL_d) return;
    int t = bt % SEQ;
    int prev = (t > 0) ? inp[bt - 1] : PAD;
    for (int v = 0; v < Vd; ++v) logits[bt * Vd + v] += Wbi[prev * Vd + v];
}

__global__ void softmax_kernel(float* logits, float* probs, int BL_d, int Vd) {
    int bt = blockIdx.x; if (bt >= BL_d) return;
    int base = bt * Vd;
    float mx = -1e30f;
    for (int v = 0; v < Vd; ++v) { float x = logits[base + v]; if (x > mx) mx = x; }
    float sum = 0;
    for (int v = 0; v < Vd; ++v) { float p = expf(logits[base + v] - mx); probs[base + v] = p; sum += p; }
    for (int v = 0; v < Vd; ++v) probs[base + v] /= sum;
}

// d_logits[v] = probs[v] - (v == target ? 1 : 0)
__global__ void d_logits_kernel(float* d_logits, const float* probs, const int* targets, int BL_d, int Vd) {
    int bt = blockIdx.x; if (bt >= BL_d) return;
    int base = bt * Vd;
    int t = targets[bt];
    for (int v = 0; v < Vd; ++v) {
        d_logits[base + v] = probs[base + v];
    }
    d_logits[base + t] -= 1.0f;
}

// Adam update W: W -= lr * m_hat / (sqrt(v_hat) + eps), with grad clip
// W (V, D) row-major, d_logits (BL, V) row-major, trit_features (BL, D) row-major
// dW[v, d] = sum_n d_logits[n, v] * trit_features[n, d]




// [V21-Phase6-EXP] Un-freeze W_sgl_out: train the output projection
// d_W_sgl_out[v, h] = sum_n d_logits[n, v] * hidden[n, h]
// hidden is the output of silu_mul (stored in d_gate_z after that kernel)
__global__ void adam_Wsglout_kernel(float* W, float* Wm, float* Wv, const float* d_logits, const float* hidden,
                                     float lr, float b1, float b2, float bc1, float bc2, float eps,
                                     int Vd, int Hd, int BL_d, float* raw_grad) {
    int v = blockIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= Hd) return;
    int idx = v * Hd + h;
    float g = 0;
    for (int n = 0; n < BL_d; ++n) g += d_logits[n * Vd + v] * hidden[n * Hd + h];
    g /= (float) BL_d;  // mean gradient (per math)
    raw_grad[idx] = g; // raw mean gradient, before clipping or Adam
}

// ---------- Backward kernels for state -> hidden -> logits ----------
// hidden = silu(g) * u, with g = Wg*state + bg, u = Wu*state + bu
// d_logits (BL, V) -> d_hidden (BL, H) via Wout^T, then -> d_gate_pre, d_up_pre.
__global__ void d_logits_to_d_hidden_kernel(float* d_hidden, const float* d_logits, const float* Wout,
                                             int Hd, int Vd, int BL_d) {
    // Computes d_hidden[n, h] = sum_v d_logits[n, v] * Wout[v, h]; one row per (n,h), sum over v.
    int n = blockIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= Hd || n >= BL_d) return;
    float s = 0.0f;
    for (int v = 0; v < Vd; ++v) s += d_logits[n * Vd + v] * Wout[v * Hd + h];
    d_hidden[n * Hd + h] = s;
}

// silu_back: from d_hidden -> d_gate_pre, d_up_pre.
// g is gate_pre (= gate_z before silu_mul, we re-load d_gate_z which holds pre-silu after the kernel update? no — silu_mul mutates gate_z in place).
// We re-compute silu(g) inside kernel from g and u.
__global__ void silu_back_kernel(float* d_gate_pre, float* d_up_pre,
                                  const float* d_hidden, const float* g_pre, const float* u_pre,
                                  int n_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_total) return;
    float dh = d_hidden[i];
    float g = g_pre[i];
    float u = u_pre[i];
    float sig = 1.0f / (1.0f + expf(-g));
    // silu'(g) = sig(g) * (1 + g*(1 - sig(g)))  numerically stable.
    float dsilg = sig * (1.0f + g * (1.0f - sig));
    d_gate_pre[i] = dh * u * dsilg;
    d_up_pre[i]   = dh * (g * sig);
}

// Mean-gradient Adam for Wg (H x H) row-major, g_pre (BL x H) row-major, state (BL x H) row-major.
// dW[i, j] = (1/BL) * sum_n d_gate_pre[n, i] * state[n, j]
__global__ void adam_W_gateup_kernel(float* W, float* Wm, float* Wv,
                                      const float* g_pre, const float* state,
                                      float lr, float b1, float b2, float bc1, float bc2, float eps,
                                      int Hd, int BL_d, float* raw_grad) {
    int i = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= Hd) return;
    int idx = i * Hd + j;
    float g = 0;
    for (int n = 0; n < BL_d; ++n) g += g_pre[n * Hd + i] * state[n * Hd + j];
    g /= (float) BL_d;
    raw_grad[idx] = g; // raw mean gradient, before clipping or Adam
}

// Mean-gradient Adam for bias (H): db[i] = mean over n of g_pre[n, i]


// d_state += W^T * d_gate_pre + W^T * d_up_pre ; here we fuse:
// d_state[n, j] = sum_i (d_gate_pre[n, i] * Wg[i, j] + d_up_pre[n, i] * Wu[i, j])
// Pass W as a single concatenated buffer [Wg | Wu] stacked vertically? No: each separately.
// Use two kernel launches (or one with two reads). We'll do two launches for clarity.
__global__ void d_state_add_kernel(float* d_state, const float* d_grad, const float* W, int Hd, int BL_d) {
    int n = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= Hd || n >= BL_d) return;
    float s = 0;
    for (int i = 0; i < Hd; ++i) s += d_grad[n * Hd + i] * W[i * Hd + j];
    d_state[n * Hd + j] += s;
}

// d_trit_f = d_state[:, :D];  d_hash_f = d_state[:, D:D+HASH_FEATURES];
// split: copy slices (we never update q1 or hash params; q1/hash are frozen w.r.t. params,
// but for state backward propagation we expose d_trit_f for diagnosis only).
__global__ void split_state_kernel(const float* d_state, float* d_trit_f, float* d_hash_f, int BL_d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = BL_d * HIDDEN;
    if (idx >= total) return;
    int b = idx / HIDDEN;
    int h = idx % HIDDEN;
    if (h < D) d_trit_f[b * D + h] = d_state[b * HIDDEN + h];
    else d_hash_f[b * HASH_FEATURES + (h - D)] = d_state[b * HIDDEN + h];
}

__global__ void adam_Wbi_kernel(float* Wbi, float* Wm, float* Wv, const float* d_logits, const int* inp, const int* targets,
                                  float lr, float b1, float b2, float bc1, float bc2, float eps,
                                  int Vd, int BL_d, int PAD, float* raw_grad) {
    // For each (prev, v): mean of d_logits[n, v] over n where inp[n-1] == prev (or PAD if n%SEQ==0).
    // Mean is computed by dividing the sum by the per-prev count.  We iterate twice: once to
    // count, once to sum — but a more efficient alternative: keep running sum + count and
    // divide at the end.  Either works; we do count-then-sum to keep numerics simple.
    int prev = blockIdx.x;
    int v = blockIdx.y * blockDim.x + threadIdx.x;
    if (v >= Vd) return;
    int idx = prev * Vd + v;
    int count = 0;
    float sum = 0.0f;
    for (int n = 0; n < BL_d; ++n) {
        int t = n % SEQ;
        int p = (t > 0) ? inp[n - 1] : PAD;
        if (p == prev) { sum += d_logits[n * Vd + v]; count++; }
    }
    float g = sum / (float)BL_d; // derivative of GLOBAL mean cross entropy
    raw_grad[idx] = g; // raw mean gradient, before clipping or Adam
}

// All parameter derivatives are materialized before any optimizer writes.
__global__ void apply_adam_kernel(float* w, float* m, float* v, const float* grad,
    int size, float lr, float b1, float b2, float bc1, float bc2, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float g = fminf(1.f, fmaxf(-1.f, grad[i]));
    m[i] = b1*m[i] + (1-b1)*g;
    v[i] = b2*v[i] + (1-b2)*g*g;
    w[i] = fminf(2.f, fmaxf(-2.f, w[i] - lr*(m[i]/bc1)/(sqrtf(v[i]/bc2)+eps)));
}

// Load model from .bin
struct Model {
    float *W, *Wm, *Wv, *Wh, *Whm, *Whv, *Wbi, *Wbim, *Wbiv;
    trit *q1;
    int q1_step, step;
    float *W_sgl_gate, *W_sgl_up, *W_sgl_out;
    float *W_sgl_gate_m, *W_sgl_gate_v;
    float *W_sgl_up_m, *W_sgl_up_v;
    float *W_sgl_out_m, *W_sgl_out_v;
    float *b_sgl_gate, *b_sgl_up;
    float *b_sgl_gate_m, *b_sgl_gate_v;
    float *b_sgl_up_m, *b_sgl_up_v;
    // Note: W/Wh legacy are loaded for file-format compatibility but NEVER updated
    // (they do not participate in logits = Wout*hidden + Wbi[prev,:]).
};

bool load_model(const char* path, Model& m) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    int magic; f.read((char*)&magic, 4);
    int version; f.read((char*)&version, 4);
    int v; f.read((char*)&v, 4);
    if (magic != 0x59414F59) return false;
    if (version != 4) return false;
    if (v != V) { printf("load_model: v mismatch, got %d expected %d\n", v, V); return false; }
    int Wsz = V * D, WhSz = V * HASH_FEATURES, WbiSz = V * V;
    int Q1Sz = B * K * D;
    m.W = (float*)malloc(Wsz * 4); m.Wm = (float*)malloc(Wsz * 4); m.Wv = (float*)malloc(Wsz * 4);
    f.read((char*)m.W, Wsz * 4); f.read((char*)m.Wm, Wsz * 4); f.read((char*)m.Wv, Wsz * 4);
    m.Wh = (float*)malloc(WhSz * 4); m.Whm = (float*)malloc(WhSz * 4); m.Whv = (float*)malloc(WhSz * 4);
    f.read((char*)m.Wh, WhSz * 4); f.read((char*)m.Whm, WhSz * 4); f.read((char*)m.Whv, WhSz * 4);
    m.Wbi = (float*)malloc(WbiSz * 4); m.Wbim = (float*)malloc(WbiSz * 4); m.Wbiv = (float*)malloc(WbiSz * 4);
    f.read((char*)m.Wbi, WbiSz * 4); f.read((char*)m.Wbim, WbiSz * 4); f.read((char*)m.Wbiv, WbiSz * 4);
    // Skip Q3, aW, ab, gW, gb
    f.ignore(5 * 2 * D * 4 * 3);
    f.ignore(2 * D * D * 4 * 3 + 2 * D * 4 * 3);
    f.ignore(2 * D * D * 4 * 3 + 2 * D * 4 * 3);
    m.q1 = (trit*)malloc(Q1Sz);
    f.read((char*)m.q1, Q1Sz);
    f.ignore(Q1Sz * 4 * 2);
    f.read((char*)&m.q1_step, 4); f.read((char*)&m.step, 4);
    int H = HIDDEN;
    m.W_sgl_gate = (float*)malloc(H*H*4); m.b_sgl_gate = (float*)malloc(H*4);
    m.W_sgl_up = (float*)malloc(H*H*4);   m.b_sgl_up = (float*)malloc(H*4);
    m.W_sgl_out = (float*)malloc(V*H*4);
    f.read((char*)m.W_sgl_gate, H*H*4); f.read((char*)m.b_sgl_gate, H*4);
    f.read((char*)m.W_sgl_up, H*H*4);   f.read((char*)m.b_sgl_up, H*4);
    f.read((char*)m.W_sgl_out, V*H*4);
    // Allocate adam state for Wg/Wu/Wout and bg/bu.
    m.W_sgl_out_m = (float*)calloc(V*H, 4);
    m.W_sgl_out_v = (float*)calloc(V*H, 4);
    m.W_sgl_gate_m = (float*)calloc(H*H, 4);
    m.W_sgl_gate_v = (float*)calloc(H*H, 4);
    m.W_sgl_up_m = (float*)calloc(H*H, 4);
    m.W_sgl_up_v = (float*)calloc(H*H, 4);
    m.b_sgl_gate_m = (float*)calloc(H, 4);
    m.b_sgl_gate_v = (float*)calloc(H, 4);
    m.b_sgl_up_m = (float*)calloc(H, 4);
    m.b_sgl_up_v = (float*)calloc(H, 4);
    if (!f) return false;
    const auto payload_end = f.tellg();
    f.seekg(0,std::ios::end); const auto file_end = f.tellg(); f.seekg(payload_end);
    const std::streamoff optimizer_bytes = (4*H*H + 2*V*H + 4*H)*4;
    if (file_end != payload_end && file_end-payload_end != optimizer_bytes) return false;
    // A legacy file may omit the ENTIRE optimizer extension, never a partial one.
    auto try_read = [&](float* dst, int count) {
        std::streampos before_pos = f.tellg();
        f.read((char*)dst, count * 4);
        std::streampos after_pos = f.tellg();
        long got = (long)(after_pos - before_pos);
        if (got != (long)(count * 4)) {
            // Short read -> older binary without these fields; zero-fill and rewind to EOF.
            f.clear();
            f.seekg(0, std::ios::end);
            f.clear();
            for (int i = 0; i < count; ++i) dst[i] = 0.0f;
        }
    };
    try_read(m.W_sgl_gate_m, H*H); try_read(m.W_sgl_gate_v, H*H);
    try_read(m.W_sgl_up_m,   H*H); try_read(m.W_sgl_up_v,   H*H);
    try_read(m.W_sgl_out_m,  V*H); try_read(m.W_sgl_out_v,  V*H);
    try_read(m.b_sgl_gate_m, H);   try_read(m.b_sgl_gate_v, H);
    try_read(m.b_sgl_up_m,   H);   try_read(m.b_sgl_up_v,   H);
    return true;
}

bool save_model(const char* path, Model& m) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    int magic = 0x59414F59, version = 4, v = V;
    f.write((char*)&magic, 4); f.write((char*)&version, 4); f.write((char*)&v, 4);
    int Wsz = V * D, WhSz = V * HASH_FEATURES, WbiSz = V * V;
    f.write((char*)m.W, Wsz * 4); f.write((char*)m.Wm, Wsz * 4); f.write((char*)m.Wv, Wsz * 4);
    f.write((char*)m.Wh, WhSz * 4); f.write((char*)m.Whm, WhSz * 4); f.write((char*)m.Whv, WhSz * 4);
    f.write((char*)m.Wbi, WbiSz * 4); f.write((char*)m.Wbim, WbiSz * 4); f.write((char*)m.Wbiv, WbiSz * 4);
    // Q3, aW, ab, gW, gb
    // Q3: 5 layers * 3 (W + m + v) * NL * D floats
    std::vector<float> zeros(5 * 3 * NL * D);
    f.write((char*)zeros.data(), zeros.size() * 4);
    // aW + ab: 3 (W + m + v) * (NL*D*D + NL*D) floats
    std::vector<float> zeros2(3 * (NL * D * D + NL * D));
    f.write((char*)zeros2.data(), zeros2.size() * 4);
    // gW + gb: same as aW
    std::vector<float> zeros3(3 * (NL * D * D + NL * D));
    f.write((char*)zeros3.data(), zeros3.size() * 4);
    int Q1Sz = B * K * D;
    f.write((char*)m.q1, Q1Sz);
    // adam state for q1 (m + v): Q1Sz floats * 2
    std::vector<float> zeros4(Q1Sz * 2);
    f.write((char*)zeros4.data(), zeros4.size() * 4);
    int q1step = 0;
    f.write((char*)&q1step, 4); f.write((char*)&m.step, 4);
    int H = HIDDEN;
    f.write((char*)m.W_sgl_gate, H*H*4); f.write((char*)m.b_sgl_gate, H*4);
    f.write((char*)m.W_sgl_up, H*H*4);   f.write((char*)m.b_sgl_up, H*4);
    f.write((char*)m.W_sgl_out, V*H*4);
    // Persist adam state for Wg/Wu/Wout and bg/bu (newly frozen into file format).
    f.write((char*)m.W_sgl_gate_m, H*H*4); f.write((char*)m.W_sgl_gate_v, H*H*4);
    f.write((char*)m.W_sgl_up_m,   H*H*4); f.write((char*)m.W_sgl_up_v,   H*H*4);
    f.write((char*)m.W_sgl_out_m,  V*H*4); f.write((char*)m.W_sgl_out_v,  V*H*4);
    f.write((char*)m.b_sgl_gate_m, H*4);   f.write((char*)m.b_sgl_gate_v, H*4);
    f.write((char*)m.b_sgl_up_m,   H*4);   f.write((char*)m.b_sgl_up_v,   H*4);
    f.flush();
    return bool(f);
}


// Independent oracle lives in verify_v21_math.py; validation uses production --export-raw.
int main(int argc, char** argv) {
    if (argc < 4) { printf("Usage:\n  %s model.bin tokens.txt output.bin [n_win=100] [epochs=1] [lr=0.005]\n  %s model.bin --fd-check tokens.txt\n", argv[0], argv[0]); return 1; }
    Model m;
    if (!load_model(argv[1], m)) { printf("load failed\n"); return 1; }
    printf("[V21-Phase5-GPU] Loaded model step=%d\n", m.step);
    int n_win = argc > 4 ? atoi(argv[4]) : 100;
    int epochs = argc > 5 ? atoi(argv[5]) : 1;
    float lr = argc > 6 ? atof(argv[6]) : 0.005f;
    // argv[7] is retained for CLI compatibility; all four active matrices are trained.
    if (n_win < 0 || epochs < 1 || !std::isfinite(lr) || lr <= 0) return 1;

    // CPU dump_tokens writes a count header, NOT a raw token stream.
    std::ifstream ft(argv[2], std::ios::binary | std::ios::ate);
    if (!ft) { fprintf(stderr, "Cannot open token file\n"); return 1; }
    const std::streamoff token_bytes = ft.tellg();
    ft.seekg(0);
    int32_t ntok = 0;
    if (!ft.read(reinterpret_cast<char*>(&ntok), 4) || ntok <= 0 ||
        token_bytes != (static_cast<std::streamoff>(ntok) + 1) * 4) {
        fprintf(stderr, "Invalid count-prefixed token file\n"); return 1;
    }
    std::vector<int> tokens(static_cast<size_t>(ntok));
    if (!ft.read(reinterpret_cast<char*>(tokens.data()), static_cast<std::streamsize>(ntok) * 4)) {
        fprintf(stderr, "Truncated token payload\n"); return 1;
    }
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] < 0 || tokens[i] >= V) {
            fprintf(stderr, "Invalid token at %zu: %d\n", i, tokens[i]); return 1;
        }
    }
    printf("Loaded %zu tokens (count header excluded)\n", tokens.size());

    // Init GPU
    cublasHandle_t handle; cublasCheck(cublasCreate(&handle));
    cudaStream_t stream; cudaCheck(cudaStreamCreate(&stream));
    cublasCheck(cublasSetStream(handle, stream));

    // Allocate GPU memory
    float *d_W, *d_Wm, *d_Wv, *d_Wh, *d_Whm, *d_Whv, *d_Wbi, *d_Wbim, *d_Wbiv;
    float *d_W_sgl_gate, *d_W_sgl_up, *d_W_sgl_out, *d_W_sgl_out_m, *d_W_sgl_out_v, *d_b_sgl_gate, *d_b_sgl_up;
    // Adam state for Wg/Wu/bg/bu (allocated & zeroed; mirrors host calloc).
    float *d_W_sgl_gate_m, *d_W_sgl_gate_v, *d_W_sgl_up_m, *d_W_sgl_up_v;
    float *d_b_sgl_gate_m, *d_b_sgl_gate_v, *d_b_sgl_up_m, *d_b_sgl_up_v;
    trit *d_q1;
    int Wsz = V * D, WhSz = V * HASH_FEATURES, WbiSz = V * V, Q1Sz = B * K * D;
    cudaCheck(cudaMalloc(&d_W, Wsz * 4)); cudaCheck(cudaMalloc(&d_Wm, Wsz * 4)); cudaCheck(cudaMalloc(&d_Wv, Wsz * 4));
    cudaCheck(cudaMalloc(&d_Wh, WhSz * 4)); cudaCheck(cudaMalloc(&d_Whm, WhSz * 4)); cudaCheck(cudaMalloc(&d_Whv, WhSz * 4));
    cudaCheck(cudaMalloc(&d_Wbi, WbiSz * 4)); cudaCheck(cudaMalloc(&d_Wbim, WbiSz * 4)); cudaCheck(cudaMalloc(&d_Wbiv, WbiSz * 4));
    cudaCheck(cudaMalloc(&d_W_sgl_gate, HIDDEN*HIDDEN*4)); cudaCheck(cudaMalloc(&d_W_sgl_up, HIDDEN*HIDDEN*4));
    cudaCheck(cudaMalloc(&d_W_sgl_out, V*HIDDEN*4));
    cudaCheck(cudaMalloc(&d_W_sgl_out_m, V*HIDDEN*4)); cudaCheck(cudaMalloc(&d_W_sgl_out_v, V*HIDDEN*4));
    cudaCheck(cudaMemset(d_W_sgl_out_m, 0, V*HIDDEN*4)); cudaCheck(cudaMemset(d_W_sgl_out_v, 0, V*HIDDEN*4));
    cudaCheck(cudaMalloc(&d_W_sgl_gate_m, HIDDEN*HIDDEN*4)); cudaCheck(cudaMalloc(&d_W_sgl_gate_v, HIDDEN*HIDDEN*4));
    cudaCheck(cudaMemset(d_W_sgl_gate_m, 0, HIDDEN*HIDDEN*4)); cudaCheck(cudaMemset(d_W_sgl_gate_v, 0, HIDDEN*HIDDEN*4));
    cudaCheck(cudaMalloc(&d_W_sgl_up_m, HIDDEN*HIDDEN*4)); cudaCheck(cudaMalloc(&d_W_sgl_up_v, HIDDEN*HIDDEN*4));
    cudaCheck(cudaMemset(d_W_sgl_up_m, 0, HIDDEN*HIDDEN*4)); cudaCheck(cudaMemset(d_W_sgl_up_v, 0, HIDDEN*HIDDEN*4));
    cudaCheck(cudaMalloc(&d_b_sgl_gate, HIDDEN*4)); cudaCheck(cudaMalloc(&d_b_sgl_up, HIDDEN*4));
    cudaCheck(cudaMalloc(&d_b_sgl_gate_m, HIDDEN*4)); cudaCheck(cudaMalloc(&d_b_sgl_gate_v, HIDDEN*4));
    cudaCheck(cudaMemset(d_b_sgl_gate_m, 0, HIDDEN*4)); cudaCheck(cudaMemset(d_b_sgl_gate_v, 0, HIDDEN*4));
    cudaCheck(cudaMalloc(&d_b_sgl_up_m, HIDDEN*4)); cudaCheck(cudaMalloc(&d_b_sgl_up_v, HIDDEN*4));
    cudaCheck(cudaMemset(d_b_sgl_up_m, 0, HIDDEN*4)); cudaCheck(cudaMemset(d_b_sgl_up_v, 0, HIDDEN*4));
    cudaCheck(cudaMalloc(&d_q1, Q1Sz));
    if (g_train_abort) { printf("[abort] cudaMalloc failed\n"); return 1; }

    cudaCheck(cudaMemcpyAsync(d_W, m.W, Wsz*4, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_Wm, m.Wm, Wsz*4, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_Wv, m.Wv, Wsz*4, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_Wh, m.Wh, WhSz*4, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_Whm, m.Whm, WhSz*4, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_Whv, m.Whv, WhSz*4, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_Wbi, m.Wbi, WbiSz*4, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_Wbim, m.Wbim, WbiSz*4, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_Wbiv, m.Wbiv, WbiSz*4, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_W_sgl_gate, m.W_sgl_gate, HIDDEN*HIDDEN*4, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_W_sgl_up, m.W_sgl_up, HIDDEN*HIDDEN*4, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_W_sgl_out, m.W_sgl_out, V*HIDDEN*4, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_b_sgl_gate, m.b_sgl_gate, HIDDEN*4, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_b_sgl_up, m.b_sgl_up, HIDDEN*4, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_q1, m.q1, Q1Sz, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_W_sgl_gate_m, m.W_sgl_gate_m, HIDDEN*HIDDEN*4, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_W_sgl_gate_v, m.W_sgl_gate_v, HIDDEN*HIDDEN*4, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_W_sgl_up_m, m.W_sgl_up_m, HIDDEN*HIDDEN*4, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_W_sgl_up_v, m.W_sgl_up_v, HIDDEN*HIDDEN*4, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_W_sgl_out_m, m.W_sgl_out_m, V*HIDDEN*4, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_W_sgl_out_v, m.W_sgl_out_v, V*HIDDEN*4, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_b_sgl_gate_m, m.b_sgl_gate_m, HIDDEN*4, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_b_sgl_gate_v, m.b_sgl_gate_v, HIDDEN*4, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_b_sgl_up_m, m.b_sgl_up_m, HIDDEN*4, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_b_sgl_up_v, m.b_sgl_up_v, HIDDEN*4, cudaMemcpyHostToDevice, stream));
    if (g_train_abort) { printf("[abort] cudaMemcpy (params) failed\n"); return 1; }

    int *d_inp, *d_tgt;
    float *d_trit_f, *d_hash_f, *d_state, *d_gate_z, *d_up_z, *d_logits, *d_probs, *d_d_logits;
    // Backward buffers: gate_pre/up_pre SNAPSHOT before silu_mul, d_hidden, d_gate_pre, d_up_pre,
    // d_state grad, d_trit_f grad, d_hash_f grad.
    float *d_gate_pre, *d_up_pre, *d_d_hidden, *d_d_gate_pre, *d_d_up_pre, *d_d_state, *d_d_trit_f, *d_d_hash_f;
    // [V21-Phase6-GPU] 因果 hash 链只保留末态 (诊断), 不再分配 4 链指针
    hash_t *d_h_last;
    cudaCheck(cudaMalloc(&d_inp, BL*4)); cudaCheck(cudaMalloc(&d_tgt, BL*4));
    cudaCheck(cudaMalloc(&d_trit_f, BL*D*4)); cudaCheck(cudaMalloc(&d_hash_f, BL*HASH_FEATURES*4));
    cudaCheck(cudaMalloc(&d_state, BL*HIDDEN*4));
    cudaCheck(cudaMalloc(&d_gate_z, BL*HIDDEN*4)); cudaCheck(cudaMalloc(&d_up_z, BL*HIDDEN*4));
    cudaCheck(cudaMalloc(&d_logits, BL*V_unit*4)); cudaCheck(cudaMalloc(&d_probs, BL*V_unit*4)); cudaCheck(cudaMalloc(&d_d_logits, BL*V_unit*4));
    cudaCheck(cudaMalloc(&d_h_last, BATCH*4));
    cudaCheck(cudaMalloc(&d_gate_pre, BL*HIDDEN*4)); cudaCheck(cudaMalloc(&d_up_pre, BL*HIDDEN*4));
    cudaCheck(cudaMalloc(&d_d_hidden, BL*HIDDEN*4));
    cudaCheck(cudaMalloc(&d_d_gate_pre, BL*HIDDEN*4)); cudaCheck(cudaMalloc(&d_d_up_pre, BL*HIDDEN*4));
    cudaCheck(cudaMalloc(&d_d_state, BL*HIDDEN*4));
    float *grad_o, *grad_g, *grad_u, *grad_bi;
    cudaCheck(cudaMalloc(&grad_o, V*HIDDEN*4));
    cudaCheck(cudaMalloc(&grad_g, HIDDEN*HIDDEN*4));
    cudaCheck(cudaMalloc(&grad_u, HIDDEN*HIDDEN*4));
    cudaCheck(cudaMalloc(&grad_bi, V*V*4));
    cudaCheck(cudaMemset(d_d_state, 0, BL*HIDDEN*4));
    cudaCheck(cudaMalloc(&d_d_trit_f, BL*D*4)); cudaCheck(cudaMalloc(&d_d_hash_f, BL*HASH_FEATURES*4));
    if (g_train_abort) { printf("[abort] cudaMalloc (acts/grad) failed\n"); return 1; }

    float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    int total_windows = 0;
    cudaEvent_t e_start, e_stop;
    cudaEventCreate(&e_start); cudaEventCreate(&e_stop);

    auto t_start = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        total_windows = 0; // epoch-local denominator
        std::mt19937 sampler(epoch * 1234u + 5678u);
        int max_offset = (int)tokens.size() - BL - SEQ - 1;
        if (max_offset <= 0) { printf("not enough tokens\n"); return 1; }
        float total_loss = 0;

        for (int w = 0; w < n_win; ++w) {
            // Random offset for each window (避免 overfit)
            int offset = std::uniform_int_distribution<int>(0, max_offset - 1)(sampler);
            std::vector<int> inpBL(BL), tgtBL(BL);
            for (int n = 0; n < BL; ++n) {
                int start = offset + (n / SEQ) * 7;
                if (start + SEQ + 1 >= (int)tokens.size()) start = offset + (n / SEQ);
                inpBL[n] = tokens[start + (n % SEQ)];
                tgtBL[n] = tokens[start + (n % SEQ) + 1];
            }

            cudaMemcpyAsync(d_inp, inpBL.data(), BL*4, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_tgt, tgtBL.data(), BL*4, cudaMemcpyHostToDevice, stream);

            // Forward
            trit_acc_kernel<<<BATCH, 1, 0, stream>>>(d_q1, d_inp, d_trit_f, BATCH, SEQ, D);
            kernelCheck();
            // [V21-Phase6-GPU] 单链因果 hash; d_h_last 仅做诊断 (BATCH 大小)
            hash_extract_kernel<<<BATCH, 1, 0, stream>>>(d_inp, d_h_last, d_hash_f, BATCH, SEQ);
            kernelCheck();
            concat_kernel<<<BL, 1, 0, stream>>>(d_state, d_trit_f, d_hash_f, BL);
            kernelCheck();

            float one = 1.0f, zero = 0.0f;
            cublasCheck(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        HIDDEN, BL, HIDDEN, &one,
                        d_W_sgl_gate, HIDDEN, d_state, HIDDEN, &zero, d_gate_z, HIDDEN));
            int total = BL * HIDDEN;
            // No gate/up bias in the specified graph.

            cublasCheck(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        HIDDEN, BL, HIDDEN, &one,
                        d_W_sgl_up, HIDDEN, d_state, HIDDEN, &zero, d_up_z, HIDDEN));
            // No gate/up bias in the specified graph.

            // Snapshot g, u BEFORE silu_mul mutates d_gate_z in place.
            cudaCheck(cudaMemcpyAsync(d_gate_pre, d_gate_z, total*4, cudaMemcpyDeviceToDevice, stream));
            cudaCheck(cudaMemcpyAsync(d_up_pre,   d_up_z,   total*4, cudaMemcpyDeviceToDevice, stream));

            silu_mul_kernel<<<(total+255)/256, 256, 0, stream>>>(d_gate_z, d_up_z, total);
            kernelCheck();

            cublasCheck(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        V_unit, BL, HIDDEN, &one,
                        d_W_sgl_out, HIDDEN, d_gate_z, HIDDEN, &zero, d_logits, V_unit));
            add_wbi_kernel<<<BL, 1, 0, stream>>>(d_logits, d_Wbi, d_inp, PAD, BL, V_unit);
            kernelCheck();

            // Softmax
            softmax_kernel<<<BL, 1, 0, stream>>>(d_logits, d_probs, BL, V_unit);
            kernelCheck();

            // Stable mean cross entropy; finite hard examples are never skipped.
            std::vector<float> h_logits(BL * V_unit);
            cudaCheck(cudaMemcpyAsync(h_logits.data(), d_logits, BL*V_unit*4, cudaMemcpyDeviceToHost, stream));
            cudaCheck(cudaStreamSynchronize(stream));
            double loss_sum = 0;
            for (int n = 0; n < BL; ++n) {
                double mx = -INFINITY;
                for (int v = 0; v < V_unit; ++v) {
                    const float z = h_logits[n*V_unit+v];
                    if (!std::isfinite(z)) {
                        fprintf(stderr,"[abort] nonfinite logit epoch=%d window=%d row=%d col=%d step=%d\n",epoch+1,w+1,n,v,m.step);
                        return 2;
                    }
                    mx = std::max(mx, double(z));
                }
                double den = 0;
                for (int v = 0; v < V_unit; ++v) den += std::exp(double(h_logits[n*V_unit+v])-mx);
                loss_sum += mx + std::log(den) - h_logits[n*V_unit+tgtBL[n]];
            }
            float loss = float(loss_sum / BL);
            if (!std::isfinite(loss)) { fprintf(stderr,"[abort] nonfinite mean loss\n"); return 2; }
            total_loss += loss;
            total_windows++;
            m.step++;

            // d_logits
            d_logits_kernel<<<BL, 1, 0, stream>>>(d_d_logits, d_probs, d_tgt, BL, V_unit);
            kernelCheck();

            // Adam updates
            float bc1 = 1 - std::pow(b1, (float)m.step), bc2 = 1 - std::pow(b2, (float)m.step);

            // ---- Backward through logits = Wout*hidden + Wbi[prev,:] ----
            // d_hidden[n, h] = sum_v d_logits[n, v] * Wout[v, h]
            {
                dim3 grid_dh((HIDDEN + 127) / 128, BL);
                d_logits_to_d_hidden_kernel<<<grid_dh, 128, 0, stream>>>(
                    d_d_hidden, d_d_logits, d_W_sgl_out, HIDDEN, V_unit, BL);
                kernelCheck();
            }

            // ---- Backward through silu_mul ----
            silu_back_kernel<<<(total+255)/256, 256, 0, stream>>>(
                d_d_gate_pre, d_d_up_pre, d_d_hidden, d_gate_pre, d_up_pre, total);
            kernelCheck();

            // ---- Backward to state (no q1 / hash params to update; expose d_trit_f / d_hash_f for diagnosis) ----
            cudaCheck(cudaMemsetAsync(d_d_state, 0, BL*HIDDEN*4, stream));
            {
                dim3 grid_ds((HIDDEN + 127) / 128, BL);
                d_state_add_kernel<<<grid_ds, 128, 0, stream>>>(
                    d_d_state, d_d_gate_pre, d_W_sgl_gate, HIDDEN, BL);
                kernelCheck();
                d_state_add_kernel<<<grid_ds, 128, 0, stream>>>(
                    d_d_state, d_d_up_pre,   d_W_sgl_up,   HIDDEN, BL);
                kernelCheck();
                split_state_kernel<<<(BL*HIDDEN + 255) / 256, 256, 0, stream>>>(
                    d_d_state, d_d_trit_f, d_d_hash_f, BL);
                kernelCheck();
            }

            // ---- Wout Adam (always update output projection; mean gradient) ----
            {
                dim3 grid_SglOut((HIDDEN + 127) / 128, V);
                adam_Wsglout_kernel<<<grid_SglOut, 128, 0, stream>>>(
                    d_W_sgl_out, d_W_sgl_out_m, d_W_sgl_out_v,
                    d_d_logits, d_gate_z,
                    lr, b1, b2, bc1, bc2, eps, V, HIDDEN, BL, grad_o);
                kernelCheck();
            }

            // ---- Wbi Adam: full V x V; mean over contributing positions per row ----
            {
                dim3 grid_Wbi(V, V); // every previous-token row
                adam_Wbi_kernel<<<grid_Wbi, 1, 0, stream>>>(d_Wbi, d_Wbim, d_Wbiv, d_d_logits, d_inp, d_tgt,
                                                             lr, b1, b2, bc1, bc2, eps, V, BL, PAD, grad_bi);
                kernelCheck();
            }

            // ---- Wg / Wu / bg / bu Adam (per math, mean gradient) ----
            {
                dim3 grid_Wgu((HIDDEN + 127) / 128, HIDDEN);
                adam_W_gateup_kernel<<<grid_Wgu, 128, 0, stream>>>(
                    d_W_sgl_gate, d_W_sgl_gate_m, d_W_sgl_gate_v,
                    d_d_gate_pre, d_state,
                    lr, b1, b2, bc1, bc2, eps, HIDDEN, BL, grad_g);
                kernelCheck();
                adam_W_gateup_kernel<<<grid_Wgu, 128, 0, stream>>>(
                    d_W_sgl_up, d_W_sgl_up_m, d_W_sgl_up_v,
                    d_d_up_pre, d_state,
                    lr, b1, b2, bc1, bc2, eps, HIDDEN, BL, grad_u);
                kernelCheck();
                // Legacy gate/up biases are not trainable parameters.
            }

            if (argc > 8 && std::string(argv[8]) == "--export-raw") {
                cudaCheck(cudaStreamSynchronize(stream));
                if (g_train_abort) return 2;
                auto dump = [&](const char* name, const float* device, size_t count) {
                    std::vector<float> host(count);
                    cudaCheck(cudaMemcpy(host.data(), device, count*4, cudaMemcpyDeviceToHost));
                    if (g_train_abort) return false;
                    std::ofstream out(std::string(argv[3])+"."+name+".f32", std::ios::binary);
                    out.write(reinterpret_cast<const char*>(host.data()), count*4);
                    return bool(out);
                };
                bool ok = dump("state",d_state,BL*HIDDEN) && dump("Wg",d_W_sgl_gate,HIDDEN*HIDDEN)
                    && dump("Wu",d_W_sgl_up,HIDDEN*HIDDEN) && dump("Wo",d_W_sgl_out,V*HIDDEN)
                    && dump("Wbi",d_Wbi,V*V) && dump("logits",d_logits,BL*V)
                    && dump("dWg",grad_g,HIDDEN*HIDDEN) && dump("dWu",grad_u,HIDDEN*HIDDEN)
                    && dump("dWo",grad_o,V*HIDDEN) && dump("dWbi",grad_bi,V*V);
                std::ofstream ids(std::string(argv[3])+".ids.i32",std::ios::binary);
                ids.write(reinterpret_cast<const char*>(inpBL.data()),BL*4);
                ids.write(reinterpret_cast<const char*>(tgtBL.data()),BL*4);
                printf("RAW EXPORT: no Adam update, no model save; BL=%d H=%d V=%d\n",BL,HIDDEN,V);
                return ok && ids ? 0 : 2;
            }
            apply_adam_kernel<<<((V*HIDDEN)+255)/256,256,0,stream>>>(d_W_sgl_out,d_W_sgl_out_m,d_W_sgl_out_v,grad_o,V*HIDDEN,lr,b1,b2,bc1,bc2,eps);
            kernelCheck();
            apply_adam_kernel<<<((HIDDEN*HIDDEN)+255)/256,256,0,stream>>>(d_W_sgl_gate,d_W_sgl_gate_m,d_W_sgl_gate_v,grad_g,HIDDEN*HIDDEN,lr,b1,b2,bc1,bc2,eps);
            kernelCheck();
            apply_adam_kernel<<<((HIDDEN*HIDDEN)+255)/256,256,0,stream>>>(d_W_sgl_up,d_W_sgl_up_m,d_W_sgl_up_v,grad_u,HIDDEN*HIDDEN,lr,b1,b2,bc1,bc2,eps);
            kernelCheck();
            apply_adam_kernel<<<((V*V)+255)/256,256,0,stream>>>(d_Wbi,d_Wbim,d_Wbiv,grad_bi,V*V,lr,b1,b2,bc1,bc2,eps);
            kernelCheck();

            // Note: legacy W (trit->logits) and Wh (hash->logits) are intentionally NOT updated;
            // logits = Wout*hidden + Wbi[prev,:] does not use them.  Storage / load / save kept for
            // file-format compatibility only.

            cudaCheck(cudaStreamSynchronize(stream));

            if ((w + 1) % 50 == 0 || w == 0) {
                auto t_now = std::chrono::steady_clock::now();
                double el = std::chrono::duration<double>(t_now - t_start).count();
                printf("  ep%d win%d/%d loss=%.4f avg=%.4f (%.1fs)\n",
                       epoch+1, w+1, n_win, loss, total_loss/total_windows, el);
            }
        }
        if (total_windows > 0)
            printf("Epoch %d avg_loss=%.4f (over %d accepted windows / %d requested)\n",
                   epoch+1, total_loss / total_windows, total_windows, n_win);
        else
            printf("Epoch %d avg_loss=NA (no accepted windows)\n", epoch+1);
    }

    // Fail-safe: if any CUDA / cuBLAS error occurred during training, DO NOT save.
    if (g_train_abort) {
        fprintf(stderr, "[abort] CUDA/cuBLAS error encountered; skipping save of %s\n", argv[3]);
        cudaFree(d_W); cudaFree(d_Wm); cudaFree(d_Wv); cudaFree(d_Wh); cudaFree(d_Whm); cudaFree(d_Whv);
        cudaFree(d_Wbi); cudaFree(d_Wbim); cudaFree(d_Wbiv);
        cudaFree(d_W_sgl_gate); cudaFree(d_W_sgl_up); cudaFree(d_W_sgl_out);
        cudaFree(d_b_sgl_gate); cudaFree(d_b_sgl_up); cudaFree(d_q1);
        cudaFree(d_inp); cudaFree(d_tgt); cudaFree(d_trit_f); cudaFree(d_hash_f);
        cudaFree(d_state); cudaFree(d_gate_z); cudaFree(d_up_z); cudaFree(d_logits); cudaFree(d_probs); cudaFree(d_d_logits);
        cudaFree(d_gate_pre); cudaFree(d_up_pre); cudaFree(d_d_hidden);
        cudaFree(d_d_gate_pre); cudaFree(d_d_up_pre); cudaFree(d_d_state);
        cudaFree(d_d_trit_f); cudaFree(d_d_hash_f);
        cudaFree(d_h_last);
        cudaFree(d_W_sgl_gate_m); cudaFree(d_W_sgl_gate_v);
        cudaFree(d_W_sgl_up_m); cudaFree(d_W_sgl_up_v);
        cudaFree(d_W_sgl_out_m); cudaFree(d_W_sgl_out_v);
        cudaFree(d_b_sgl_gate_m); cudaFree(d_b_sgl_gate_v);
        cudaFree(d_b_sgl_up_m); cudaFree(d_b_sgl_up_v);
        cudaFree(d_h_last);
        cublasDestroy(handle);
        return 2;
    }

    // Copy weights back
    cudaCheck(cudaMemcpy(m.W, d_W, Wsz*4, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(m.Wm, d_Wm, Wsz*4, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(m.Wv, d_Wv, Wsz*4, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(m.Wh, d_Wh, WhSz*4, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(m.Whm, d_Whm, WhSz*4, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(m.Whv, d_Whv, WhSz*4, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(m.Wbi, d_Wbi, WbiSz*4, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(m.Wbim, d_Wbim, WbiSz*4, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(m.Wbiv, d_Wbiv, WbiSz*4, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(m.W_sgl_gate, d_W_sgl_gate, HIDDEN*HIDDEN*4, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(m.W_sgl_up,   d_W_sgl_up,   HIDDEN*HIDDEN*4, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(m.W_sgl_out,  d_W_sgl_out,  V*HIDDEN*4, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(m.W_sgl_gate_m, d_W_sgl_gate_m, HIDDEN*HIDDEN*4, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(m.W_sgl_gate_v, d_W_sgl_gate_v, HIDDEN*HIDDEN*4, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(m.W_sgl_up_m,   d_W_sgl_up_m,   HIDDEN*HIDDEN*4, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(m.W_sgl_up_v,   d_W_sgl_up_v,   HIDDEN*HIDDEN*4, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(m.W_sgl_out_m,  d_W_sgl_out_m,  V*HIDDEN*4, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(m.W_sgl_out_v,  d_W_sgl_out_v,  V*HIDDEN*4, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(m.b_sgl_gate, d_b_sgl_gate, HIDDEN*4, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(m.b_sgl_up,   d_b_sgl_up,   HIDDEN*4, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(m.b_sgl_gate_m, d_b_sgl_gate_m, HIDDEN*4, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(m.b_sgl_gate_v, d_b_sgl_gate_v, HIDDEN*4, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(m.b_sgl_up_m,   d_b_sgl_up_m,   HIDDEN*4, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(m.b_sgl_up_v,   d_b_sgl_up_v,   HIDDEN*4, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(m.q1, d_q1, Q1Sz, cudaMemcpyDeviceToHost));
    if (g_train_abort) {
        fprintf(stderr, "[abort] cudaMemcpy (D2H final) failed; skipping save of %s\n", argv[3]);
        return 2;
    }

    if (!save_model(argv[3], m)) {
        fprintf(stderr, "[abort] save_model failed for %s\n", argv[3]);
        return 2;
    }
    printf("[V21-Phase5-GPU] Saved to %s (step=%d)\n", argv[3], m.step);

    cudaFree(d_W); cudaFree(d_Wm); cudaFree(d_Wv); cudaFree(d_Wh); cudaFree(d_Whm); cudaFree(d_Whv);
    cudaFree(d_Wbi); cudaFree(d_Wbim); cudaFree(d_Wbiv);
    cudaFree(d_W_sgl_gate); cudaFree(d_W_sgl_up); cudaFree(d_W_sgl_out);
    cudaFree(d_W_sgl_gate_m); cudaFree(d_W_sgl_gate_v);
    cudaFree(d_W_sgl_up_m); cudaFree(d_W_sgl_up_v);
    cudaFree(d_W_sgl_out_m); cudaFree(d_W_sgl_out_v);
    cudaFree(d_b_sgl_gate); cudaFree(d_b_sgl_up);
    cudaFree(d_b_sgl_gate_m); cudaFree(d_b_sgl_gate_v);
    cudaFree(d_b_sgl_up_m); cudaFree(d_b_sgl_up_v);
    cudaFree(d_q1);
    cudaFree(d_inp); cudaFree(d_tgt); cudaFree(d_trit_f); cudaFree(d_hash_f);
    cudaFree(d_state); cudaFree(d_gate_z); cudaFree(d_up_z); cudaFree(d_logits); cudaFree(d_probs); cudaFree(d_d_logits);
    cudaFree(d_gate_pre); cudaFree(d_up_pre); cudaFree(d_d_hidden);
    cudaFree(d_d_gate_pre); cudaFree(d_d_up_pre); cudaFree(d_d_state);
    cudaFree(d_d_trit_f); cudaFree(d_d_hash_f);
    cudaFree(d_h_last);
    cublasDestroy(handle);
    return 0;
}
