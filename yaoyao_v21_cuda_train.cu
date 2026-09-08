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

typedef int8_t trit; typedef uint32_t hash_t;
const int D = 128, V = 1024, V_unit = 1024;
const int B = 128, K = 16, NL = 2;
const int HIDDEN = 192, HASH_FEATURES = 64;
const int BATCH = 16, SEQ = 64, BL = BATCH * SEQ;

__global__ void hash_extract_kernel(const int* inp, hash_t* h0, hash_t* h1, hash_t* h2, hash_t* h3,
                                     float* hf_out, int Bd, int Sd) {
    int b = blockIdx.x; if (b >= Bd) return;
    hash_t h0_save = 5381u, h1_save = 5387u, h2_save = 5393u, h3_save = 5399u;
    for (int t = 0; t < Sd; ++t) {
        int id = inp[b*Sd + t];
        h0_save = (t > 0) ? (h0_save * 33u + (hash_t)id + 7u) : (5381u * 33u + (hash_t)id + 7u);
        h1_save = (t > 0) ? (h1_save * 37u + (hash_t)id + 11u) : 5387u;
        h2_save = (t > 0) ? (h2_save * 41u + (hash_t)id + 13u) : 5393u;
        h3_save = (t > 0) ? (h3_save * 43u + (hash_t)id + 17u) : 5399u;
    }
    h0[b] = h0_save; h1[b] = h1_save; h2[b] = h2_save; h3[b] = h3_save;
    uint32_t v0 = h0_save, v1 = h1_save, v2 = h2_save, v3 = h3_save;
    for (int t = 0; t < Sd; ++t) {
        int base = (b*Sd + t) * HASH_FEATURES;
        uint32_t t0 = v0;
        for (int i = 0; i < 16; ++i) {
            hf_out[base + i] = ((float)(t0 & 0xFFFFu) / 65535.0f - 0.5f);
            t0 >>= 16; if (t0 == 0) t0 = v0;
        }
        uint32_t t1 = v1;
        for (int i = 16; i < 32; ++i) {
            t1 = t1 * 0x85ebca6bu + 0xc2b2ae35u;
            hf_out[base + i] = ((float)(t1 & 0xFFFFu) / 65535.0f - 0.5f);
        }
        uint32_t t2 = v2;
        for (int i = 32; i < 48; ++i) {
            t2 = (t2 ^ (t2 >> 16)) * 0x9e3779b9u;
            hf_out[base + i] = ((float)(t2 & 0xFFFFu) / 65535.0f - 0.5f);
        }
        uint32_t t3 = v3 ^ 0xdeadbeefu;
        for (int i = 48; i < 64; ++i) {
            t3 = t3 * 0x27d4eb2fu + 0x165667b1u;
            hf_out[base + i] = ((float)(t3 & 0xFFFFu) / 65535.0f - 0.5f);
        }
    }
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

__global__ void add_bias_kernel(float* y, const float* bias, int rows, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] += bias[i % rows];
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
__global__ void adam_W_kernel(float* W, float* Wm, float* Wv, const float* d_logits, const float* trit_f,
                                float lr, float b1, float b2, float bc1, float bc2, float eps, int step,
                                int Vd, int Dd, int BL_d) {
    int v = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= Dd) return;
    int idx = v * Dd + d;
    float g = 0;
    for (int n = 0; n < BL_d; ++n) g += d_logits[n * Vd + v] * trit_f[n * Dd + d];
    if (g > 1) g = 1; if (g < -1) g = -1;
    float m = b1 * Wm[idx] + (1 - b1) * g;
    float vv = b2 * Wv[idx] + (1 - b2) * g * g;
    Wm[idx] = m; Wv[idx] = vv;
    float st = lr * (m / bc1) / (sqrtf(vv / bc2) + eps);
    float w = W[idx] - st;
    if (w > 2) w = 2; if (w < -2) w = -2;
    W[idx] = w;
}

__global__ void adam_Whash_kernel(float* Wh, float* Wm, float* Wv, const float* d_logits, const float* hash_f,
                                   float lr, float b1, float b2, float bc1, float bc2, float eps,
                                   int Vd, int Hd, int BL_d) {
    int v = blockIdx.y;
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= Hd) return;
    int idx = v * Hd + f;
    float g = 0;
    for (int n = 0; n < BL_d; ++n) g += d_logits[n * Vd + v] * hash_f[n * Hd + f];
    if (g > 1) g = 1; if (g < -1) g = -1;
    float m = b1 * Wm[idx] + (1 - b1) * g;
    float vv = b2 * Wv[idx] + (1 - b2) * g * g;
    Wm[idx] = m; Wv[idx] = vv;
    float st = lr * (m / bc1) / (sqrtf(vv / bc2) + eps);
    float w = Wh[idx] - st;
    if (w > 1) w = 1; if (w < -1) w = -1;
    Wh[idx] = w;
}

__global__ void adam_Wbi_kernel(float* Wbi, float* Wm, float* Wv, const float* d_logits, const int* inp, const int* targets,
                                  float lr, float b1, float b2, float bc1, float bc2, float eps,
                                  int Vd, int BL_d, int PAD) {
    // For each (prev, v): sum d_logits[n, v] for n where inp[n-1] == prev (or PAD if n%SEQ==0)
    int prev = blockIdx.x;
    int v = blockIdx.y * blockDim.x + threadIdx.x;
    if (v >= Vd) return;
    int idx = prev * Vd + v;
    float g = 0;
    for (int n = 0; n < BL_d; ++n) {
        int t = n % SEQ;
        int p = (t > 0) ? inp[n - 1] : PAD;
        if (p == prev) g += d_logits[n * Vd + v];
    }
    if (g > 1) g = 1; if (g < -1) g = -1;
    float m = b1 * Wm[idx] + (1 - b1) * g;
    float vv = b2 * Wv[idx] + (1 - b2) * g * g;
    Wm[idx] = m; Wv[idx] = vv;
    float st = lr * (m / bc1) / (sqrtf(vv / bc2) + eps);
    float w = Wbi[idx] - st;
    if (w > 2) w = 2; if (w < -2) w = -2;
    Wbi[idx] = w;
}

// Load model from .bin
struct Model {
    float *W, *Wm, *Wv, *Wh, *Whm, *Whv, *Wbi, *Wbim, *Wbiv;
    trit *q1;
    int q1_step, step;
    float *W_sgl_gate, *W_sgl_up, *W_sgl_out;
    float *b_sgl_gate, *b_sgl_up;
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
    return true;
}


int main(int argc, char** argv) {
    if (argc < 4) { printf("Usage: %s model.bin tokens.txt output.bin [n_win=100] [epochs=1] [lr=0.005]\n", argv[0]); return 1; }
    Model m;
    if (!load_model(argv[1], m)) { printf("load failed\n"); return 1; }
    printf("[V21-Phase5-GPU] Loaded model step=%d\n", m.step);
    int n_win = argc > 4 ? atoi(argv[4]) : 100;
    int epochs = argc > 5 ? atoi(argv[5]) : 1;
    float lr = argc > 6 ? atof(argv[6]) : 0.005f;
    
    // Read tokens from file
    std::ifstream ft(argv[2], std::ios::binary);
    if (!ft) { printf("no tokens file\n"); return 1; }
    
    // For simplicity, treat argv[2] as binary int32 token file
    ft.seekg(0, std::ios::end);
    long fsz = ft.tellg();
    ft.seekg(0);
    std::vector<int> tokens(fsz / 4);
    ft.read((char*)tokens.data(), fsz);
    printf("Loaded %zu tokens\n", tokens.size());
    
    // Init GPU
    cublasHandle_t handle; cublasCreate(&handle);
    cudaStream_t stream; cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);
    
    // Allocate GPU memory
    float *d_W, *d_Wm, *d_Wv, *d_Wh, *d_Whm, *d_Whv, *d_Wbi, *d_Wbim, *d_Wbiv;
    float *d_W_sgl_gate, *d_W_sgl_up, *d_W_sgl_out, *d_b_sgl_gate, *d_b_sgl_up;
    trit *d_q1;
    int Wsz = V * D, WhSz = V * HASH_FEATURES, WbiSz = V * V, Q1Sz = B * K * D;
    cudaMalloc(&d_W, Wsz * 4); cudaMalloc(&d_Wm, Wsz * 4); cudaMalloc(&d_Wv, Wsz * 4);
    cudaMalloc(&d_Wh, WhSz * 4); cudaMalloc(&d_Whm, WhSz * 4); cudaMalloc(&d_Whv, WhSz * 4);
    cudaMalloc(&d_Wbi, WbiSz * 4); cudaMalloc(&d_Wbim, WbiSz * 4); cudaMalloc(&d_Wbiv, WbiSz * 4);
    cudaMalloc(&d_W_sgl_gate, HIDDEN*HIDDEN*4); cudaMalloc(&d_W_sgl_up, HIDDEN*HIDDEN*4);
    cudaMalloc(&d_W_sgl_out, V*HIDDEN*4);
    cudaMalloc(&d_b_sgl_gate, HIDDEN*4); cudaMalloc(&d_b_sgl_up, HIDDEN*4);
    cudaMalloc(&d_q1, Q1Sz);
    
    cudaMemcpyAsync(d_W, m.W, Wsz*4, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_Wm, m.Wm, Wsz*4, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_Wv, m.Wv, Wsz*4, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_Wh, m.Wh, WhSz*4, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_Whm, m.Whm, WhSz*4, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_Whv, m.Whv, WhSz*4, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_Wbi, m.Wbi, WbiSz*4, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_Wbim, m.Wbim, WbiSz*4, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_Wbiv, m.Wbiv, WbiSz*4, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_W_sgl_gate, m.W_sgl_gate, HIDDEN*HIDDEN*4, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_W_sgl_up, m.W_sgl_up, HIDDEN*HIDDEN*4, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_W_sgl_out, m.W_sgl_out, V*HIDDEN*4, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b_sgl_gate, m.b_sgl_gate, HIDDEN*4, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b_sgl_up, m.b_sgl_up, HIDDEN*4, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_q1, m.q1, Q1Sz, cudaMemcpyHostToDevice, stream);
    
    int *d_inp, *d_tgt;
    float *d_trit_f, *d_hash_f, *d_state, *d_gate_z, *d_up_z, *d_logits, *d_probs, *d_d_logits;
    hash_t *d_h0, *d_h1, *d_h2, *d_h3;
    cudaMalloc(&d_inp, BL*4); cudaMalloc(&d_tgt, BL*4);
    cudaMalloc(&d_trit_f, BL*D*4); cudaMalloc(&d_hash_f, BL*HASH_FEATURES*4);
    cudaMalloc(&d_state, BL*HIDDEN*4);
    cudaMalloc(&d_gate_z, BL*HIDDEN*4); cudaMalloc(&d_up_z, BL*HIDDEN*4);
    cudaMalloc(&d_logits, BL*V_unit*4); cudaMalloc(&d_probs, BL*V_unit*4); cudaMalloc(&d_d_logits, BL*V_unit*4);
    cudaMalloc(&d_h0, BATCH*4); cudaMalloc(&d_h1, BATCH*4);
    cudaMalloc(&d_h2, BATCH*4); cudaMalloc(&d_h3, BATCH*4);
    
    float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    int total_windows = 0;
    cudaEvent_t e_start, e_stop;
    cudaEventCreate(&e_start); cudaEventCreate(&e_stop);
    
    auto t_start = std::chrono::steady_clock::now();
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::srand(epoch * 1234 + 5678);
        int max_offset = (int)tokens.size() - BL - SEQ - 1;
        if (max_offset <= 0) { printf("not enough tokens\n"); return 1; }
        float total_loss = 0;
        
        for (int w = 0; w < n_win; ++w) {
            // Random offset for each window (避免 overfit)
            int offset = std::rand() % max_offset;
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
            hash_extract_kernel<<<BATCH, 1, 0, stream>>>(d_inp, d_h0, d_h1, d_h2, d_h3, d_hash_f, BATCH, SEQ);
            concat_kernel<<<BL, 1, 0, stream>>>(d_state, d_trit_f, d_hash_f, BL);
            
            float one = 1.0f, zero = 0.0f;
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        HIDDEN, BL, HIDDEN, &one,
                        d_W_sgl_gate, HIDDEN, d_state, HIDDEN, &zero, d_gate_z, HIDDEN);
            int total = BL * HIDDEN;
            add_bias_kernel<<<(total+255)/256, 256, 0, stream>>>(d_gate_z, d_b_sgl_gate, HIDDEN, total);
            
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        HIDDEN, BL, HIDDEN, &one,
                        d_W_sgl_up, HIDDEN, d_state, HIDDEN, &zero, d_up_z, HIDDEN);
            add_bias_kernel<<<(total+255)/256, 256, 0, stream>>>(d_up_z, d_b_sgl_up, HIDDEN, total);
            
            silu_mul_kernel<<<(total+255)/256, 256, 0, stream>>>(d_gate_z, d_up_z, total);
            
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        V_unit, BL, HIDDEN, &one,
                        d_W_sgl_out, HIDDEN, d_gate_z, HIDDEN, &zero, d_logits, V_unit);
            add_wbi_kernel<<<BL, 1, 0, stream>>>(d_logits, d_Wbi, d_inp, 0, BL, V_unit);
            
            // Softmax
            softmax_kernel<<<BL, 1, 0, stream>>>(d_logits, d_probs, BL, V_unit);
            
            // Compute Loss (CPU)
            std::vector<float> h_probs(BL * V_unit);
            cudaMemcpyAsync(h_probs.data(), d_probs, BL*V_unit*4, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            float loss = 0;
            bool bad = false;
            for (int n = 0; n < BL; ++n) {
                float p = h_probs[n*V_unit + tgtBL[n]];
                if (p < 1e-12f || std::isnan(p) || std::isinf(p)) { bad = true; break; }
                loss += -std::log(std::max(p, 1e-9f));
            }
            if (bad) {
                if (w < 5 || w % 100 == 0) printf("  [skip] win%d bad probs\n", w+1);
                continue;
            }
            loss /= BL;
            if (loss > 12.0f) {
                if (w < 5 || w % 100 == 0) printf("  [skip] win%d loss=%.2f too high\n", w+1, loss);
                continue;
            }
            total_loss += loss;
            total_windows++;
            m.step++;
            
            // d_logits
            d_logits_kernel<<<BL, 1, 0, stream>>>(d_d_logits, d_probs, d_tgt, BL, V_unit);
            
            // Adam updates
            float bc1 = 1 - std::pow(b1, (float)m.step), bc2 = 1 - std::pow(b2, (float)m.step);
            dim3 block_D(128); dim3 grid_D((D + 127) / 128, V);
            adam_W_kernel<<<grid_D, block_D, 0, stream>>>(d_W, d_Wm, d_Wv, d_d_logits, d_trit_f,
                                                          lr, b1, b2, bc1, bc2, eps, m.step, V, D, BL);
            dim3 grid_H((HASH_FEATURES + 127) / 128, V);
            adam_Whash_kernel<<<grid_H, block_D, 0, stream>>>(d_Wh, d_Whm, d_Whv, d_d_logits, d_hash_f,
                                                                lr, b1, b2, bc1, bc2, eps, V, HASH_FEATURES, BL);
            dim3 grid_Wbi(128, V);
            adam_Wbi_kernel<<<grid_Wbi, 1, 0, stream>>>(d_Wbi, d_Wbim, d_Wbiv, d_d_logits, d_inp, d_tgt,
                                                         lr, b1, b2, bc1, bc2, eps, V, BL, 0);
            
            cudaStreamSynchronize(stream);
            
            if ((w + 1) % 50 == 0 || w == 0) {
                auto t_now = std::chrono::steady_clock::now();
                double el = std::chrono::duration<double>(t_now - t_start).count();
                printf("  ep%d win%d/%d loss=%.4f avg=%.4f (%.1fs)\n",
                       epoch+1, w+1, n_win, loss, total_loss/total_windows, el);
            }
        }
        printf("Epoch %d avg_loss=%.4f\n", epoch+1, total_loss / n_win);
    }
    
    // Copy weights back
    cudaMemcpy(m.W, d_W, Wsz*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(m.Wm, d_Wm, Wsz*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(m.Wv, d_Wv, Wsz*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(m.Wh, d_Wh, WhSz*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(m.Whm, d_Whm, WhSz*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(m.Whv, d_Whv, WhSz*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(m.Wbi, d_Wbi, WbiSz*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(m.Wbim, d_Wbim, WbiSz*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(m.Wbiv, d_Wbiv, WbiSz*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(m.q1, d_q1, Q1Sz, cudaMemcpyDeviceToHost);
    
    save_model(argv[3], m);
    printf("[V21-Phase5-GPU] Saved to %s (step=%d)\n", argv[3], m.step);
    
    cudaFree(d_W); cudaFree(d_Wm); cudaFree(d_Wv); cudaFree(d_Wh); cudaFree(d_Whm); cudaFree(d_Whv);
    cudaFree(d_Wbi); cudaFree(d_Wbim); cudaFree(d_Wbiv);
    cudaFree(d_W_sgl_gate); cudaFree(d_W_sgl_up); cudaFree(d_W_sgl_out);
    cudaFree(d_b_sgl_gate); cudaFree(d_b_sgl_up); cudaFree(d_q1);
    cudaFree(d_inp); cudaFree(d_tgt); cudaFree(d_trit_f); cudaFree(d_hash_f);
    cudaFree(d_state); cudaFree(d_gate_z); cudaFree(d_up_z); cudaFree(d_logits); cudaFree(d_probs); cudaFree(d_d_logits);
    cudaFree(d_h0); cudaFree(d_h1); cudaFree(d_h2); cudaFree(d_h3);
    cublasDestroy(handle);
    return 0;
}
