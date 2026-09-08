#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <fstream>
#include <vector>

typedef int8_t trit; typedef uint32_t hash_t;
const int D = 128, V = 1024, V_unit = 1024;
const int B = 128, K = 16, NL = 2;
const int HIDDEN = 192, HASH_FEATURES = 64;
const int BATCH = 16, SEQ = 64, BL = BATCH * SEQ;

__device__ inline uint32_t jenkins(uint32_t h) {
    h = (h + 0x6d2b79f5u) ^ (h >> 11); h = (h + (h << 3)) ^ (h >> 5);
    h = (h * 0x6d2b79f5u) ^ (h >> 15); h = (h + (h << 2)) + (h << 14); return h ^ (h >> 16);
}

// hash + extract (CPU-compatible: batch-end hash)
__global__ void hash_extract(const int* inp, hash_t* h0, hash_t* h1, hash_t* h2, hash_t* h3,
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

// trit accumulate
__global__ void trit_acc(const trit* q1, const int* inp, float* trit_out, int Bd, int Sd, int Dd) {
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

// concat state
__global__ void concat_st(float* state, const float* trit_f, const float* hash_f, int BL_d) {
    int bt = blockIdx.x; if (bt >= BL_d) return;
    for (int d = 0; d < D; ++d) state[bt*HIDDEN + d] = trit_f[bt*D + d];
    for (int f = 0; f < HASH_FEATURES; ++f) state[bt*HIDDEN + D + f] = hash_f[bt*HASH_FEATURES + f];
}

// silu(gate) * up
__global__ void silu_mul(float* gate, const float* up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i], u = up[i];
    float sig = 1.0f / (1.0f + expf(-g));
    gate[i] = g * sig * u;
}

// add bias to each row of length rows (col-major style)
__global__ void add_bias(float* y, const float* bias, int rows, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] += bias[i % rows];
}

// add Wbi bias per position
__global__ void add_wbi_bias(float* logits, const float* Wbi, const int* inp, int PAD, int BL_d, int Vd) {
    int bt = blockIdx.x;
    if (bt >= BL_d) return;
    int t = bt % SEQ;
    int prev = (t > 0) ? inp[bt - 1] : PAD;
    for (int v = 0; v < Vd; ++v) logits[bt * Vd + v] += Wbi[prev * Vd + v];
}

// Load model
struct Model { 
    float *Wbi, *W_sgl_gate, *W_sgl_up, *W_sgl_out, *b_sgl_gate, *b_sgl_up;
    trit *q1_trits;
};

bool load_model(const char* path, Model& m) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    int magic, version; f.read((char*)&magic, 4); f.read((char*)&version, 4);
    f.read((char*)&version, 4);
    if (magic != 0x59414F59) return false;
    // Skip to Wbi
    int Wsz = V * D;
    f.ignore(Wsz * 4 * 3);  // W + adam_m + adam_v
    int WHashSz = V * HASH_FEATURES;
    f.ignore(WHashSz * 4 * 3);
    int WbiSz = V * V;
    m.Wbi = (float*)malloc(WbiSz * 4);
    f.read((char*)m.Wbi, WbiSz * 4); f.ignore(WbiSz * 4 * 2);
    // Skip Q3, aW, ab, gW, gb
    f.ignore(5 * 2 * D * 4 * 3);
    f.ignore(2 * D * D * 4 * 3 + 2 * D * 4 * 3);
    f.ignore(2 * D * D * 4 * 3 + 2 * D * 4 * 3);
    int Q1Sz = B * K * D;
    m.q1_trits = (trit*)malloc(Q1Sz);
    f.read((char*)m.q1_trits, Q1Sz);
    f.ignore(Q1Sz * 4 * 2);
    // skip step
    int dummy; f.read((char*)&dummy, 4); f.read((char*)&dummy, 4);
    int H = HIDDEN;
    m.W_sgl_gate = (float*)malloc(H*H*4); m.b_sgl_gate = (float*)malloc(H*4);
    m.W_sgl_up = (float*)malloc(H*H*4);   m.b_sgl_up = (float*)malloc(H*4);
    m.W_sgl_out = (float*)malloc(V*H*4);
    f.read((char*)m.W_sgl_gate, H*H*4); f.read((char*)m.b_sgl_gate, H*4);
    f.read((char*)m.W_sgl_up, H*H*4);   f.read((char*)m.b_sgl_up, H*4);
    f.read((char*)m.W_sgl_out, V*H*4);
    return true;
}

int main(int argc, char** argv) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    if (argc < 3) return 1;
    Model m;
    if (!load_model(argv[1], m)) { printf("load failed\n"); return 1; }
    
    // Load CPU dump
    std::ifstream fd(argv[2], std::ios::binary);
    int magic, n_inp; fd.read((char*)&magic, 4); fd.read((char*)&n_inp, 4);
    std::vector<int> inputs(n_inp);
    fd.read((char*)inputs.data(), 4 * n_inp);
    std::vector<float> cpu_logits(BL * V_unit);
    fd.read((char*)cpu_logits.data(), 4 * BL * V_unit);
    std::vector<float> cpu_trit(BL * D);
    fd.read((char*)cpu_trit.data(), 4 * BL * D);
    std::vector<float> cpu_hash(BL * HASH_FEATURES);
    fd.read((char*)cpu_hash.data(), 4 * BL * HASH_FEATURES);
    fd.close();
    
    cublasHandle_t handle; cublasCreate(&handle);
    cudaEventRecord(start);
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    printf("GPU: %s\n", p.name);
    
    // Upload
    float *d_Wbi, *d_W_sgl_gate, *d_W_sgl_up, *d_W_sgl_out;
    float *d_b_sgl_gate, *d_b_sgl_up;
    trit *d_q1_trits;
    cudaMalloc(&d_Wbi, V*V*4);
    cudaMalloc(&d_W_sgl_gate, HIDDEN*HIDDEN*4);
    cudaMalloc(&d_W_sgl_up, HIDDEN*HIDDEN*4);
    cudaMalloc(&d_W_sgl_out, V*HIDDEN*4);
    cudaMalloc(&d_b_sgl_gate, HIDDEN*4);
    cudaMalloc(&d_b_sgl_up, HIDDEN*4);
    cudaMalloc(&d_q1_trits, B*K*D);
    cudaMemcpy(d_Wbi, m.Wbi, V*V*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_sgl_gate, m.W_sgl_gate, HIDDEN*HIDDEN*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_sgl_up, m.W_sgl_up, HIDDEN*HIDDEN*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_sgl_out, m.W_sgl_out, V*HIDDEN*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_sgl_gate, m.b_sgl_gate, HIDDEN*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_sgl_up, m.b_sgl_up, HIDDEN*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q1_trits, m.q1_trits, B*K*D, cudaMemcpyHostToDevice);
    
    int *d_inp;
    float *d_trit_f, *d_hash_f, *d_state, *d_gate_z, *d_up_z, *d_hidden, *d_logits;
    hash_t *d_h0, *d_h1, *d_h2, *d_h3;
    cudaMalloc(&d_inp, BL*4);
    cudaMalloc(&d_trit_f, BL*D*4);
    cudaMalloc(&d_hash_f, BL*HASH_FEATURES*4);
    cudaMalloc(&d_state, BL*HIDDEN*4);
    cudaMalloc(&d_gate_z, BL*HIDDEN*4);
    cudaMalloc(&d_up_z, BL*HIDDEN*4);
    cudaMalloc(&d_hidden, BL*HIDDEN*4);
    cudaMalloc(&d_logits, BL*V_unit*4);
    cudaMalloc(&d_h0, BATCH*4); cudaMalloc(&d_h1, BATCH*4);
    cudaMalloc(&d_h2, BATCH*4); cudaMalloc(&d_h3, BATCH*4);
    cudaMemcpy(d_inp, inputs.data(), BL*4, cudaMemcpyHostToDevice);
    
    // Step 1: trit_acc
    trit_acc<<<BATCH, 1>>>(d_q1_trits, d_inp, d_trit_f, BATCH, SEQ, D);
    
    // Step 2: hash_extract
    hash_extract<<<BATCH, 1>>>(d_inp, d_h0, d_h1, d_h2, d_h3, d_hash_f, BATCH, SEQ);
    
    // Step 3: concat state
    concat_st<<<BL, 1>>>(d_state, d_trit_f, d_hash_f, BL);
    
    // Step 4: SwiGLU
    float one = 1.0f, zero = 0.0f;
    // gate_z (BL, HIDDEN) row-major = state (BL, HIDDEN) × W_gate^T
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                HIDDEN, BL, HIDDEN, &one,
                d_W_sgl_gate, HIDDEN, d_state, HIDDEN, &zero, d_gate_z, HIDDEN);
    int total = BL * HIDDEN;
    add_bias<<<(total+255)/256, 256>>>(d_gate_z, d_b_sgl_gate, HIDDEN, total);
    
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                HIDDEN, BL, HIDDEN, &one,
                d_W_sgl_up, HIDDEN, d_state, HIDDEN, &zero, d_up_z, HIDDEN);
    add_bias<<<(total+255)/256, 256>>>(d_up_z, d_b_sgl_up, HIDDEN, total);
    
    silu_mul<<<(total+255)/256, 256>>>(d_gate_z, d_up_z, total);
    
    // Dump state and hidden before sgemm
    std::vector<float> gpu_state(BL * HIDDEN), gpu_hidden(BL * HIDDEN);
    cudaMemcpy(gpu_state.data(), d_state, BL*HIDDEN*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_hidden.data(), d_gate_z, total*4, cudaMemcpyDeviceToHost);
    
    printf("GPU state[0, 0..9]: ");
    for (int i = 0; i < 10; i++) printf("%.4f ", gpu_state[i]);
    printf("\n");
    printf("GPU hidden[0, 0..9]: ");
    for (int i = 0; i < 10; i++) printf("%.4f ", gpu_hidden[i]);
    printf("\n");
    
    // CPU state[0, 0..9]
    printf("CPU state[0, 0..9]: ");
    for (int i = 0; i < 5; i++) printf("%.4f ", cpu_trit[i]);
    for (int i = 0; i < 5; i++) printf("%.4f ", cpu_hash[i]);
    printf("\n");
    
    // Compute CPU hidden using GPU state (to isolate sgemm vs CPU matmul)
    // ...
    
    FILE* f_dump = fopen("yaoyao_v21_gpu_state.bin", "wb");
    if (f_dump) {
        fwrite(gpu_state.data(), 4, BL*HIDDEN, f_dump);
        fwrite(gpu_hidden.data(), 4, BL*HIDDEN, f_dump);
        fclose(f_dump);
    }
    
    // Step 5: logits
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                V_unit, BL, HIDDEN, &one,
                d_W_sgl_out, HIDDEN, d_gate_z, HIDDEN, &zero, d_logits, V_unit);
    add_wbi_bias<<<BL, 1>>>(d_logits, d_Wbi, d_inp, 0, BL, V_unit);
    
    cudaDeviceSynchronize();
    
    std::vector<float> gpu_logits(BL * V_unit);
    cudaMemcpy(gpu_logits.data(), d_logits, BL*V_unit*4, cudaMemcpyDeviceToHost);
    
    f_dump = fopen("yaoyao_v21_gpu_logits.bin", "wb");
    if (f_dump) {
        fwrite(gpu_logits.data(), 4, BL*V_unit, f_dump);
        fclose(f_dump);
    }
    
    float max_diff = 0, avg_diff = 0;
    int n_diff = 0;
    for (int i = 0; i < BL * V_unit; ++i) {
        float d = std::abs(cpu_logits[i] - gpu_logits[i]);
        if (d > max_diff) max_diff = d;
        avg_diff += d;
        if (d > 0.001f) n_diff++;
    }
    avg_diff /= (BL * V_unit);
    printf("\nMax diff: %.6f\n", max_diff);
    printf("Avg diff: %.6f\n", avg_diff);
    printf("Diffs > 0.001: %d / %d (%.2f%%)\n", n_diff, BL*V_unit, 100.0f*n_diff/(BL*V_unit));
    
    cudaFree(d_Wbi); cudaFree(d_W_sgl_gate); cudaFree(d_W_sgl_up); cudaFree(d_W_sgl_out);
    cudaFree(d_b_sgl_gate); cudaFree(d_b_sgl_up); cudaFree(d_q1_trits);
    cudaFree(d_inp); cudaFree(d_trit_f); cudaFree(d_hash_f);
    cudaFree(d_state); cudaFree(d_gate_z); cudaFree(d_up_z); cudaFree(d_hidden);
    cudaFree(d_logits);
    cudaFree(d_h0); cudaFree(d_h1); cudaFree(d_h2); cudaFree(d_h3);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("\nGPU forward time: %.3f ms for %d positions\n", ms, BL);
    printf("Throughput: %.1f tokens/sec\n", BL * 1000.0f / ms);
    cublasDestroy(handle);
    free(m.Wbi); free(m.q1_trits);
    free(m.W_sgl_gate); free(m.b_sgl_gate); free(m.W_sgl_up); free(m.b_sgl_up); free(m.W_sgl_out);
    return 0;
}
