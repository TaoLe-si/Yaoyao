// yaoyao_v21_cuda_train.cu
// [V21-Phase5] GPU 加速训练 (RTX 4070, sm_89)
// CPU 推理: yaoyao_gen_v21_fast.cpp (格式兼容)
//
// 编译:
//   call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44
//   nvcc -O3 -arch=sm_89 -lcublas -o yaoyao_v21_cuda_train.exe yaoyao_v21_cuda_train.cu
// 运行:
//   yaoyao_v21_cuda_train.exe tinystories_train.txt yaoyao_v21.bin [windows] [lr]
//
// 设计:
// - 读取现有 .bin (yaoyao_v21.bin, ~19MB)
// - 用 GPU 训练若干 windows
// - 写回 .bin (格式与 CPU 版完全一致)

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

typedef int8_t trit;
typedef uint32_t hash_t;

// ============================================================
//  [V] 架构常量 (与 yaoyao_v21_full.cpp 完全一致)
// ============================================================
const int D = 128;                // trit 维度
const int V = 1024;               // vocab
const int V_unit = 1024;          // logits 输出维度
const int B = 128;                // Q1 bucket pool 大小
const int K = 16;                 // Q1 每桶 trits 数
const int NL = 2;                 // layer
const int HIDDEN = 192;           // state 维度 (D + HASH_FEATURES)
const int HASH_FEATURES = 64;     // 4 chains × 16 features
const int Q3_K = 5;               // Q3 kernel (训练时跳过, init=identity)
const int BATCH = 16;
const int SEQ = 64;
const int BL = BATCH * SEQ;       // 1024
const float LR = 0.005f;
const float BETA1 = 0.9f, BETA2 = 0.999f, EPS = 1e-8f;

// 4-chain hash 基数
__device__ const uint32_t HASH_BASES[4]     = {33, 37, 41, 43};
__device__ const uint32_t HASH_INV_BASES[4]  = {0x3e0f83e1u, 0x914c1badu, 0xc18f9c19u, 0x2fa0be83u};
__device__ const uint32_t HASH_ADDS[4]       = {7, 11, 13, 17};
__device__ const uint32_t HASH_SEEDS[4]      = {5381, 5387, 5393, 5399};

// ============================================================
//  [V] Kernels
// ============================================================

// mod3 累积: new_trit = mod3(prev_trit + embed)
__global__ void mod3_accumulate_kernel(
    const float* prev, const float* embed, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int p = (int)prev[i], e = (int)embed[i];
    int s = p + e;
    int r = s % 3;
    if (r > 1) r -= 3;
    if (r < -1) r += 3;
    out[i] = (float)r;
}

// 4-chain rolling hash
__global__ void hash4_forward_kernel(
    hash_t* h0, hash_t* h1, hash_t* h2, hash_t* h3,
    const int* tok, int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    hash_t t = (hash_t)tok[b];
    h0[b] = h0[b] * 33u + t + 7u;
    h1[b] = h1[b] * 37u + t + 11u;
    h2[b] = h2[b] * 41u + t + 13u;
    h3[b] = h3[b] * 43u + t + 17u;
}

// Q1 lookup: x[b,d] = sum_k weights[b,k] * trits[aux.h*K+k, d]
// aux.h 由 hash state 决定 (这里简化为常数 0, 实际是 h%NL)
// 使用 precomputed query/dot/trits
__global__ void q1_lookup_kernel(
    const float* query,        // [BATCH, SEQ, D]  来自 yao_forward
    const float* trits,        // [NL*B*K, D]      Q1 trits
    const float* weights,      // [BATCH*SEQ, K]   attention weights
    const int* h_sel,          // [BATCH*SEQ]      选哪个 layer
    float* out,                // [BATCH, SEQ, D]
    int total_pos, int NL_H, int K_per, int D_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pos = idx / D_dim;
    int d = idx % D_dim;
    if (pos >= total_pos) return;
    int hs = h_sel[pos];
    float sum = 0;
    for (int k = 0; k < K_per; ++k) {
        sum += weights[pos * K_per + k] * trits[(hs * K_per + k) * D_dim + d];
    }
    out[idx] = sum;
}

// alpha gate: alpha[b,d] = sigmoid(aW[l*D*D + d*D + k] * x[b*D+k] + ab[l*D+d])
// 简化: 先算 z = aW · x (BL × NL*D), 然后 sigmoid
// 用 cuBLAS sgemv 做矩阵乘

// SwiGLU forward: hidden[h] = silu(W_gate[h,:]·state + b_gate[h]) * (W_up[h,:]·state + b_up[h])
// 用 cuBLAS sgemv

// Logits: lv[v] = Wbi[prev, v] + W_out[v,:]·hidden
// 用 cuBLAS sgemv

// 暂不实现完整 backward (Phase 5 第一步只做 forward + simple LBFGS-like)
// 先做 forward-only verify

// ============================================================
//  [V] Test main
// ============================================================
int main(int argc, char** argv) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    printf("Yaoyao v21 CUDA Training\n");
    printf("GPU: %s, sm_%d%d\n", p.name, p.major, p.minor);
    printf("Total mem: %zu MB\n", p.totalGlobalMem/1024/1024);
    
    // 实际训练流程: 1) 读 .bin 2) 上传到 GPU 3) 训练 windows 4) 写回 .bin
    // 这个完整版留待 Phase 5.2
    
    printf("\n[V21-Phase5] GPU CUDA hello success!\n");
    printf("Next: implement full forward + backward in Phase 5.1\n");
    
    cublasDestroy(handle);
    return 0;
}
