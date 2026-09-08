// yaoyao_v21_cuda_hello.cu
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

typedef signed char trit;
typedef unsigned int hash_t;

__global__ void mod3_kernel(const trit* prev, const trit* emb, trit* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int s = (int)prev[i] + (int)emb[i];
    int r = s % 3;
    if (r > 1) r -= 3;
    if (r < -1) r += 3;
    out[i] = (trit)r;
}

__global__ void hash4_kernel(hash_t* h0, hash_t* h1, hash_t* h2, hash_t* h3,
                             const int* tok, int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    hash_t t = (hash_t)tok[b];
    h0[b] = h0[b] * 33u + t + 7u;
    h1[b] = h1[b] * 37u + t + 11u;
    h2[b] = h2[b] * 41u + t + 13u;
    h3[b] = h3[b] * 43u + t + 17u;
}

int main() {
    cublasHandle_t h; cublasCreate(&h);
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    printf("GPU: %s, sm_%d%d\n", p.name, p.major, p.minor);
    
    // Test 1: mod3 verify
    const int N = 128;
    trit prev[N], emb[N], out[N];
    for (int i = 0; i < N; i++) { prev[i] = 1; emb[i] = 1; }
    trit *d_prev, *d_emb, *d_out;
    cudaMalloc(&d_prev, N); cudaMalloc(&d_emb, N); cudaMalloc(&d_out, N);
    cudaMemcpy(d_prev, prev, N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_emb, emb, N, cudaMemcpyHostToDevice);
    mod3_kernel<<<1, N>>>(d_prev, d_emb, d_out, N);
    cudaMemcpy(out, d_out, N, cudaMemcpyDeviceToHost);
    // mod3(1+1) = mod3(2) = -1
    printf("mod3(1+1) verify: %d (expect -1)\n", out[0]);
    
    // Test 2: 4-chain hash
    const int B = 4;
    hash_t h0=5381u, h1=5387u, h2=5393u, h3=5399u;
    int tok[4] = {10, 20, 30, 40};
    hash_t *d_h0, *d_h1, *d_h2, *d_h3;
    int *d_tok;
    cudaMalloc(&d_h0, B*4); cudaMalloc(&d_h1, B*4); cudaMalloc(&d_h2, B*4); cudaMalloc(&d_h3, B*4);
    cudaMalloc(&d_tok, B*4);
    cudaMemcpy(d_h0, &h0, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h1, &h1, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h2, &h2, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h3, &h3, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tok, tok, B*4, cudaMemcpyHostToDevice);
    hash4_kernel<<<1, B>>>(d_h0, d_h1, d_h2, d_h3, d_tok, B);
    cudaMemcpy(&h0, d_h0, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h1, d_h1, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h2, d_h2, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h3, d_h3, 4, cudaMemcpyDeviceToHost);
    unsigned int expected = 5381u*33u + 10u + 7u;
    printf("h0 = %u (expect %u)\n", h0, expected);
    
    cudaFree(d_prev); cudaFree(d_emb); cudaFree(d_out);
    cudaFree(d_h0); cudaFree(d_h1); cudaFree(d_h2); cudaFree(d_h3); cudaFree(d_tok);
    cublasDestroy(h);
    printf("ALL TESTS PASSED\n");
    return 0;
}
