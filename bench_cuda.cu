
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <random>

int main() {
    printf("=== cuBLAS fp32 matmul benchmark ===\n");
    fflush(stdout);

    cublasHandle_t handle;
    cublasCreate(&handle);

    int V = 142, D = 256, L = 64, B = 32;

    // Allocate float on GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, B*L*D*sizeof(float));
    cudaMalloc(&d_B, D*V*sizeof(float));
    cudaMalloc(&d_C, B*L*V*sizeof(float));

    // Init
    float* h_A = (float*)malloc(B*L*D*sizeof(float));
    float* h_B = (float*)malloc(D*V*sizeof(float));
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0, 0.5f);
    for (int i = 0; i < B*L*D; ++i) h_A[i] = nd(rng);
    for (int i = 0; i < D*V; ++i) h_B[i] = nd(rng);
    cudaMemcpy(d_A, h_A, B*L*D*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, D*V*sizeof(float), cudaMemcpyHostToDevice);

    // Warmup
    float alpha = 1.0f, beta = 0.0f;
    // C[B,L,V] = A[B,L,D] @ B[D,V]
    // Compute per batch: C[b] = A[b] @ B
    for (int b = 0; b < B; ++b) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    V, L, D, &alpha,
                    d_B, V,
                    d_A + b*L*D, D,
                    &beta,
                    d_C + b*L*V, V);
    }
    cudaDeviceSynchronize();

    // Time it
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    int reps = 100;
    for (int rep = 0; rep < reps; ++rep) {
        for (int b = 0; b < B; ++b) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        V, L, D, &alpha,
                        d_B, V,
                        d_A + b*L*D, D,
                        &beta,
                        d_C + b*L*V, V);
        }
    }
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0;
    cudaEventElapsedTime(&ms, s, e);

    double flops = 2.0 * B * L * V * D * reps;
    printf("cuBLAS fp32 (B=%d, L=%d, V=%d, D=%d): %.3f ms / %d reps\n", B, L, V, D, ms, reps);
    printf("  Per batch-call: %.3f us\n", ms*1000.0/reps/B);
    printf("  Per epoch (30K win): %.1f ms\n", ms*30000/(reps*5000));  // if N_WINDOWS=5000, B=32
    printf("  Throughput: %.2f GFLOPs/s\n", flops/(ms*1e6));

    // Big matmul: float32 full BPE-style
    int V2 = 50000, D2 = 1024, L2 = 128;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaMalloc(&d_A, B*L2*D2*sizeof(float));
    cudaMalloc(&d_B, D2*V2*sizeof(float));
    cudaMalloc(&d_C, B*L2*V2*sizeof(float));
    free(h_A); free(h_B);
    h_A = (float*)malloc(B*L2*D2*sizeof(float));
    h_B = (float*)malloc(D2*V2*sizeof(float));
    for (int i = 0; i < B*L2*D2; ++i) h_A[i] = nd(rng);
    for (int i = 0; i < D2*V2; ++i) h_B[i] = nd(rng);
    cudaMemcpy(d_A, h_A, B*L2*D2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, D2*V2*sizeof(float), cudaMemcpyHostToDevice);

    for (int b = 0; b < B; ++b) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    V2, L2, D2, &alpha,
                    d_B, V2,
                    d_A + b*L2*D2, D2,
                    &beta,
                    d_C + b*L2*V2, V2);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(s);
    reps = 30;
    for (int rep = 0; rep < reps; ++rep) {
        for (int b = 0; b < B; ++b) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        V2, L2, D2, &alpha,
                        d_B, V2,
                        d_A + b*L2*D2, D2,
                        &beta,
                        d_C + b*L2*V2, V2);
        }
    }
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    cudaEventElapsedTime(&ms, s, e);
    flops = 2.0 * B * L2 * V2 * D2 * reps;
    printf("\ncuBLAS fp32 BIG (B=%d, L=%d, V=%d, D=%d): %.3f ms / %d reps\n", B, L2, V2, D2, ms, reps);
    printf("  Throughput: %.2f TFLOPs/s\n", flops/(ms*1e9));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B);
    cublasDestroy(handle);
    printf("Done\n");
    fflush(stdout);
    return 0;
}
