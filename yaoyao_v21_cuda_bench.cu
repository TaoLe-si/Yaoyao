// yaoyao_v21_cuda_bench.cu - benchmark cuBLAS sgemv
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <chrono>

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    int N = 192;  // HIDDEN (rows)
    int K = 192;  // HIDDEN (cols)
    
    float *d_A, *d_x, *d_y;
    cudaMalloc(&d_A, N*K*4);
    cudaMalloc(&d_x, K*4);
    cudaMalloc(&d_y, N*4);
    
    // 预热
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemv(handle, CUBLAS_OP_N, N, K, &alpha, d_A, N, d_x, 1, &beta, d_y, 1);
    cudaDeviceSynchronize();
    
    // benchmark 1000 次
    const int N_ITERS = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_ITERS; ++i) {
        cublasSgemv(handle, CUBLAS_OP_N, N, K, &alpha, d_A, N, d_x, 1, &beta, d_y, 1);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    double ms = std::chrono::duration<double, std::milli>(end - start).count() / N_ITERS;
    double gflops = 2.0 * N * K / ms / 1e6;
    printf("cuBLAS sgemv (%dxx %d): %.3f ms/iter, %.2f GFLOPs\n", N, K, ms, gflops);
    printf("CPU 对比 (Naive): 192*192*2 = 73,728 ops in ~0.05 ms ≈ 1.5 GFLOPs\n");
    
    // 现在 benchmark 更大的矩阵: V × HIDDEN = 1024 × 192
    N = 1024; K = 192;
    cudaFree(d_A); cudaFree(d_x); cudaFree(d_y);
    cudaMalloc(&d_A, N*K*4);
    cudaMalloc(&d_x, K*4);
    cudaMalloc(&d_y, N*4);
    
    cublasSgemv(handle, CUBLAS_OP_N, N, K, &alpha, d_A, N, d_x, 1, &beta, d_y, 1);
    cudaDeviceSynchronize();
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_ITERS; ++i) {
        cublasSgemv(handle, CUBLAS_OP_N, N, K, &alpha, d_A, N, d_x, 1, &beta, d_y, 1);
    }
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    
    ms = std::chrono::duration<double, std::milli>(end - start).count() / N_ITERS;
    gflops = 2.0 * N * K / ms / 1e6;
    printf("\ncuBLAS sgemv (%dx %d): %.3f ms/iter, %.2f GFLOPs\n", N, K, ms, gflops);
    printf("(V × HIDDEN, logits projection)\n");
    
    // batched: BL × HIDDEN @ HIDDEN, 实际训练中是 BL=1024 个矩阵乘
    int BL = 1024;
    cudaFree(d_A); cudaFree(d_x); cudaFree(d_y);
    cudaMalloc(&d_A, N*K*4);
    cudaMalloc(&d_x, K*BL*4);  // BL 个 vector
    cudaMalloc(&d_y, N*BL*4);
    
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, BL, K, &alpha, d_A, N, d_x, K, &beta, d_y, N);
    cudaDeviceSynchronize();
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, BL, K, &alpha, d_A, N, d_x, K, &beta, d_y, N);
    }
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    
    ms = std::chrono::duration<double, std::milli>(end - start).count() / 100;
    gflops = 2.0 * N * BL * K / ms / 1e6;
    printf("\ncuBLAS sgemm (BL=%d batched %dx %d): %.3f ms/iter, %.2f GFLOPs\n", BL, N, K, ms, gflops);
    printf("(batch sgemm, 全 batch 一次完成)\n");
    
    cudaFree(d_A); cudaFree(d_x); cudaFree(d_y);
    cublasDestroy(handle);
    return 0;
}
