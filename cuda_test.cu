
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void test_kernel(float* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 1024) x[i] = x[i] * 2.0f;
}

int main() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    printf("CUDA err: %s\n", cudaGetErrorString(err));
    printf("Device count: %d\n", count);
    if (count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("GPU: %s\n", prop.name);
        printf("Compute: %d.%d\n", prop.major, prop.minor);
        printf("Total memory: %zu MB\n", prop.totalGlobalMem / 1024 / 1024);
        printf("SM count: %d\n", prop.multiProcessorCount);
        float* d_x;
        cudaMalloc(&d_x, 1024 * sizeof(float));
        float h_x[1024];
        for (int i = 0; i < 1024; ++i) h_x[i] = i;
        cudaMemcpy(d_x, h_x, 1024 * sizeof(float), cudaMemcpyHostToDevice);
        test_kernel<<<4, 256>>>(d_x);
        cudaMemcpy(h_x, d_x, 1024 * sizeof(float), cudaMemcpyDeviceToHost);
        printf("Test: h_x[10] = %f (expected 20)\n", h_x[10]);
        cudaFree(d_x);
    }
    return 0;
}
