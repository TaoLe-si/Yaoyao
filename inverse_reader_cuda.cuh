#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
// Tao Attention (桃注意力): ternary mixing over exactly recoverable inverse-chain states.
// GPU training can materialize exact prefix states. CPU inference reconstructs
// the same prefixes from node payload; no attention or KV cache is introduced.
__global__ void inverse_reader_mix(const float *prefix, const int8_t *a, float *mixed, int rows,
                                   int seq, int dim, int horizon) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * dim)
        return;
    int n = i / dim, d = i % dim, t = n % seq;
    float sum = 0;
    for (int k = 0; k <= horizon && k <= t; ++k) {
        int w = a[k * dim + d];
        float r = prefix[(n - k) * dim + d];
        if (w == 1)
            sum += r;
        else if (w == -1)
            sum -= r;
    }
    // k=t+1 would access initial zero trit and contributes zero.
    mixed[i] = sum;
}
// One selected coordinate can be evaluated without recomputing reader mixture.
__global__ void inverse_reader_candidate(const float *base, const float *prefix, float *out,
                                         int rows, int seq, int dim, int k, int d, int delta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * dim)
        return;
    int n = i / dim, j = i % dim;
    float x = base[i];
    if (j == d && k <= n % seq)
        x += delta * prefix[(n - k) * dim + d];
    out[i] = x;
}
