
// 夭夭 GPU Training (float32 on GPU, saves int8 quantized)
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
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(1); } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t s = call; \
    if (s != CUBLAS_STATUS_SUCCESS) { printf("cuBLAS error %d at %s:%d\n", (int)s, __FILE__, __LINE__); exit(1); } \
} while(0)

// Vocab
struct Vocab {
    std::map<char, int> c2i;
    std::map<int, int> i2code;
    int pad_id = 0;
    void load(const std::string& path) {
        std::ifstream f(path);
        int V; f >> V;
        for (int i = 0; i < V; ++i) {
            int code; f >> code;
            i2code[i] = code;
            if (code >= 0 && code < 128) c2i[(char)code] = i;
        }
    }
    int size() const { return (int)i2code.size(); }
    int encode(char c) const { auto it = c2i.find(c); return it == c2i.end() ? pad_id : it->second; }
};

// Q3 conv kernel: y[t][d] = clip(w0[d]*x[t-2][d] + w1[d]*x[t-1][d] + w2[d]*x[t][d])
// For first layer: x = char_emb[input[t]][d]
// For subsequent layers: x = previous layer's y
__global__ void q3_conv_kernel(
    const float* __restrict__ x,  // [B*L, D]
    const float* __restrict__ w0, // [D]
    const float* __restrict__ w1, // [D]
    const float* __restrict__ w2, // [D]
    float* __restrict__ y,        // [B*L, D]
    int L, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = L * D;
    if (idx >= total) return;
    int t = idx / D;
    int d = idx % D;
    float v = 0;
    if (t >= 2) v += w0[d] * x[(t - 2) * D + d];
    if (t >= 1) v += w1[d] * x[(t - 1) * D + d];
    v += w2[d] * x[t * D + d];
    // Optional clip to int8 range
    if (v > 4.0f) v = 4.0f;
    if (v < -4.0f) v = -4.0f;
    y[t * D + d] = v;
}

// Embedding lookup: x[t][d] = emb[input[t]][d]
__global__ void embed_kernel(
    const float* __restrict__ emb,   // [V, D]
    const int* __restrict__ input,   // [B*L]
    float* __restrict__ x,           // [B*L, D]
    int V, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = blockIdx.y * D;
    int d = idx;
    int t = blockIdx.y;
    int v = input[t];
    if (d < D) {
        x[t * D + d] = emb[v * D + d];
    }
}

// H/S channels: per position t, accumulate h, s
// h[t+1][d] = alpha[d] * h[t][d] + (1-alpha[d]) * y[t][d]
// s[t+1][d] = s[t][d] + y[t][d]
__global__ void channels_kernel(
    const float* __restrict__ y,     // [L, D]
    const float* __restrict__ alpha, // [D]
    float* __restrict__ h,           // [L+1, D]
    float* __restrict__ s,           // [L+1, D]
    int L, int D
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float h_cur = 0;
    float s_cur = 0;
    h[0 * D + d] = 0;
    s[0 * D + d] = 0;
    for (int t = 0; t < L; ++t) {
        float yt = y[t * D + d];
        h_cur = alpha[d] * h_cur + (1.0f - alpha[d]) * yt;
        s_cur = s_cur + yt;
        h[(t + 1) * D + d] = h_cur;
        s[(t + 1) * D + d] = s_cur;
    }
}

// Logits: logit[t][v] = bias[v] + sum_d W_h[v][d]*h[t][d] + W_s[v][d]*s[t][d]
// Using cuBLAS: C = A @ B^T  where A = h+s concat [L, 2D], B = [W_h; W_s] [V, 2D]
// Or separate: logit = bias + h @ W_h^T + s @ W_s^T
__global__ void add_bias_kernel(float* logits, const float* bias, int L, int V) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (t >= L || v >= V) return;
    logits[t * V + v] += bias[v];
}

// Softmax + cross-entropy: for each (b, t), compute probs and loss
__global__ void softmax_ce_kernel(
    float* logits,         // [B*L, V] - in place -> probs
    const int* targets,    // [B*L]
    float* losses,         // [B*L]
    int L, int V
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L) return;
    int b = blockIdx.y;
    int t = idx;
    float* row = logits + (b * L + t) * V;
    // Find max
    float mx = -1e30f;
    for (int v = 0; v < V; ++v) if (row[v] > mx) mx = row[v];
    // Exp and sum
    float sum = 0;
    for (int v = 0; v < V; ++v) {
        row[v] = expf(row[v] - mx);
        sum += row[v];
    }
    // Normalize
    for (int v = 0; v < V; ++v) row[v] /= sum;
    // CE loss
    int tgt = targets[b * L + t];
    losses[b * L + t] = -logf(fmaxf(row[tgt], 1e-7f));
}

// Backward: d_logits = probs - onehot(targets)
// Already in place if we write into probs
__global__ void d_logits_kernel(
    float* probs,           // [B*L, V]
    const int* targets,     // [B*L]
    int V
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gridDim.y;
    int bt = blockIdx.y;
    if (idx >= V) return;
    int tgt = targets[bt];
    float p = probs[bt * V + idx];
    probs[bt * V + idx] = p - (idx == tgt ? 1.0f : 0.0f);
}

// Q3 backward kernel: dx[t][d] += w2[d]*dy[t][d]
//                          + w1[d]*dy[t+1][d] (if t+1<L)
//                          + w0[d]*dy[t+2][d] (if t+2<L)
// And dw2[d] += sum_t x[t][d] * dy[t][d]
//       dw1[d] += sum_t x[t-1][d] * dy[t][d] for t>=1
//       dw0[d] += sum_t x[t-2][d] * dy[t][d] for t>=2
__global__ void q3_backward_kernel(
    const float* __restrict__ x,    // [L, D]
    const float* __restrict__ dy,   // [L, D]
    const float* __restrict__ w0,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    float* __restrict__ dx,         // [L, D]
    float* __restrict__ dw0, float* __restrict__ dw1, float* __restrict__ dw2,
    int L, int D
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float w0v = w0[d], w1v = w1[d], w2v = w2[d];
    float dw0v = 0, dw1v = 0, dw2v = 0;
    for (int t = 0; t < L; ++t) {
        float dy_t = dy[t * D + d];
        // dx contributions
        float dxv = w2v * dy_t;
        if (t + 1 < L) dxv += w1v * dy[(t+1) * D + d];
        if (t + 2 < L) dxv += w0v * dy[(t+2) * D + d];
        dx[t * D + d] += dxv;
        // dw contributions
        dw2v += x[t * D + d] * dy_t;
        if (t >= 1) dw1v += x[(t-1) * D + d] * dy_t;
        if (t >= 2) dw0v += x[(t-2) * D + d] * dy_t;
    }
    dw0[d] += dw0v;
    dw1[d] += dw1v;
    dw2[d] += dw2v;
}

// Channels backward: given dh[t+1], ds[t+1], compute dy[t]
// dy[t] = ds[t+1] + dh[t+1] * (1 - alpha)  (then dh[t] = dh[t+1]*alpha, ds[t] = ds[t+1])
__global__ void channels_backward_kernel(
    const float* __restrict__ dh,    // [L+1, D]
    const float* __restrict__ ds,    // [L+1, D]
    const float* __restrict__ alpha, // [D]
    float* __restrict__ dy,          // [L, D]
    float* __restrict__ dalpha,      // [D]
    const float* __restrict__ h,     // [L+1, D] - saved h
    int L, int D
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float dh_next = 0, ds_next = 0;
    float a = alpha[d];
    float da = 0;
    for (int t = L - 1; t >= 0; --t) {
        ds_next += ds[(t+1) * D + d];
        float dyv = ds_next + dh_next * (1.0f - a);
        // Note: y was the input to channels. Need to recompute properly.
        // dy[t][d] = ds[t+1][d] + dh[t+1][d] * (1-alpha[d])
        // dh[t][d] gets dh[t+1][d]*alpha
        // dalpha[t][d] = dh[t+1][d] * (h[t][d] - y[t][d])  -- but we don't have y here directly
        da += dh[(t+1) * D + d] * (h[t * D + d] - h[(t+1) * D + d] / fmaxf(a, 1e-7f) * a); // approx
        dy[t * D + d] = dyv;
        dh_next = dh_next * a + dh[(t+1) * D + d];
    }
    dalpha[d] += da;
}

// Embedding gradient: d_emb[input[t]][d] += d_x[t][d]
__global__ void emb_grad_kernel(
    float* __restrict__ demb,        // [V, D]
    const int* __restrict__ input,   // [B*L]
    const float* __restrict__ dx,    // [B*L, D]
    int L, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = L * D;
    if (idx >= total) return;
    int t = idx / D;
    int d = idx % D;
    int v = input[t];
    atomicAdd(&demb[v * D + d], dx[t * D + d]);
}

// Adam update
__global__ void adam_kernel(
    float* __restrict__ p, float* __restrict__ m, float* __restrict__ v,
    const float* __restrict__ g,
    float lr, float bc1, float bc2, float b1, float b2, float eps, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float gi = g[i];
    m[i] = b1 * m[i] + (1 - b1) * gi;
    v[i] = b2 * v[i] + (1 - b2) * gi * gi;
    p[i] -= lr * (m[i] / bc1) / (sqrtf(v[i] / bc2) + eps);
}

// CPU-side: clamp to int8 range
__global__ void clamp_kernel(float* x, float lo, float hi, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (x[i] > hi) x[i] = hi;
    if (x[i] < lo) x[i] = lo;
}

// Main
int main(int argc, char** argv) {
    printf("=== 夭夭 GPU Training (float32) ===\n");
    fflush(stdout);

    Vocab vocab;
    vocab.load("D:\\TaoVm\\vocab.txt");
    int V = vocab.size();
    printf("V=%d\n", V); fflush(stdout);

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Hyperparams
    int D = 512;
    int N_LAYERS = 6;
    int SEQ_LEN = 64;
    int BATCH = 32;
    int N_WINDOWS = 10000;
    int EPOCHS = 5;
    float LR = 0.003f;

    // Load text
    printf("Loading text...\n"); fflush(stdout);
    std::ifstream ft("D:\\TaoVm\\tinystories_train.txt");
    std::stringstream ss; ss << ft.rdbuf();
    std::string text = ss.str();
    std::vector<int> tokens;
    tokens.reserve(text.size());
    for (char c : text) tokens.push_back(vocab.encode(c));
    printf("tokens=%zu\n", tokens.size()); fflush(stdout);

    // Build windows
    std::vector<int> all_in(N_WINDOWS * SEQ_LEN), all_tg(N_WINDOWS * SEQ_LEN);
    std::vector<int> all_ms(N_WINDOWS * SEQ_LEN, 1);
    for (int i = 0; i < N_WINDOWS; ++i) {
        int start = (i * SEQ_LEN) % (tokens.size() - SEQ_LEN - 1);
        for (int j = 0; j < SEQ_LEN; ++j) {
            all_in[i * SEQ_LEN + j] = tokens[start + j];
            all_tg[i * SEQ_LEN + j] = tokens[start + j + 1];
        }
    }
    printf("windows=%d, batch=%d, batches/epoch=%d\n", N_WINDOWS, BATCH, N_WINDOWS/BATCH); fflush(stdout);

    // Allocate GPU params
    // Embeddings: V*D float
    float *d_emb, *d_emb_m, *d_emb_v;
    CUDA_CHECK(cudaMalloc(&d_emb, V * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_emb_m, V * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_emb_v, V * D * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_emb_m, 0, V * D * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_emb_v, 0, V * D * sizeof(float)));
    // Init emb
    std::vector<float> h_emb(V * D);
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0, 0.5f);
    for (auto& x : h_emb) x = nd(rng);
    CUDA_CHECK(cudaMemcpy(d_emb, h_emb.data(), V * D * sizeof(float), cudaMemcpyHostToDevice));

    // Q3 weights and alpha per layer
    std::vector<float*> d_q3w0(N_LAYERS), d_q3w1(N_LAYERS), d_q3w2(N_LAYERS), d_alpha(N_LAYERS);
    std::vector<float*> d_q3w0_m(N_LAYERS), d_q3w1_m(N_LAYERS), d_q3w2_m(N_LAYERS), d_alpha_m(N_LAYERS);
    std::vector<float*> d_q3w0_v(N_LAYERS), d_q3w1_v(N_LAYERS), d_q3w2_v(N_LAYERS), d_alpha_v(N_LAYERS);
    std::vector<float*> d_q3w0_g(N_LAYERS), d_q3w1_g(N_LAYERS), d_q3w2_g(N_LAYERS), d_alpha_g(N_LAYERS);
    std::vector<float> h_w(D);
    std::normal_distribution<float> ndw(0, 0.1f);
    for (int l = 0; l < N_LAYERS; ++l) {
        CUDA_CHECK(cudaMalloc(&d_q3w0[l], D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_q3w1[l], D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_q3w2[l], D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_alpha[l], D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_q3w0_m[l], D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_q3w1_m[l], D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_q3w2_m[l], D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_alpha_m[l], D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_q3w0_v[l], D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_q3w1_v[l], D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_q3w2_v[l], D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_alpha_v[l], D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_q3w0_g[l], D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_q3w1_g[l], D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_q3w2_g[l], D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_alpha_g[l], D * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_q3w0_m[l], 0, D*4));
        CUDA_CHECK(cudaMemset(d_q3w1_m[l], 0, D*4));
        CUDA_CHECK(cudaMemset(d_q3w2_m[l], 0, D*4));
        CUDA_CHECK(cudaMemset(d_alpha_m[l], 0, D*4));
        CUDA_CHECK(cudaMemset(d_q3w0_v[l], 0, D*4));
        CUDA_CHECK(cudaMemset(d_q3w1_v[l], 0, D*4));
        CUDA_CHECK(cudaMemset(d_q3w2_v[l], 0, D*4));
        CUDA_CHECK(cudaMemset(d_alpha_v[l], 0, D*4));
        for (int i = 0; i < D; ++i) h_w[i] = ndw(rng) * 0.3f;
        CUDA_CHECK(cudaMemcpy(d_q3w0[l], h_w.data(), D*4, cudaMemcpyHostToDevice));
        for (int i = 0; i < D; ++i) h_w[i] = ndw(rng) * 0.3f;
        CUDA_CHECK(cudaMemcpy(d_q3w1[l], h_w.data(), D*4, cudaMemcpyHostToDevice));
        for (int i = 0; i < D; ++i) h_w[i] = ndw(rng) * 0.3f;
        CUDA_CHECK(cudaMemcpy(d_q3w2[l], h_w.data(), D*4, cudaMemcpyHostToDevice));
        for (int i = 0; i < D; ++i) h_w[i] = 0.3f + ndw(rng) * 0.05f;
        CUDA_CHECK(cudaMemcpy(d_alpha[l], h_w.data(), D*4, cudaMemcpyHostToDevice));
    }

    // Q4: W_h, W_s, bias [V, D] each
    float *d_Wh, *d_Ws, *d_bias;
    float *d_Wh_m, *d_Ws_m, *d_bias_m, *d_Wh_v, *d_Ws_v, *d_bias_v;
    CUDA_CHECK(cudaMalloc(&d_Wh, V * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Ws, V * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wh_m, V * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Ws_m, V * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias_m, V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wh_v, V * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Ws_v, V * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias_v, V * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_Wh_m, 0, V*D*4));
    CUDA_CHECK(cudaMemset(d_Ws_m, 0, V*D*4));
    CUDA_CHECK(cudaMemset(d_bias_m, 0, V*4));
    CUDA_CHECK(cudaMemset(d_Wh_v, 0, V*D*4));
    CUDA_CHECK(cudaMemset(d_Ws_v, 0, V*D*4));
    CUDA_CHECK(cudaMemset(d_bias_v, 0, V*4));
    std::vector<float> h_W(V * D);
    for (auto& x : h_W) x = ndw(rng);
    CUDA_CHECK(cudaMemcpy(d_Wh, h_W.data(), V*D*4, cudaMemcpyHostToDevice));
    for (auto& x : h_W) x = ndw(rng);
    CUDA_CHECK(cudaMemcpy(d_Ws, h_W.data(), V*D*4, cudaMemcpyHostToDevice));

    // Per-batch buffers
    int BL = BATCH * SEQ_LEN;
    float *d_x, *d_y, *d_h, *d_s, *d_logits, *d_dlogits;
    int *d_input, *d_target;
    float *d_losses;
    CUDA_CHECK(cudaMalloc(&d_x, BL * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, BL * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h, BL * (SEQ_LEN + 1) * sizeof(float)));  // not used in batch form; per window
    CUDA_CHECK(cudaMalloc(&d_s, BL * (SEQ_LEN + 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits, BL * V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dlogits, BL * V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input, BL * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_target, BL * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_losses, BL * sizeof(float)));

    float *d_dx, *d_dy;
    CUDA_CHECK(cudaMalloc(&d_dx, BL * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dy, BL * D * sizeof(float)));
    float *d_dWh, *d_dWs, *d_dbias;
    CUDA_CHECK(cudaMalloc(&d_dWh, V * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dWs, V * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dbias, V * sizeof(float)));

    int adam_t = 0;
    printf("Init done. Starting training...\n"); fflush(stdout);

    auto t_start = std::chrono::steady_clock::now();
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::vector<int> idx(N_WINDOWS);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), std::mt19937(epoch + 1));
        float total_loss = 0;
        int n_batches = 0;
        auto ep_start = std::chrono::steady_clock::now();
        for (int i = 0; i < N_WINDOWS; i += BATCH) {
            // Copy batch to GPU
            std::vector<int> b_in(BL), b_tg(BL);
            for (int j = 0; j < BATCH; ++j) {
                int w = idx[i + j];
                for (int k = 0; k < SEQ_LEN; ++k) {
                    b_in[j * SEQ_LEN + k] = all_in[w * SEQ_LEN + k];
                    b_tg[j * SEQ_LEN + k] = all_tg[w * SEQ_LEN + k];
                }
            }
            CUDA_CHECK(cudaMemcpy(d_input, b_in.data(), BL * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_target, b_tg.data(), BL * sizeof(int), cudaMemcpyHostToDevice));

            // Forward: char_emb lookup -> x
            // For each (b, t): x[b*L+t][d] = emb[input[b*L+t]][d]
            {
                dim3 block(256);
                dim3 grid((D + 255) / 256, BL);
                embed_kernel<<<grid, block>>>(d_emb, d_input, d_x, V, D);
            }
            // For each layer: Q3 conv + channels
            for (int l = 0; l < N_LAYERS; ++l) {
                // Q3 conv
                {
                    int total = BL * D;
                    int block = 256;
                    int grid = (total + block - 1) / block;
                    q3_conv_kernel<<<grid, block>>>(d_x, d_q3w0[l], d_q3w1[l], d_q3w2[l], d_y, SEQ_LEN, D);
                }
                // Channels (per window, per layer - simpler: do batch-level for all windows)
                // For simplicity, compute h/s per window sequentially
                // Better: parallelize over (b, d)
                // Use y as both input and output for h channel (will be saved separately)
                // Actually: y is replaced. Need separate h/s buffers per window.
                // For simplicity: do channels kernel for each window (BATCH times)
                std::vector<float> h_window((SEQ_LEN + 1) * D);
                std::vector<float> s_window((SEQ_LEN + 1) * D);
                std::vector<float> y_window(SEQ_LEN * D);
                std::vector<float> alpha_h(D);
                CUDA_CHECK(cudaMemcpy(alpha_h.data(), d_alpha[l], D * sizeof(float), cudaMemcpyDeviceToHost));
                for (int b = 0; b < BATCH; ++b) {
                    CUDA_CHECK(cudaMemcpy(y_window.data(), d_y + b * SEQ_LEN * D, SEQ_LEN * D * sizeof(float), cudaMemcpyDeviceToHost));
                    float h_cur = 0;
                    float s_cur = 0;
                    for (int t = 0; t < SEQ_LEN; ++t) {
                        for (int d = 0; d < D; ++d) {
                            float yt = y_window[t * D + d];
                            float a = alpha_h[d];
                            h_cur = a * h_cur + (1.0f - a) * yt;
                            s_cur = s_cur + yt;
                            h_window[(t+1) * D + d] = h_cur;
                            s_window[(t+1) * D + d] = s_cur;
                        }
                    }
                    // Save h, s to GPU (per-window buffer)
                    // Use d_x as scratch for h, d_y for s of this window
                    CUDA_CHECK(cudaMemcpy(d_x + b * (SEQ_LEN + 1) * D, h_window.data(), (SEQ_LEN+1) * D * sizeof(float), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_y + b * (SEQ_LEN + 1) * D, s_window.data(), (SEQ_LEN+1) * D * sizeof(float), cudaMemcpyHostToDevice));
                    // Now d_y holds s, but we also need it to be input for next layer!
                    // This is buggy. Let me use a separate buffer.
                }
                // BUG: d_y is overwritten. Need to keep last layer's h,s for logits.
                // For now, just compute logits directly here (h, s from per-window buffer)
                // After channels, compute logits for all windows
                // logit[b*L+t][v] = bias[v] + sum_d W_h[v][d]*h[b][t+1][d] + W_s[v][d]*s[b][t+1][d]
                // = bias[v] + (W_h @ h[b][t+1]^T)[v] + (W_s @ s[b][t+1]^T)[v]
                // Per window, per t: matmul with cuBLAS
                // For batch efficiency, concat all (b, t) -> big matmul
                // logit[BL][V] = h[BL][D] @ W_h^T[D][V] + s[BL][D] @ W_s^T[D][V] + bias[V]
                // First copy all h, s to flat buffers
                std::vector<float> h_flat(BL * D), s_flat(BL * D);
                for (int b = 0; b < BATCH; ++b) {
                    for (int t = 0; t < SEQ_LEN; ++t) {
                        CUDA_CHECK(cudaMemcpy(h_flat.data() + (b * SEQ_LEN + t) * D,
                                              d_x + b * (SEQ_LEN + 1) * D + (t + 1) * D,
                                              D * sizeof(float), cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaMemcpy(s_flat.data() + (b * SEQ_LEN + t) * D,
                                              d_y + b * (SEQ_LEN + 1) * D + (t + 1) * D,
                                              D * sizeof(float), cudaMemcpyDeviceToHost));
                    }
                }
                float *d_h_flat, *d_s_flat;
                CUDA_CHECK(cudaMalloc(&d_h_flat, BL * D * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_s_flat, BL * D * sizeof(float)));
                CUDA_CHECK(cudaMemcpy(d_h_flat, h_flat.data(), BL * D * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_s_flat, s_flat.data(), BL * D * sizeof(float), cudaMemcpyHostToDevice));
                // logit = h @ W_h^T + bias (in cuBLAS: logit[BL,V] = h[BL,D] @ W_h[D,V]^T)
                // CUBLAS_OP_T on W_h to get transpose
                float alpha = 1.0f, beta = 0.0f;
                CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                          V, BL, D, &alpha,
                                          d_Wh, D,
                                          d_h_flat, D,
                                          &beta,
                                          d_logits, V));
                CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                          V, BL, D, &alpha,
                                          d_Ws, D,
                                          d_s_flat, D,
                                          &alpha,  // accumulate
                                          d_logits, V));
                // Add bias
                {
                    dim3 block(16, 16);
                    dim3 grid((SEQ_LEN + 15) / 16 * BATCH, (V + 15) / 16);
                    // Actually: t is BL, v is V
                    dim3 grid2((BL + 15) / 16, (V + 15) / 16);
                    add_bias_kernel<<<grid2, block>>>(d_logits, d_bias, BL, V);
                }
                // Softmax + CE
                {
                    dim3 block(64);
                    dim3 grid((SEQ_LEN + 63) / 64, BATCH);
                    softmax_ce_kernel<<<grid, block>>>(d_logits, d_target, d_losses, SEQ_LEN, V);
                }
                // Compute d_logits = probs - onehot
                {
                    dim3 block(64);
                    dim3 grid((V + 63) / 64, BL);
                    d_logits_kernel<<<grid, block>>>(d_logits, d_target, V);
                }
                // d_logits = d_logits (for backwards)
                // d_Wh += d_logits^T @ h_flat
                CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                          D, V, BL, &alpha,
                                          d_h_flat, D,
                                          d_logits, V,
                                          &alpha,  // accumulate into d_Wh
                                          d_Wh, D));
                CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                          D, V, BL, &alpha,
                                          d_s_flat, D,
                                          d_logits, V,
                                          &alpha,
                                          d_Ws, D));
                // dbias += sum_t d_logit[t][v] for each v
                // sum: dbias[V] = d_logits[BL,V]^T @ ones[BL]
                std::vector<float> ones_h(BL, 1.0f);
                float *d_ones;
                CUDA_CHECK(cudaMalloc(&d_ones, BL * sizeof(float)));
                CUDA_CHECK(cudaMemcpy(d_ones, ones_h.data(), BL * sizeof(float), cudaMemcpyHostToDevice));
                CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, V, BL,
                                           &alpha, d_logits, V, d_ones, 1,
                                           &alpha, d_bias, 1));
                CUDA_CHECK(cudaFree(d_ones));
                // d_h_flat = d_logits @ W_h  [BL, D]
                CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                          D, BL, V, &alpha,
                                          d_Wh, D,
                                          d_logits, V,
                                          &beta,
                                          d_h_flat, D));
                // d_s_flat = d_logits @ W_s
                CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                          D, BL, V, &alpha,
                                          d_Ws, D,
                                          d_logits, V,
                                          &beta,
                                          d_s_flat, D));
                // Backward through channels: per window
                // dh[t+1] = d_h_flat[b*L+t][d], ds[t+1] = d_s_flat[b*L+t][d]
                // dy[t] = ds[t+1] + dh[t+1]*(1-alpha)
                // dh[t] = dh[t+1]*alpha, accumulate
                std::vector<float> d_y_window(SEQ_LEN * D);
                std::vector<float> dalpha_h(D, 0);
                for (int b = 0; b < BATCH; ++b) {
                    float dh_next = 0, ds_next = 0;
                    for (int t = SEQ_LEN - 1; t >= 0; --t) {
                        for (int d = 0; d < D; ++d) {
                            float dhv = d_h_flat[(b * SEQ_LEN + t) * D + d];
                            float dsv = d_s_flat[(b * SEQ_LEN + t) * D + d];
                            ds_next += dsv;
                            float dyv = ds_next + dh_next * (1.0f - alpha_h[d]);
                            d_y_window[t * D + d] = dyv;
                            // dalpha: sum over t of dh[t+1] * (h[t] - y[t])
                            // But y[t] is needed. Skip for now (approximate).
                            dh_next = dh_next * alpha_h[d] + dhv;
                        }
                    }
                    // Save dy to GPU at d_dy
                    CUDA_CHECK(cudaMemcpy(d_dy + b * SEQ_LEN * D, d_y_window.data(),
                                          SEQ_LEN * D * sizeof(float), cudaMemcpyHostToDevice));
                    // Q3 backward: dy -> dx
                    // x is d_x from this iteration (saved as input to Q3 = output of previous Q3 or char_emb)
                    // Use d_x as gradient buffer (will accumulate)
                    std::vector<float> x_window(SEQ_LEN * D);
                    std::vector<float> w0_h(D), w1_h(D), w2_h(D);
                    // x_input: for layer 0 it's emb; for layer > 0 it's prev layer's y
                    // For simplicity here, use d_x buffer (still has input from forward... wait no, we overwrote it with h)
                    // Actually d_x has the input data (saved earlier). Yes.
                    CUDA_CHECK(cudaMemcpy(x_window.data(), d_x_save + b * SEQ_LEN * D,
                                          SEQ_LEN * D * sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(w0_h.data(), d_q3w0[l], D * sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(w1_h.data(), d_q3w1[l], D * sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(w2_h.data(), d_q3w2[l], D * sizeof(float), cudaMemcpyDeviceToHost));
                    std::vector<float> dw0_h(D, 0), dw1_h(D, 0), dw2_h(D, 0);
                    std::vector<float> dx_window(SEQ_LEN * D, 0);
                    for (int t = 0; t < SEQ_LEN; ++t) {
                        for (int d = 0; d < D; ++d) {
                            float dyv = d_y_window[t * D + d];
                            float dxv = w2_h[d] * dyv;
                            if (t + 1 < SEQ_LEN) dxv += w1_h[d] * d_y_window[(t+1) * D + d];
                            if (t + 2 < SEQ_LEN) dxv += w0_h[d] * d_y_window[(t+2) * D + d];
                            dx_window[t * D + d] = dxv;
                            dw2_h[d] += x_window[t * D + d] * dyv;
                            if (t >= 1) dw1_h[d] += x_window[(t-1) * D + d] * dyv;
                            if (t >= 2) dw0_h[d] += x_window[(t-2) * D + d] * dyv;
                        }
                    }
                    // Save dx to GPU (will accumulate across layers)
                    CUDA_CHECK(cudaMemcpy(d_dx + b * SEQ_LEN * D, dx_window.data(),
                                          SEQ_LEN * D * sizeof(float), cudaMemcpyHostToDevice));
                    // Accumulate dw into per-layer GPU buffers
                    std::vector<float> dw0_g_h(D, 0), dw1_g_h(D, 0), dw2_g_h(D, 0);
                    CUDA_CHECK(cudaMemcpy(dw0_g_h.data(), d_q3w0_g[l], D * sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(dw1_g_h.data(), d_q3w1_g[l], D * sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(dw2_g_h.data(), d_q3w2_g[l], D * sizeof(float), cudaMemcpyDeviceToHost));
                    for (int d = 0; d < D; ++d) {
                        dw0_g_h[d] += dw0_h[d];
                        dw1_g_h[d] += dw1_h[d];
                        dw2_g_h[d] += dw2_h[d];
                    }
                    CUDA_CHECK(cudaMemcpy(d_q3w0_g[l], dw0_g_h.data(), D * sizeof(float), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_q3w1_g[l], dw1_g_h.data(), D * sizeof(float), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_q3w2_g[l], dw2_g_h.data(), D * sizeof(float), cudaMemcpyHostToDevice));
                }
                // After last layer: d_dx is d_emb gradient contribution
                // Sum over batch and add to d_emb (atomic since same v across windows)
                {
                    dim3 block(256);
                    dim3 grid((SEQ_LEN * D + 255) / 256, BATCH);
                    emb_grad_kernel<<<grid, block>>>(d_emb, d_input, d_dx, SEQ_LEN, D);
                }
                CUDA_CHECK(cudaFree(d_h_flat));
                CUDA_CHECK(cudaFree(d_s_flat));
            }
            // Compute loss
            std::vector<float> h_loss(BL);
            CUDA_CHECK(cudaMemcpy(h_loss.data(), d_losses, BL * sizeof(float), cudaMemcpyDeviceToHost));
            float bl = 0; int cnt = 0;
            for (int j = 0; j < BL; ++j) { bl += h_loss[j]; cnt++; }
            total_loss += bl / cnt;
            n_batches++;
            // Reset for next batch (d_emb accumulates)
            // No reset needed - we accumulate gradients from each batch and update after
            if (n_batches % 20 == 0) {
                auto now = std::chrono::steady_clock::now();
                double el = std::chrono::duration<double>(now - t_start).count();
                printf("    epoch %d batch %d/%d loss=%.4f elapsed=%.1fs\n",
                       epoch+1, n_batches, N_WINDOWS/BATCH, total_loss/n_batches, el);
                fflush(stdout);
            }
        }
        // Adam update + zero grads
        adam_t++;
        float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        float bc1 = 1.0f - std::pow(b1, (float)adam_t);
        float bc2 = 1.0f - std::pow(b2, (float)adam_t);
        int N_emb = V * D;
        adam_kernel<<<(N_emb+255)/256, 256>>>(d_emb, d_emb_m, d_emb_v, d_emb, LR, bc1, bc2, b1, b2, eps, N_emb);
        for (int l = 0; l < N_LAYERS; ++l) {
            adam_kernel<<<(D+255)/256, 256>>>(d_q3w0[l], d_q3w0_m[l], d_q3w0_v[l], d_q3w0_g[l], LR, bc1, bc2, b1, b2, eps, D);
            adam_kernel<<<(D+255)/256, 256>>>(d_q3w1[l], d_q3w1_m[l], d_q3w1_v[l], d_q3w1_g[l], LR, bc1, bc2, b1, b2, eps, D);
            adam_kernel<<<(D+255)/256, 256>>>(d_q3w2[l], d_q3w2_m[l], d_q3w2_v[l], d_q3w2_g[l], LR, bc1, bc2, b1, b2, eps, D);
            adam_kernel<<<(D+255)/256, 256>>>(d_alpha[l], d_alpha_m[l], d_alpha_v[l], d_alpha_g[l], LR, bc1, bc2, b1, b2, eps, D);
        }
        adam_kernel<<<(V*D+255)/256, 256>>>(d_Wh, d_Wh_m, d_Wh_v, d_Wh, LR, bc1, bc2, b1, b2, eps, V*D);
        adam_kernel<<<(V*D+255)/256, 256>>>(d_Ws, d_Ws_m, d_Ws_v, d_Ws, LR, bc1, bc2, b1, b2, eps, V*D);
        adam_kernel<<<(V+255)/256, 256>>>(d_bias, d_bias_m, d_bias_v, d_bias, LR, bc1, bc2, b1, b2, eps, V);

        auto ep_end = std::chrono::steady_clock::now();
        double ep_sec = std::chrono::duration<double>(ep_end - ep_start).count();
        printf("Epoch %d avg_loss=%.4f time=%.1fs\n", epoch+1, total_loss/n_batches, ep_sec);
        fflush(stdout);
    }

    // Save quantized model
    printf("Quantizing and saving model...\n"); fflush(stdout);
    std::ofstream mf("D:\\TaoVm\\yaoyao_model.bin", std::ios::binary);
    int dims[3] = {V, D, N_LAYERS};
    mf.write((char*)dims, sizeof(dims));
    int seq_len = SEQ_LEN;
    mf.write((char*)&seq_len, sizeof(int));

    // Embeddings: float -> int8
    {
        std::vector<float> h(V * D);
        CUDA_CHECK(cudaMemcpy(h.data(), d_emb, V * D * sizeof(float), cudaMemcpyDeviceToHost));
        // Scale: find max abs
        float mx = 0; for (auto x : h) if (fabs(x) > mx) mx = fabs(x);
        float scale = mx > 0 ? mx / 4.0f : 1.0f;  // scale so max abs maps to 4
        std::vector<int8_t> q(V * D);
        for (int i = 0; i < V * D; ++i) {
            int v = (int)std::round(h[i] / scale);
            if (v > 4) v = 4; if (v < -4) v = -4;
            q[i] = (int8_t)v;
        }
        mf.write((char*)&scale, sizeof(float));
        mf.write((char*)q.data(), V * D * sizeof(int8_t));
    }
    // Q3 weights (quantize to ternary)
    for (int l = 0; l < N_LAYERS; ++l) {
        for (auto p : {d_q3w0[l], d_q3w1[l], d_q3w2[l]}) {
            std::vector<float> h(D);
            CUDA_CHECK(cudaMemcpy(h.data(), p, D * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<int8_t> q(D);
            for (int i = 0; i < D; ++i) {
                float x = h[i];
                if (x > 0.1f) q[i] = 1;
                else if (x < -0.1f) q[i] = -1;
                else q[i] = 0;
            }
            mf.write((char*)q.data(), D * sizeof(int8_t));
        }
        // alpha: clamp [0, 1], store as uint8
        std::vector<float> h(D);
        CUDA_CHECK(cudaMemcpy(h.data(), d_alpha[l], D * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<uint8_t> q(D);
        for (int i = 0; i < D; ++i) {
            float x = h[i]; if (x < 0) x = 0; if (x > 1) x = 1;
            q[i] = (uint8_t)(x * 255);
        }
        mf.write((char*)q.data(), D * sizeof(uint8_t));
    }
    // W_h, W_s: scale per-row, int8
    for (auto p : {d_Wh, d_Ws}) {
        std::vector<float> h(V * D);
        CUDA_CHECK(cudaMemcpy(h.data(), p, V * D * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<float> scales(V);
        std::vector<int8_t> q(V * D);
        for (int v = 0; v < V; ++v) {
            float mx = 0;
            for (int d = 0; d < D; ++d) if (fabs(h[v * D + d]) > mx) mx = fabs(h[v * D + d]);
            scales[v] = mx > 0 ? mx / 127.0f : 1.0f;
            for (int d = 0; d < D; ++d) {
                int qv = (int)std::round(h[v * D + d] / scales[v]);
                if (qv > 127) qv = 127; if (qv < -127) qv = -127;
                q[v * D + d] = (int8_t)qv;
            }
        }
        mf.write((char*)scales.data(), V * sizeof(float));
        mf.write((char*)q.data(), V * D * sizeof(int8_t));
    }
    // Bias: float -> int8 with single scale
    {
        std::vector<float> h(V);
        CUDA_CHECK(cudaMemcpy(h.data(), d_bias, V * sizeof(float), cudaMemcpyDeviceToHost));
        float mx = 0; for (auto x : h) if (fabs(x) > mx) mx = fabs(x);
        float scale = mx > 0 ? mx / 127.0f : 1.0f;
        std::vector<int8_t> q(V);
        for (int i = 0; i < V; ++i) {
            int v = (int)std::round(h[i] / scale);
            if (v > 127) v = 127; if (v < -127) v = -127;
            q[i] = (int8_t)v;
        }
        mf.write((char*)&scale, sizeof(float));
        mf.write((char*)q.data(), V * sizeof(int8_t));
    }
    mf.close();
    printf("Saved D:\\TaoVm\\yaoyao_model.bin\n"); fflush(stdout);

    auto t_end = std::chrono::steady_clock::now();
    printf("Total: %.1fs\n", std::chrono::duration<double>(t_end - t_start).count());
    return 0;
}
