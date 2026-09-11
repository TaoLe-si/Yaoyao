// R4 一致性验证（doc 24）：训练器（GPU）与解码器（CPU）的激活必须逐位同式。
//
// 检验三件事：
//   1) 前向一致：ds_act_tanh/ds_act_sigmoid（GPU，真实头文件）== fast_tanh/fast_sigmoid（CPU 参考）
//   2) 导数正确：ds_act_tanh_grad 对比双精度中心差分（捕捉手推导数写错）
//   3) 门控更新一致：GPU ds_update 与 CPU fast_gated_update 在同一输入上逐元素一致
//
// 构建：
//   nvcc -O2 -std=c++17 -arch=sm_89 -DNOMINMAX -DTAO_DELTA_MEM -I src \
//        src/act_parity_test.cu -o build/act_parity_test.exe
#define NOMINMAX
#include "dual_state_cuda_resident.cuh"
#include "cpu_fast_activation.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

using namespace tao::dual;

__global__ void k_act(const float* x, float* t, float* s, float* tg, float* sg, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { t[i] = ds_act_tanh(x[i]); s[i] = ds_act_sigmoid(x[i]);
                 tg[i] = ds_act_tanh_grad(x[i]); sg[i] = ds_act_sigmoid_grad(x[i]); }
}
__global__ void k_update(float* st, const float* u, const float* g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) st[i] += ds_act_sigmoid(g[i]) * (ds_act_tanh(u[i]) - st[i]);
}
// Old trainer operator (exact expf/tanhf) - quantifies the defect this change fixes.
__global__ void k_old(const float* x, float* t, float* s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { t[i] = tanhf(x[i]); s[i] = ds_sigmoid(x[i]); }
}

// 双精度参考：与 Padé 式子同式，用于差分校验导数
static double ref_tanh(double x) {
    if (x > 3.0) return 1.0;
    if (x < -3.0) return -1.0;
    return x * (27.0 + x * x) / (27.0 + 9.0 * x * x);
}

int main() {
    const int N = 400001;
    std::vector<float> hx(N);
    { std::mt19937 rng(20260913); std::uniform_real_distribution<float> d(-8.0f, 8.0f);
      for (int i = 0; i < N; ++i) hx[i] = d(rng); }
    // 边界与特殊点必须覆盖：夹紧点 ±3、0、极小值
    const float special[] = {0.f, 1e-8f, -1e-8f, 3.f, -3.f, 2.999999f, -2.999999f, 3.000001f, -3.000001f};
    for (int i = 0; i < 9; ++i) hx[i] = special[i];

    float *dx, *dt, *ds, *dtg, *dsg;
    check(cudaMalloc(&dx, N * 4)); check(cudaMalloc(&dt, N * 4)); check(cudaMalloc(&ds, N * 4));
    check(cudaMalloc(&dtg, N * 4)); check(cudaMalloc(&dsg, N * 4));
    check(cudaMemcpy(dx, hx.data(), N * 4, cudaMemcpyHostToDevice));
    k_act<<<(N + 127) / 128, 128>>>(dx, dt, ds, dtg, dsg, N);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    std::vector<float> ht(N), hs(N), htg(N), hsg(N);
    check(cudaMemcpy(ht.data(), dt, N * 4, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(hs.data(), ds, N * 4, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(htg.data(), dtg, N * 4, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(hsg.data(), dsg, N * 4, cudaMemcpyDeviceToHost));

    // ---- 1) 前向一致（GPU 训练器 vs CPU 解码器） ----
    double mt = 0, ms = 0; int nt = 0, ns = 0;
    for (int i = 0; i < N; ++i) {
        const double a = std::fabs(double(ht[i]) - double(fast_tanh(hx[i])));
        const double b = std::fabs(double(hs[i]) - double(fast_sigmoid(hx[i])));
        if (a > mt) mt = a; if (b > ms) ms = b;
        if (a != 0.0) ++nt; if (b != 0.0) ++ns;
    }
    std::printf("[1] forward tanh    max|GPU-CPU| = %.3e   differing %d/%d\n", mt, nt, N);
    std::printf("[1] forward sigmoid max|GPU-CPU| = %.3e   differing %d/%d\n", ms, ns, N);

    // ---- 2) 导数 vs 双精度中心差分 ----
    double mg = 0, mgs = 0; int bad = 0;
    const double h = 1e-5;
    for (int i = 0; i < N; ++i) {
        const double x = hx[i];
        if (std::fabs(x) > 2.9) continue;           // 夹紧边界附近差分无意义
        const double num = (ref_tanh(x + h) - ref_tanh(x - h)) / (2 * h);
        const double e = std::fabs(double(htg[i]) - num);
        if (e > mg) mg = e;
        if (e > 1e-5) ++bad;
        const double e2 = std::fabs(double(hsg[i]) - 0.5 * num);
        if (e2 > mgs) mgs = e2;
    }
    std::printf("[2] dtanh  max|analytic-fd| = %.3e   over-tol %d\n", mg, bad);
    std::printf("[2] dsigmoid (=0.5*dtanh) max dev = %.3e\n", mgs);

    // ---- 3) 门控更新一致（GPU ds_update vs CPU fast_gated_update） ----
    const int M = 100000;
    std::vector<float> s0(M), u(M), g(M);
    { std::mt19937 rng(4242); std::uniform_real_distribution<float> d(-4.f, 4.f);
      for (int i = 0; i < M; ++i) { s0[i] = d(rng); u[i] = d(rng); g[i] = d(rng); } }
    std::vector<float> cpu_s = s0, cpu_u = u;
    fast_gated_update(cpu_s.data(), cpu_u.data(), g.data(), M);   // CPU 解码器路径
    float *ds_, *du_, *dg_;
    check(cudaMalloc(&ds_, M * 4)); check(cudaMalloc(&du_, M * 4)); check(cudaMalloc(&dg_, M * 4));
    check(cudaMemcpy(ds_, s0.data(), M * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(du_, u.data(), M * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dg_, g.data(), M * 4, cudaMemcpyHostToDevice));
    k_update<<<(M + 127) / 128, 128>>>(ds_, du_, dg_, M);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    std::vector<float> gpu_s(M);
    check(cudaMemcpy(gpu_s.data(), ds_, M * 4, cudaMemcpyDeviceToHost));
    double mu = 0; int nu = 0;
    for (int i = 0; i < M; ++i) { const double e = std::fabs(double(gpu_s[i]) - double(cpu_s[i]));
        if (e > mu) mu = e; if (e != 0.0) ++nu; }
    std::printf("[3] gated update GPU ds_update vs CPU fast_gated_update: max = %.3e  differing %d/%d\n", mu, nu, M);

    // ---- 4) 量化被修复的缺陷：旧训练器算子 vs 解码器算子 ----
    float* dot_; float* dos_;
    check(cudaMalloc(&dot_, N * 4)); check(cudaMalloc(&dos_, N * 4));
    k_old<<<(N + 127) / 128, 128>>>(dx, dot_, dos_, N);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    std::vector<float> hot(N), hos(N);
    check(cudaMemcpy(hot.data(), dot_, N * 4, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(hos.data(), dos_, N * 4, cudaMemcpyDeviceToHost));
    double mo_t = 0, mo_s = 0;
    for (int i = 0; i < N; ++i) {
        mo_t = std::max(mo_t, std::fabs(double(hot[i]) - double(fast_tanh(hx[i]))));
        mo_s = std::max(mo_s, std::fabs(double(hos[i]) - double(fast_sigmoid(hx[i]))));
    }
    std::printf("[4] OLD trainer (exact) vs decoder (fast): tanh %.3e  sigmoid %.3e  <- defect now fixed\n", mo_t, mo_s);

    // 判据：语义等价（远小于旧缺陷），而非跨 ISA 逐位相等。
    // 1 ULP 浮点差异来自 nvcc 的 FMA 收缩，属正常范围；
    // 旧缺陷是 1e-2 量级，两者相差 5 个数量级。
    const double ULP = 1.2e-7;
    const bool forward_ok = (mt <= 4 * ULP && ms <= 4 * ULP);
    const bool update_ok  = (mu <= 64 * ULP);
    const bool grad_ok    = (bad == 0);
    const bool ok = forward_ok && update_ok && grad_ok;
    std::printf("\nULP(1.0)=%.3e  forward_ok=%d update_ok=%d grad_ok=%d\n", ULP, int(forward_ok), int(update_ok), int(grad_ok));
    std::printf("%s\n", ok ? "PARITY_OK trainer and decoder use the same operator (rounding-level agreement only)"
                           : "PARITY_FAIL semantic mismatch; do not train with this build");
    return ok ? 0 : 1;
}
