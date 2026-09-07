// bench_q2a_sum.cpp
// Performance benchmark: Q2-A + sum + polynomial head.
// Tests scaling across D and SEQ_LEN, and isolates cost of each component.

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>

struct BenchResult {
    double us_per_step;
    double total_ms;
    double tokens_per_sec;
};

// Q2-A only: h update + linear head
BenchResult bench_q2a_only(int D, int seq_len, int iters = 5000) {
    std::vector<int8_t> h_old(D, 0), h_new(D);
    std::vector<int8_t> x_t(D);
    std::vector<float> alpha(D, 0.9f);
    std::vector<float> w(D);
    std::default_random_engine rng(42);
    std::normal_distribution<float> nd(0, 0.1f);
    for (auto& v : w) v = nd(rng);
    std::uniform_int_distribution<int> ud(-1, 1);
    for (auto& v : x_t) v = ud(rng);

    // Warmup
    for (int i = 0; i < 100; ++i) {
        for (int d = 0; d < D; ++d) {
            float v = alpha[d] * h_old[d] + (1 - alpha[d]) * x_t[d];
            int r = std::lroundf(v);
            if (r > 4) r = 4;
            if (r < -4) r = -4;
            h_new[d] = (int8_t)r;
        }
        std::swap(h_old, h_new);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    float acc = 0;
    for (int i = 0; i < iters; ++i) {
        for (int d = 0; d < D; ++d) {
            float v = alpha[d] * h_old[d] + (1 - alpha[d]) * x_t[d];
            int r = std::lroundf(v);
            if (r > 4) r = 4;
            if (r < -4) r = -4;
            h_new[d] = (int8_t)r;
            acc += h_new[d] * w[d];
        }
        std::swap(h_old, h_new);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    double ups = us / iters;
    if (acc == 0) std::cout << "";  // prevent optimization
    return {ups, ups * seq_len / 1000.0, 1e6 / ups};
}

// Q2-A + sum channel + linear head
BenchResult bench_q2a_sum(int D, int seq_len, int iters = 5000) {
    std::vector<int8_t> h_old(D, 0), h_new(D);
    std::vector<int16_t> s_old(D, 0), s_new(D);
    std::vector<int8_t> x_t(D);
    std::vector<float> alpha(D, 0.9f);
    std::vector<float> w_h(D), w_s(D);
    float b = 0;
    std::default_random_engine rng(42);
    std::normal_distribution<float> nd(0, 0.1f);
    for (auto& v : w_h) v = nd(rng);
    for (auto& v : w_s) v = nd(rng);
    std::uniform_int_distribution<int> ud(-1, 1);
    for (auto& v : x_t) v = ud(rng);

    for (int i = 0; i < 100; ++i) {
        for (int d = 0; d < D; ++d) {
            float v = alpha[d] * h_old[d] + (1 - alpha[d]) * x_t[d];
            int r = std::lroundf(v);
            if (r > 4) r = 4;
            if (r < -4) r = -4;
            h_new[d] = (int8_t)r;
            int s = (int)s_old[d] + (int)x_t[d];
            if (s > 64) s = 64;
            if (s < -64) s = -64;
            s_new[d] = (int16_t)s;
        }
        std::swap(h_old, h_new);
        std::swap(s_old, s_new);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    float acc = 0;
    for (int i = 0; i < iters; ++i) {
        for (int d = 0; d < D; ++d) {
            float v = alpha[d] * h_old[d] + (1 - alpha[d]) * x_t[d];
            int r = std::lroundf(v);
            if (r > 4) r = 4;
            if (r < -4) r = -4;
            h_new[d] = (int8_t)r;
            int s = (int)s_old[d] + (int)x_t[d];
            if (s > 64) s = 64;
            if (s < -64) s = -64;
            s_new[d] = (int16_t)s;
            acc += h_new[d] * w_h[d] + s_new[d] * w_s[d] + b;
        }
        std::swap(h_old, h_new);
        std::swap(s_old, s_new);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    double ups = us / iters;
    if (acc == 0) std::cout << "";
    return {ups, ups * seq_len / 1000.0, 1e6 / ups};
}

// Q2-A + sum + POLYNOMIAL head (s, s^2, h*s)
BenchResult bench_q2a_sum_poly(int D, int seq_len, int iters = 5000) {
    std::vector<int8_t> h_old(D, 0), h_new(D);
    std::vector<int16_t> s_old(D, 0), s_new(D);
    std::vector<int8_t> x_t(D);
    std::vector<float> alpha(D, 0.9f);
    std::vector<float> w_h(D), w_s(D), w_s2(D), w_hs(D);
    float b = 0;
    std::default_random_engine rng(42);
    std::normal_distribution<float> nd(0, 0.1f);
    for (auto& v : w_h) v = nd(rng);
    for (auto& v : w_s) v = nd(rng);
    for (auto& v : w_s2) v = nd(rng);
    for (auto& v : w_hs) v = nd(rng);
    std::uniform_int_distribution<int> ud(-1, 1);
    for (auto& v : x_t) v = ud(rng);

    for (int i = 0; i < 100; ++i) {
        for (int d = 0; d < D; ++d) {
            float v = alpha[d] * h_old[d] + (1 - alpha[d]) * x_t[d];
            int r = std::lroundf(v);
            if (r > 4) r = 4;
            if (r < -4) r = -4;
            h_new[d] = (int8_t)r;
            int s = (int)s_old[d] + (int)x_t[d];
            if (s > 64) s = 64;
            if (s < -64) s = -64;
            s_new[d] = (int16_t)s;
        }
        std::swap(h_old, h_new);
        std::swap(s_old, s_new);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    float acc = 0;
    for (int i = 0; i < iters; ++i) {
        for (int d = 0; d < D; ++d) {
            float v = alpha[d] * h_old[d] + (1 - alpha[d]) * x_t[d];
            int r = std::lroundf(v);
            if (r > 4) r = 4;
            if (r < -4) r = -4;
            h_new[d] = (int8_t)r;
            int s = (int)s_old[d] + (int)x_t[d];
            if (s > 64) s = 64;
            if (s < -64) s = -64;
            s_new[d] = (int16_t)s;
            float hd = h_new[d], sd = s_new[d];
            acc += w_h[d] * hd + w_s[d] * sd + w_s2[d] * sd * sd + w_hs[d] * hd * sd;
        }
        std::swap(h_old, h_new);
        std::swap(s_old, s_new);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    double ups = us / iters;
    if (acc == 0) std::cout << "";
    return {ups, ups * seq_len / 1000.0, 1e6 / ups};
}

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Q2-A + Sum + Polynomial head: Performance Benchmark\n";
    std::cout << "================================================================\n\n";

    std::cout << "  Machine: Windows, DDR4 single-channel (~18 GB/s RAM)\n";
    std::cout << "  Compiler: clang++ -O2 -march=native\n\n";

    const int D_LIST[] = {1024, 2048, 4096, 8192};
    const int SEQ_LIST[] = {64, 256, 1024, 4096};

    std::cout << "  ===== PER-STEP LATENCY (Q2-A + sum + poly) =====\n\n";
    std::cout << "  | D    | us/step | ns/dim |\n";
    std::cout << "  |------|---------|--------|\n";
    for (int D : D_LIST) {
        auto r = bench_q2a_sum_poly(D, 256, 3000);
        std::cout << "  | " << std::setw(4) << D
                  << " | " << std::setw(7) << std::fixed << std::setprecision(2) << r.us_per_step
                  << " | " << std::setw(6) << std::fixed << std::setprecision(2) << r.us_per_step * 1000 / D
                  << " |\n";
    }

    std::cout << "\n  ===== COST BREAKDOWN at D=4096 =====\n\n";
    std::cout << "  | Component              | us/step | overhead vs baseline |\n";
    std::cout << "  |------------------------|---------|----------------------|\n";
    auto r_a = bench_q2a_only(4096, 256, 5000);
    auto r_as = bench_q2a_sum(4096, 256, 5000);
    auto r_asp = bench_q2a_sum_poly(4096, 256, 5000);
    std::cout << "  | Q2-A only (h + linear) | " << std::setw(7) << std::fixed << std::setprecision(2) << r_a.us_per_step
              << " | baseline              |\n";
    std::cout << "  | Q2-A + sum (linear)    | " << std::setw(7) << std::fixed << std::setprecision(2) << r_as.us_per_step
              << " | +" << std::setw(5) << std::fixed << std::setprecision(1) << (r_as.us_per_step - r_a.us_per_step) / r_a.us_per_step * 100
              << "%               |\n";
    std::cout << "  | Q2-A + sum + poly      | " << std::setw(7) << std::fixed << std::setprecision(2) << r_asp.us_per_step
              << " | +" << std::setw(5) << std::fixed << std::setprecision(1) << (r_asp.us_per_step - r_a.us_per_step) / r_a.us_per_step * 100
              << "%               |\n";

    std::cout << "\n  ===== SCALING: D x SEQ_LEN (Q2-A + sum + poly) =====\n\n";
    std::cout << "  | D    | SEQ    | us/step | total ms | tokens/s | mem KB |\n";
    std::cout << "  |------|--------|---------|----------|----------|--------|\n";
    for (int D : D_LIST) {
        for (int seq_len : SEQ_LIST) {
            auto r = bench_q2a_sum_poly(D, seq_len, 2000);
            // Memory: h (D bytes) + s (2D) + x_t (D) + alpha (4D) + 4*D*4 (weights)
            double mem = (D + 2 * D + D + 4 * D + 4 * D * 4.0) / 1024.0;
            std::cout << "  | " << std::setw(4) << D
                      << " | " << std::setw(6) << seq_len
                      << " | " << std::setw(7) << std::fixed << std::setprecision(2) << r.us_per_step
                      << " | " << std::setw(8) << std::fixed << std::setprecision(2) << r.total_ms
                      << " | " << std::setw(8) << std::fixed << std::setprecision(0) << r.tokens_per_sec
                      << " | " << std::setw(6) << std::fixed << std::setprecision(1) << mem
                      << " |\n";
        }
    }

    std::cout << "\n  ===== REAL-WORLD EXAMPLES =====\n\n";

    // 27B-class scaling estimate (just inference)
    // Assume 1 layer = 1 step
    auto r_27b = bench_q2a_sum_poly(4096, 1024, 1000);
    std::cout << "  One inference pass for 1K tokens at D=4096:\n";
    std::cout << "    Time:  " << std::fixed << std::setprecision(2) << r_27b.total_ms << " ms\n";
    std::cout << "    Speed: " << std::fixed << std::setprecision(0) << r_27b.tokens_per_sec << " tokens/s\n\n";

    auto r_27b_4k = bench_q2a_sum_poly(4096, 4096, 1000);
    std::cout << "  One inference pass for 4K tokens at D=4096:\n";
    std::cout << "    Time:  " << std::fixed << std::setprecision(2) << r_27b_4k.total_ms << " ms\n";
    std::cout << "    Speed: " << std::fixed << std::setprecision(0) << r_27b_4k.tokens_per_sec << " tokens/s\n\n";

    auto r_27b_8k = bench_q2a_sum_poly(8192, 1024, 1000);
    std::cout << "  One inference pass for 1K tokens at D=8192 (qwen-class):\n";
    std::cout << "    Time:  " << std::fixed << std::setprecision(2) << r_27b_8k.total_ms << " ms\n";
    std::cout << "    Speed: " << std::fixed << std::setprecision(0) << r_27b_8k.tokens_per_sec << " tokens/s\n";

    std::cout << "    Memory: ~" << std::fixed << std::setprecision(0) << (8192.0 * 8 + 8192.0 * 4 * 4) / 1024 << " KB per-token state\n";

    std::cout << "\n================================================================\n";
    return 0;
}
