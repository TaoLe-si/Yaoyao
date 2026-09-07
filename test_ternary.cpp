// test_ternary.cpp
// Ternary {-1, 0, +1} design with pure-addition operations.
// Validates CPU-friendly storage and computation for the hash bucket table.
//
// Compile: clang++ -O2 -std=c++17 -march=native -o test_ternary.exe test_ternary.cpp

#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <chrono>
#include <random>

// ============================================================================
// PACKED TERNARY VECTOR
// 4 trits per byte, 2 bits each:
//   value  0  -> code 0b00
//   value +1  -> code 0b01
//   value -1  -> code 0b10
// ============================================================================

class TritVec {
public:
    std::vector<uint8_t> bytes;
    size_t D;

    explicit TritVec(size_t D) : D(D), bytes((D + 3) / 4, 0) {}

    int get(size_t i) const {
        size_t byte_idx = i >> 2;          // / 4
        size_t bit_off  = (i & 3) << 1;    // % 4 then * 2
        uint8_t code = (bytes[byte_idx] >> bit_off) & 0b11;
        if (code == 0b01) return  1;
        if (code == 0b10) return -1;
        return 0;
    }

    void set(size_t i, int v) {
        size_t byte_idx = i >> 2;
        size_t bit_off  = (i & 3) << 1;
        uint8_t code = (v == 1) ? 0b01 : (v == -1) ? 0b10 : 0b00;
        uint8_t mask = 0b11 << bit_off;
        bytes[byte_idx] = (bytes[byte_idx] & ~mask) | (code << bit_off);
    }

    size_t bytes_used() const { return bytes.size(); }
};

// Unpacked version: int8 per trit (1 byte per trit, easier for arithmetic)
using UnpackedTrits = std::vector<int8_t>;

// ============================================================================
// TESTS
// ============================================================================

void test_storage_roundtrip() {
    std::cout << "\n[1] Packed storage roundtrip\n";
    TritVec v(10);
    int values[10] = {1, -1, 0, 1, 1, -1, 0, 0, 1, -1};
    for (int i = 0; i < 10; ++i) v.set(i, values[i]);
    
    std::cout << "  Original: ";
    for (int i = 0; i < 10; ++i) std::cout << values[i] << " ";
    std::cout << "\n  Packed bytes used: " << v.bytes_used() 
              << " (vs 10 bytes if unpacked, vs 40 bytes if float)\n";
    
    bool ok = true;
    for (int i = 0; i < 10; ++i) {
        if (v.get(i) != values[i]) ok = false;
    }
    std::cout << "  Roundtrip " << (ok ? "OK" : "FAILED") << "\n";
}

void test_element_add() {
    std::cout << "\n[2] Element-wise add (a + b), result in {-2,-1,0,1,2}\n";
    int a[] = {1, -1, 0, 1, -1, 0, 1, 0};
    int b[] = {1, 1, -1, 0, 1, -1, -1, 0};
    std::cout << "  a: "; for (int x : a) std::cout << std::setw(2) << x << " "; std::cout << "\n";
    std::cout << "  b: "; for (int x : b) std::cout << std::setw(2) << x << " "; std::cout << "\n";
    std::cout << "  a+b:";
    for (size_t i = 0; i < 8; ++i) std::cout << std::setw(3) << (a[i] + b[i]) << " ";
    std::cout << "\n";
}

// Show that ternary "multiplication" is just sign agreement:
//   (+1)*(+1) = +1  -> same sign, contribute +1
//   (-1)*(-1) = +1  -> same sign, contribute +1
//   (+1)*(-1) = -1  -> opposite, contribute -1
//   anything * 0 = 0 -> skip
void test_sign_agreement() {
    std::cout << "\n[3] Sign agreement table (replaces multiplication)\n";
    int a_vals[] = {1, 1, 1, -1, -1, 0, 0, 0};
    int b_vals[] = {1, -1, 0, 1, 0, 1, -1, 0};
    std::cout << "  a \\ b |  +1  -1   0\n";
    std::cout << "  ------+--------------\n";
    std::cout << "  +1    | " 
              << "  +1   -1   0\n";
    std::cout << "  -1    |   -1   +1   0\n";
    std::cout << "   0    |    0    0   0\n";
    std::cout << "  \n";
    std::cout << "  Rule: same_sign -> +1, opposite_sign -> -1, any_zero -> 0\n";
    std::cout << "  Implementation: pure comparison + integer add/sub, NO multiply.\n";
}

// Pure-addition dot product using packed storage
int dot_packed(const TritVec& a, const TritVec& b, size_t D) {
    int sum = 0;
    for (size_t i = 0; i < D; ++i) {
        int ai = a.get(i);
        int bi = b.get(i);
        if (ai == 0 || bi == 0) continue;
        if (ai == bi) sum += 1;
        else           sum -= 1;
    }
    return sum;
}

// Pure-addition dot product using unpacked storage (faster)
int dot_unpacked(const UnpackedTrits& a, const UnpackedTrits& b, size_t D) {
    int sum = 0;
    for (size_t i = 0; i < D; ++i) {
        int ai = a[i];
        int bi = b[i];
        if (ai == 0 || bi == 0) continue;
        if (ai == bi) sum += 1;
        else           sum -= 1;
    }
    return sum;
}

void test_dot_correctness() {
    std::cout << "\n[4] Dot product correctness check\n";
    const size_t D = 8;
    UnpackedTrits a = {1, -1, 1, 0, -1, 1, 0, 1};
    UnpackedTrits b = {1, 1, -1, 0, 1, -1, 0, 1};
    
    // Manual: count of same-sign nonzero minus count of opposite-sign nonzero
    // a[0]=1, b[0]=1 -> +1
    // a[1]=-1, b[1]=1 -> -1
    // a[2]=1, b[2]=-1 -> -1
    // a[3]=0 -> 0
    // a[4]=-1, b[4]=1 -> -1
    // a[5]=1, b[5]=-1 -> -1
    // a[6]=0 -> 0
    // a[7]=1, b[7]=1 -> +1
    // sum = 1-1-1-1-1+1 = -2
    int expected = -2;
    
    int got = dot_unpacked(a, b, D);
    std::cout << "  Expected: " << expected << ", Got: " << got 
              << (got == expected ? " (OK)" : " (FAILED!)") << "\n";
}

// Float reference for comparison
float dot_float(const std::vector<float>& a, const std::vector<float>& b, size_t D) {
    float sum = 0;
    for (size_t i = 0; i < D; ++i) sum += a[i] * b[i];
    return sum;
}

void test_memory_compression() {
    std::cout << "\n[5] Memory compression (hash bucket pool, 4096 buckets x 64 dims)\n";
    const int H = 4096;
    const int D = 64;
    
    size_t float_bytes = H * D * sizeof(float);
    size_t trit_packed_bytes = H * ((D + 3) / 4);   // 16 bytes per bucket
    size_t trit_unpacked_bytes = H * D * sizeof(int8_t); // 64 bytes per bucket
    
    std::cout << "  Float (32-bit):     " << std::setw(8) << float_bytes << " bytes  ("
              << float_bytes/1024.0 << " KB)\n";
    std::cout << "  Trit (unpacked):    " << std::setw(8) << trit_unpacked_bytes << " bytes  ("
              << trit_unpacked_bytes/1024.0 << " KB)\n";
    std::cout << "  Trit (packed 2bit): " << std::setw(8) << trit_packed_bytes << " bytes  ("
              << trit_packed_bytes << " B = " << trit_packed_bytes/1024.0 << " KB)\n";
    std::cout << "  Compression vs float: " 
              << std::fixed << std::setprecision(1)
              << (float_bytes / (double)trit_packed_bytes) << "x smaller\n";
    std::cout << "  Triton packed pool easily fits in L1 cache (typical 32-64 KB).\n";
}

// Performance benchmark: dot product on random 64-dim ternary vectors
void benchmark_dot() {
    std::cout << "\n[6] PERFORMANCE: dot product (64-dim, 1M random vectors)\n";
    const size_t D = 64;
    const int N = 1000000;
    
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> trit(-1, 1);
    std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
    
    // Generate test data
    std::vector<UnpackedTrits> trits_a(N, UnpackedTrits(D));
    std::vector<UnpackedTrits> trits_b(N, UnpackedTrits(D));
    std::vector<std::vector<float>> floats_a(N, std::vector<float>(D));
    std::vector<std::vector<float>> floats_b(N, std::vector<float>(D));
    
    for (int i = 0; i < N; ++i) {
        for (size_t d = 0; d < D; ++d) {
            int t = trit(rng);
            trits_a[i][d] = (int8_t)t;
            trits_b[i][d] = (int8_t)trit(rng);
            floats_a[i][d] = (float)t;
            floats_b[i][d] = (float)trits_b[i][d];
        }
    }
    
    // Float version
    volatile int sum_f = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        float r = dot_float(floats_a[i], floats_b[i], D);
        sum_f += (int)r;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto us_float = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    // Trit version (unpacked int8)
    volatile int sum_t = 0;
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        sum_t += dot_unpacked(trits_a[i], trits_b[i], D);
    }
    t1 = std::chrono::high_resolution_clock::now();
    auto us_trit = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    // Trit version (packed 2-bit, with get() overhead)
    // Pre-pack
    std::vector<TritVec> packed_a(N, TritVec(D));
    std::vector<TritVec> packed_b(N, TritVec(D));
    for (int i = 0; i < N; ++i) {
        for (size_t d = 0; d < D; ++d) {
            packed_a[i].set(d, trits_a[i][d]);
            packed_b[i].set(d, trits_b[i][d]);
        }
    }
    volatile int sum_p = 0;
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        sum_p += dot_packed(packed_a[i], packed_b[i], D);
    }
    t1 = std::chrono::high_resolution_clock::now();
    auto us_packed = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    std::cout << "  D=" << D << " N=" << N << "\n";
    std::cout << "  ---- Float (multiply) ----\n";
    std::cout << "  Time: " << us_float << " us = " << us_float/1000.0 << " ms\n";
    std::cout << "  Throughput: " << (long long)N * 1000000 / us_float << " dots/sec\n";
    std::cout << "  ---- Trit unpacked (int8 + add) ----\n";
    std::cout << "  Time: " << us_trit << " us = " << us_trit/1000.0 << " ms\n";
    std::cout << "  Throughput: " << (long long)N * 1000000 / us_trit << " dots/sec\n";
    std::cout << "  ---- Trit packed 2-bit (int8 + add, with bit unpacking) ----\n";
    std::cout << "  Time: " << us_packed << " us = " << us_packed/1000.0 << " ms\n";
    std::cout << "  Throughput: " << (long long)N * 1000000 / us_packed << " dots/sec\n";
    std::cout << "  ---- Speedups ----\n";
    std::cout << "  Trit unpacked vs Float: " << std::fixed << std::setprecision(2) 
              << (double)us_float / us_trit << "x\n";
    std::cout << "  Trit packed vs Float:   " << std::fixed << std::setprecision(2) 
              << (double)us_float / us_packed << "x\n";
    std::cout << "  (sum_f=" << sum_f << " sum_t=" << sum_t << " sum_p=" << sum_p << ")\n";
}

// Real-world scenario: lookup 3 buckets and dot-product with a query vector
void benchmark_hash_lookup_dot() {
    std::cout << "\n[7] REAL-WORKLOAD: hash bucket lookup + dot product (the core inference path)\n";
    const int H = 4096;
    const int D = 64;
    const int K = 3;
    const int N = 100000;  // tokens to process
    
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> trit(-1, 1);
    std::uniform_int_distribution<int> idist(0, 49999);
    
    // Float pool
    std::vector<std::vector<float>> pool_f(H, std::vector<float>(D));
    for (int i = 0; i < H; ++i)
        for (int d = 0; d < D; ++d) pool_f[i][d] = (float)trit(rng);
    
    // Trit pool (packed)
    std::vector<TritVec> pool_t(H, TritVec(D));
    for (int i = 0; i < H; ++i)
        for (int d = 0; d < D; ++d) pool_t[i].set(d, (i*31 + d*7) % 3 - 1);
    
    // Token IDs
    std::vector<int> tokens(N);
    for (auto& t : tokens) t = idist(rng);
    
    auto h1 = [](int id) { return id % 4096; };
    auto h2 = [](int id) { return (id * 3 + 1) % 4096; };
    auto h3 = [](int id) { return (id * 7 + 5) % 4096; };
    
    // ----- Float version: gather 3 buckets, sum into embedding, then with a query -----
    std::vector<float> query_f(D);
    for (auto& v : query_f) v = (float)trit(rng);
    
    volatile float sum_f = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < N; ++t) {
        int id = tokens[t];
        std::array<float, 64> emb;
        for (size_t d = 0; d < D; ++d) emb[d] = 0;
        for (int d = 0; d < D; ++d) {
            emb[d] = pool_f[h1(id)][d] + pool_f[h2(id)][d] + pool_f[h3(id)][d];
        }
        // dot with query
        float s = 0;
        for (int d = 0; d < D; ++d) s += emb[d] * query_f[d];
        sum_f += s;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto us_float = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    // ----- Trit version: gather 3 buckets (packed), sum into embedding, dot with query -----
    // For dot with float query, we'd need to convert. Let's do trit-query for fair compare.
    UnpackedTrits query_t(D);
    for (size_t d = 0; d < D; ++d) query_t[d] = (int8_t)trit(rng);
    
    volatile int sum_t = 0;
    t0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < N; ++t) {
        int id = tokens[t];
        const TritVec& b1 = pool_t[h1(id)];
        const TritVec& b2 = pool_t[h2(id)];
        const TritVec& b3 = pool_t[h3(id)];
        int s = 0;
        for (int d = 0; d < D; ++d) {
            int a1 = b1.get(d), a2 = b2.get(d), a3 = b3.get(d);
            // Sum the three trits -> value in {-3,-2,-1,0,1,2,3}
            int emb = a1 + a2 + a3;
            int q = query_t[d];
            // dot contribution via sign agreement
            if (emb == 0 || q == 0) continue;
            // For multi-trit emb, clamp to {-1,+1} by sign
            int es = (emb > 0) ? 1 : -1;
            if (es == q) s += 1;
            else s -= 1;
        }
        sum_t += s;
    }
    t1 = std::chrono::high_resolution_clock::now();
    auto us_trit = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    std::cout << "  H=" << H << " D=" << D << " K=" << K << " N=" << N << "\n";
    std::cout << "  ---- Float version (with multiply) ----\n";
    std::cout << "  Time: " << us_float << " us = " << us_float/1000.0 << " ms\n";
    std::cout << "  Throughput: " << (long long)N * 1000000 / us_float << " tokens/sec\n";
    std::cout << "  ---- Trit version (pure addition) ----\n";
    std::cout << "  Time: " << us_trit << " us = " << us_trit/1000.0 << " ms\n";
    std::cout << "  Throughput: " << (long long)N * 1000000 / us_trit << " tokens/sec\n";
    std::cout << "  Speedup: " << std::fixed << std::setprecision(2) 
              << (double)us_float / us_trit << "x\n";
    std::cout << "  (sum_f=" << sum_f << " sum_t=" << sum_t << ")\n";
}

int main() {
    std::cout << "==========================================================\n";
    std::cout << "  Ternary {-1, 0, +1} Test (pure-addition design)\n";
    std::cout << "==========================================================\n";
    
    test_storage_roundtrip();
    test_element_add();
    test_sign_agreement();
    test_dot_correctness();
    test_memory_compression();
    benchmark_dot();
    benchmark_hash_lookup_dot();
    
    std::cout << "\n==========================================================\n";
    std::cout << "  All tests complete.\n";
    std::cout << "==========================================================\n";
    return 0;
}
