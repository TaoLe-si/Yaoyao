// test_mod3_reversible.cpp
// Test 1: 验证 mod 3 模运算的可逆性
// 核心思想: h_new = (a * h + b * x) mod 3
// 反向: h = a * (h_new - b * x) mod 3

#include <cstdio>
#include <cstdint>
#include <cstring>

typedef int8_t trit;

// Mod 3 操作: 映射到 {-1, 0, +1}
inline trit mod3(int x) {
    int r = x % 3;
    if (r > 1) r -= 3;
    if (r < -1) r += 3;
    return (trit)r;
}

// 前向: h_new = (a * h + b * x) mod 3
inline trit forward_op(trit h, trit x, trit a, trit b) {
    return mod3(a * h + b * x);
}

// 反向: h = a * (h_new - b * x) mod 3
inline trit reverse_op(trit h_new, trit x, trit a, trit b) {
    return mod3(a * (h_new - b * x));
}

void test_case(const char* name, trit a, trit b) {
    printf("\n=== Test Case: %s (a=%d, b=%d) ===\n", name, a, b);
    
    trit h[] = {1, -1, 0, 1, -1, 0, 1, -1};
    trit h_orig[8];
    memcpy(h_orig, h, sizeof(h));
    
    trit x_seq[][8] = {
        {-1, 1, 0, -1, 1, 0, -1, 1},
        {1, -1, 1, 1, -1, 1, 1, -1},
        {-1, 0, 1, -1, 0, 1, -1, 0},
        {0, 1, -1, 0, 1, -1, 0, 1},
        {1, 1, -1, 1, 1, -1, 1, 1},
    };
    int T = 5;
    
    printf("Forward:\n");
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < 8; d++) {
            h[d] = forward_op(h[d], x_seq[t][d], a, b);
        }
    }
    
    printf("  h[T]:   ");
    for (int d = 0; d < 8; d++) printf("%2d ", h[d]);
    printf("\n");
    
    printf("Backward:\n");
    for (int t = T-1; t >= 0; t--) {
        for (int d = 0; d < 8; d++) {
            h[d] = reverse_op(h[d], x_seq[t][d], a, b);
        }
    }
    
    printf("  h[0]:   ");
    for (int d = 0; d < 8; d++) printf("%2d ", h[d]);
    printf("\n");
    
    bool match = true;
    for (int d = 0; d < 8; d++) {
        if (h[d] != h_orig[d]) {
            match = false;
            printf("  Mismatch at d=%d: orig=%d, recovered=%d\n",
                   d, h_orig[d], h[d]);
        }
    }
    
    printf("Original: ");
    for (int d = 0; d < 8; d++) printf("%2d ", h_orig[d]);
    printf("\n");
    printf("Result: %s\n", match ? "PASS" : "FAIL");
}

int main() {
    printf("================================================\n");
    printf("  Test 1: Mod 3 Reversibility Verification\n");
    printf("================================================\n");
    
    test_case("Identity (a=1, b=1)", 1, 1);
    test_case("Negative (a=-1, b=1)", -1, 1);
    test_case("Both negative (a=-1, b=-1)", -1, -1);
    
    printf("\n================================================\n");
    printf("  All tests done\n");
    printf("================================================\n");
    
    return 0;
}
