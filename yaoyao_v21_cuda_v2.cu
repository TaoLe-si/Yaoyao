// yaoyao_v21_cuda_v2.cu
// GPU forward + 与 CPU 对比
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <fstream>
#include <vector>

typedef int8_t trit;
typedef uint32_t hash_t;

const int D = 128, V = 1024, V_unit = 1024;
const int B = 128, K = 16, NL = 2;
const int HIDDEN = 192, HASH_FEATURES = 64;
const int Q3_K = 5;
const int BATCH = 16, SEQ = 64;
const int BL = BATCH * SEQ;  // 1024

// 加载模型 (与 v21_full.cpp 完全一致)
struct Model {
    float *W, *W_hash, *Wbi;
    float *aW, *ab, *gW, *gb;
    trit *q1_trits;
    float *q1_query;
    float *W_sgl_gate, *W_sgl_up, *W_sgl_out;
    float *b_sgl_gate, *b_sgl_up;
    int q1_step, step;
};

bool load_model(const char* path, Model& m) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    int magic, version, v;
    f.read((char*)&magic, 4); f.read((char*)&version, 4); f.read((char*)&v, 4);
    if (magic != 0x59414F59) return false;
    
    int Wsz = V * D, WbiSz = V * V, NLDD = NL * D * D;
    int NLD = NL * D, Q3Sz = NL * D;
    int Q1Sz = B * K * D;  // 修正: 不带 NL
    
    m.W = (float*)malloc(Wsz * 4);
    f.read((char*)m.W, Wsz * 4);
    f.ignore(Wsz * 4 * 2);  // W_m, W_v
    
    int WHashSz = V * HASH_FEATURES;
    m.W_hash = (float*)malloc(WHashSz * 4);
    f.read((char*)m.W_hash, WHashSz * 4);
    f.ignore(WHashSz * 4 * 2);
    
    m.Wbi = (float*)malloc(WbiSz * 4);
    f.read((char*)m.Wbi, WbiSz * 4);
    f.ignore(WbiSz * 4 * 2);
    
    for (int kk = 0; kk < Q3_K; ++kk) f.ignore(Q3Sz * 4 * 3);
    
    m.aW = (float*)malloc(NLDD * 4); m.ab = (float*)malloc(NLD * 4);
    m.gW = (float*)malloc(NLDD * 4); m.gb = (float*)malloc(NLD * 4);
    f.read((char*)m.aW, NLDD * 4); f.ignore(NLDD * 4 * 2);
    f.read((char*)m.ab, NLD * 4);  f.ignore(NLD * 4 * 2);
    f.read((char*)m.gW, NLDD * 4); f.ignore(NLDD * 4 * 2);
    f.read((char*)m.gb, NLD * 4);  f.ignore(NLD * 4 * 2);
    
    m.q1_trits = (trit*)malloc(Q1Sz);
    f.read((char*)m.q1_trits, Q1Sz);
    f.ignore(Q1Sz * 4 * 2);  // adam_m, adam_v
    
    f.read((char*)&m.q1_step, 4);
    f.read((char*)&m.step, 4);
    
    int H = HIDDEN;
    m.W_sgl_gate = (float*)malloc(H * H * 4);
    m.b_sgl_gate = (float*)malloc(H * 4);
    m.W_sgl_up = (float*)malloc(H * H * 4);
    m.b_sgl_up = (float*)malloc(H * 4);
    m.W_sgl_out = (float*)malloc(V * H * 4);
    f.read((char*)m.W_sgl_gate, H * H * 4);
    f.read((char*)m.b_sgl_gate, H * 4);
    f.read((char*)m.W_sgl_up, H * H * 4);
    f.read((char*)m.b_sgl_up, H * 4);
    f.read((char*)m.W_sgl_out, V * H * 4);
    
    printf("Loaded step=%d, q1_step=%d\n", m.step, m.q1_step);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("Usage: %s model.bin\n", argv[0]); return 1; }
    Model m;
    if (!load_model(argv[1], m)) { printf("Failed to load\n"); return 1; }
    
    // 验证 weights 加载正确 (打印 first few)
    printf("W[0]=%f, W[100]=%f\n", m.W[0], m.W[100]);
    printf("Wbi[0]=%f\n", m.Wbi[0]);
    printf("W_sgl_gate[0]=%f\n", m.W_sgl_gate[0]);
    printf("W_sgl_out[0]=%f\n", m.W_sgl_out[0]);
    
    return 0;
}
