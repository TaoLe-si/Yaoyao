#include <numeric>
#include "self_decoding_node.hpp"
#include "native_reader_checkpoint.hpp"
#define main legacy_cuda_main
#include "yaoyao_v21_cuda_train_stable_ce.cu"
#undef main
#include "inverse_reader_cuda.cuh"
#include <cassert>
#include <algorithm>
__global__ void reader_head(const float *s, const float *g, const float *u, float *h, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * HIDDEN)
        return;
    int n = i / HIDDEN, j = i % HIDDEN;
    float a = 0, b = 0;
    for (int d = 0; d < HIDDEN; ++d) {
        a += g[j * HIDDEN + d] * s[n * HIDDEN + d];
        b += u[j * HIDDEN + d] * s[n * HIDDEN + d];
    }
    h[i] = a / (1 + expf(-a)) * b;
}
__global__ void reader_logits(const float *h, const float *o, const float *bi, const int *ids,
                              float *z, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * V)
        return;
    int n = i / V, v = i % V, prev = n % SEQ ? ids[n - 1] : 0;
    float x = bi[prev * V + v];
    for (int d = 0; d < HIDDEN; ++d)
        x += o[v * HIDDEN + d] * h[n * HIDDEN + d];
    z[i] = x;
}
template <class T> T *dev(const T *p, size_t n) {
    T *q;
    cudaCheck(cudaMalloc(&q, n * sizeof(T)));
    if (p)
        cudaCheck(cudaMemcpy(q, p, n * sizeof(T), cudaMemcpyHostToDevice));
    return q;
}

#include "reader_training_data.hpp"

struct GpuBatch {
    int rows;
    int *ids;
    float *prefix, *hash, *mixed, *state, *gate, *up, *logits;
    std::vector<int> targets;
    GpuBatch(const ReaderBatch &batch, trit *q1) {
        rows = int(batch.input.size());
        targets = batch.target;
        ids = dev(batch.input.data(), rows);
        prefix = dev<float>(nullptr, rows * D);
        hash = dev<float>(nullptr, rows * HASH_FEATURES);
        mixed = dev<float>(nullptr, rows * D);
        state = dev<float>(nullptr, rows * HIDDEN);
        gate = dev<float>(nullptr, rows * HIDDEN);
        up = dev<float>(nullptr, rows * HIDDEN);
        logits = dev<float>(nullptr, rows * V);
        trit_acc_kernel<<<rows / SEQ, 1>>>(q1, ids, prefix, rows / SEQ, SEQ, D);
        kernelCheck();
        hash_extract_kernel<<<rows / SEQ, 1>>>(ids, nullptr, hash, rows / SEQ, SEQ);
        kernelCheck();
    }
    ~GpuBatch() {
        for (void *p : {(void *)ids, (void *)prefix, (void *)hash, (void *)mixed, (void *)state,
                        (void *)gate, (void *)up, (void *)logits})
            cudaFree(p);
    }
};

__global__ void tao_word_rows(float *z, const float *w, const int *ids, const int8_t *g, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * V)
        return;
    int n = i / V, v = i % V;
    float x = z[i];
    for (int k = 1; k < 16 && k <= n % SEQ; ++k) {
        int q = g[ids[n] * 15 + k - 1];
        if (q)
            x += float(q) * (1.0f / 32) * w[ids[n - k] * V + v];
    }
    z[i] = x;
}
int main(int argc, char **argv) {
    if (argc != 8) {
        printf("model train-corpus eval-corpus new-output-prefix coordinates input-reader "
               "start-coordinate\n");
        return 1;
    }
    const std::string output = argv[4];
    if (std::ifstream(output + ".csv").good() || std::ifstream(output + ".reader").good())
        return 2;
    const std::string arg = argv[5];
    if (arg.empty() || arg.find_first_not_of("0123456789") != std::string::npos)
        return 2;
    unsigned rounds = unsigned(std::stoul(arg));
    if (rounds > 4096)
        return 2;
    const unsigned start = unsigned(std::stoul(argv[7]));
    if (start > 4096 || rounds > 4096 - start)
        return 2;
    auto model_sha = file_sha256(argv[1]);
    auto train_sha = file_sha256(argv[2]);
    auto eval_sha = file_sha256(argv[3]);
    if (train_sha == eval_sha)
        throw std::runtime_error("train/eval must be distinct corpora");
    Model m;
    if (!load_model(argv[1], m))
        return 3;
    // Separate files from established prefix/tail split. Fixed diagnostic batch for stage1.
    const auto train_host = read_reader_batch(argv[2], 2000000, 64);
    const auto eval_host = read_reader_batch(argv[3], 0, 16);
    auto q1 = dev(m.q1, B * K * D);
    GpuBatch train(train_host, q1), evaluation(eval_host, q1);
    const auto train_host2 = read_reader_batch(argv[2], 12000000, 64);
    const auto train_host3 = read_reader_batch(argv[2], 42000000, 64);
    const auto train_host4 = read_reader_batch(argv[2], 82000000, 64);
    GpuBatch train2(train_host2, q1), train3(train_host3, q1), train4(train_host4, q1);
    auto wg = dev(m.W_sgl_gate, HIDDEN * HIDDEN);
    auto wu = dev(m.W_sgl_up, HIDDEN * HIDDEN);
    auto wo = dev(m.W_sgl_out, V * HIDDEN);
    auto bi = dev(m.Wbi, V * V);
    std::vector<int8_t> coefficients(17 * D, 0);
    std::fill(coefficients.begin(), coefficients.begin() + D, 1);
    std::ifstream source(argv[6], std::ios::binary | std::ios::ate);
    if (!source || source.tellg() != std::streamoff(4484))
        throw std::runtime_error("RTS1 size");
    source.seekg(0);
    std::vector<unsigned char> raw(4484);
    if (!source.read((char *)raw.data(), raw.size()))
        throw std::runtime_error("RTS1 read");
    auto word = [&](size_t p) {
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i)
            v |= uint32_t(raw[p + i]) << (8 * i);
        return v;
    };
    if (word(0) != 0x31535452u || word(4) != 1 || word(8) != 16 || word(12) != 256 ||
        word(16) != 64)
        throw std::runtime_error("RTS1 contract");
    if (word(4480) != reader_crc32(raw.data(), 4480))
        throw std::runtime_error("RTS1 checksum");
    if (!std::equal(train_sha.begin(), train_sha.end(), raw.begin() + 64))
        throw std::runtime_error("RTS1 corpus");
    // Transfer coefficients from parent model to its adapted descendant, not exact optimizer
    // resume.
    for (size_t i = 0; i < coefficients.size(); ++i) {
        if (raw[128 + i] > 2)
            throw std::runtime_error("RTS1 selector");
        coefficients[i] = int8_t(raw[128 + i]) - 1;
    }
    std::vector<int8_t> gates(V * 15, 0);
    auto dg = dev(gates.data(), gates.size());
    auto da = dev(coefficients.data(), coefficients.size());
    cublasHandle_t handle;
    cublasCheck(cublasCreate(&handle));
    cublasCheck(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    auto forward = [&](GpuBatch &batch, bool reference) {
        cudaCheck(cudaMemcpy(da, coefficients.data(), coefficients.size(), cudaMemcpyHostToDevice));
        inverse_reader_mix<<<(batch.rows * D + 255) / 256, 256>>>(batch.prefix, da, batch.mixed,
                                                                  batch.rows, SEQ, D, 16);
        kernelCheck();
        concat_kernel<<<batch.rows, 1>>>(batch.state, batch.mixed, batch.hash, batch.rows);
        kernelCheck();
        if (reference) {
            reader_head<<<(batch.rows * HIDDEN + 255) / 256, 256>>>(batch.state, wg, wu, batch.gate,
                                                                    batch.rows);
            kernelCheck();
            reader_logits<<<(batch.rows * V + 255) / 256, 256>>>(batch.gate, wo, bi, batch.ids,
                                                                 batch.logits, batch.rows);
            kernelCheck();
        } else {
            float one = 1, zero = 0;
            cublasCheck(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, HIDDEN, batch.rows, HIDDEN,
                                    &one, wg, HIDDEN, batch.state, HIDDEN, &zero, batch.gate,
                                    HIDDEN));
            cublasCheck(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, HIDDEN, batch.rows, HIDDEN,
                                    &one, wu, HIDDEN, batch.state, HIDDEN, &zero, batch.up,
                                    HIDDEN));
            silu_mul_kernel<<<(batch.rows * HIDDEN + 255) / 256, 256>>>(batch.gate, batch.up,
                                                                        batch.rows * HIDDEN);
            kernelCheck();
            cublasCheck(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, V, batch.rows, HIDDEN, &one,
                                    wo, HIDDEN, batch.gate, HIDDEN, &zero, batch.logits, V));
            add_wbi_kernel<<<batch.rows, 1>>>(batch.logits, bi, batch.ids, PAD, batch.rows, V);
            kernelCheck();
        }
        cudaCheck(cudaMemcpy(dg, gates.data(), gates.size(), cudaMemcpyHostToDevice));
        tao_word_rows<<<(batch.rows * V + 255) / 256, 256>>>(batch.logits, bi, batch.ids, dg,
                                                             batch.rows);
        kernelCheck();
        std::vector<float> logits(batch.rows * V);
        cudaCheck(
            cudaMemcpy(logits.data(), batch.logits, logits.size() * 4, cudaMemcpyDeviceToHost));
        return logits;
    };
    auto objective = [&](GpuBatch &batch) {
        auto logits = forward(batch, false);
        double total = 0;
        for (int n = 0; n < batch.rows; ++n) {
            const float *row = logits.data() + n * V;
            double mx = *std::max_element(row, row + V), sum = 0;
            for (int v = 0; v < V; ++v) {
                if (!std::isfinite(row[v]))
                    throw std::runtime_error("nonfinite logits");
                sum += std::exp(double(row[v]) - mx);
            }
            total += mx + std::log(sum) - row[batch.targets[n]];
        }
        return total / batch.rows;
    };
    auto ref = forward(train, true), actual = forward(train, false);
    double error = 0;
    for (size_t i = 0; i < ref.size(); ++i)
        error = std::max(error, double(std::abs(ref[i] - actual[i])));
    printf("GATE cuBLAS vs reference all training positions max_abs=%.9g\n", error);
    if (error > 1e-4)
        return 4;
    std::ofstream log(output + ".csv");
    log << "coordinate,accepted,train_ce,eval_ce\n";
    log.precision(12);
    auto training_objective = [&]() {
        return (objective(train) + objective(train2) + objective(train3) + objective(train4)) / 4.0;
    };
    double current = training_objective();
    const double baseline_eval = objective(evaluation);
    log << "0,0," << current << ',' << baseline_eval << '\n';
    printf("STEP 0 train=%.9f eval=%.9f\n", current, baseline_eval);
    std::vector<int> counts(V, 0), order(V);
    for (auto *host : {&train_host, &train_host2, &train_host3, &train_host4})
        for (int id : host->input)
            ++counts[id];
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return counts[a] > counts[b]; });
    int accepted = 0;
    for (int rank = 0; rank < V; ++rank) {
        int token = order[rank];
        if (counts[token] < 64)
            continue;
        for (int k = 1; k <= 15; ++k) {
            int idx = token * 15 + k - 1;
            int8_t old = gates[idx], chosen = old;
            double best = current;
            for (int q : {-8, -4, -2, -1, 0, 1, 2, 4, 8}) {
                if (q == old)
                    continue;
                int active = 0, magnitude = 0;
                for (int j = 0; j < 15; ++j) {
                    int x = j == k - 1 ? q : gates[token * 15 + j];
                    active += x != 0;
                    magnitude += std::abs(x);
                }
                if (active > 2 || magnitude > 8)
                    continue;
                gates[idx] = q;
                double trial = training_objective();
                if (trial < best) {
                    best = trial;
                    chosen = q;
                }
            }
            gates[idx] = chosen;
            current = best;
            accepted += chosen != old;
        }
        double held = objective(evaluation);
        printf("TOKEN %d count=%d train=%.9f eval=%.9f changes=%d\n", token, counts[token], current,
               held, accepted);
        fflush(stdout);
        log << token << ',' << accepted << ',' << current << ',' << held << '\n';
        log.flush();
    }
    // Verify nonzero gate arithmetic against CPU node recovery independently of head.
    std::vector<int8_t> probe(V * 15, 0);
    for (int id = 0; id < V; ++id) {
        probe[id * 15] = 1;
        probe[id * 15 + 14] = -2;
    }
    cudaCheck(cudaMemcpy(dg, probe.data(), probe.size(), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(evaluation.logits, 0, evaluation.rows * V * 4));
    tao_word_rows<<<(evaluation.rows * V + 255) / 256, 256>>>(evaluation.logits, bi, evaluation.ids,
                                                              dg, evaluation.rows);
    kernelCheck();
    std::vector<float> result(evaluation.rows * V);
    cudaCheck(
        cudaMemcpy(result.data(), evaluation.logits, result.size() * 4, cudaMemcpyDeviceToHost));
    auto code = [&](int id) {
        uint32_t h = uint32_t(id) * 2654435761u;
        h = (h >> 16) ^ h;
        return m.q1 + (h % B) * K * D;
    };
    SelfDecodingNode<16, 256> node;
    double error2 = 0;
    for (int n = 0; n < evaluation.rows; ++n) {
        if (n % SEQ == 0)
            node = {};
        node.push(eval_host.input[n], code);
        auto cur = node;
        std::vector<float> cpu(V, 0);
        for (int k = 0; cur.count; ++k) {
            int id = cur.pop(code);
            for (int v = 0; v < V; ++v) {
                if (k > 0) {
                    int q = probe[eval_host.input[n] * 15 + k - 1];
                    if (q)
                        cpu[v] += float(q) * (1.0f / 32) * m.Wbi[id * V + v];
                }
            }
        }
        for (int v = 0; v < V; ++v)
            error2 = std::max(error2, double(std::abs(cpu[v] - result[n * V + v])));
    }
    printf("NODE GPU full vocabulary max_abs=%.9g\n", error2);
    if (error2 != 0)
        return 8;
    std::vector<unsigned char> bytes;
    auto put = [&](uint32_t x) {
        for (int j = 0; j < 4; ++j)
            bytes.push_back((x >> (8 * j)) & 255);
    };
    for (uint32_t x : {0x31474354u, 1u, 1024u, 256u, 16u})
        put(x);
    for (auto sha : {model_sha, train_sha, file_sha256(argv[6])})
        bytes.insert(bytes.end(), sha.begin(), sha.end());
    for (auto x : gates)
        bytes.push_back(x + 8);
    put(reader_crc32(bytes.data(), bytes.size()));
    std::string path = output + ".tcg";
    HANDLE f = CreateFileA(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE)
        return 9;
    DWORD written = 0;
    bool ok = WriteFile(f, bytes.data(), DWORD(bytes.size()), &written, nullptr) &&
              written == bytes.size();
    ok = ok && FlushFileBuffers(f);
    CloseHandle(f);
    if (!ok)
        return 9;
    std::ifstream in(path, std::ios::binary);
    std::vector<unsigned char> restored((std::istreambuf_iterator<char>(in)), {});
    if (restored != bytes)
        return 9;
    printf("TCG1 byte roundtrip exact\n");
    cublasDestroy(handle);
}