#include <numeric>
#include "tao_state_gates.hpp"
#include "self_decoding_node.hpp"
#include "native_reader_checkpoint.hpp"
#define main legacy_cuda_main
#include "yaoyao_v21_cuda_train_stable_ce.cu"
#undef main
#include "inverse_reader_cuda.cuh"
#include <cassert>
#include <algorithm>
__global__ void reader_head(const float *s, const float *g, const float *u, const float *bg, const float *bu, float *h, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * HIDDEN)
        return;
    int n = i / HIDDEN, j = i % HIDDEN;
    float a = bg[j], b = bu[j];
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
__global__ void state_mix(const float* prefix,const int8_t* a,const uint8_t* gates,const int* ids,float* mixed,int rows) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*D)return;int n=i/D,d=i%D,t=n%SEQ;float sum=0;
 for(int k=0;k<=16&&k<=t;++k){float r=prefix[(n-k)*D+d]*tds::distance_scale(gates,ids[n],k);int w=a[k*D+d];if(w==1)sum+=r;else if(w==-1)sum-=r;}mixed[i]=sum;
}
__global__ void biased_silu(float* gate,const float* up,const float* bg,const float* bu,int n) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float g=gate[i]+bg[i%HIDDEN];gate[i]=g/(1+expf(-g))*(up[i]+bu[i%HIDDEN]);}
}
int state_main(int argc, char **argv) {
    setvbuf(stdout,nullptr,_IONBF,0);
    if (argc != 10) {
        printf("model train-corpus eval-corpus new-output-prefix coordinates input-reader "
               "start-coordinate parent-model frozen-tcg\n");
        return 1;
    }
    const std::string output = argv[4];
    if (std::ifstream(output + ".csv").good() || std::ifstream(output + ".tds").good())
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
    Model parent;if(!load_model(argv[8],parent))throw std::runtime_error("parent load");
    tds::validate_transfer(m.q1,parent.q1,B*K*D,m.Wbi,parent.Wbi,V*V*sizeof(float));
    tds::Dependencies deps;deps.model=model_sha;deps.parent=file_sha256(argv[8]);deps.reader=file_sha256(argv[6]);deps.tcg=file_sha256(argv[9]);deps.train=train_sha;deps.heldout=eval_sha;deps.q1=tds::memory_sha256(m.q1,B*K*D);deps.wbi=tds::memory_sha256(m.Wbi,V*V*sizeof(float));
    const char* names[]={"model","parent","reader","tcg","train","heldout","q1","wbi"};int hi=0;for(auto sha:deps.all())printf("SHA256 %s %s\n",names[hi++],tds::hex(sha).c_str());
    auto frozen=tds::load_tcg(argv[9],deps.parent,train_sha,deps.reader);
    auto dfrozen=dev(frozen.data(),frozen.size());
    auto bg=dev(m.b_sgl_gate,HIDDEN),bu=dev(m.b_sgl_up,HIDDEN);
    // Separate files from established prefix/tail split.
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
    auto coefficients=tds::load_reader(argv[6],train_sha,eval_sha);
    std::vector<uint8_t> gates(V*15,0);
    auto dg=dev(gates.data(),gates.size());
    auto da = dev(coefficients.data(), coefficients.size());
    cublasHandle_t handle;
    cublasCheck(cublasCreate(&handle));
    cublasCheck(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    auto forward = [&](GpuBatch &batch, bool reference) {
        cudaCheck(cudaMemcpy(da, coefficients.data(), coefficients.size(), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dg,gates.data(),gates.size(),cudaMemcpyHostToDevice));
        state_mix<<<(batch.rows*D+255)/256,256>>>(batch.prefix,da,dg,batch.ids,batch.mixed,batch.rows);
        kernelCheck();
        concat_kernel<<<batch.rows, 1>>>(batch.state, batch.mixed, batch.hash, batch.rows);
        kernelCheck();
        if (reference) {
            reader_head<<<(batch.rows * HIDDEN + 255) / 256, 256>>>(batch.state, wg, wu, bg, bu, batch.gate,
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
            biased_silu<<<(batch.rows*HIDDEN+255)/256,256>>>(batch.gate,batch.up,bg,bu,batch.rows*HIDDEN);
            kernelCheck();
            cublasCheck(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, V, batch.rows, HIDDEN, &one,
                                    wo, HIDDEN, batch.gate, HIDDEN, &zero, batch.logits, V));
            add_wbi_kernel<<<batch.rows, 1>>>(batch.logits, bi, batch.ids, PAD, batch.rows, V);
            kernelCheck();
        }
        cudaCheck(cudaMemcpy(dg, gates.data(), gates.size(), cudaMemcpyHostToDevice));
        tao_word_rows<<<(batch.rows * V + 255) / 256, 256>>>(batch.logits, bi, batch.ids, dfrozen,
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
    // Identity is exact versus original frozen reader kernel; probe includes k15.
    auto check_mix=[&](bool probe) {
        std::fill(gates.begin(),gates.end(),0);
        if(probe)for(int id=0;id<V;++id){gates[id*15]=1;gates[id*15+14]=6;}
        forward(evaluation,false);std::vector<float> gpu(evaluation.rows*D),original(gpu.size());
        cudaCheck(cudaMemcpy(gpu.data(),evaluation.mixed,gpu.size()*4,cudaMemcpyDeviceToHost));
        if(!probe){inverse_reader_mix<<<(evaluation.rows*D+255)/256,256>>>(evaluation.prefix,da,evaluation.mixed,evaluation.rows,SEQ,D,16);kernelCheck();cudaCheck(cudaMemcpy(original.data(),evaluation.mixed,original.size()*4,cudaMemcpyDeviceToHost));if(original!=gpu)throw std::runtime_error("identity kernel mismatch");}
        auto code=[&](int id){uint32_t h=uint32_t(id)*2654435761u;h=(h>>16)^h;return m.q1+(h%B)*K*D;};
        SelfDecodingNode<16,256> node;double err=0;unsigned touched=0;
        for(int n=0;n<evaluation.rows;++n){if(n%SEQ==0)node={};node.push(eval_host.input[n],code);auto cur=node;float cpu[D]={};
            for(int k=0;;++k){float scale=tds::distance_scale(gates.data(),eval_host.input[n],k);for(int d=0;d<D;++d){float r=float(cur.trit[d])*scale;int w=coefficients[k*D+d];if(w==1)cpu[d]+=r;else if(w==-1)cpu[d]-=r;}if(!cur.count)break;cur.pop(code);}
            for(int d=0;d<D;++d)err=std::max(err,double(std::abs(cpu[d]-gpu[n*D+d])));
            int t=n%SEQ;if(t==0||t==15||t==16||t==63)++touched;
        }
        printf("MIX probe=%d CPU/GPU max_abs=%.9g boundary_positions=%u t0,t15,t16,t63\n",int(probe),err,touched);
        if(err!=0)throw std::runtime_error("CPU/GPU state mismatch");std::fill(gates.begin(),gates.end(),0);
    };
    check_mix(false);check_mix(true);
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
    {auto z=forward(evaluation,false);std::ofstream dump(output+".baseline.logits.f32",std::ios::binary);dump.write(reinterpret_cast<const char*>(z.data()),z.size()*sizeof(float));if(!dump)throw std::runtime_error("logit dump write");}

    std::vector<int> counts(V, 0), order(V);
    for (auto *host : {&train_host, &train_host2, &train_host3, &train_host4})
        for (int id : host->input)
            ++counts[id];
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return counts[a] > counts[b]; });
    int accepted = 0;
    unsigned visited=0;
    for (int rank = int(start); rank < V && visited<rounds; ++rank) {
        int token = order[rank];
        if (counts[token] < 64)
            continue;
        ++visited;
        for (int k = 1; k <= 15; ++k) {
            int idx = token * 15 + k - 1;
            uint8_t old = gates[idx], chosen = old;
            double best = current;
            for (int q : {0,1,2,3,4,5,6,7,8}) {
                if (q == old)
                    continue;
                int active = 0, magnitude = 0;
                for (int j = 0; j < 15; ++j) {
                    int x = j == k - 1 ? q : gates[token * 15 + j];
                    active += x != 0 || frozen[token * 15 + j] != 0;
                    magnitude += tds::magnitude(uint8_t(x));
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
    auto trained=gates;double final_train=training_objective(),final_eval=objective(evaluation);
    std::fill(gates.begin(),gates.end(),0);double check_baseline=objective(evaluation);gates=trained;
    if(check_baseline!=baseline_eval)throw std::runtime_error("baseline identity drift");
    tds::validate_union(gates, frozen);
    tds::Artifact artifact;artifact.deps=deps;artifact.codes=gates;artifact.save_new((output+".tds").c_str());
    printf("FINAL baseline_eval=%.12f trained_eval=%.12f trained_train=%.12f changes=%d TDS roundtrip exact\n",baseline_eval,final_eval,final_train,accepted);
    cublasDestroy(handle);
    return 0;
}
int main(int argc,char** argv){try{return state_main(argc,argv);}catch(const std::exception& e){fprintf(stderr,"STATE FAIL: %s\n",e.what());return 10;}}
