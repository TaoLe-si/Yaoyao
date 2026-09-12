#define NOMINMAX
#define TAO_INPUT_SCALE
#define TAO_CPU_AVX2
#include "tokenizer_file.hpp"
#include "greedy_resident_lifecycle.hpp"
#include <cstdio>
#include <cstdlib>
int main(int argc, char** argv) {
    using namespace tao::dual;
    using Clock = std::chrono::steady_clock;
    auto seconds = [](auto a) { return std::chrono::duration<double>(Clock::now() - a).count(); };
    const char* dsb = argc > 1 ? argv[1] : "build/L1_qa_cot/step_1200/final.dsb";
    const char* tokpath = std::getenv("TAO_TOKENIZER");
    if (!tokpath) tokpath = "build/tok_qa.bbp";
    unsigned nt = 8;
    if (const char* e = std::getenv("TAO_CPU_THREADS")) {
        int v = std::atoi(e);
        if (v > 0) nt = unsigned(v);
    }
    std::string hash;
    auto tok = tao::text::load_tokenizer(tokpath, hash);
    auto t0 = Clock::now();
    GreedyPipelineGroupedModel model(read_compact_bundle(dsb, hash));
    if (nt > 1u) model.set_cpu_threads(nt);
    printf("LOAD seconds=%.4f  cfg=%s\n", seconds(t0), model.config_dump().c_str());
    printf("POOL_SPIN=%s  VNNI_ENV=%s\n",
           std::getenv("TAO_POOL_SPIN") ? std::getenv("TAO_POOL_SPIN") : "(unset)",
           std::getenv("TAO_VNNI") ? std::getenv("TAO_VNNI") : "(unset=default)");
    auto prompt = tok.encode("鸡有40只，是鸭的20%，鸡鸭共多少只？");
    auto state = model.initial();
    uint32_t next = 258;
    for (int i = 0; i < 32; i++) next = model.greedy_step(next, state);
    const int steps = 512;
    double best_tps = 0, sum_tps = 0;
    uint32_t cksum = 0;
    for (int round = 0; round < 5; round++) {
        state = model.initial();
        model.advance(256, state);
        model.advance(257, state);
        for (auto t : prompt) model.advance(t, state);
        model.advance(259, state);
        t0 = Clock::now();
        next = model.greedy_step(258, state);
        double first = seconds(t0);
        t0 = Clock::now();
        for (int i = 0; i < steps; i++) next = model.greedy_step(next, state);
        double elapsed = seconds(t0);
        double tps = steps / elapsed;
        sum_tps += tps;
        if (tps > best_tps) best_tps = tps;
        cksum = next;
        printf("FORCED round=%d steps=%d seconds=%.6f tps=%.1f ms_per_step=%.4f first_ms=%.3f tok=%u\n",
               round, steps, elapsed, tps, elapsed * 1000 / steps, first * 1000, next);
    }
    printf("BASELINE mean_tps=%.1f  best_tps=%.1f  last_tok=%u  prompt_tokens=%zu  threads=%u\n",
           sum_tps / 5, best_tps, cksum, prompt.size(), model.cpu_threads());
    return 0;
}
