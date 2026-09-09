#define main tao_server_entry
#include "tao_d256_api.cpp"
#undef main
#include <stdexcept>

int main(int argc, char **argv) {
    if (argc != 2) return 1;
    FixedBoundariesModel model;
    if (!model.load(argv[1])) return 2;
    std::ifstream input(argv[1], std::ios::binary);
    // v4: header, W/Adam trio, Wh/Adam trio, then Wbi/Adam trio.
    input.seekg(12 + 3LL * 1024 * 256 * 4 + 3LL * 1024 * 64 * 4);
    std::vector<float> expected(1024 * 1024);
    input.read(reinterpret_cast<char *>(expected.data()), expected.size() * sizeof(float));
    if (!input || expected != model.Wbi) return 3;
    double max_abs = 0;
    for (float value : model.Wbi) {
        if (!std::isfinite(value)) return 4;
        max_abs = std::max(max_abs, double(std::abs(value)));
    }
    double bias_max = 0;
    for (float value : model.b_sgl_gate) bias_max = std::max(bias_max, double(std::abs(value)));
    for (float value : model.b_sgl_up) bias_max = std::max(bias_max, double(std::abs(value)));
    std::printf("Wbi checkpoint equality PASS elements=%zu weight_max_abs=%.9g head_bias_max_abs=%.9g step=%d\n",
                expected.size(), max_abs, bias_max, model.step);
    std::mt19937 rng(42);
    std::vector<float> logits{std::log(0.6f), std::log(0.3f), std::log(0.1f)};
    int counts[3] = {};
    for (int i = 0; i < 100000; ++i) ++counts[ServerCore::sample_top_p(logits, 1, 0.8f, rng)];
    if (counts[2] != 0 || counts[0] < 65000 || counts[0] > 68500) return 5;
    if (ServerCore::sample_top_p({10000, 9999, -10000}, 0.001f, 1, rng) != 0) return 6;
    if (ServerCore::sample_top_p({-3, 5, 1}, 0, 1, rng) != 1) return 7;
    std::printf("Sampler nucleus renormalization PASS counts=%d,%d,%d; low-temperature and greedy PASS\n", counts[0], counts[1], counts[2]);
    std::ifstream corpus("tinystories_train.txt", std::ios::binary);
    std::string text(5000000, '\0');
    corpus.read(text.data(), text.size());
    text.resize(size_t(corpus.gcount()));
    Vocab vocab;
    vocab.build(text, 1024);
    auto ids = vocab.encode(text);
    std::ifstream tokens("experiments/d256_nibble64_baseline/train_tokens.bin", std::ios::binary);
    uint32_t count = 0;
    tokens.read(reinterpret_cast<char *>(&count), 4);
    size_t checked = std::min(size_t(4096), ids.size());
    for (size_t i = 0; i < checked; ++i) {
        int expected_id = -1;
        tokens.read(reinterpret_cast<char *>(&expected_id), 4);
        if (!tokens || expected_id != ids[i]) {
            std::printf("Vocabulary corpus mismatch position=%zu expected=%d actual=%d\n", i, expected_id, ids[i]);
            return 8;
        }
    }
    std::printf("Vocabulary corpus prefix PASS tokens=%zu\n", checked);
    return 0;
}
