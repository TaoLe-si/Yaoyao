#include "native_reader_checkpoint.hpp"
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <vector>

struct Record {
    std::array<int32_t, 64> input;
    int32_t target, alternate, distance, pair;
};
static_assert(sizeof(Record) == 272, "record layout");

int main(int argc, char **argv) {
    try {
        if (argc != 2) return 1;
        std::filesystem::path root(argv[1]);
        if (std::filesystem::exists(root)) throw std::runtime_error("refuse existing output directory");
        std::filesystem::create_directories(root);
        const std::array<int, 7> distances{2,4,8,15,16,32,64};
        std::set<std::array<int32_t,64>> training;
        std::ofstream manifest(root / "manifest.tsv");
        manifest << "split\trecords\tsha256\n";
        for (int split = 0; split < 3; ++split) {
            const char *name = split == 0 ? "train" : split == 1 ? "test_context" : "test_unseen_answers";
            int repetitions = split == 0 ? 64 : 16;
            int answer_start = split == 2 ? 144 : 128;
            std::mt19937 rng(41001 + split * 1009);
            std::vector<Record> records;
            std::map<std::pair<int,int>, int> counts;
            int pair = 0;
            for (int qi = 0; qi < 7; ++qi) {
                int d = distances[qi];
                for (int rep = 0; rep < repetitions; ++rep) {
                    for (int label = 0; label < 16; label += 2) {
                        Record a{};
                        for (auto &id : a.input) id = 3 + int(rng() % 125);
                        a.input[63] = 900 + qi;
                        a.target = answer_start + label;
                        a.alternate = a.target + 1;
                        a.distance = d; a.pair = pair;
                        Record b = a;
                        b.target = a.alternate; b.alternate = a.target;
                        if (d < 64) {a.input[63-d] = a.target; b.input[63-d] = b.target;}
                        int differences = 0;
                        for (int i = 0; i < 64; ++i) differences += a.input[i] != b.input[i];
                        if (differences != (d < 64 ? 1 : 0)) throw std::runtime_error("pair structure");
                        for (auto &r : {a,b}) {
                            if (split == 0) training.insert(r.input);
                            else if (training.count(r.input)) throw std::runtime_error("train/test overlap");
                            ++counts[{d,r.target}]; records.push_back(r);
                        }
                        ++pair;
                    }
                }
            }
            for (int d : distances) for (int target = answer_start; target < answer_start+16; ++target)
                if (counts[{d,target}] != repetitions) throw std::runtime_error("label imbalance");
            // Shuffling complete pairs preserves adjacency of matched A/B records.
            std::vector<int> order(pair);
            for (int i = 0; i < pair; ++i) order[i] = i;
            std::shuffle(order.begin(),order.end(),rng);
            auto path = root / (std::string(name) + ".bin");
            std::ofstream out(path,std::ios::binary);
            uint32_t header[]{0x31525443u,1u,uint32_t(records.size()),64u,272u};
            out.write((char*)header,sizeof(header));
            for(int p : order) {out.write((char*)&records[p*2],sizeof(Record));out.write((char*)&records[p*2+1],sizeof(Record));}
            out.close();
            if (!out) throw std::runtime_error("write failed");
            auto sha=file_sha256(path.string().c_str());
            const char *hex="0123456789abcdef";std::string digest;
            for(auto x:sha){digest+=hex[x>>4];digest+=hex[x&15];}
            manifest << name << "	" << records.size() << "	" << digest << "\n";
            std::cout << "PASS " << name << " records=" << records.size() << " balanced=16_answers_x7_distances pairs=" << pair << " sha256=" << digest << "\n";
        }
        std::cout << "PASS no train/test input overlap; d64 identical-input conflicting-label negative control; seen answers128..143 unseen144..159; query900..906; filler3..127\n";
        return 0;
    } catch(const std::exception &e) {std::cerr << e.what() << "\n";return 2;}
}
