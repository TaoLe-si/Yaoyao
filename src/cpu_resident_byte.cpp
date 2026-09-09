#define NOMINMAX
#define TAO_INPUT_SCALE
#define TAO_CPU_AVX2
#include "dual_model_bundle.hpp"
#include "byte_cpu_model.hpp"
#include <iostream>
namespace tao::dual {
inline ByteCpuModel load_byte_bundle(const std::string&path,const std::string&hash){ByteCpuModel result(load_bundle(path,hash));std::cerr<<"CPU_BACKEND explicit_AVX2 byte_ternary weight_payload_bytes="<<result.weight_bytes()<<std::endl;return result;}
}
// Reuse unchanged dialogue/token accounting; do not copy truncated source lines.
#define load_bundle load_byte_bundle
#include "cpu_resident_worker.cpp"
#undef load_bundle
