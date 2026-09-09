#define NOMINMAX
#define TAO_INPUT_SCALE
#define TAO_CPU_AVX2
#include "byte_cpu_model.hpp"
#include <iostream>
namespace tao::dual {inline ByteCpuModel load_byte_direct(const std::string&p,const std::string&h){ByteCpuModel m(read_compact_bundle(p,h));std::cerr<<"CPU_BACKEND explicit_AVX2 byte_ternary_direct payload_bytes="<<m.weight_bytes()<<std::endl;return m;}}
#define load_bundle load_byte_direct
#include "cpu_resident_worker.cpp"
#undef load_bundle
