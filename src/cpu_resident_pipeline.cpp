#define NOMINMAX
#define TAO_INPUT_SCALE
#define TAO_CPU_AVX2
#include "pipeline_grouped_model.hpp"
#include "advance_row_parallel_byte_cpu_model.hpp"
#define AdvanceRowParallelByteCpuModel PipelineGroupedModel
#include "cpu_resident_advance_row_parallel_byte.cpp"
#undef AdvanceRowParallelByteCpuModel
