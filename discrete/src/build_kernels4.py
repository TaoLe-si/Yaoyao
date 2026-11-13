import os, torch
os.environ["TORCH_CUDA_ARCH_LIST"]="8.9"
from torch.utils.cpp_extension import load_inline
cuda_src=open("estep_kernels.cu",encoding="utf-8").read()
cpp=("void estep2_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);\n"
     "void argmax_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);\n"
     "void estep_out_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);")
mod=load_inline(name="dna_estep4k",cpp_sources=cpp,cuda_sources=cuda_src,
                functions=["estep2_run","argmax_run","estep_out_run"],verbose=False,
                extra_cuda_cflags=["-O3","-Xcompiler","/Zc:preprocessor","--use_fast_math"])
print("KERNELS4 BUILT")
