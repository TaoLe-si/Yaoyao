#include "cpu_ternary_avx2.hpp"
#include "cpu_dot_avx2.hpp"
#include <cstdio>
int main(){for(size_t n:{7,128,512,1024}){std::vector<float>w(n*3),x(n);for(size_t i=0;i<n;++i)x[i]=float(int(i%19)-9)*.13f;for(size_t i=0;i<w.size();++i)w[i]=float(int(i%3)-1)*.03125f;CpuTernaryRows packed(w,3,n);for(size_t r=0;r<3;++r)if(packed.dot(r,x.data())!=tao_dot_avx2(w.data()+r*n,x.data(),n))return 1;}puts("PASS exact row ternary AVX2 vs expanded FP32 including tail");}
