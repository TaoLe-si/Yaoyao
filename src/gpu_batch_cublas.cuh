#pragma once
#include "dual_state_autograd.cuh"
#include <cublas_v2.h>
namespace tao::dual {
inline void blas_check(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS status "+std::to_string(int(s)));}
struct BatchBlas{cublasHandle_t h;BatchBlas(){blas_check(cublasCreate(&h));blas_check(cublasSetMathMode(h,CUBLAS_PEDANTIC_MATH));}~BatchBlas(){cublasDestroy(h);}BatchBlas(const BatchBlas&)=delete;
// Row-major W[rows,cols], X[slots,cols], Y[slots,rows].
void forward(const float*w,const float*x,float*y,int rows,int cols,int slots){float a=1,b=0;blas_check(cublasSgemm(h,CUBLAS_OP_T,CUBLAS_OP_N,rows,slots,cols,&a,w,cols,x,cols,&b,y,rows));}
void backward(const float*w,const float*x,const float*dy,float*dx,float*dw,int rows,int cols,int slots){float a=1,b=1;blas_check(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,cols,slots,rows,&a,w,cols,dy,rows,&b,dx,cols));blas_check(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_T,cols,rows,slots,&a,x,cols,dy,rows,&b,dw,cols));}
};
}
