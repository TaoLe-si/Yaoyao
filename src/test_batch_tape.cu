#define TAO_DEVICE_ZERO_GRAD
#include "batch_tape.cuh"
#include <cstdio>
int main(){using namespace tao::dual;try{BatchTape t;auto w=t.leaf(Vec(3*5,.2f)),x=t.leaf(Vec(4*5,.3f));auto y=t.linear_batch(w,x,3,5,4);Vec seed(12,1);check(cudaMemcpy(y->grad.p,seed.data(),48,cudaMemcpyHostToDevice));t.backward();auto o=y->value.host(),dx=x->grad.host(),dw=w->grad.host();for(float z:o)if(std::abs(z-.3f)>1e-6)return 1;for(float z:dx)if(std::abs(z-.6f)>1e-6)return 2;for(float z:dw)if(std::abs(z-1.2f)>1e-6)return 3;bool reject=false;try{t.linear_batch(w,x,3,6,4);}catch(...){reject=true;}if(!reject)return 4;printf("PASS batched tape forward/backward shared parameters and shape rejection\n");return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 5;}}
