#define TAO_DEVICE_ZERO_GRAD
#define TAO_FUSED_ADD_BACKWARD
#include "dual_state_autograd.cuh"
#include <cstdio>
int main(){using namespace tao::dual;try{for(bool alias:{false,true}){Tape t;auto a=t.leaf(Vec(257,.2f)),b=alias?a:t.leaf(Vec(257,.3f));auto y=t.add(a,b);Vec prior(257,.1f),seed(257,.7f);check(cudaMemcpy(a->grad.p,prior.data(),prior.size()*4,cudaMemcpyHostToDevice));check(cudaMemcpy(y->grad.p,seed.data(),seed.size()*4,cudaMemcpyHostToDevice));t.backward();auto da=a->grad.host(),db=b->grad.host();float expected=.1f;expected+=.7f;if(alias)expected+=.7f;for(int i=0;i<257;++i)if(da[i]!=expected||(!alias&&db[i]!=.7f))return 1;}printf("PASS fused add backward nonzero gradients and aliased inputs\n");return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 2;}}
