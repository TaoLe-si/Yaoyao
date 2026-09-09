#define TAO_DEVICE_ZERO_GRAD
#include "batch_tape.cuh"
#include <cstdio>
int main(){using namespace tao::dual;try{BatchTape t;auto a=t.leaf(Vec(12,2)),b=t.leaf(Vec(12,3));auto mask=std::make_shared<Device>(Vec{0,1,0,1});auto y=t.select_batch(a,b,mask);Vec seed(12,1);check(cudaMemcpy(y->grad.p,seed.data(),48,cudaMemcpyHostToDevice));t.backward();auto v=y->value.host(),da=a->grad.host(),db=b->grad.host();for(int i=0;i<12;++i){bool selected=(i/3)%2;if(v[i]!=(selected?3:2)||da[i]!=(selected?0:1)||db[i]!=(selected?1:0))return 1;}printf("PASS batch select values and gradients route per slot\n");return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 2;}}
