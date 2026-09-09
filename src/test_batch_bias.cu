#define TAO_DEVICE_ZERO_GRAD
#include "batch_tape.cuh"
#include <cstdio>
int main(){using namespace tao::dual;try{BatchTape t;auto x=t.leaf(Vec(12,.25f)),b=t.leaf(Vec(3,.5f));auto y=t.bias_batch(x,b,4);Vec dy(12);for(int i=0;i<12;++i)dy[i]=float(i);check(cudaMemcpy(y->grad.p,dy.data(),48,cudaMemcpyHostToDevice));t.backward();for(float z:y->value.host())if(z!=.75f)return 1;if(x->grad.host()!=dy)return 2;auto db=b->grad.host();for(int j=0;j<3;++j)if(db[j]!=18+4*j)return 3;printf("PASS batch bias broadcast and shared reduction across4slots\n");return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 4;}}
