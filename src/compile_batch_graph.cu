#define TAO_INPUT_SCALE
#define TAO_DEVICE_ZERO_GRAD
#include "batch_train_graph.cuh"
void compile_graph(tao::dual::BatchTrainGraph&g){g.step(std::vector<unsigned>(g.slots,256),std::vector<bool>(g.slots,true),std::vector<bool>(g.slots,true));}
int main(){return 0;}
