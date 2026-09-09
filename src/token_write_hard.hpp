#pragma once
#include <array>
#include <cmath>
#include <stdexcept>
namespace tao {
// FP64 diagnostic model. Not ternary deployment format. Shape 44->16->9.
struct TokenWriteHard {
 std::array<double,857> weights;int previous=12,older=12,memory=-1;size_t writes=0;
 explicit TokenWriteHard(const std::array<double,857>&w):weights(w){for(double v:w)if(!std::isfinite(v))throw std::invalid_argument("nonfinite weight");}
 void reset(){previous=older=12;memory=-1;writes=0;}
 int step(int query,int token){if(query<0||query>=4||token<0||token>=13)throw std::invalid_argument("token/query range");double h[16];int ix[5]={query,4+token,17+previous,30+older,43};for(int j=0;j<16;++j){double a=0;for(int i:ix)a+=weights[j*44+i];h[j]=std::tanh(a);}older=previous;previous=token;double gate=weights[704+16];for(int j=0;j<16;++j)gate+=weights[704+j]*h[j];if(gate<0)return memory;double best=0;int winner=-1;for(int o=1;o<9;++o){double z=weights[704+o*17+16];for(int j=0;j<16;++j)z+=weights[704+o*17+j]*h[j];if(winner<0||z>best){best=z;winner=o-1;}}memory=winner;++writes;return memory;}
};
}
