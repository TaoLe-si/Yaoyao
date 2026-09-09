#pragma once
#include <array>
#include <cmath>
#include <stdexcept>
namespace tao {
// FP64 diagnostic model. Not ternary deployment format. Shape 57->16->9.
struct TokenWriteHard4 {
 std::array<double,1065> weights;int previous=12,older=12,oldest=12,memory=-1;size_t writes=0;
 explicit TokenWriteHard4(const std::array<double,1065>&w):weights(w){for(double v:w)if(!std::isfinite(v))throw std::invalid_argument("nonfinite weight");}
 void reset(){previous=older=oldest=12;memory=-1;writes=0;}
 int step(int query,int token){if(query<0||query>=4||token<0||token>=13)throw std::invalid_argument("token/query range");double h[16];int ix[6]={query,4+token,17+previous,30+older,43+oldest,56};for(int j=0;j<16;++j){double a=0;for(int i:ix)a+=weights[j*57+i];h[j]=std::tanh(a);}oldest=older;older=previous;previous=token;double gate=weights[912+16];for(int j=0;j<16;++j)gate+=weights[912+j]*h[j];if(gate<0)return memory;double best=0;int winner=-1;for(int o=1;o<9;++o){double z=weights[912+o*17+16];for(int j=0;j<16;++j)z+=weights[912+o*17+j]*h[j];if(winner<0||z>best){best=z;winner=o-1;}}memory=winner;++writes;return memory;}
};
}
