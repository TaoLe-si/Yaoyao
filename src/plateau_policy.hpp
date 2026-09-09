#pragma once
#include <cmath>
#include <algorithm>
#include <stdexcept>
namespace tao::dual {
struct PlateauPolicy {double best=1e100;unsigned stale=0,last_step=0;float lr=.0005f;bool reached=false;
void observe(unsigned step,double nll){if(step<=last_step||!std::isfinite(nll)||nll<0)throw std::runtime_error("invalid or repeated validation");last_step=step;if(nll<=2.5){reached=true;return;}if(nll<=best-.02){best=nll;stale=0;}else if(++stale>=3){lr=std::max(.00003f,lr*.5f);stale=0;}}
};
}
