#pragma once
#include <fstream>
#include <string>
#include <cmath>
#include <stdexcept>
namespace tao::dual {
struct TrainingControl {float lr=.0005f;unsigned target=0;bool stop=false;};
inline TrainingControl read_control(const std::string&path,TrainingControl fallback){std::ifstream f(path);if(!f)return fallback;TrainingControl c=fallback;std::string tag,extra;int stop;if(!(f>>tag>>c.lr>>c.target>>stop)||tag!="TC1"||(f>>extra)||!std::isfinite(c.lr)||c.lr<1e-6f||c.lr>.001f||(stop!=0&&stop!=1)||!c.target)throw std::runtime_error("invalid control; retain prior checkpoint");c.stop=stop!=0;return c;}
}
