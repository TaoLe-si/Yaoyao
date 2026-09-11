#pragma once
#include "dual_state_cpu.hpp"
#include <random>
namespace tao::dual {
inline CpuModel initialize(const Config&c,uint32_t seed){CpuModel model(c);std::mt19937 rng(seed);for(auto&t:schema(c)){auto&v=model.w.at(t.name);if(!t.ternary){float value=0;if(t.name.find("norm")!=std::string::npos)value=t.name.find("read.norm")!=std::string::npos?1/std::sqrt(2.f*c.layers):1;if(t.name.find("m.gate.bias")!=std::string::npos)value=-2;std::fill(v.begin(),v.end(),value);continue;}float fan=0;if(t.name=="embedding"||t.name.find("ff.up")!=std::string::npos)fan=c.d;else if(t.name.find("ff.down")!=std::string::npos)fan=2.f*c.layers*c.e;else if(t.name.find("mem.key")!=std::string::npos||t.name.find("mem.query")!=std::string::npos)fan=c.d;
  else if(t.name.find("mem.value")!=std::string::npos)fan=c.d;
  else if(t.name.find("read.")!=std::string::npos)fan=c.s+c.m;else if(t.name.find(".s.candidate.")!=std::string::npos||t.name.find(".s.gate.")!=std::string::npos)fan=c.d+c.s;else fan=c.d+c.s+c.m;std::normal_distribution<float>dist(0,1/std::sqrt(fan));for(auto&x:v)x=dist(rng);}return model;}
}
