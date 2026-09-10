#pragma once
#include "dual_model_stream.hpp"
#include <sstream>
#include <filesystem>
#ifdef TAO_NO_FFN
#define TAO_OPERATOR_ID "dual-state-3-noffn-input-sqrt-d"
#elif defined(TAO_INPUT_SCALE)
#define TAO_OPERATOR_ID "dual-state-2-input-sqrt-d"
#else
#define TAO_OPERATOR_ID "dual-state-1"
#endif
namespace tao::dual {
inline uint64_t bundle_hash(const std::string&s){uint64_t h=14695981039346656037ull;for(unsigned char c:s){h^=c;h*=1099511628211ull;}return h;}
inline std::string bundle_manifest(const Config&c,const std::string&tokenizer){if(tokenizer.size()!=64||tokenizer.find_first_not_of("0123456789abcdef")!=std::string::npos)throw std::runtime_error("tokenizer SHA256");std::ostringstream o;o<<TAO_OPERATOR_ID<<"\n"<<tokenizer<<"\n";for(auto&t:schema(c))o<<t.name<<" "<<t.rows<<" "<<t.cols<<" "<<t.ternary<<"\n";return o.str();}
inline uint64_t model_bytes(const Config&c){uint64_t n=28;for(auto&t:schema(c))n+=t.ternary?uint64_t(t.rows)*(4+(uint64_t(t.cols)+3)/4):4*t.elements();return n;}
inline void save_bundle(const CpuModel&m,const std::string&path,const std::string&tokenizer){if(std::filesystem::exists(path)||std::filesystem::exists(path+".tmp"))throw std::runtime_error("refuse overwrite");auto manifest=bundle_manifest(m.c,tokenizer);std::ostringstream payload(std::ios::binary|std::ios::out);save_model_stream(m,payload);auto p=payload.str();std::string body=manifest+p;std::ofstream f(path+".tmp",std::ios::binary);f.write("DSB2",4);auto put=[&](uint64_t v){for(int i=0;i<8;++i)f.put(char(v>>(8*i)));};put(manifest.size());put(p.size());put(bundle_hash(body));f.write(body.data(),body.size());f.close();if(!f)throw std::runtime_error("bundle write");std::filesystem::rename(path+".tmp",path);}
inline CpuModel load_bundle(const std::string&path,const std::string&expected_tokenizer){std::ifstream f(path,std::ios::binary|std::ios::ate);auto length=f.tellg();if(length<28||length>512ll*1024*1024)throw std::runtime_error("bundle length");f.seekg(0);std::string bytes(size_t(length),char(0));f.read(bytes.data(),bytes.size());if(!f||bytes.substr(0,4)!="DSB2")throw std::runtime_error("bundle header");auto get=[&](size_t pos){uint64_t v=0;for(int i=0;i<8;++i)v|=uint64_t((unsigned char)bytes[pos+i])<<(8*i);return v;};auto mn=get(4),pn=get(12);if(mn>1024*1024||pn<28||mn>bytes.size()-28||pn!=bytes.size()-28-mn)throw std::runtime_error("bundle lengths");if(bundle_hash(bytes.substr(28))!=get(20))throw std::runtime_error("checksum");auto payload=bytes.substr(28+mn);auto u=[&](size_t off){uint32_t v=0;for(int i=0;i<4;++i)v|=uint32_t((unsigned char)payload[off+i])<<(8*i);return v;};Config c;c.layers=u(4);c.d=u(8);c.s=u(12);c.m=u(16);c.e=u(20);c.vocab=u(24);c.validate();if(c.layers>64||c.d>8192||c.s>8192||c.m>8192||c.e>32768||c.vocab>262144)throw std::runtime_error("dimension bounds");if(model_bytes(c)!=pn)throw std::runtime_error("expected payload length");if(bytes.substr(28,mn)!=bundle_manifest(c,expected_tokenizer))throw std::runtime_error("manifest/tokenizer mismatch");std::istringstream in(payload,std::ios::binary|std::ios::in);return load_model_stream(in);}
}
