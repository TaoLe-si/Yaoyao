#pragma once
#include "dual_model_bundle.hpp"
namespace tao::dual {
// 三值矩阵以磁盘上的 2-bit 打包形式原样常驻（每字节 4 个权重）。
// 旧实现把每个 2-bit 码展开成 1 字节 int8，使常驻量放大 4 倍：
// 119,033,986 权重 -> 119 MB/token 而不是 29.8 MB/token。解码是带宽受限的，
// 这个展开直接把解码速度砍到四分之一。解码器现在直接读打包字节。
struct CompactBundleData {Config c;struct Matrix{std::vector<uint8_t>q;Vec scale;};std::map<std::string,Matrix>matrices;std::map<std::string,Vec>vectors;};
inline CompactBundleData read_compact_bundle(const std::string&path,const std::string&tokenizer){
std::ifstream f(path,std::ios::binary|std::ios::ate);auto length=f.tellg();if(length<28||length>512ll*1024*1024)throw std::runtime_error("bundle length");f.seekg(0);std::string bytes(size_t(length),0);f.read(bytes.data(),bytes.size());if(!f||bytes.substr(0,4)!="DSB2")throw std::runtime_error("bundle header");
{auto nl=bytes.find('\n',28);if(nl==std::string::npos||nl<=28||bytes.compare(28,nl-28,TAO_OPERATOR_ID)!=0)throw std::runtime_error("operator identity mismatch");}
auto u64=[&](size_t p){uint64_t v=0;for(int i=0;i<8;++i)v|=uint64_t((unsigned char)bytes.at(p+i))<<(8*i);return v;};auto mn=u64(4),pn=u64(12);if(mn>1024*1024||pn<28||mn>bytes.size()-28||pn!=bytes.size()-28-mn||bundle_hash(bytes.substr(28))!=u64(20))throw std::runtime_error("bundle length/checksum");
size_t pos=28+mn;auto byte=[&](){if(pos>=bytes.size())throw std::runtime_error("truncated");return unsigned((unsigned char)bytes[pos++]);};auto u32=[&](){uint32_t v=0;for(int i=0;i<4;++i)v|=byte()<<(8*i);return v;};auto fp=[&](){uint32_t v=u32();float x;std::memcpy(&x,&v,4);if(!std::isfinite(x))throw std::runtime_error("nonfinite");return x;};
if(bytes.substr(pos,4)!="DSM1")throw std::runtime_error("model magic");pos+=4;CompactBundleData out;auto&c=out.c;c.layers=u32();c.d=u32();c.s=u32();c.m=u32();c.e=u32();c.vocab=u32();
#ifdef TAO_DELTA_MEM
c.dk=u32();
#endif
// 元素数上限。原值 1e8 是 <=27M 参数时代的遗留：本项目目标模型是 119,033,986 参数
// (d=3200 L=2 dk=400)，会被这道门直接拒载 —— 训得出来却加载不了，等于交付不出来。
// 这里只作为"损坏头部导致巨额分配"的护栏；三值紧凑格式实际占用远小于等值 float32。
constexpr uint64_t kMaxModelElements = 500000000ull;   // 约 2 GB 等值 float32
c.validate();if(c.layers>64||c.d>8192||c.s>8192||c.m>8192||c.e>32768||c.vocab>262144)throw std::runtime_error("bounds");uint64_t count=0;for(auto&t:schema(c))count+=t.elements();if(count>kMaxModelElements||model_bytes(c)!=pn||bytes.substr(28,mn)!=bundle_manifest(c,tokenizer))throw std::runtime_error("schema/identity");
for(auto&t:schema(c)){if(!t.ternary){Vec v(t.elements());for(float&x:v)x=fp();out.vectors.emplace(t.name,std::move(v));continue;}CompactBundleData::Matrix m;const size_t pst=(size_t(t.cols)+3)/4;m.q.assign(pst*t.rows,0);m.scale.resize(t.rows);for(unsigned r=0;r<t.rows;++r){m.scale[r]=fp();if(m.scale[r]<=0)throw std::runtime_error("scale");for(unsigned j=0;j<t.cols;j+=4){const unsigned b=byte();for(unsigned k=0;k<4;++k){const unsigned code=(b>>(2*k))&3;if(code==3||(j+k>=t.cols&&code))throw std::runtime_error("symbol/padding");}m.q[size_t(r)*pst+(j>>2)]=uint8_t(b);}}out.matrices.emplace(t.name,std::move(m));}if(pos!=bytes.size())throw std::runtime_error("trailing");return out;
}
}
