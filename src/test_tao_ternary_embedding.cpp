#include "tao_ternary_embedding.hpp"
#include <cstdio>
void demand(bool b){if(!b)throw std::runtime_error("test failure");}
template<class F>void reject(F f){bool ok=false;try{f();}catch(const std::exception&){ok=true;}demand(ok);}
// Independently re-sign malformed fixtures so semantic checks are exercised past CRC.
void resign(std::vector<uint8_t>&b){uint32_t c=~0u;for(size_t i=0;i<b.size()-4;++i){c^=b[i];for(int j=0;j<8;++j)c=c&1?(c>>1)^0xedb88320u:c>>1;}c=~c;for(int j=0;j<4;++j)b[b.size()-4+j]=uint8_t(c>>(8*j));}
int main(){try{std::array<uint8_t,32> hash{};hash[0]=42;tao::TernaryEmbedding t(8192,257,hash);
for(uint32_t id:{0u,1u,1023u,1024u,8191u}){std::vector<int8_t>r(257);for(int i=0;i<257;++i)r[i]=int((i+id)%3)-1;t.set_row(id,r,.25f);demand(t.codes(id)==r);auto e=t.lookup(id);for(int i=0;i<257;++i)demand(e[i]==r[i]*.25f);}
auto bytes=t.bytes();auto loaded=tao::TernaryEmbedding::load(bytes,hash);demand(loaded.bytes()==bytes);demand(bytes.size()==56+8192*(4+65));
reject([&]{t.lookup(8192);});reject([&]{t.set_row(0,{1},1);});reject([&]{t.set_row(0,std::vector<int8_t>(257,2),1);});reject([&]{t.set_row(0,std::vector<int8_t>(257),0);});
auto other=hash;other[0]++;reject([&]{tao::TernaryEmbedding::load(bytes,other);});
for(int kind=0;kind<8;++kind){auto bad=bytes;if(kind==0)bad.pop_back();if(kind==1)bad.push_back(0);if(kind==2)bad[4]=2;if(kind==3)bad[60]^=1;if(kind==4){bad[52]=0;bad[53]=0;bad[54]=128;bad[55]=127;resign(bad);}if(kind==5){bad[52+8192*4]|=3;resign(bad);}if(kind==6){bad[52+8192*4+64]|=4;resign(bad);}if(kind==7){bad[8]=255;bad[9]=255;bad[10]=255;bad[11]=255;resign(bad);}reject([&]{tao::TernaryEmbedding::load(bad,hash);});}
printf("PASS TTE1 V8192 D257 high-ID lookup, row scaling, canonical roundtrip, tokenizer binding, CRC, lengths, invalid code/padding/scale/dimension rejection bytes=%zu\n",bytes.size());return 0;}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
