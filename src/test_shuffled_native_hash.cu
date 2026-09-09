#define NOMINMAX
#include "shuffled_native_checkpoint.cuh"
#include "tokenizer_file.hpp"
#include <cstdio>
int main(){try{for(size_t n:{size_t(0),size_t(1),size_t(55),size_t(56),size_t(63),size_t(64),size_t(65),size_t(127),size_t(128),size_t(4097),size_t(1000000)}){std::string s(n,0);for(size_t i=0;i<n;++i)s[i]=char((i*71+i/13)%256);auto a=tao::dual::shuffled_native::detail::sha256(s),b=tao::text::sha256(s);if(a!=b)throw std::runtime_error("SHA256 mismatch");}printf("PASS native SHA256 vs Windows provider 11 boundary/binary cases; no CUDA calls\n");return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
