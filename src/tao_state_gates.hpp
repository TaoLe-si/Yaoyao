#pragma once
#include "native_reader_checkpoint.hpp"
#include <algorithm>
#include <string>
#include <cstring>
#ifdef __CUDACC__
#define TDS_HD __host__ __device__
#else
#define TDS_HD
#endif
// TDS1 is an experimental artifact only; this header does not enable serving.
namespace tds {
using Sha = std::array<unsigned char, 32>;
TDS_HD inline int magnitude(uint8_t c) { return c == 0 ? 0 : c <= 4 ? (1 << (c - 1)) : c <= 8 ? (1 << (c - 5)) : -1; }
TDS_HD inline int delta(uint8_t c) { return c <= 4 ? magnitude(c) : -magnitude(c); }
TDS_HD inline float scale(uint8_t c) { return c == 0 ? 1.0f : float(delta(c)) * (1.0f / 32.0f); }
// The boundary check belongs here rather than at each CPU/CUDA call site.
TDS_HD inline float distance_scale(const uint8_t *codes, int token, int k) {
    return k >= 1 && k <= 15 ? scale(codes[token * 15 + k - 1]) : 1.0f;
}
inline void validate_codes(const std::vector<uint8_t>& codes) {
    if (codes.size() != 1024 * 15) throw std::runtime_error("TDS table size");
    for (int token = 0; token < 1024; ++token) {
        int count = 0, total = 0;
        for (int k = 0; k < 15; ++k) {
            auto c = codes[token * 15 + k];
            if (c > 8) throw std::runtime_error("TDS invalid code");
            count += c != 0; total += magnitude(c);
        }
        if (count > 2 || total > 8) throw std::runtime_error("TDS row budget");
    }
}
inline void validate_union(const std::vector<uint8_t>& codes, const std::vector<int8_t>& frozen) {
    validate_codes(codes);
    if (frozen.size() != codes.size()) throw std::runtime_error("TCG/TDS shape mismatch");
    for (int token = 0; token < 1024; ++token) {
        int active = 0;
        for (int k = 0; k < 15; ++k)
            active += codes[token * 15 + k] != 0 || frozen[token * 15 + k] != 0;
        if (active > 2) throw std::runtime_error("combined history distance budget");
    }
}
inline uint32_t word(const std::vector<unsigned char>& b, size_t p) {
    if (p + 4 > b.size()) throw std::runtime_error("artifact word bounds");
    uint32_t v = 0; for (int j = 0; j < 4; ++j) v |= uint32_t(b[p+j]) << (8*j); return v;
}
inline void put(std::vector<unsigned char>& b, uint32_t x) { for(int j=0;j<4;++j) b.push_back((x>>(8*j))&255); }
inline std::vector<unsigned char> read_exact(const char *path, size_t size) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in || in.tellg() != std::streamoff(size)) throw std::runtime_error(std::string("artifact size: ")+path);
    in.seekg(0); std::vector<unsigned char> b(size);
    if (!in.read((char*)b.data(), size)) throw std::runtime_error("artifact read");
    if (word(b,size-4) != reader_crc32(b.data(),size-4)) throw std::runtime_error("artifact CRC");
    return b;
}
inline void require_sha(const std::vector<unsigned char>& b,size_t p,const Sha& sha) {
    if(p+32>b.size() || !std::equal(sha.begin(),sha.end(),b.begin()+p)) throw std::runtime_error("artifact dependency mismatch");
}
inline Sha memory_sha256(const void* data,size_t size) {
    BCRYPT_ALG_HANDLE alg=nullptr; BCRYPT_HASH_HANDLE hash=nullptr;
    auto check=[](NTSTATUS s){if(s<0) throw std::runtime_error("SHA256 failure");};
    check(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0));
    try {
        check(BCryptCreateHash(alg,&hash,nullptr,0,nullptr,0,0));
        auto p=static_cast<const unsigned char*>(data);
        while(size) { ULONG n=ULONG((std::min)(size,size_t(65536))); check(BCryptHashData(hash,const_cast<PUCHAR>(p),n,0)); p+=n;size-=n; }
        Sha out{};check(BCryptFinishHash(hash,out.data(),32,0));BCryptDestroyHash(hash);BCryptCloseAlgorithmProvider(alg,0);return out;
    } catch(...) {if(hash)BCryptDestroyHash(hash);BCryptCloseAlgorithmProvider(alg,0);throw;}
}
inline std::string hex(const Sha& sha) { const char* digits="0123456789abcdef";std::string s;for(auto c:sha){s+=digits[c>>4];s+=digits[c&15];}return s; }
struct Dependencies {
    Sha model{},parent{},reader{},tcg{},train{},heldout{},q1{},wbi{};
    std::array<Sha,8> all() const {return {model,parent,reader,tcg,train,heldout,q1,wbi};}
};
struct Artifact {
    Dependencies deps;
    std::vector<uint8_t> codes = std::vector<uint8_t>(1024*15,0);
    std::vector<unsigned char> bytes() const {
        validate_codes(codes);std::vector<unsigned char> b;
        // Fixed protocol: vocabulary, dimension, horizon, sequence, slots,
        // support floor, max overrides, max L1 magnitude, denominator, batch count.
        for(uint32_t x:{0x31534454u,2u,1024u,256u,16u,64u,15u,64u,2u,8u,32u,4u})put(b,x);
        for(uint32_t x:{2000000u,12000000u,42000000u,82000000u,64u,0u,16u})put(b,x);
        for(const auto& sha:deps.all())b.insert(b.end(),sha.begin(),sha.end());
        b.insert(b.end(),codes.begin(),codes.end());put(b,reader_crc32(b.data(),b.size()));return b;
    }
    static Artifact load(const char* path,const Dependencies& expected) {
        Artifact a;a.deps=expected;auto b=read_exact(path,76+256+1024*15+4);
        auto canonical=a.bytes();
        if(!std::equal(b.begin(),b.begin()+332,canonical.begin()))throw std::runtime_error("TDS contract/dependency mismatch");
        a.codes.assign(b.begin()+332,b.end()-4);validate_codes(a.codes);return a;
    }
    void save_new(const char* path) const {
        auto b=bytes();HANDLE h=CreateFileA(path,GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
        if(h==INVALID_HANDLE_VALUE)throw std::runtime_error("TDS refuse overwrite/create failure");
        DWORD n=0;bool ok=WriteFile(h,b.data(),DWORD(b.size()),&n,nullptr)&&n==b.size();
        if(ok)ok=FlushFileBuffers(h)!=0;CloseHandle(h);if(!ok)throw std::runtime_error("TDS write failure");
        auto restored=load(path,deps);if(restored.codes!=codes)throw std::runtime_error("TDS roundtrip failure");
    }
};
inline std::vector<int8_t> load_reader(const char* path,const Sha& train,const Sha& heldout) {
    auto b=read_exact(path,4484);
    for(auto pair: std::array<std::pair<size_t,uint32_t>,7>{{{0,0x31535452u},{4,1},{8,16},{12,256},{16,64},{20,64},{24,2000000}}})
        if(word(b,pair.first)!=pair.second)throw std::runtime_error("RTS1 contract");
    if(word(b,28)>4096)throw std::runtime_error("RTS1 rounds");
    require_sha(b,64,train);require_sha(b,96,heldout);
    std::vector<int8_t> a(17*256);for(size_t i=0;i<a.size();++i){if(b[128+i]>2)throw std::runtime_error("RTS1 selector");a[i]=int8_t(b[128+i])-1;}
    for(int d=0;d<256;++d)if(a[d]!=1)throw std::runtime_error("RTS1 current-state boundary");
    return a;
}
inline std::vector<int8_t> load_tcg(const char* path,const Sha& parent,const Sha& train,const Sha& reader) {
    auto b=read_exact(path,15480);size_t p=0;
    for(uint32_t x:{0x31474354u,1u,1024u,256u,16u}){if(word(b,p)!=x)throw std::runtime_error("TCG contract");p+=4;}
    require_sha(b,20,parent);require_sha(b,52,train);require_sha(b,84,reader);
    std::vector<int8_t> out(1024*15);
    for(int token=0;token<1024;++token){int active=0,total=0;for(int k=0;k<15;++k){int i=token*15+k,q=int(b[116+i])-8;
        int m=q<0?-q:q;if(q && m!=1 && m!=2 && m!=4 && m!=8)throw std::runtime_error("TCG code");out[i]=int8_t(q);active+=q!=0;total+=m;}
        if(active>2 || total>8)throw std::runtime_error("TCG budget");}
    return out;
}
// Caller must load both checkpoints strictly before invoking this. Comparison is
// byte-exact (including signed zeros), not a numeric tolerance or filename claim.
inline void validate_transfer(const void* q1,const void* parent_q1,size_t q1_bytes,
                              const void* wbi,const void* parent_wbi,size_t wbi_bytes) {
    if(std::memcmp(q1,parent_q1,q1_bytes) || std::memcmp(wbi,parent_wbi,wbi_bytes))
        throw std::runtime_error("TCG transfer forbidden: Q1/Wbi changed");
}
} // namespace tds
#undef TDS_HD
