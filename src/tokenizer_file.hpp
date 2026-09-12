#pragma once
#include "byte_bpe.hpp"
#include <fstream>
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib,"bcrypt.lib")
namespace tao::text {
inline std::string sha256(const std::string&bytes){BCRYPT_ALG_HANDLE alg=nullptr;BCRYPT_HASH_HANDLE hash=nullptr;auto ck=[](NTSTATUS s){if(s<0)throw std::runtime_error("SHA256 provider");};try{ck(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0));ck(BCryptCreateHash(alg,&hash,nullptr,0,nullptr,0,0));size_t p=0;while(p<bytes.size()){ULONG n=ULONG((std::min)(bytes.size()-p,size_t(1048576)));ck(BCryptHashData(hash,(PUCHAR)bytes.data()+p,n,0));p+=n;}unsigned char digest[32];ck(BCryptFinishHash(hash,digest,32,0));BCryptDestroyHash(hash);BCryptCloseAlgorithmProvider(alg,0);const char*hex="0123456789abcdef";std::string out;for(auto c:digest){out+=hex[c>>4];out+=hex[c&15];}return out;}catch(...){if(hash)BCryptDestroyHash(hash);if(alg)BCryptCloseAlgorithmProvider(alg,0);throw;}}
inline std::string serialize_tokenizer(const ByteBpe&b){b.validate();std::string s="BBP1";auto u=[&](uint32_t v){for(int i=0;i<4;++i)s.push_back(char(v>>(8*i)));};u(uint32_t(b.merges.size()));for(auto p:b.merges){u(p.first);u(p.second);}return s;}
inline ByteBpe parse_tokenizer(const std::string&s){if(s.size()<8||s.substr(0,4)!="BBP1")throw std::runtime_error("tokenizer header");auto u=[&](size_t p){uint32_t v=0;for(int i=0;i<4;++i)v|=uint32_t((unsigned char)s.at(p+i))<<(8*i);return v;};uint32_t n=u(4);if(n>16121||s.size()!=8+size_t(n)*8)throw std::runtime_error("tokenizer length");ByteBpe b;for(uint32_t i=0;i<n;++i)b.merges.push_back({u(8+8*i),u(12+8*i)});b.validate();return b;}
inline ByteBpe load_tokenizer(const std::string&path,std::string&fingerprint){std::ifstream f(path,std::ios::binary|std::ios::ate);auto n=f.tellg();if(n<8||n>129000)throw std::runtime_error("tokenizer file size");f.seekg(0);std::string s(size_t(n),char(0));f.read(s.data(),s.size());if(!f)throw std::runtime_error("tokenizer read");auto b=parse_tokenizer(s);fingerprint=sha256(s);return b;}
}
