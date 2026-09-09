#define NOMINMAX
#include "tokenizer_file.hpp"
#include "bpe_pilot_reader.hpp"
#include <cstdio>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <map>
#include <string>
namespace fs=std::filesystem;
struct Counter{size_t reject_huge_doc=0,reject_no_assistant=0,reject_short=0,reject_long=0,reject_short_turn=0;size_t docs_in=0,docs_out=0;size_t tokens=0,sup=0,ends=0;};
static std::string sha_file(const std::string& path){std::ifstream f(path,std::ios::binary);std::string b((std::istreambuf_iterator<char>(f)),{});return tao::text::sha256(b);}
int main(int argc,char**argv){try{if(argc!=4&&argc!=5)throw std::runtime_error("usage: filter_expanded_corpus SRC_BIN SRC_MANIFEST OUT_BIN [OUT_MANIFEST]");std::string th;tao::text::load_tokenizer("build/formal_tokenizer.bbp",th);std::ifstream sb(argv[1],std::ios::binary);if(!sb)throw std::runtime_error("open source");std::string raw((std::istreambuf_iterator<char>(sb)),{});std::istringstream in(raw);auto docs=tao::data::read_bpe_pilot(in,th);std::ofstream out(std::string(argv[3])+".partial",std::ios::binary);if(!out)throw std::runtime_error("open output");out.write("TLP2",4);out<<th;std::map<std::string,std::string> meta;{std::ifstream sm(argv[2],std::ios::binary);std::string mraw((std::istreambuf_iterator<char>(sm)),{});std::istringstream mi(mraw);std::string row;while(std::getline(mi,row)){if(!row.empty()&&row.back()==0x0d)row.pop_back();auto tab=row.find("\t");if(tab==std::string::npos)continue;meta[row.substr(0,tab)]=row.substr(tab+1);}}
Counter c;
for(const auto& d:docs){c.docs_in++;
if(d.size()>4096){c.reject_huge_doc++;continue;}
std::vector<size_t> lens;
for(size_t i=1;i<d.size();){if(d[i].loss&&d[i].id==259){i++;continue;}if(d[i].loss){size_t run=0,j=i;while(j<d.size()&&!(d[j].loss&&d[j].id==259)){run++;j++;}lens.push_back(run);i=j;continue;}i++;}
if(lens.empty()){c.reject_no_assistant++;continue;}
bool any_long=false;for(auto l:lens)if(l>=16){any_long=true;break;}
if(!any_long){c.reject_short_turn++;continue;}
bool too_long=false;for(auto l:lens)if(l>512){too_long=true;break;}
if(too_long){c.reject_long++;continue;}
size_t sup=0;for(size_t i=1;i<d.size();++i)sup+=d[i].loss;
if(sup==0){c.reject_short++;continue;}
uint32_t n=uint32_t(d.size());out.put(char(n&255));out.put(char((n>>8)&255));out.put(char((n>>16)&255));out.put(char((n>>24)&255));for(const auto& t:d){out.put(char(t.id&255));out.put(char((t.id>>8)&255));out.put(char(t.loss));if(t.loss){c.sup++;if(t.id==259)c.ends++;}c.tokens++;}
c.docs_out++;}
out.close();if(!out)throw std::runtime_error("close output");
auto hash=sha_file(std::string(argv[3])+".partial");fs::rename(std::string(argv[3])+".partial",std::string(argv[3]));
auto bytes=fs::file_size(std::string(argv[3]));
printf("INPUT docs=%zu OUTPUT docs=%zu tokens=%zu supervised=%zu turn_end=%zu bytes=%zu SHA256=%s\n",c.docs_in,c.docs_out,c.tokens,c.sup,c.ends,bytes,hash.c_str());
printf("REJECTS huge=%zu no_assistant=%zu short_turn=%zu long_turn=%zu short_doc=%zu\n",c.reject_huge_doc,c.reject_no_assistant,c.reject_short_turn,c.reject_long,c.reject_short);
std::string out_manifest=argc==5?argv[4]:(std::string(argv[3])+".manifest.tsv");
std::ofstream m(std::string(out_manifest)+".partial",std::ios::binary);if(!m)throw std::runtime_error("open manifest");
auto val_hash=sha_file("build/bpe_pilot_validation.bin");auto test_hash=sha_file("build/bpe_pilot_test.bin");
m<<"status\tcomplete\n";
m<<"format\tTLP2\n";
m<<"policy\tfiltered-v1;min-turn-16;max-turn-512;max-doc-4096\n";
m<<"source_corpus_sha256\t"<<meta["bin_sha256"]<<"\n";
m<<"tokenizer_path\tD:\\x5cTaoVm\\x5cbuild\\x5cformal_tokenizer.bbp\n";
m<<"tokenizer_sha256\t"<<th<<"\n";
m<<"validation_path\tD:\\x5cTaoVm\\x5cbuild\\x5cbpe_pilot_validation.bin\n";
m<<"validation_sha256\t"<<val_hash<<"\n";
m<<"test_path\tD:\\x5cTaoVm\\x5cbuild\\x5cbpe_pilot_test.bin\n";
m<<"test_sha256\t"<<test_hash<<"\n";
m<<"bin_sha256\t"<<hash<<"\n";
m<<"tokens\t"<<c.tokens<<"\n";
m<<"supervised\t"<<c.sup<<"\n";
m<<"accepted\t"<<c.docs_out<<"\n";
m<<"rejected_huge_doc\t"<<c.reject_huge_doc<<"\n";
m<<"rejected_no_assistant\t"<<c.reject_no_assistant<<"\n";
m<<"rejected_short_turn\t"<<c.reject_short_turn<<"\n";
m<<"rejected_long_turn\t"<<c.reject_long<<"\n";
m<<"rejected_zero_supervised\t"<<c.reject_short<<"\n";
m.close();if(!m)throw std::runtime_error("close manifest");fs::rename(std::string(out_manifest)+".partial",out_manifest);
printf("MANIFEST %s\n",out_manifest.c_str());
return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
