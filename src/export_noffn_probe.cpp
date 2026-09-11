// CPU-only tiny TLP2 export for the no-FFN learning probe. Never trains.
// Usage: export_noffn_probe conversations.txt train.bin tokenizer.bbp
#define NOMINMAX
#include "tokenizer_file.hpp"
#include "language_data_contract.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iterator>
#include <stdexcept>
namespace fs = std::filesystem;
using namespace tao::data;
static void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
static void le32(std::ostream& out, uint32_t n) { for (int i=0;i<4;++i) out.put(char(n>>(8*i))); }
int main(int argc, char** argv) {
    try {
        require(argc==4, "usage: export_noffn_probe CONVERSATIONS.txt TRAIN.bin TOKENIZER.bbp");
        fs::path textp=fs::absolute(argv[1]).lexically_normal();
        fs::path bin=fs::absolute(argv[2]).lexically_normal();
        fs::path tok=fs::absolute(argv[3]).lexically_normal();
        require(fs::is_regular_file(textp), "conversations missing");
        require(fs::is_regular_file(tok), "tokenizer missing");
        require(!fs::exists(bin) && !fs::exists(std::string(bin.string())+".partial")
            && !fs::exists(bin.string()+".manifest.tsv"), "refuse overwrite train outputs");
        std::string digest; auto b=tao::text::load_tokenizer(tok.string(), digest);
        require(b.merges.size()==16123, "frozen 16384 tokenizer required");
        std::ifstream in(textp, std::ios::binary); require(bool(in), "conversations open");
        std::string raw((std::istreambuf_iterator<char>(in)), {});
        require(!raw.empty() && !in.bad(), "conversations read");
        if (raw.size()>=3 && (unsigned char)raw[0]==0xef && (unsigned char)raw[1]==0xbb && (unsigned char)raw[2]==0xbf)
            raw.erase(0,3);
        std::istringstream lines(raw); std::string line;
        std::vector<std::vector<Message>> docs; std::vector<Message> cur;
        auto flush=[&](){ if(!cur.empty()){ require(cur.size()%2==0 && cur.front().assistant==false, "role order");
            docs.push_back(std::move(cur)); cur.clear(); } };
        while (std::getline(lines, line)) {
            if (!line.empty() && line.back()=='\r') line.pop_back();
            if (line.empty() || line[0]=='#') continue;
            if (line=="DOC") { flush(); continue; }
            require(line.size()>=3 && (line.compare(0,2,"U ")==0 || line.compare(0,2,"A ")==0), "line prefix");
            bool assistant=line[0]=='A';
            require(cur.empty()? !assistant : assistant==((cur.size()%2)==1), "alternating roles");
            cur.push_back({assistant, line.substr(2)});
        }
        flush();
        require(docs.size()>=10, "too few documents");
        const fs::path tmp=bin.string()+".partial";
        const fs::path tsv=bin.string()+".tsv";
        const fs::path tsv_tmp=tsv.string()+".partial";
        const fs::path man=bin.string()+".manifest.tsv";
        const fs::path man_tmp=man.string()+".partial";
        std::ofstream out(tmp, std::ios::binary), rows(tsv_tmp, std::ios::binary);
        require(bool(out)&&bool(rows), "output open");
        out.write("TLP2",4); out<<digest;
        rows<<"doc_id\tturns\tutf8_bytes\ttokens\tsupervised\n";
        uint64_t total_tokens=0, total_loss=0, total_bytes=0;
        for (size_t d=0;d<docs.size();++d) {
            std::vector<Token> tokens{{BOS,false}};
            size_t bytes=0;
            for (const auto& m: docs[d]) {
                tokens.push_back({m.assistant?ASSISTANT:USER,false});
                auto ids=b.encode(m.utf8); require(b.decode(ids)==m.utf8, "BPE roundtrip");
                for (auto id: ids) { require(id<16384 && (id<256||id>=261), "BPE text ID"); tokens.push_back({int(id), m.assistant}); }
                tokens.push_back({TURN_END, m.assistant});
                bytes+=m.utf8.size();
            }
            require(tokens.size()>=5 && tokens.size()<=1000000, "record length");
            uint64_t supervised=0; le32(out, uint32_t(tokens.size()));
            for (const auto& t: tokens) { out.put(char(t.id&255)); out.put(char(t.id>>8)); out.put(char(t.loss)); supervised+=t.loss; }
            require(supervised>0, "zero supervision");
            rows<<d<<'\t'<<docs[d].size()<<'\t'<<bytes<<'\t'<<tokens.size()<<'\t'<<supervised<<'\n';
            total_tokens+=tokens.size(); total_loss+=supervised; total_bytes+=bytes;
        }
        out.close(); rows.close(); require(bool(out)&&bool(rows), "output close");
        const auto bin_hash=[&](){
            std::ifstream f(tmp, std::ios::binary); std::string s((std::istreambuf_iterator<char>(f)), {});
            require(!s.empty()&&!f.bad(),"hash read"); return tao::text::sha256(s);
        }();
        std::ofstream meta(man_tmp, std::ios::binary); require(bool(meta), "manifest open");
        meta<<"status\tcomplete\nformat\tTLP2\npolicy\ttiny-closed-probe;human-written;no-holdout-in-train\n"
            <<"docs\t"<<docs.size()<<"\ntokens\t"<<total_tokens<<"\nsupervised\t"<<total_loss
            <<"\nutf8_bytes\t"<<total_bytes<<"\ntokenizer_sha256\t"<<digest<<"\nbin_sha256\t"<<bin_hash<<"\n";
        meta.close(); require(bool(meta), "manifest close");
        fs::rename(tmp, bin); fs::rename(tsv_tmp, tsv); fs::rename(man_tmp, man);
        std::cout<<"PROBE_EXPORT_COMPLETE docs="<<docs.size()<<" tokens="<<total_tokens
            <<" supervised="<<total_loss<<" sha256="<<bin_hash<<"\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr<<"PROBE_EXPORT_FAILED "<<e.what()<<"; do not consume partial outputs\n"; return 1;
    }
}
