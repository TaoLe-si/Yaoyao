// Native CPU-only export. Never trains or modifies existing artifacts.
// Usage: export_repair_bpe_train NEW_DIRECTORY/PREFIX [DATA_ROOT [TOKENIZER [VALIDATION TEST]]]
// NEW_DIRECTORY must not exist; its parent must exist. Manifest is completion marker.
#define NOMINMAX
#include <algorithm>
#include "tokenizer_file.hpp"
#include "conversation_partition.hpp"
#include "bpe_pilot_reader.hpp"
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>
#include <filesystem>
#include <set>
#include <map>
#include <iostream>
#include <sstream>
#include <array>
#include <cstdio>
namespace fs = std::filesystem;
using namespace tao::data;

static void require(bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error(why);
}
static void le32(std::ostream& out, uint32_t n) {
    for (int i=0;i<4;++i) out.put(char(n>>(8*i)));
}
static std::string framed(const std::vector<Message>& messages) {
    std::string result;
    auto number=[&](uint64_t n){for(int i=0;i<8;++i) result.push_back(char(n>>(8*i)));};
    number(messages.size());
    for (const auto& m:messages) {
        result.push_back(m.assistant ? 1 : 0); number(m.utf8.size()); result+=m.utf8;
    }
    return result;
}
// Escape all non-ASCII, control and backslash bytes: one reversible TSV field.
static std::string safe(const std::string& text) {
    const char* hex="0123456789abcdef"; std::string out;
    for(unsigned char c:text) {
        if(c>=32 && c<=126 && c!=92) out.push_back(char(c));
        else {out+="\\x";out+=hex[c>>4];out+=hex[c&15];}
    }
    return out;
}
static std::string file_hash(const fs::path& path) {
    std::ifstream in(path,std::ios::binary); require(bool(in),"hash input open");
    BCRYPT_ALG_HANDLE alg=nullptr; BCRYPT_HASH_HANDLE hash=nullptr;
    auto ck=[](NTSTATUS s){require(s>=0,"stream SHA256 failure");};
    try {
        ck(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0));
        ck(BCryptCreateHash(alg,&hash,nullptr,0,nullptr,0,0));
        std::array<char,65536> buffer;
        while(in) {in.read(buffer.data(),buffer.size());auto n=in.gcount();
            if(n) ck(BCryptHashData(hash,reinterpret_cast<PUCHAR>(buffer.data()),ULONG(n),0));}
        require(in.eof()&&!in.bad(),"hash input read");
        unsigned char digest[32];ck(BCryptFinishHash(hash,digest,32,0));
        BCryptDestroyHash(hash);hash=nullptr;BCryptCloseAlgorithmProvider(alg,0);alg=nullptr;
        std::string out;const char* hex="0123456789abcdef";
        for(auto c:digest){out+=hex[c>>4];out+=hex[c&15];}return out;
    } catch(...) {if(hash)BCryptDestroyHash(hash);if(alg)BCryptCloseAlgorithmProvider(alg,0);throw;}
}
struct Holdouts {
    std::set<std::string> prompt_groups, individual_prompts;
    uint64_t docs=0;
};
static void load_holdout(const fs::path& path,const tao::text::ByteBpe& b,
                         const std::string& digest,Holdouts& h) {
    std::ifstream in(path,std::ios::binary);require(bool(in),"required holdout open");
    const auto docs=read_bpe_pilot(in,digest);
    require(!in.bad()&&!docs.empty(),"required holdout empty or IO failure");
    for(const auto& t:docs) {
        std::vector<Message> prompts;
        size_t i=1;
        while(i<t.size()) {
            bool assistant=t.at(i++).id==ASSISTANT;
            std::vector<uint32_t> ids;
            while(i<t.size() && t[i].id!=TURN_END) ids.push_back(uint32_t(t.at(i++).id));
            require(i<t.size(),"holdout turn boundary");++i;
            auto text=b.decode(ids);
            if(!assistant) {prompts.push_back({false,text});h.individual_prompts.insert(text);}
        }
        if(!prompts.empty()) h.prompt_groups.insert(framed(prompts));
        ++h.docs;
    }
}
static std::shared_ptr<arrow::StringArray> optional_strings(
    const std::shared_ptr<arrow::Table>& table,const char* name) {
    auto c=table->GetColumnByName(name);
    if(!c || c->num_chunks()!=1 || c->type()->id()!=arrow::Type::STRING) return nullptr;
    return std::static_pointer_cast<arrow::StringArray>(c->chunk(0));
}
static std::string metadata(const std::shared_ptr<arrow::StringArray>& a,int64_t row) {
    return (!a || a->IsNull(row)) ? "" : safe(a->GetString(row));
}
int main(int argc,char** argv) {
    try {
        require(argc==2||argc==3||argc==4||argc==6,
            "usage: export_repair_bpe_train NEW_DIRECTORY/PREFIX [DATA_ROOT [TOKENIZER [VALIDATION TEST]]]");
        fs::path prefix=fs::absolute(argv[1]).lexically_normal();
        require(!prefix.filename().empty(),"output prefix needs filename");
        fs::path dir=prefix.parent_path();
        require(!fs::exists(dir)&&fs::is_directory(dir.parent_path()),
                "output directory must be NEW, with existing parent; never reuse failed directory");
        fs::path root=argc>=3?argv[2]:"D:/Datasets/Infinity-Instruct/7M";
        fs::path tokenizer=argc>=4?argv[3]:"D:/TaoVm/build/formal_tokenizer.bbp";
        fs::path validation=argc==6?argv[4]:"D:/TaoVm/build/bpe_pilot_validation.bin";
        fs::path test=argc==6?argv[5]:"D:/TaoVm/build/bpe_pilot_test.bin";
        std::string digest;auto b=tao::text::load_tokenizer(tokenizer.string(),digest);
        require(b.merges.size()==16123,"requires frozen full 16384-token vocabulary");
        auto val_hash=file_hash(validation),test_hash=file_hash(test);
        Holdouts holdouts;load_holdout(validation,b,digest,holdouts);load_holdout(test,b,digest,holdouts);
        require(file_hash(validation)==val_hash&&file_hash(test)==test_hash,"holdouts changed during load");
        // create_directory is the exclusive reservation; fail rather than reuse on races.
        require(fs::create_directory(dir),"cannot reserve fresh output directory");
        const fs::path bin=prefix.string()+".bin",tsv=prefix.string()+".tsv";
        const fs::path bin_tmp=bin.string()+".partial",tsv_tmp=tsv.string()+".partial";
        const fs::path manifest=prefix.string()+".manifest.tsv";
        std::ofstream out(bin_tmp,std::ios::binary),rows(tsv_tmp,std::ios::binary);
        require(bool(out)&&bool(rows),"output open");out.write("TLP2",4);out<<digest;
        rows<<"doc_id\tshard\trow\tsource_escaped\tlang_escaped\tconversation_fnv64\tprompt_fnv64\tconversation_sha256\tprompt_sha256\tutf8_bytes\ttokens\tsupervised\n";
        std::set<std::string> seen; // Full framed content comparison, NOT hash-only dedup.
        std::map<std::string,uint64_t> counts;
        for(const char* name:{"rows_scanned","not_sampled","sampled","accepted",
            "reject_empty_or_null_conversation","reject_null_message","reject_unknown_role",
            "reject_role_order","reject_over65536_bytes","reject_unfinished_user_turn",
            "exclude_nontrain_partition","exclude_exact_holdout_prompt","duplicate_exact_content"}) counts[name]=0;
        std::ostringstream sources;
        uint64_t total_tokens=0,total_loss=0,total_bytes=0;
        for(int shard:{0,25,50}) {
            char name[80];std::snprintf(name,sizeof(name),"train-%05d-of-00075.parquet",shard);
            fs::path path=root/name;
            const auto size_before=fs::file_size(path);const auto time_before=fs::last_write_time(path);
            auto opened=arrow::io::ReadableFile::Open(path.string());require(opened.ok(),"parquet file open");
            auto result=parquet::arrow::OpenFile(opened.ValueOrDie(),arrow::default_memory_pool());
            require(result.ok(),"parquet reader open");auto reader=std::move(result).ValueOrDie();
            std::shared_ptr<arrow::Table> table;
            require(reader->ReadTable(&table).ok(),"parquet ReadTable");
            auto combined=table->CombineChunks();require(combined.ok(),"CombineChunks");table=combined.ValueOrDie();
            auto col=table->GetColumnByName("conversations");
            require(col&&col->num_chunks()==1&&col->type()->id()==arrow::Type::LIST,"conversation list schema");
            auto lists=std::static_pointer_cast<arrow::ListArray>(col->chunk(0));
            require(lists->values()->type_id()==arrow::Type::STRUCT,"message struct schema");
            auto structs=std::static_pointer_cast<arrow::StructArray>(lists->values());
            auto role_field=structs->GetFieldByName("from"),text_field=structs->GetFieldByName("value");
            require(role_field&&text_field&&role_field->type_id()==arrow::Type::STRING&&
                    text_field->type_id()==arrow::Type::STRING,"message string schema");
            auto roles=std::static_pointer_cast<arrow::StringArray>(role_field);
            auto texts=std::static_pointer_cast<arrow::StringArray>(text_field);
            auto source=optional_strings(table,"source"),lang=optional_strings(table,"langdetect");
            const uint64_t before=counts["accepted"];
            for(int64_t row=0;row<lists->length();++row) {
                ++counts["rows_scanned"];
                uint64_t mix=uint64_t(shard)*100000+uint64_t(row)+0x9e3779b97f4a7c15ull;
                mix=(mix^(mix>>30))*0xbf58476d1ce4e5b9ull;
                mix=(mix^(mix>>27))*0x94d049bb133111ebull;mix^=mix>>31;
                if(mix%100>=10){++counts["not_sampled"];continue;}++counts["sampled"];
                if(lists->IsNull(row)||lists->value_length(row)==0){++counts["reject_empty_or_null_conversation"];continue;}
                std::vector<Message> messages,prompts;size_t bytes=0;std::string reject;
                const auto start=lists->value_offset(row),end=start+lists->value_length(row);
                for(int64_t j=start;j<end;++j) {
                    if(structs->IsNull(j)||roles->IsNull(j)||texts->IsNull(j)){reject="reject_null_message";break;}
                    auto role=roles->GetString(j);
                    if(role!="human"&&role!="gpt"){reject="reject_unknown_role";break;}
                    const bool assistant=role=="gpt";
                    if(assistant!=(messages.size()%2==1)){reject="reject_role_order";break;}
                    auto text=texts->GetString(j);
                    if(text.size()>65536-bytes){reject="reject_over65536_bytes";break;}
                    bytes+=text.size();messages.push_back({assistant,text});
                    if(!assistant)prompts.push_back({false,text});
                }
                if(reject.empty()&&(messages.size()%2!=0))reject="reject_unfinished_user_turn";
                if(!reject.empty()){++counts[reject];continue;}
                auto ph=fingerprint(prompts);
                if(partition(ph)!=Partition::train){++counts["exclude_nontrain_partition"];continue;}
                auto prompt_key=framed(prompts);bool heldout=holdouts.prompt_groups.count(prompt_key)!=0;
                for(const auto& p:prompts) heldout=heldout||holdouts.individual_prompts.count(p.utf8)!=0;
                if(heldout){++counts["exclude_exact_holdout_prompt"];continue;}
                auto key=framed(messages);
                if(!seen.insert(key).second){++counts["duplicate_exact_content"];continue;}
                std::vector<Token> tokens{{BOS,false}};
                for(const auto& m:messages) {
                    tokens.push_back({m.assistant?ASSISTANT:USER,false});
                    auto ids=b.encode(m.utf8);require(b.decode(ids)==m.utf8,"BPE roundtrip");
                    for(auto id:ids){require(id<16384&&(id<256||id>=261),"BPE text ID");tokens.push_back({int(id),m.assistant});}
                    tokens.push_back({TURN_END,m.assistant});
                }
                require(tokens.size()<=1000000,"record exceeds existing reader bound");
                uint64_t planned_loss=0;for(const auto& t:tokens)planned_loss+=t.loss;
                if(counts["accepted"]>=30000||planned_loss>10000000-total_loss){++counts["exclude_budget_cap"];continue;}
                uint64_t supervised=0;le32(out,uint32_t(tokens.size()));
                for(const auto& t:tokens){out.put(char(t.id&255));out.put(char(t.id>>8));out.put(char(t.loss));supervised+=t.loss;}
                require(supervised>0,"zero supervision");
                rows<<counts["accepted"]<<'\t'<<shard<<'\t'<<row<<'\t'<<metadata(source,row)<<'\t'<<metadata(lang,row)
                    <<'\t'<<fingerprint(messages)<<'\t'<<ph<<'\t'<<tao::text::sha256(key)<<'\t'<<tao::text::sha256(prompt_key)
                    <<'\t'<<bytes<<'\t'<<tokens.size()<<'\t'<<supervised<<'\n';
                require(bool(out)&&bool(rows),"record write");
                ++counts["accepted"];total_tokens+=tokens.size();total_loss+=supervised;total_bytes+=bytes;
            }
            require(fs::file_size(path)==size_before&&fs::last_write_time(path)==time_before,"source changed during read");
            sources<<"source_"<<shard<<"_path\t"<<safe(fs::absolute(path).string())<<'\n'
                <<"source_"<<shard<<"_bytes\t"<<size_before<<'\n'
                <<"source_"<<shard<<"_rows\t"<<lists->length()<<'\n'
                <<"source_"<<shard<<"_accepted\t"<<counts["accepted"]-before<<'\n';
        }
        out.close();rows.close();require(bool(out)&&bool(rows)&&counts["accepted"]>0,"output close or empty corpus");
        require(file_hash(tokenizer)==digest&&file_hash(validation)==val_hash&&file_hash(test)==test_hash,
                "frozen dependencies changed during export");
        const auto bin_hash=file_hash(bin_tmp),tsv_hash=file_hash(tsv_tmp);
        fs::rename(bin_tmp,bin);fs::rename(tsv_tmp,tsv);
        // Consumers MUST require final manifest; partial/failure directories are never reusable.
        const fs::path manifest_tmp=manifest.string()+".partial";
        std::ofstream meta(manifest_tmp,std::ios::binary);require(bool(meta),"manifest open");
        meta<<"status\tcomplete\nformat\tTLP2\npolicy\tshards0,25,50;SplitMixGlobalRow10percent;prompt98/1/1_train_only\n"
            <<"budget\tmax30000docs;max10000000supervised;whole_doc_only\n"
            <<"ordering\tshard_then_source_row_not_shuffled\n"
            <<"dedup\texact_framed_content;no_near_duplicate_claim\n"
            <<"holdout_policy\texact_prompt_groups_and_any_exact_individual_user_prompt\n"
            <<"metadata_encoding\tASCII_with_backslash_x_hex_byte_escapes;empty_if_missing_or_null\n"
            <<"source_integrity\tpath_size_mtime_guard_only;not_source_SHA256\n"
            <<"tokenizer_path\t"<<safe(fs::absolute(tokenizer).string())<<"\ntokenizer_sha256\t"<<digest
            <<"\nvalidation_path\t"<<safe(fs::absolute(validation).string())<<"\nvalidation_sha256\t"<<val_hash
            <<"\ntest_path\t"<<safe(fs::absolute(test).string())<<"\ntest_sha256\t"<<test_hash
            <<"\nholdout_docs\t"<<holdouts.docs<<"\nbin_sha256\t"<<bin_hash<<"\ntsv_sha256\t"<<tsv_hash
            <<"\ntokens\t"<<total_tokens<<"\nsupervised\t"<<total_loss<<"\nutf8_bytes\t"<<total_bytes<<'\n';
        for(const auto& c:counts)meta<<c.first<<'\t'<<c.second<<'\n';meta<<sources.str();
        meta.close();require(bool(meta),"manifest close");fs::rename(manifest_tmp,manifest);
        std::cout<<"EXPORT_COMPLETE docs="<<counts["accepted"]<<" tokens="<<total_tokens
            <<" supervised="<<total_loss<<" manifest="<<manifest.string()<<'\n';return 0;
    }catch(const std::exception& e){std::cerr<<"EXPORT_FAILED "<<e.what()
        <<"; do not consume outputs without final manifest; use a new directory for retry\n";return 1;}
}
