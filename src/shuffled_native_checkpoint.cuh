#pragma once
// EXPERIMENTAL, not wired into a trainer. Includes existing native/CUDA snapshot
// interfaces but launches nothing unless the explicit capture/restore API is used.
#include "slot_checkpoint_file.cuh"
#include "shuffled_epoch_cursor.hpp"
#include "shuffled_checkpoint_metadata.hpp"
#include <array>
#include <fstream>
#include <iomanip>
#include <limits>

namespace tao::dual::shuffled_native {
namespace detail {
inline void require(bool ok,const char* why) {if(!ok)throw std::runtime_error(why);}
inline bool digest(const std::string& s) {
    return s.size()==64 && s.find_first_not_of("0123456789abcdef")==std::string::npos;
}
// SHA-256 (FIPS 180-4), byte input, standard big-endian digest. No claim of
// authentication without a separately trusted expected manifest digest.
inline std::string sha256(const std::string& bytes) {
    static constexpr uint32_t k[64]={
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::array<uint32_t,8> h{{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}};
    auto rotr=[](uint32_t x,unsigned n){return (x>>n)|(x<<(32-n));};
    auto block=[&](const unsigned char* p) {
        uint32_t w[64];
        for(unsigned i=0;i<16;++i) w[i]=(uint32_t(p[4*i])<<24)|(uint32_t(p[4*i+1])<<16)|(uint32_t(p[4*i+2])<<8)|p[4*i+3];
        for(unsigned i=16;i<64;++i) {
            uint32_t x=w[i-15],y=w[i-2];
            w[i]=w[i-16]+(rotr(x,7)^rotr(x,18)^(x>>3))+w[i-7]+(rotr(y,17)^rotr(y,19)^(y>>10));
        }
        auto a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],z=h[7];
        for(unsigned i=0;i<64;++i) {
            uint32_t t1=z+(rotr(e,6)^rotr(e,11)^rotr(e,25))+((e&f)^(~e&g))+k[i]+w[i];
            uint32_t t2=(rotr(a,2)^rotr(a,13)^rotr(a,22))+((a&b)^(a&c)^(b&c));
            z=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=z;
    };
    require(bytes.size()<=UINT64_MAX/8,"SHA256 input length");
    size_t full=bytes.size()/64*64;
    for(size_t p=0;p<full;p+=64)block(reinterpret_cast<const unsigned char*>(bytes.data()+p));
    unsigned char tail[128]{};size_t rem=bytes.size()-full;
    for(size_t i=0;i<rem;++i)tail[i]=static_cast<unsigned char>(bytes[full+i]);
    tail[rem]=0x80;size_t padded=rem<56?64:128;uint64_t bits=uint64_t(bytes.size())*8;
    for(unsigned i=0;i<8;++i)tail[padded-1-i]=static_cast<unsigned char>(bits>>(8*i));
    block(tail);if(padded==128)block(tail+64);
    std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto x:h)out<<std::setw(8)<<x;
    return out.str();
}
inline void hash_self_test() {
    require(sha256("")=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","SHA256 empty KAT");
    require(sha256("abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","SHA256 abc KAT");
    require(sha256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")=="248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1","SHA256 multiblock KAT");
}
inline std::filesystem::path directory(const std::filesystem::path& input) {
    require(!input.empty() && !input.filename().empty(),"checkpoint directory name");
    // Absolute paths and ordinary relative paths are accepted. Drive-relative
    // Windows paths, traversal, dot components and control bytes are rejected.
    require(!(input.has_root_name() && !input.has_root_directory()),"drive-relative checkpoint path");
    for(const auto& part:input) {
        auto s=part.string();require(s!="." && s!="..","checkpoint traversal");
        for(unsigned char c:s)require(c>=32 && c!=127,"checkpoint path control byte");
    }
    auto p=std::filesystem::absolute(input).lexically_normal();
    require(p!=p.root_path(),"checkpoint root directory");
    // Reject symlink components (not a defense against concurrent hostile changes).
    auto parent=p.parent_path();
    for(auto q=parent;!q.empty();) {
        require(!std::filesystem::is_symlink(std::filesystem::symlink_status(q)),"checkpoint symlink ancestor");
        auto up=q.parent_path();if(up==q)break;q=up;
    }
    require(std::filesystem::is_directory(parent),"checkpoint parent missing");return p;
}
inline std::string read_regular(const std::filesystem::path& p,uint64_t cap) {
    auto st=std::filesystem::symlink_status(p);
    require(std::filesystem::is_regular_file(st),"checkpoint artifact missing/nonregular");
    auto n=std::filesystem::file_size(p);
    require(n<=cap && n<=uint64_t(std::numeric_limits<std::streamsize>::max()) && n<=SIZE_MAX,"checkpoint artifact size");
    std::ifstream in(p,std::ios::binary);require(bool(in),"checkpoint artifact open");
    std::string b(static_cast<size_t>(n),'\0');in.read(b.data(),static_cast<std::streamsize>(n));
    require(bool(in) && in.peek()==std::char_traits<char>::eof(),"checkpoint artifact read/changed");return b;
}
inline void publish_text(const std::filesystem::path& p,const std::string& bytes) {
    auto tmp=p;tmp+=".tmp";
    require(!std::filesystem::exists(p) && !std::filesystem::exists(tmp),"checkpoint overwrite");
    std::ofstream out(tmp,std::ios::binary);require(bool(out),"checkpoint artifact create");
    out.write(bytes.data(),static_cast<std::streamsize>(bytes.size()));out.close();require(bool(out),"checkpoint artifact write");
    require(!std::filesystem::exists(p),"checkpoint overwrite before rename");
    std::filesystem::rename(tmp,p);
}
inline std::string permutation_digest(const std::vector<size_t>& order) {
    std::string b="TAO_PERMUTATION_RANK_TO_CANONICAL_V1";
    auto u=[&](uint64_t x){for(unsigned i=0;i<8;++i)b.push_back(char(x>>(8*i)));};
    u(order.size());for(auto d:order)u(d);return sha256(b);
}
// Verify the sidecar's permutation against the existing algorithm, not just
// its bijectivity. No epoch iteration or training work is performed.
inline void check_order(const tao::data::ShuffledEpochCursor& e,const std::string& side,uint64_t seed) {
    std::istringstream in(side);std::string magic,id,cp;uint64_t parsed_seed,epoch;int shuffled;
    require(bool(in>>magic>>std::quoted(id)>>std::quoted(cp)>>parsed_seed>>epoch>>shuffled),"sidecar prefix");
    require(parsed_seed==seed && epoch==e.epoch() && (shuffled==0 || shuffled==1) && !(epoch==0 && shuffled),"sidecar epoch/seed");
    std::vector<size_t> order(e.order().size());std::iota(order.begin(),order.end(),size_t(0));
    uint64_t state=seed^(epoch*UINT64_C(0xd1342543de82ef95));
    auto draw=[&](){uint64_t z=(state+=UINT64_C(0x9e3779b97f4a7c15));z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);return z^(z>>31);};
    if(shuffled)for(size_t i=order.size();i>1;--i) {
        uint64_t bound=i,threshold=(uint64_t(0)-bound)%bound,r;do {r=draw();}while(r<threshold);
        std::swap(order[i-1],order[size_t(r%bound)]);
    }
    require(order==e.order(),"sidecar seed-derived permutation");
}
struct Manifest {
    uint64_t scp_bytes=0,side_bytes=0,seed=0,epoch=0,step=0,docs=0,slots=0;
    std::string scp_hash,side_hash,order_hash,identity_hash,scp_identity_hash;
    std::string text() const {
        std::ostringstream o;o<<"TAO_SHUFFLED_NATIVE_V1 SCP1 permutation-rank-v1 splitmix64-fisher-yates-v1\n"
            <<"state.scp "<<scp_bytes<<' '<<scp_hash<<"\nepoch.sidecar "<<side_bytes<<' '<<side_hash<<'\n'
            <<seed<<' '<<epoch<<' '<<step<<' '<<docs<<' '<<slots<<'\n'
            <<order_hash<<' '<<identity_hash<<' '<<scp_identity_hash<<'\n';return o.str();
    }
    static Manifest parse(const std::string& text) {
        Manifest m;std::istringstream in(text);std::string magic,format,encoding,algorithm,scp,side;
        require(bool(in>>magic>>format>>encoding>>algorithm>>scp>>m.scp_bytes>>m.scp_hash>>side>>m.side_bytes>>m.side_hash
            >>m.seed>>m.epoch>>m.step>>m.docs>>m.slots>>m.order_hash>>m.identity_hash>>m.scp_identity_hash),"outer manifest parse");
        require(magic=="TAO_SHUFFLED_NATIVE_V1" && format=="SCP1" && encoding=="permutation-rank-v1" && algorithm=="splitmix64-fisher-yates-v1"
            && scp=="state.scp" && side=="epoch.sidecar","outer manifest schema/path");
        require(digest(m.scp_hash)&&digest(m.side_hash)&&digest(m.order_hash)&&digest(m.identity_hash)&&digest(m.scp_identity_hash),"outer manifest digest");
        require(text==m.text(),"outer manifest noncanonical/trailing");return m;
    }
};
} // namespace detail

struct Limits {uint64_t checkpoint_bytes=UINT64_C(2)*1024*1024*1024,sidecar_bytes=UINT64_C(64)*1024*1024;};
struct Commit {std::filesystem::path directory;std::string manifest_sha256;};
struct Loaded {SlotSnapshot snapshot;tao::data::ShuffledEpochCursor epoch;};

// The directory is the transaction identifier. create_directory reserves it
// exclusively; existing directories, artifacts or stale attempts are not reused.
// Caller supplies a private, non-live parent and must exclude concurrent writers.
inline Commit save_snapshot(const std::filesystem::path& path,const SlotSnapshot& captured,
        const tao::data::ShuffledEpochCursor& epoch,uint64_t seed,
        const std::string& verified_identity,const std::string& scp_identity,Limits limits={}) {
    using namespace detail;hash_self_test();require(!verified_identity.empty()&&!scp_identity.empty(),"checkpoint identities");
    require(captured.s.size()==epoch.cursor().slots.size() && captured.mem.size()==epoch.cursor().slots.size(),"snapshot tensor slot count");
    tao::data::ShuffledCheckpointMapping mapping(epoch.cursor(),epoch.order());
    auto rank=mapping.before_save(captured.cursor,captured.next);
    mapping.after_load(rank.cursor,rank.next,epoch.cursor()); // snapshot/sidecar agree
    auto snapshot=captured;snapshot.cursor=rank.cursor;snapshot.next=rank.next;
    // Validate supplied seed/identity before filesystem publication.
    auto probe=epoch.save(std::string(64,'0'));
    auto verified=tao::data::ShuffledEpochCursor::restore(epoch.cursor(),probe,seed,verified_identity,std::string(64,'0'));
    check_order(verified,probe,seed);
    auto dir=directory(path);require(std::filesystem::create_directory(dir),"checkpoint transaction already exists");
    // Failure deliberately leaves an uncommitted directory; never overwrite/retry it.
    save_slot_file(snapshot,(dir/"state.scp").string(),scp_identity);
    auto scp=read_regular(dir/"state.scp",limits.checkpoint_bytes);
    Manifest m;m.scp_bytes=scp.size();m.scp_hash=sha256(scp);
    auto side=epoch.save(m.scp_hash);require(side.size()<=limits.sidecar_bytes,"sidecar size");
    m.side_bytes=side.size();m.side_hash=sha256(side);m.seed=seed;m.epoch=epoch.epoch();m.step=snapshot.steps;
    m.docs=epoch.cursor().docs.size();m.slots=epoch.cursor().slots.size();m.order_hash=permutation_digest(epoch.order());
    m.identity_hash=sha256(verified_identity);m.scp_identity_hash=sha256(scp_identity);
    publish_text(dir/"epoch.sidecar",side);
    auto manifest=m.text();publish_text(dir/"manifest",manifest); // LAST rename = logical commit
    return {dir,sha256(manifest)};
}

// Native convenience capture: mutable signature is satisfied with a COPY of
// canonical metadata. Tensor states remain by slot, unchanged and never permuted.
inline Commit save(const std::filesystem::path& path,SequenceSlots& slots,
        const tao::data::ShuffledEpochCursor& epoch,uint64_t seed,
        const std::string& verified_identity,const std::string& scp_identity,Limits limits={}) {
    auto canonical=epoch.cursor();auto snapshot=tao::dual::capture(slots,canonical);
    return save_snapshot(path,snapshot,epoch,seed,verified_identity,scp_identity,limits);
}

// No tensor mutation. trusted_manifest_sha256 MUST come from a separate trusted
// receipt, not be calculated from the candidate manifest immediately before load.
inline Loaded load_snapshot(const std::filesystem::path& path,const std::string& trusted_manifest_sha256,
        const SlotSnapshot& expected,const tao::data::PilotCursor& canonical,uint64_t expected_seed,
        const std::string& verified_identity,const std::string& scp_identity,Limits limits={}) {
    using namespace detail;hash_self_test();require(digest(trusted_manifest_sha256),"trusted manifest digest required");
    require(!verified_identity.empty()&&!scp_identity.empty(),"checkpoint identities");
    auto dir=directory(path);require(std::filesystem::is_directory(std::filesystem::symlink_status(dir)),"checkpoint directory missing/symlink");
    auto manifest=read_regular(dir/"manifest",4096);require(sha256(manifest)==trusted_manifest_sha256,"outer manifest checksum");
    auto m=Manifest::parse(manifest);
    require(m.seed==expected_seed && m.docs==canonical.docs.size() && m.slots==canonical.slots.size()
        && m.slots==expected.cursor.size() && m.step<=UINT32_MAX && m.identity_hash==sha256(verified_identity)
        && m.scp_identity_hash==sha256(scp_identity),"outer manifest expectations");
    require(m.scp_bytes<=limits.checkpoint_bytes && m.side_bytes<=limits.sidecar_bytes,"outer manifest size limits");
    auto scp=read_regular(dir/"state.scp",m.scp_bytes),side=read_regular(dir/"epoch.sidecar",m.side_bytes);
    require(scp.size()==m.scp_bytes && side.size()==m.side_bytes && sha256(scp)==m.scp_hash && sha256(side)==m.side_hash,"checkpoint artifact checksum");
    auto epoch=tao::data::ShuffledEpochCursor::restore(canonical,side,expected_seed,verified_identity,m.scp_hash);
    check_order(epoch,side,expected_seed);
    require(epoch.epoch()==m.epoch && permutation_digest(epoch.order())==m.order_hash,"outer permutation/epoch");
    tao::data::ShuffledCheckpointMapping mapping(canonical,epoch.order());auto physical=mapping.load_template(canonical);
    auto loaded=load_slot_file((dir/"state.scp").string(),scp_identity,expected,physical);
    require(loaded.steps==m.step,"outer checkpoint step");
    auto decoded=mapping.after_load(loaded.cursor,loaded.next,epoch.cursor());loaded.cursor=decoded.cursor;loaded.next=decoded.next;
    // Existing loader reopens by pathname; recheck before exposing loaded state.
    // Private immutable directory is required; this is not hostile-race protection.
    require(read_regular(dir/"state.scp",m.scp_bytes)==scp && read_regular(dir/"epoch.sidecar",m.side_bytes)==side
        && read_regular(dir/"manifest",4096)==manifest,"checkpoint changed during load");
    return {std::move(loaded),std::move(epoch)};
}

// Explicit native adoption. Existing capture checks a safe current boundary;
// all file/hash/order/metadata checks precede existing restore's first GPU write.
// Existing restore is NOT rollback-atomic on device/shape errors.
inline void load_and_restore(const std::filesystem::path& path,const std::string& trusted_manifest_sha256,
        SequenceSlots& slots,tao::data::ShuffledEpochCursor& epoch,uint64_t expected_seed,
        const std::string& verified_identity,const std::string& scp_identity,Limits limits={}) {
    auto canonical=epoch.cursor();auto expected=tao::dual::capture(slots,canonical);
    auto loaded=load_snapshot(path,trusted_manifest_sha256,expected,canonical,expected_seed,verified_identity,scp_identity,limits);
    auto restored_canonical=loaded.epoch.cursor();
    tao::dual::restore(loaded.snapshot,slots,restored_canonical);
    epoch=std::move(loaded.epoch);
}
} // namespace tao::dual::shuffled_native
