// Native CPU parity: actual serving forward, not a copy of its implementation.
#define Q1 ServerQ1
#define Vocab ServerVocab
#define main tao_server_entry
#include "tao_d256_api.cpp"
#undef main
#undef Vocab
#undef Q1
#define INVERSE_READER_HEAD_ONLY
#include "test_integrated_inverse_reader.cpp"
#include "reader_training_data.hpp"

static void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}
static std::vector<unsigned char> artifact(const char* path, size_t size) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    require(bool(f) && f.tellg() == std::streamoff(size), "artifact size");
    f.seekg(0); std::vector<unsigned char> b(size);
    require(bool(f.read((char*)b.data(), size)), "artifact read");
    uint32_t crc=0; for(int j=0;j<4;++j) crc |= uint32_t(b[size-4+j]) << (8*j);
    require(crc == reader_crc32(b.data(),size-4), "artifact CRC");
    return b;
}
template<class T> static void equal_weights(const std::vector<T>& a, const std::vector<T>& b, const char* name) {
    require(a == b, name);
    for(auto x:a) require(std::isfinite(double(x)), "nonfinite loaded weight");
    std::printf("loaded equality PASS %s elements=%zu\n",name,a.size());
}
int main(int argc, char** argv) {
  try {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    const char* model_path=argc>1?argv[1]:"tao_fixed_step50002.bin";
    const char* gate_path=argc>2?argv[2]:"tao_context_supported.tcg";
    const char* reader_path=argc>3?argv[3]:"tao_coef_rts1_128.reader";
    const char* text_path=argc>4?argv[4]:"tinystories_train.txt";
    const char* token_path=argc>5?argv[5]:"experiments/d256_nibble64_baseline/train_tokens.bin";
    std::printf("CPU forward parity model=%s gates=%s reader=%s tokens=%s\n",model_path,gate_path,reader_path,token_path);
    ServerCore server;
    require(server.load(model_path,gate_path,reader_path,text_path,token_path), "server load");
    if (argc > 6) server.load_state_gates(argv[6], model_path, "tao_alternating_step47002.bin", reader_path, gate_path, token_path, "experiments/d256_nibble64_baseline/multislice/slice_0.bin");
    M reference; std::mt19937 rng(42); reference.init(rng,1024);
    require(reference.load(model_path), "reference load");
    require(server.model.step==50002, "expected fixed50002");
    equal_weights(server.model.Wbi,reference.Wbi,"Wbi");
    equal_weights(server.model.q1,reference.q1.trits,"Q1");
    equal_weights(server.model.W_sgl_gate,reference.W_sgl_gate,"Wgate");
    equal_weights(server.model.W_sgl_up,reference.W_sgl_up,"Wup");
    equal_weights(server.model.W_sgl_out,reference.W_sgl_out,"Wout");
    equal_weights(server.model.b_sgl_gate,reference.b_sgl_gate,"bgate");
    equal_weights(server.model.b_sgl_up,reference.b_sgl_up,"bup");
    // fixed_boundaries training omits gate/up biases; reference head does too.
    for(float x:server.model.b_sgl_gate) require(x==0,"nonzero fixed-boundary gate bias");
    for(float x:server.model.b_sgl_up) require(x==0,"nonzero fixed-boundary up bias");
    Reader reader; auto rb=artifact(reader_path,4484); auto gb=artifact(gate_path,15480);
    for(int k=0;k<=16;++k) for(int d=0;d<256;++d) {
        reader.selectors[k][d]=int8_t(rb[128+k*256+d])-1;
        require(reader.selectors[k][d]==server.reader_sel[k*256+d],"reader selector mismatch");
    }
    std::vector<int> gates(1024*15);
    for(size_t i=0;i<gates.size();++i) { gates[i]=int(gb[116+i])-8; require(gates[i]==server.gates[i],"TCG mismatch"); }
    for(int id=0;id<1024;++id) require(Q1::hash(id,128)==ServerQ1::hash(id,128),"Q1 hash mismatch");
    std::printf("reader selectors=4352 TCG entries=15360 hash IDs=1024 biases=zero PASS\n");
    auto code=[&](int id){return reference.q1.trits.data()+Q1::hash(id,128)*16*256;};
    double worst=0,squares=0,ce_delta=0; size_t values=0,windows=0,top_matches=0,active=0;
    auto check=[&](const std::vector<int>& win,int target) {
        std::vector<float> actual,expected(1024);
        server.forward_window(win,actual);
        Node node; for(int id:win) node.push(id,code);
        // Included reference head and inverse reader, independently loaded M.
        auto mixed = reader.read(node, code);
        if (argc > 6) {
            mixed.fill(0);
            // Independent saved-prefix oracle, avoiding inverse pops for the new state gate test.
            Node prefix; std::vector<std::array<int8_t,256>> states;
            for (int id : win) {prefix.push(id,code); states.push_back(prefix.trit);}
            for (int k=0;k<=16;++k) {
                int c=k>=1 && k<=15?server.state_gates[win.back()*15+k-1]:0;
                int magnitudes[]={0,1,2,4,8,-1,-2,-4,-8};
                float scale=c?float(magnitudes[c])/32:1;
                for(int d=0;d<256;++d) mixed[d]+=float(reader.selectors[k][d])*scale*states[63-k][d];
            }
        }
        head(reference,mixed,node.hash,win[62],expected);
        auto history=node;
        for(int k=0;history.count;++k) {
            int id=history.pop(code); if(k==0) continue;
            int q=gates[win.back()*15+k-1]; if(q) ++active;
            for(int v=0;v<1024;++v) expected[v]+=float(q)*(1.0f/32)*reference.Wbi[id*1024+v];
        }
        require(actual.size()==1024,"logit size");
        for(int v=0;v<1024;++v) {
            require(std::isfinite(actual[v])&&std::isfinite(expected[v]),"nonfinite logits");
            double err=std::abs(double(actual[v])-expected[v]); worst=std::max(worst,err); squares+=err*err; ++values;
            // Different accumulation order: reference seeds output dot with Wbi.
            if(err>2e-4+2e-6*std::abs(double(expected[v]))) {
                std::printf("FAIL window=%zu token=%d actual=%.9g reference=%.9g error=%.9g\n",windows,v,actual[v],expected[v],err);
                throw std::runtime_error("logit tolerance");
            }
        }
        auto top=[](const std::vector<float>& z){return int(std::max_element(z.begin(),z.end())-z.begin());};
        top_matches+=top(actual)==top(expected);
        auto ce=[&](const std::vector<float>& z){double mx=*std::max_element(z.begin(),z.end()),sum=0;for(float x:z)sum+=std::exp(double(x)-mx);return mx+std::log(sum)-z[target];};
        ce_delta=std::max(ce_delta,std::abs(ce(actual)-ce(expected))); ++windows;
    };
    // Teacher-forced contiguous windows (no sampled IDs), including far corpus offsets.
    for(uint64_t offset:{0ULL,2000000ULL,12000000ULL,42000000ULL,82000000ULL}) {
        auto batch=read_reader_batch(token_path,offset,8);
        for(int b=0;b<8;++b) {
            std::vector<int> win(batch.input.begin()+b*64,batch.input.begin()+(b+1)*64);
            check(win,batch.target[b*64+63]);
        }
        std::printf("corpus offset=%llu full windows=8 PASS\n",(unsigned long long)offset);
    }
    auto batch=read_reader_batch(token_path,0,8);
    std::ifstream stream(token_path,std::ios::binary); stream.seekg(4);
    std::vector<int> contiguous(513);
    require(bool(stream.read((char*)contiguous.data(),contiguous.size()*4)),"contiguous token read");
    for(int end=64;end<=512;++end) {
        std::vector<int> win(contiguous.begin()+end-64,contiguous.begin()+end);
        check(win,contiguous[end]);
    }
    // Left-padding matches serving, intentionally not an unpadded training prefix.
    for(int n=1;n<=64;++n) {
        std::vector<int> win(64,0); std::copy(batch.input.begin(),batch.input.begin()+n,win.end()-n);
        check(win,batch.target[n-1]);
    }
    for(int n=0;n<64;++n) {std::vector<int> win(64);for(int& id:win)id=int(rng()%1024);check(win,int(rng()%1024));}
    require(active>0,"TCG branch not exercised");
    std::printf("PASS windows=%zu logits=%zu max_abs=%.9g rms=%.9g max_CE_delta=%.9g top1_equal=%zu/%zu active_TCG_terms=%zu\n",windows,values,worst,std::sqrt(squares/values),ce_delta,top_matches,windows,active);
    std::printf("Scope: actual forward_window used by generate; raw logits, no sampling/repetition/masking. Shared SelfDecodingNode; independent reference head/reader and checkpoint load.\n");
    return 0;
  } catch(const std::exception& e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
