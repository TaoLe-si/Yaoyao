#define main serving_main
#include "tao_d256_api.cpp"
#undef main
#include "reader_training_data.hpp"
#include <set>

static double nll(const std::vector<float>& z,int target) {
    double mx=*std::max_element(z.begin(),z.end()),sum=0;
    for(float x:z) sum+=std::exp(double(x)-mx);
    return mx+std::log(sum)-z[target];
}
int main(int argc,char** argv) {
    try {
        if(argc!=2) return 1;
        int stage=std::stoi(argv[1]);
        ServerCore core;
        const char* train="experiments/d256_nibble64_baseline/train_tokens.bin";
        const char* held="experiments/d256_nibble64_baseline/multislice/slice_0.bin";
        if(!core.load("tao_fixed_step50002.bin","tao_context_supported.tcg","tao_coef_rts1_128.reader","tinystories_train.txt",train)) return 2;
        core.load_state_gates("tao_state_supported_v2_full.tds","tao_fixed_step50002.bin","tao_alternating_step47002.bin","tao_coef_rts1_128.reader","tao_context_supported.tcg",train,held);
        auto frozen=core.gates; auto state=core.state_gates;
        auto mode=[&](int m){core.gates=frozen;core.state_gates=state;if(m<2)std::fill(core.state_gates.begin(),core.state_gates.end(),0);if(m==0)std::fill(core.gates.begin(),core.gates.end(),0);};
        std::vector<float> z;
        if(stage==1) {
            std::vector<std::vector<int>> windows;std::vector<int> targets;
            for(int slice=0;slice<3;++slice) {
                auto path="experiments/d256_nibble64_baseline/multislice/slice_"+std::to_string(slice)+".bin";
                std::ifstream in(path,std::ios::binary);in.seekg(4);std::vector<int> ids(1088);in.read((char*)ids.data(),ids.size()*4);if(!in) return 3;
                for(int i=63;i<1087;++i){windows.emplace_back(ids.begin()+i-63,ids.begin()+i+1);targets.push_back(ids[i+1]);}
            }
            printf("STAGE1 same binary fixed50002 3 slices x1024 sliding-window targets; frozen-head ablation not retraining\n");
            for(int m=0;m<3;++m){mode(m);double losses[3]={};int correct=0;for(size_t i=0;i<windows.size();++i){core.forward_window(windows[i],z);losses[i/1024]+=nll(z,targets[i]);correct+=int(std::max_element(z.begin(),z.end())-z.begin())==targets[i];}printf("MODE %d CE %.12f %.12f %.12f top1=%d/%zu\n",m,losses[0]/1024,losses[1]/1024,losses[2]/1024,correct,windows.size());}
            for(int warm=0;warm<3;++warm){mode(warm);for(int i=0;i<64;++i)core.forward_window(windows[i],z);}
            double times[3][9];volatile float sink=0;
            for(int round=0;round<9;++round)for(int order=0;order<3;++order){int m=(round+order)%3;mode(m);auto begin=std::chrono::steady_clock::now();for(int i=0;i<512;++i){core.forward_window(windows[i],z);sink+=z[targets[i]];}times[m][round]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count()/512;printf("TIMING round=%d mode=%d us=%.6f\n",round,m,times[m][round]);}
            for(int m=0;m<3;++m){std::sort(times[m],times[m]+9);printf("TIMING_SUMMARY mode=%d min_us=%.6f median_us=%.6f max_us=%.6f\n",m,times[m][0],times[m][4],times[m][8]);}
            printf("checksum %.9g\n",double(sink));
        } else if(stage==2) {
            printf("STAGE2 paired synthetic repeat retrieval: predict key A/B seen distance d from current last input; balanced target, identical filler/query, chance pair accuracy .5. NOT trained task or language benchmark. d64 outside window.\n");
            std::mt19937 rng(4242);
            std::vector<std::vector<int>> contexts(128,std::vector<int>(64));
            std::vector<int> keysA,keysB;
            for(auto& w:contexts){for(int& id:w)id=3+int(rng()%125);int a=3+int(rng()%125),b;do{b=3+int(rng()%125);}while(a==b);keysA.push_back(a);keysB.push_back(b);}
            for(int m=0;m<3;++m){mode(m);for(int d:{2,4,8,15,16,32,64}){double loss=0,effect=0,maxdiff=0;int acc=0,top=0;for(int trial=0;trial<128;++trial){int a=keysA[trial],b=keysB[trial];auto wa=contexts[trial],wb=wa;if(d<64){wa[63-d]=a;wb[63-d]=b;}std::vector<float> za,zb;core.forward_window(wa,za);core.forward_window(wb,zb);acc+=za[a]>za[b];acc+=zb[b]>zb[a];top+=int(std::max_element(za.begin(),za.end())-za.begin())==a;top+=int(std::max_element(zb.begin(),zb.end())-zb.begin())==b;loss+=nll(za,a)+nll(zb,b);effect+=(za[a]-za[b])-(zb[a]-zb[b]);for(int v=0;v<1024;++v)maxdiff=std::max(maxdiff,double(std::abs(za[v]-zb[v])));}printf("DIST mode=%d d=%d pairs=128 binary_accuracy=%.6f top1=%.6f NLL=%.9f paired_logodds_effect=%.9f max_logit_change=%.9f\n",m,d,acc/256.,top/256.,loss/256,effect/128,maxdiff);}}
            for(int k=1;k<=15;++k){int old=0,added=0;for(int x=0;x<1024;++x){old+=frozen[x*15+k-1]!=0;added+=state[x*15+k-1]!=0;}printf("GATE_DISTANCE %d Wbi=%d state=%d\n",k,old,added);}
        } else if(stage==3) {
            mode(2);
            struct Policy {const char* name;float T,p,penalty;bool suppress;};
            std::vector<Policy> policies={{"baseline",.9f,.9f,3,true},{"lower_temperature",.6f,.9f,3,true},{"narrow_top_p",.9f,.7f,3,true},{"no_penalty",.9f,.9f,0,true},{"normal_eos_unk",.9f,.9f,3,false},{"allow_eos_only",.9f,.9f,3,false},{"greedy",0,1,3,true}};
            for(const char* prompt:{"Once upon a time","The little girl lost her toy."})for(int seed:{42,123,2026})for(auto policy:policies){std::mt19937 rng(seed);auto ids=core.vocab.encode(prompt);std::vector<int> output;int repeated=0,special=0;std::string finish="length";for(int step=0;step<96;++step){std::vector<int> win(64,0);size_t size=std::min(ids.size(),size_t(64));std::copy(ids.end()-size,ids.end(),win.end()-size);core.forward_window(win,z);for(int back=0;back<6&&back<int(ids.size());++back){int id=ids[ids.size()-1-back];if(id>=2)z[id]-=policy.penalty*std::pow(.65f,float(back));}if(policy.suppress){z[1]=-1e9f;z[2]=-1e9f;}if(std::string(policy.name)=="allow_eos_only")z[1]=-1e9f;int next=core.sample_top_p(z,policy.T,policy.p,rng);if(!output.empty()&&output.back()==next)++repeated;output.push_back(next);special+=next<3;ids.push_back(next);if(next==2){finish="eos";break;}}std::set<int> unique(output.begin(),output.end());printf("SAMPLE policy=%s seed=%d prompt=%s tokens=%zu unique=%zu adjacent_repeat=%d special=%d finish=%s\n",policy.name,seed,prompt,output.size(),unique.size(),repeated,special,finish.c_str());printf("TEXT ");for(int id:output)printf("%s ",core.vocab.i2w[id].c_str());printf("\nIDS ");for(int id:output)printf("%d ",id);printf("\n");}
        } else return 4;
        return 0;
    } catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 10;}
}
