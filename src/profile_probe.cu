#define TAO_NO_FFN
// 训练循环分阶段计时器。不改动任何生产文件。
#define main unused_profile_graph_main
#include "train_yaoyao_graph_gpuval.cu"
#undef main
#include "shuffled_epoch_batch_plan.hpp"
#include "dual_model_bundle.hpp"
#include <chrono>
namespace {
using namespace tao::dual;
namespace fs = std::filesystem;
double now_ms(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct Scoped{const char*n;double t0;double*tot;Scoped(const char*name,double&acc):n(name),t0(now_ms()),tot(&acc){}~Scoped(){*tot+=now_ms()-t0;}};
}
int main(int argc,char**argv){
  try{
    setvbuf(stdout,NULL,_IONBF,0);
    if(argc<5){printf("usage: profile_probe TRAIN.bin TOK.bbp SLOTS WIDTH [UPDATES]\n");return 2;}
    unsigned slots=unsigned(std::stoul(argv[3])), width=unsigned(std::stoul(argv[4]));
    unsigned updates=argc>5?unsigned(std::stoul(argv[5])):3u;
    std::string th; tao::text::load_tokenizer(argv[2],th);
    std::ifstream f(argv[1],std::ios::binary);
    std::string raw((std::istreambuf_iterator<char>(f)),{});
    std::istringstream in(raw);
    auto docs=tao::data::read_bpe_pilot(in,th);
    size_t positions_total=0,targets_total=0;
    for(const auto&d:docs){positions_total+=d.size()-1;for(size_t i=1;i<d.size();++i)targets_total+=d[i].loss;}
    size_t maxlen=0; for(const auto&d:docs) if(d.size()>maxlen)maxlen=d.size();
    printf("CORPUS docs=%zu positions=%zu targets=%zu max_doc_len=%zu\n",docs.size(),positions_total,targets_total,maxlen);
    const std::string identity="profile;seed20260912\ncorpus=profile\n";
    SortedGpuTrainer tr(initialize(Config{},20260912));
    SequenceSlots seq(tr,slots);
    tao::data::ShuffledEpochCursor epoch(tao::data::PilotCursor(docs,slots,false),20260912u,identity);
    double t_cap0=now_ms();
    ReusableBatchGraph replay(tr,slots,width);
    printf("CAPTURE ms=%.1f  slots=%u width=%u  graph_steps=%zu\n",now_ms()-t_cap0,slots,width,replay.data.steps);
    DeferredLoss loss(1024);
    auto mem=[&](const char*tag){size_t f=0,t=0;cudaMemGetInfo(&f,&t);printf("VRAM %s free=%.0fMB used=%.0fMB\n",tag,f/1048576.0,(t-f)/1048576.0);};
    mem("after-capture");
    // 计划数据统计
    {auto p=tao::data::take_batch(epoch,width);
     printf("PLAN timesteps=%zu slots=%zu positions=%zu supervised=%zu  (占用率 %.1f%%)\n",
        p.timesteps,p.slots,p.positions,p.supervised,100.0*p.positions/double(p.slots*width));}
    double t_replay=0,t_loss=0,t_update=0,t_alloc=0;
    for(unsigned u=0;u<updates;++u){
      size_t positions=0,targets=0;
      for(unsigned r=0;r<8;++r){
        if(epoch.exhausted()){check(cudaStreamSynchronize(0));epoch.begin_next_epoch(true);}
        auto p=tao::data::take_batch(epoch,width);
        {Scoped s("replay",t_replay);replay.run(p,seq);}
        accumulate_loss<<<1,1>>>(replay.data.loss.p,replay.data.loss.n,loss.total,loss.bad);
        check(cudaGetLastError());
        positions+=p.positions;targets+=p.supervised;
      }
      check(cudaDeviceSynchronize());
      double tl=now_ms(); (void)loss.collect(); t_loss+=now_ms()-tl;
      double tu=now_ms(); float nrm=tr.update(targets,0.001f); t_update+=now_ms()-tu;
      check(cudaDeviceSynchronize());
      const double tot=t_replay+t_loss+t_update;
      printf("UPDATE %u positions=%zu targets=%zu norm=%.4f  replay=%.1fms loss=%.1fms update=%.1fms TOTAL=%.1fms  ms/position=%.3f\n",
        u+1,positions,targets,nrm,t_replay,t_loss,t_update,tot,tot/double(positions));
      mem("after-update");
      t_replay=t_loss=t_update=0;
    }
    return 0;
  }catch(const std::exception&e){printf("PROFILE_FAIL %s\n",e.what());return 1;}
}
