#define NOMINMAX
#include "bpe_pilot_reader.hpp"
#include "batched_slot_plan.hpp"
#include "tokenizer_file.hpp"
#include <cstdio>
int main(){try{std::string hash;tao::text::load_tokenizer("build/formal_tokenizer.bbp",hash);std::ifstream f("build/bpe_pilot_train.bin",std::ios::binary);tao::data::PilotCursor c(tao::data::read_bpe_pilot(f,hash),4,false);size_t step=0,allp=0,alln=0;for(;;){bool done=c.next==c.docs.size();for(auto&s:c.slots)if(s.doc!=std::numeric_limits<size_t>::max()&&s.target<c.docs[s.doc].size())done=false;if(done)break;size_t n=0,p=0,rounds=0;for(int k=0;k<8;++k){auto b=tao::data::take_batch(c,256);p+=b.positions;n+=b.supervised;rounds+=b.timesteps!=0;}++step;allp+=p;alln+=n;printf("BATCH epoch_step=%zu positions=%zu supervised=%zu rounds=%zu next_doc=%zu occupancy=%.6f supervision_fraction=%.6f\n",step,p,n,rounds,c.next,double(p)/8192,p?double(n)/p:0);}printf("TOTAL steps=%zu positions=%zu supervised=%zu docs=%zu\n",step,allp,alln,c.docs.size());if(alln!=398141||allp!=655038)return 2;return 0;}catch(const std::exception&e){puts(e.what());return 1;}}
