#include <array>
#include <cstdio>
#include <stdexcept>
struct Item{int key=-1,value=-1;bool important=false;unsigned age=0;};
struct Store{std::array<Item,4>a;unsigned clock=0;bool priority;
explicit Store(bool p):priority(p){}
void put(int key,int value,bool important){++clock;for(auto&x:a)if(x.key==key){x.value=value;x.important=x.important||important;x.age=clock;return;}int slot=-1;for(int i=0;i<4;++i)if(a[i].key<0){slot=i;break;}if(slot<0){for(int i=0;i<4;++i){if(priority&&a[i].important)continue;if(slot<0||a[i].age<a[slot].age)slot=i;}if(slot<0){if(!important)return;slot=0;for(int i=1;i<4;++i)if(a[i].age<a[slot].age)slot=i;}}a[slot]={key,value,important,clock};}
int get(int key)const{for(auto&x:a)if(x.key==key)return x.value;return -1;}
void clear_priority(int key){for(auto&x:a)if(x.key==key)x.important=false;}
};
void check(bool b){if(!b)throw std::runtime_error("assertion");}
int main(){try{for(bool priority:{false,true}){int kept=0,updated=0,unmarked=0,released=0;for(int trial=0;trial<256;++trial){Store s(priority);s.put(0,trial,true);for(int i=1;i<=128;++i)s.put(i,i,false);kept+=s.get(0)==trial;s.put(0,trial+1,false);for(int i=129;i<=256;++i)s.put(i,i,false);updated+=s.get(0)==trial+1;Store plain(priority);plain.put(0,trial,false);for(int i=1;i<=128;++i)plain.put(i,i,false);unmarked+=plain.get(0)==trial;s.clear_priority(0);for(int i=257;i<270;++i)s.put(i,i,false);released+=s.get(0)<0;}check(kept==(priority?256:0));check(updated==(priority?256:0));check(unmarked==0&&released==256);printf("priority=%d marked_retained=%d/256 updated_retained=%d/256 unmarked_retained=%d/256 released=%d/256\n",int(priority),kept,updated,unmarked,released);}
Store full(true);for(int i=0;i<4;++i)full.put(i,i,true);full.put(99,99,false);check(full.get(99)<0);full.put(4,4,true);int kept=0;for(int i=0;i<5;++i)kept+=full.get(i)==i;check(kept==4&&full.get(0)<0);printf("important_over_capacity retained=%d/5 oldest_evicted=true ordinary_rejected=true\n",kept);return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
