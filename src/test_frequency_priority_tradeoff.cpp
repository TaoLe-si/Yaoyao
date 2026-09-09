#include <array>
#include <cstdio>
#include <stdexcept>
struct Item{int key=-1,value=0,count=0,age=0;bool marked=false;};
struct Memory{std::array<Item,4>a;int time=0,mode;explicit Memory(int m):mode(m){}
void put(int key,int value,bool mark){++time;for(auto&x:a)if(x.key==key){x.value=value;++x.count;x.age=time;x.marked|=mark;return;}int slot=-1;for(int i=0;i<4;++i)if(a[i].key<0){slot=i;break;}if(slot<0){for(int i=0;i<4;++i){if(mode==2&&a[i].marked)continue;if(slot<0||(mode==1?a[i].count<a[slot].count||(a[i].count==a[slot].count&&a[i].age<a[slot].age):a[i].age<a[slot].age))slot=i;}if(slot<0){if(!mark)return;slot=0;for(int i=1;i<4;++i)if(a[i].age<a[slot].age)slot=i;}}a[slot]={key,value,1,time,mark};}
int get(int key){for(auto&x:a)if(x.key==key)return x.value;return -1;}
};
void check(bool x){if(!x)throw std::runtime_error("assertion");}
int main(){try{for(int mode=0;mode<3;++mode){int single=0,latest=0,ordinary=0;for(int n=0;n<128;++n){Memory m(mode);m.put(0,n,true);for(int k=1;k<=3;++k)for(int r=0;r<64;++r)m.put(k,r,false);m.put(4,4,false);single+=m.get(0)==n;
Memory update(mode);update.put(0,n,true);for(int r=0;r<128;++r)update.put(0,n,true);update.put(0,n+1,false);latest+=update.get(0)==n+1;
Memory busy(mode);for(int k=0;k<3;++k)busy.put(k,k,true);busy.put(3,n,false);busy.put(4,4,false);ordinary+=busy.get(3)==n;}
check(single==(mode==2?128:0));check(latest==128);check(ordinary==(mode==2?0:128));printf("mode=%d singleton_marked=%d/128 repeated_old_then_update=%d/128 unmarked_later_query=%d/128\n",mode,single,latest,ordinary);}return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
