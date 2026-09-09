#include <vector>
#include <map>
#include <cstdio>
#include <stdexcept>
using Seq=std::vector<int>;
std::map<Seq,int>hist(const Seq&s,int k){Seq window(k,99);std::map<Seq,int>out;for(int token:s){window.erase(window.begin());window.push_back(token);++out[window];}return out;}
int main(){try{for(int k=1;k<=16;++k){int collisions=0;for(int a=0;a<8;++a)for(int b=0;b<8;++b)if(a!=b){Seq A{80,81,a},B{80,81,b};A.insert(A.end(),k,99);B.insert(B.end(),k,99);Seq x=A;x.insert(x.end(),B.begin(),B.end());x.push_back(82);Seq y=B;y.insert(y.end(),A.begin(),A.end());y.push_back(82);if(hist(x,k)!=hist(y,k))throw std::runtime_error("unexpected histogram difference");++collisions;}printf("window=%d exact_ngram_histogram_collisions=%d/56 different_latest_labels\n",k,collisions);}return 0;}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
