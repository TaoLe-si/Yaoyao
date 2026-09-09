#include "conversation_partition.hpp"
#include <cstdio>
using namespace tao::data;
void check(bool b){if(!b)throw std::runtime_error("contract");}
int main(){try{std::vector<Message>a{{false,"ab"},{true,"c"}},b{{false,"a"},{true,"bc"}};check(fingerprint(a)==fingerprint(a));check(fingerprint(a)!=fingerprint(b));b=a;b[0].utf8.push_back(char(0));check(fingerprint(a)!=fingerprint(b));b=a;b[0].assistant=true;check(fingerprint(a)!=fingerprint(b));check(partition(9799)==Partition::train);check(partition(9800)==Partition::validation);check(partition(9900)==Partition::test);printf("PASS framing roles NUL and partition boundaries\n");return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
