#include "plateau_policy.hpp"
#include <cstdio>
int main(){using namespace tao::dual;PlateauPolicy p;p.observe(10,8);p.observe(20,7.97);if(p.stale)return 1;p.observe(30,7.96);p.observe(40,7.96);p.observe(50,7.96);if(p.lr!=.00025f||p.stale)return 2;bool rejected=false;try{p.observe(50,7);}catch(...){rejected=true;}if(!rejected)return 3;p.observe(60,2.5);if(!p.reached)return 4;printf("PASS significant improvement, patience3 LRhalf, repeatedmetric rejection, target stop\n");}
