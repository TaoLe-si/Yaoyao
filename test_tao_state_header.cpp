#include "tao_state_gates.hpp"
#include <cassert>
#include <cstdio>
int main(){tds::Artifact a;int d[]={0,1,2,4,8,-1,-2,-4,-8};for(int c=0;c<9;++c){assert(tds::delta(c)==d[c]);assert(tds::scale(c)==(c==0?1.0f:d[c]/32.0f));}assert(a.bytes().size()==15696);a.codes[0]=2;a.codes[14]=7;tds::validate_codes(a.codes);assert(tds::distance_scale(a.codes.data(),0,0)==1);assert(tds::distance_scale(a.codes.data(),0,16)==1);assert(tds::distance_scale(a.codes.data(),0,1)==0.0625f);a.codes[1]=1;bool rejected=false;try{tds::validate_codes(a.codes);}catch(...){rejected=true;}assert(rejected);printf("PASS encoding, identity boundaries, artifact size, row budget\n");}
