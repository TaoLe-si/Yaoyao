#include <intrin.h>
#include <cstdio>
int main(){int r[4];__cpuid(r,1);bool os=(r[2]&(1<<27))!=0;unsigned long long x=os?_xgetbv(0):0;__cpuidex(r,7,0);printf("xcr0=%llx AVX2=%d AVX512F=%d AVX512BW=%d AVX512VL=%d AVX512VNNI=%d os_zmm=%d\n",x,(r[1]>>5)&1,(r[1]>>16)&1,(r[1]>>30)&1,(r[1]>>31)&1,(r[2]>>11)&1,(x&0xe6)==0xe6);}
