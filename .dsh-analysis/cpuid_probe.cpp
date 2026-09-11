#include <cstdio>
#include <intrin.h>
int main(){
  int r[4];
  __cpuidex(r,7,0);
  printf("leaf7.0 EBX=0x%08x ECX=0x%08x EDX=0x%08x\n", r[1], r[2], r[3]);
  auto b=[&](int reg,int bit){int v = (reg==0?r[0]:reg==1?r[1]:reg==2?r[2]:r[3]);return (v>>bit)&1;};
  printf("AVX512F=%d DQ=%d IFMA=%d BW=%d VL=%d\n", b(1,16), b(1,17), b(1,21), b(1,30), b(1,31));
  printf("AVX512VBMI=%d VBMI2=%d VNNI=%d BITALG=%d VPOPCNTDQ=%d BF16=%d\n", b(2,1), b(2,6), b(2,11), b(2,12), b(2,14), b(1,5));
  printf("AVX512VNNI(EBX bit4 AVX-VNNI)=%d\n", b(1,4));
  int a[4]; __cpuidex(a,0x80000006,0);
  printf("L2=%uKB L3=%uKB\n", (a[2]>>16)&0xffff, (a[3]>>18)*512);
  return 0;
}
