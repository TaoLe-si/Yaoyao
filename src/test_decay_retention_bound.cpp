#include <cmath>
#include <cstdio>
#include <stdexcept>
int main(){try{for(double p:{.01,.05,.1})for(int steps:{128,1024,10000}){double rate=p*p,old=1;for(int i=0;i<steps;++i)old*=1-rate;double formula=std::pow(1-rate,steps);if(std::abs(old-formula)>1e-12)throw std::runtime_error("recurrence");printf("p=%.2f n=%d squared_retention=%.9f exact_rejection_retention=1\n",p,steps,old);}for(double p:{.25,.5,.75})printf("one_update p=%.2f baseline_new=%.3f square_new=%.3f square_old=%.3f\n",p,p,p*p,1-p*p);return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}
