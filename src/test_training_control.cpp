#include "training_control.hpp"
#include <sstream>
#include <cstdio>
int main(){auto c=tao::dual::read_control("build/nonexistent_control_fixture",{.0005f,224,false});if(c.lr!=.0005f||c.target!=224||c.stop)return 1;auto v=tao::dual::read_control("build/control_valid_fixture.txt",c);if(!v.stop||v.target!=224||v.lr!=.0005f)return 2;bool rejected=false;try{tao::dual::read_control("build/control_invalid_fixture.txt",c);}catch(...){rejected=true;}if(!rejected)return 3;printf("PASS defaults, explicit LR/target/stop, invalid control rejection\n");}
