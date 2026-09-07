
#include <omp.h>
#include <iostream>
int main() {
    int n; 
    #pragma omp parallel
    {
        #pragma omp single
        n = omp_get_num_threads();
    }
    std::cout << "threads: " << n << std::endl;
    return 0;
}
