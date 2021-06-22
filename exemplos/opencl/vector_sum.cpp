#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <chrono>
#include "../../lei/gpu/gpu_vector.cpp"

int main(int argc, char** argv){

    int N = atoi(argv[1]);

	srand(time(NULL));
    
    std::vector<double> aux;

    for(auto i = 0; i < N; i++)
        aux.push_back(rand());

    std::string sum =
        "double sum(double x, double y){ return x+y; }";

    auto start = std::chrono::high_resolution_clock::now();
    lei::gpu_vector<double> vec(aux, N);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Allocation Time: " << duration.count() << "ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    vec.reduce(sum, "sum");
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Execution Time: " << duration.count() << "ms" << std::endl;

    return 0;
}