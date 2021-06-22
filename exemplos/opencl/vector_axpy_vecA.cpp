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

    std::vector<double> aux1;
    std::vector<double> aux2;
    std::vector<double> aux3;

    for(auto i = 0; i < N; i++){
        aux1.push_back(rand());
        aux2.push_back(rand());
        aux3.push_back(rand());
    }

    std::string axpy =
        "double axpy(double x, double a, double y){ return (a*x)+y; }";

    auto start = std::chrono::high_resolution_clock::now();
    lei::gpu_vector<double> vec(aux1, N);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Allocation Time: " << duration.count() << "ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    vec.map3vectors(aux2, aux3, axpy, "axpy");
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Execution Time: " << duration.count() << "ms" << std::endl;

    return 0;
}