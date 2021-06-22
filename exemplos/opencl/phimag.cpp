#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "../../lei/gpu/gpu_vector.cpp"

int main(int argc, char** argv){

    int N = 10;
    
    std::vector<float> aux;

    for(auto i = 0; i < N; i++)
        aux.push_back(i);

    std::string phimag =
        "float phimag(float phiMag, float real, float imag){ return ((real*real) + (imag*imag)); }";

    lei::gpu_vector<float> vec(aux, N);

    vec.map3vectors(aux, aux, phimag, "phimag");

    vec.show();

    return 0;
}