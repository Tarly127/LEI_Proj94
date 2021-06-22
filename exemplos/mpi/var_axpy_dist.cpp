#include "../../lei/lei.cpp"
#include <chrono>
#include <iostream>
#include <tuple>

#define s3 (unsigned long long)(2 << 27)
#define s2 (unsigned long long)(2 << 25)
#define s1 (unsigned long long)(2 << 23)
#define s0 (unsigned long long)(2 << 21)

#define t2d std::tuple<double, double>

void axpy_kernel(t2d * elem, double& multiplier)
{
    std::get<0>(*elem) = std::get<1>(*elem);
}


int main(int argc, char** argv)
{
    auto fstart = std::chrono::high_resolution_clock::now();

    if( argc != 2 )
        return 0;

    unsigned long long size = 0;

    switch(argv[1][0])
    {
        case '0' : size = s0; break;
        case '1' : size = s1; break;
        case '2' : size = s2; break;
        case '3' : size = s3; break;
        default  : size = s0;
    }

    auto start = std::chrono::high_resolution_clock::now();

    lei::init_lei(argc, argv);

    t2d * input = (t2d*)malloc(sizeof(t2d) * size);

    for(auto i = 0; i < size; i++)
    {
        input[i] = std::make_tuple(1.0,1.0);
    }

    lei::mpi_vector<t2d> v(input, size, lei::BLOCK);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << ","; // Initialization Time

    start = std::chrono::high_resolution_clock::now();

    double multiplier = 2.0;

    v.map(axpy_kernel, multiplier);

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << ","; // Exec Time
    

    free(input);

    auto fstop = std::chrono::high_resolution_clock::now();
    auto fduration = std::chrono::duration_cast<std::chrono::milliseconds>(fstop - fstart);
    std::cout << fduration.count() << std::endl;

    lei::finish_lei();
}