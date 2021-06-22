#include "../../lei/lei.cpp"
#include <chrono>
#include <iostream>

#define s3 (unsigned long long)(2 << 28)
#define s2 (unsigned long long)(2 << 26)
#define s1 (unsigned long long)(2 << 24)
#define s0 (unsigned long long)(2 << 22)

double sum_kernel(double a, double b)
{
    return a + b;
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
    // start time measurement

    auto start = std::chrono::high_resolution_clock::now();

    lei::init_lei(argc, argv);

    double * input = (double*)malloc(sizeof(double) * size);

    for(unsigned long long i = 0; i < size; i++)
    {
        input[i] = 1.0;    
    }

    // initialize structure
    lei::mpi_vector<double> v(input, size, lei::BLOCK);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Initialization time: " << (double)duration.count()/1000.0 << std::endl;
    start = std::chrono::high_resolution_clock::now();

    //std::cout << v.size() << std::endl;


    //perform the reduction
    double reduction = v.reduce(&sum_kernel);

    //printf("%f\n", reduction);


    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Execution time: " << (double)duration.count()/1000.0 << std::endl;


    free(input);

    auto fstop = std::chrono::high_resolution_clock::now();
    auto fduration = std::chrono::duration_cast<std::chrono::microseconds>(fstop - fstart);
    std::cout << "Total Execution time: " << (double)fduration.count()/1000.0 << std::endl;

    lei::finish_lei();




}