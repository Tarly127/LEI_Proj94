#include "../../lei/lei.cpp"
#include <cmath>
#include <chrono>

#define s3 (unsigned long long)(2 << 14)
#define s2 (unsigned long long)(2 << 13)
#define s1 (unsigned long long)(2 << 12)
#define s0 (unsigned long long)(2 << 11)

double frobenius_kernel(double a, double b)
{
    return a + b * b;
}

double frobenius_norm(lei::mpi_matrix<double>& A)
{    
    double norm = A.reduce(&frobenius_kernel);
    return sqrt(norm);
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

    double** input = (double**)malloc(sizeof(double*) * size);

    for(unsigned long long i = 0; i < size; i++)
    {
        input[i] = (double*)malloc(sizeof(double) * size);

        for(unsigned long long j = 0; j < size; j++)
        {
            input[i][j] = 1.0;
        }
    }

    lei::mpi_matrix<double> v(input, size, size, lei::ROW_WISE);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << ",";

    start = std::chrono::high_resolution_clock::now();

    double fn = frobenius_norm(v);

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << ",";

    for(auto i = 0; i < size; i++)
        free(input[i]);

    free(input);
    
    auto fstop = std::chrono::high_resolution_clock::now();
    auto fduration = std::chrono::duration_cast<std::chrono::milliseconds>(fstop - fstart);
    std::cout << fduration.count() << std::endl;

    lei::finish_lei();


}