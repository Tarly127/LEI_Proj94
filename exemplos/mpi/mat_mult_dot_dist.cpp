#include "../../lei/lei.cpp"
#include <cmath>
#include <tuple>
#include <chrono>

#define s3 (unsigned long long)(4096)
#define s2 (unsigned long long)(2048)
#define s1 (unsigned long long)(1024)
#define s0 (unsigned long long)(512)

using namespace lei;

void mat_mult_dot(mpi_matrix<double>& A, mpi_matrix<double>& B, mpi_matrix<double>& C)
{
    if(   A.total_lines()     == B.total_lines() 
       && A.total_lines()     == C.total_lines() 
       && B.total_lines()     == C.total_columns()
       && A.getDistribution() == ROW_WISE
       && B.getDistribution() == MCOPY_TRANSPOSE
       && C.getDistribution() == ROW_WISE
      )
    {
        for(unsigned i = 0; i < A.lines(); i++)
        {
            for(unsigned j = 0; j < B.lines(); j++)
            {
                double accum = 0.0;

                for(unsigned k = 0; k < A.columns(); k++)
                {
                    accum += *A.get(i,k) * *B.get(j,k);
                }

                C.set(i,j,accum);
            }
        }
    }
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

    init_lei(argc, argv);

    // Initialize the matrices
    double **A = (double**)malloc(sizeof(double*) * size);
    double **B = (double**)malloc(sizeof(double*) * size);

    for(unsigned i = 0; i < size; i++)
    {
        A[i] = (double*)malloc(sizeof(double) * size);
        B[i] = (double*)malloc(sizeof(double) * size);

        for(unsigned j = 0; j < size; j++)
        {
            A[i][j] = i * size + j;
            B[i][j] = i == j ? 1 : 0;
        }
    }


    mpi_matrix<double> Ad (A, size, size, ROW_WISE);
    mpi_matrix<double> Bd (B, size, size, MCOPY_TRANSPOSE);
    mpi_matrix<double> Cd    (size, size, ROW_WISE);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << ","; // Init Time

    start = std::chrono::high_resolution_clock::now();

    mat_mult_dot(Ad, Bd, Cd);

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << ","; // Exec Time


    for(unsigned i = 0; i < size; i++)
    {
        free(A[i]); free(B[i]);
    }

    free(A); free(B);

    auto fstop = std::chrono::high_resolution_clock::now();
    auto fduration = std::chrono::duration_cast<std::chrono::milliseconds>(fstop - fstart);
    std::cout << fduration.count() << std::endl; // Tot Time

    finish_lei();
}