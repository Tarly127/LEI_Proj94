#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <chrono>


// Example of a simple matrix multiplication algorithm
// using the ijk method (with dot product)
// No bells and whistles

int main(int argc, char* argv[])
{

    if (argc < 2) return 0;

    int size = atoi(argv[1]);


    // Initialize the matrices
    double **A = (double**)malloc(sizeof(double*) * size);
    double **B = (double**)malloc(sizeof(double*) * size);
    double **C = (double**)malloc(sizeof(double*) * size);

    for(unsigned i = 0; i < size; i++)
    {
        A[i] = (double*)malloc(sizeof(double) * size);
        B[i] = (double*)malloc(sizeof(double) * size);
        C[i] = (double*)malloc(sizeof(double) * size);

        for(unsigned j = 0; j < size; j++)
        {
            A[i][j] = i * size + j;
            B[i][j] = i == j ? 1 : 0;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Matrix Multiplication
    for(unsigned i = 0; i < size; i++)
    {
        for(unsigned j = 0; j < size; j++)
        {
            double tmp = 0;

            for(unsigned k = 0; k < size; k++)
            {
                tmp += A[i][k] * B[k][j];
            }
            C[i][j] = tmp;
            tmp = 0;
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    //for(unsigned i = 0; i < size; i++)
    //{
    //    for(unsigned j = 0; j < size; j++)
    //        printf("%.1f ", C[i][j]);
    //    printf("\n");
    //}

    std::cout << "Duration: " << duration.count() << "ms" << std::endl;


    for(unsigned i = 0; i < size; i++)
    {
        free(A[i]); free(B[i]); free(C[i]);
    }

    free(A); free(B); free(C);

    return 0;

}