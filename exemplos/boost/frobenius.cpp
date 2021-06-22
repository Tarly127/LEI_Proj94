#include <stdio.h>
#include <iostream>
#include <vector>
#include "../../lei/boost/boost_vector.cpp"

namespace lei {

BOOST_COMPUTE_FUNCTION(double, mapsquare, (double x),
							{
								return x * x;
							});

// o primeiro argumento para o reduce é o acumulador

BOOST_COMPUTE_FUNCTION(double, reducesum, (double x, double y),
							{
								return x + y;
							});

int main(int argc, char** argv){
    int rows = atoi(argv[1]);
    int cols = atoi(argv[2]);
    double** matrix = new double*[rows];
    for (int i = 0; i < rows; ++i)
        matrix[i] = new double[cols];    
	
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i][j] = (i * rows) + j + 1;

	boost_matrix<double> A(matrix, rows, cols);

    auto start = std::chrono::high_resolution_clock::now();

    A.map(mapsquare);

    double n = A.reduce(reducesum);

    n = sqrt(n);


    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  
    std::cout << "Time taken by function: "
         << duration.count() << " microseconds" << std::endl;
         
    std::cout << n << std::endl;
    std::cout << "\n" << std::endl;
    
    for (int i = 0; i < rows; ++i)
        delete [] matrix[i];
    delete [] matrix;

	return 0;
}
}
