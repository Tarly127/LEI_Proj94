#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>
#include "../../lei/boost/boost_vector.cpp"

namespace lei {

struct Pair {
    double x;
    double y;
};

BOOST_COMPUTE_ADAPT_STRUCT(Pair, Pair, (x, y))

Pair make_pair() {
    Pair p;
    p.x = 0;
    p.y = 0;
    return p;
}

Pair make_pair2(double x, double y) {
    Pair p;
    p.x = x;
    p.y = y;
    return p;
}

BOOST_COMPUTE_FUNCTION(double, ax, (Pair p),
							{
                                return p.x + (2 * p.y);
							});
                            

void print_pair(Pair p){
    std::cout << "x ";
    std::cout << p.x << std::endl;
    std::cout << "y ";
    std::cout << p.y << std::endl;
    std::cout << "\n";
}

int main(int argc, char** argv){

    std::vector<Pair> vector = {};

    int test_num = atoi(argv[1]);
    int test_size = pow(2, test_num);

    for(int i = 0; i < test_size; i++){
        vector.push_back(make_pair2(1,1));
    }

	boost_vector<Pair> A(vector);

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<double> return_vector = A.maptoother(ax);

    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  
    std::cout << "Time taken by function: "
         << duration.count() << " microseconds" << std::endl;

	return 0;
}
}
