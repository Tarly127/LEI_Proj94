#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>
#include "../../lei/boost/boost_vector.cpp"

namespace lei {

struct Trio {
    double x;
    double y;
    double a;
};

BOOST_COMPUTE_ADAPT_STRUCT(Trio, Trio, (x, y, a))

Trio make_pair() {
    Trio p;
    p.x = 0;
    p.y = 0;
    p.a = 0;
    return p;
}

Trio make_pair2(double x, double y, double a) {
    Trio p;
    p.x = x;
    p.y = y;
    p.a = a;
    return p;
}

BOOST_COMPUTE_FUNCTION(double, ax, (Trio p),
							{
                                return p.x + (p.a * p.y);
							});
                            

void print_pair(Trio p){
    std::cout << "x ";
    std::cout << p.x << std::endl;
    std::cout << "y ";
    std::cout << p.y << std::endl;
    std::cout << "a ";
    std::cout << p.a << std::endl;
    std::cout << "\n";
}

int main(int argc, char** argv){

    std::vector<Trio> vector = {};

    int test_num = atoi(argv[1]);
    int test_size = pow(2, test_num);

    for(int i = 0; i < test_size; i++){
        vector.push_back(make_pair2(1,1,2));
    }

	boost_vector<Trio> A(vector);

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<double> return_vector = A.maptoother(ax);

    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  
    std::cout << "Time taken by function: "
         << duration.count() << " microseconds" << std::endl;

	return 0;
}
}
