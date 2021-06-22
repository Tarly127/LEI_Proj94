#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <functional>

#include <boost/compute/core.hpp>
#include <boost/compute/utility/program_cache.hpp>
#include <boost/compute/function.hpp>
#include <boost/compute/utility/source.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/algorithm/reduce.hpp>
#include <boost/compute/container/vector.hpp>
#include <CL/cl.hpp>

namespace lei
{

template<typename T>
class boost_matrix
{
	private:
		// vector size
		size_t _size;
		// matrix rows
		size_t _rows;
        // matrix columns
        size_t _cols;
        // is column wise
        char _wise;
		// flattened matrix
		std::vector<T> flattened_matrix;
		// default device
		boost::compute::device device;
		// default context
		boost::compute::context ctx;
		// default queue
		boost::compute::command_queue queue;

	public:
		// class constructor
		boost_matrix(T** in_mat, size_t rows, size_t columns);
		// class constructor column wise
		boost_matrix(T** in_mat, size_t rows, size_t columns, char wise);
		//boost_matrix(T in_arr[], size_t in_size);
		// copy constructor
		//boost_matrix(boost_matrix<T> &in_mat);
		// empty constructor
		//boost_matrix(size_t size);

		// instance methods
		void show_myself();
		void show_myself(void (*f)(T));
		T& get(const int x, const int y);
		T& operator[](const int index);
		~boost_matrix();
		size_t size();


		void map(boost::compute::function<T(T)> fn);
		std::vector<T> maptoother(boost::compute::function<T(T)> fn);
		T reduce(boost::compute::function<T(T, T)> fn);
		T reduce(boost::compute::function<T(T, T)> fn, T (*init)(void));
		T mapreduce(boost::compute::function<T(T, T)> fn, T (*init)(void));
		template <typename T2>
		//T2 reduce(boost::compute::function<T2(T2, T)> fn, T2 (*init)(void));
		std::vector<T2> maptoother(boost::compute::function<T2(T)> fn);
};

template<typename T> 
boost_matrix<T>::boost_matrix(T** in_mat, size_t rows, size_t columns){
	// getting size
    _size = rows*columns;
    // getting rows
    _rows = rows;
    // getting columns
    _cols = columns;
    // column wise
    _wise = 0;

	// flatten matrix into vector
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
                flattened_matrix.push_back(in_mat[i][j]);
        }
    }	

	// setup context, device and queue
	device = boost::compute::system::default_device();	
	ctx = boost::compute::system::default_context();
	queue = boost::compute::system::default_queue();
}

template<typename T> 
boost_matrix<T>::boost_matrix(T** in_mat, size_t rows, size_t columns, char wise){
	// getting size
    _size = rows*columns;
    // getting rows
    _rows = rows;
    // getting columns
    _cols = columns;
    // column wise
    _wise = 1;

    // initializing matrix vector
    std::vector<T> flattened_matrix = {};

	// flatten matrix into vector
    for(int i = 0; i < columns; i++){
        for(int j = 0; j < rows; j++){
                flattened_matrix.push_back(in_mat[i][j]);
        }
    }

	// setup context, device and queue
	device = boost::compute::system::default_device();	
	ctx = boost::compute::system::default_context();
	queue = boost::compute::system::default_queue();
}

template<typename T> 
void boost_matrix<T>::map(boost::compute::function<T(T)> fn){
	// create a vector on the device
    boost::compute::vector<T> device_vector(flattened_matrix.size(), ctx);

    // transfer data from the host to the device
    boost::compute::copy(
        flattened_matrix.begin(), flattened_matrix.end(), device_vector.begin(), queue
    );

	// map
    boost::compute::transform(
		device_vector.begin(), device_vector.end(), device_vector.begin(), fn, queue
	);

	// copy values back to the host
    boost::compute::copy(
        device_vector.begin(), device_vector.end(), flattened_matrix.begin(), queue
    );
}

/* (Irrelevante?)
template<typename T> 
std::vector<T> boost_matrix<T>::maptoother(boost::compute::function<T(T)> fn){
	// create a vector on the device
    boost::compute::vector<T> device_vector(flattened_matrix.size(), ctx);

    // transfer data from the host to the device
    boost::compute::copy(
        flattened_matrix.begin(), flattened_matrix.end(), device_vector.begin(), queue
    );

	// map
    boost::compute::transform(
		device_vector.begin(), device_vector.end(), device_vector.begin(), fn, queue
	);

	// create a new return host vector
	std::vector<T> flattened_matrix_r;
	flattened_matrix_r.resize(flattened_matrix.size());

	// copy values back to the host
    boost::compute::copy(
        device_vector.begin(), device_vector.end(), flattened_matrix_r.begin(), queue
    );
	
	return flattened_matrix_r;
}
*/

template<typename T> 
template<typename T2>
std::vector<T2> boost_matrix<T>::maptoother(boost::compute::function<T2(T)> fn){
	// create a vector on the device
    boost::compute::vector<T> device_vector(flattened_matrix.size(), ctx);

    // transfer data from the host to the device
    boost::compute::copy(
        flattened_matrix.begin(), flattened_matrix.end(), device_vector.begin(), queue
    );

	// create a return vector on the device
    boost::compute::vector<T2> device_vector_r(flattened_matrix.size(), ctx);

	// create a new return host vector
	std::vector<T2> flattened_matrix_r;
	flattened_matrix_r.resize(flattened_matrix.size());

	// map
    boost::compute::transform(
		device_vector.begin(), device_vector.end(), device_vector_r.begin(), fn, queue
	);

	// copy values back to the host
    boost::compute::copy(
        device_vector_r.begin(), device_vector_r.end(), flattened_matrix_r.begin(), queue
    );
	
	return flattened_matrix_r;
}


template<typename T>
T boost_matrix<T>::reduce(boost::compute::function<T(T, T)> fn){
	// create a vector on the device
    boost::compute::vector<T> device_vector(flattened_matrix.size(), ctx);

    // transfer data from the host to the device
    boost::compute::copy(
        flattened_matrix.begin(), flattened_matrix.end(), device_vector.begin(), queue
    );

	T result = 0;

	// reduce
    boost::compute::reduce(
		device_vector.begin(), device_vector.end(), &result, fn, queue
	);

	return result;
}

template<typename T>
T boost_matrix<T>::reduce(boost::compute::function<T(T, T)> fn, T (*init)(void)){
	// create a vector on the device
    boost::compute::vector<T> device_vector(flattened_matrix.size(), ctx);

    // transfer data from the host to the device
    boost::compute::copy(
        flattened_matrix.begin(), flattened_matrix.end(), device_vector.begin(), queue
    );

	T result = init();

	// reduce
    boost::compute::reduce(
		device_vector.begin(), device_vector.end(), &result, fn, queue
	);

	return result;
}

template<typename T>
T boost_matrix<T>::mapreduce(boost::compute::function<T(T, T)> fn, T (*init)(void)){
	// create a vector on the device
    boost::compute::vector<T> device_vector(flattened_matrix.size(), ctx);

    // transfer data from the host to the device
    boost::compute::copy(
        flattened_matrix.begin(), flattened_matrix.end(), device_vector.begin(), queue
    );
	
	// map
    boost::compute::transform(
		device_vector.begin(), device_vector.end(), device_vector.begin(), fn, queue
	);

	// copy values back to the host
    boost::compute::copy(
        device_vector.begin(), device_vector.end(), flattened_matrix.begin(), queue
    );

	T result = init();

	// reduce
    boost::compute::reduce(
		device_vector.begin(), device_vector.end(), &result, fn, queue
	);

	return result;
}

/*
template<typename T>
template<typename T2>
T2 boost_matrix<T>::reduce(boost::compute::function<T2(T2, T)> fn, T2 (*init)(void)){
	// create a vector on the device
    boost::compute::vector<T> device_vector(flattened_matrix.size(), ctx);

    // transfer data from the host to the device
    boost::compute::copy(
        flattened_matrix.begin(), flattened_matrix.end(), device_vector.begin(), queue
    );

	T2 result = init();

	// reduce
    boost::compute::reduce(device_vector.begin(), device_vector.end(), &result, fn, queue);

	return result;
}
*/

template<typename T>
T& boost_matrix<T>  ::get(const int x, const int y){
    if(_wise){
        return flattened_matrix[(y * _cols) + x];
    } else {
        return flattened_matrix[(x * _rows) + y];
    }    
}

template<typename T>
void boost_matrix<T>::show_myself(){
    std::cout << "------------------------------------" << std::endl;
    std::cout << "Host Vector: \n";
	// print vector
    for (auto i = flattened_matrix.begin(); i != flattened_matrix.end(); ++i)
        std::cout << *i << ' ';
    std::cout << '\n';
    std::cout << "------------------------------------" << std::endl;
}

template<typename T>
void boost_matrix<T>::show_myself(void (*f)(T)){
    std::cout << "------------------------------------" << std::endl;
    std::cout << "Host Vector: \n";
	// print vector
    for (auto i = flattened_matrix.begin(); i != flattened_matrix.end(); ++i)
        f(*i);
    std::cout << "------------------------------------" << std::endl;
}

template<typename T>
size_t boost_matrix<T>::size(){ return this->_size; }

template<typename T>
boost_matrix<T>::~boost_matrix(){
    #ifndef __APPLE__
    #endif
}

}