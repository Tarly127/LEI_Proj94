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

template<typename T>
class boost_vector
{
	private:
		// vector size
		size_t _size;
		// host vector
		std::vector<T> host_vector;
		// default device
		boost::compute::device device;
		// default context
		boost::compute::context ctx;
		// default queue
		boost::compute::command_queue queue;

	public:
		// class constructors
		boost_vector(std::vector<T> in_vec);
		//boost_vector(T in_arr[], size_t in_size);
		// copy constructor
		//boost_vector(boost_vector<T> &in_vec);
		// empty constructor
		//boost_vector(size_t size);

		// instance methods
		void show_myself();
		void show_myself(void (*f)(T));
		T& get(const int index);
		T& operator[](const int index);
		~boost_vector();
		size_t size();


		void map(boost::compute::function<T(T)> fn);
		//std::vector<T> maptoother(boost::compute::function<T(T)> fn);
		T reduce(boost::compute::function<T(T, T)> fn);
		T reduce(boost::compute::function<T(T, T)> fn, T (*init)(void));
		T mapreduce(boost::compute::function<T(T)> fn1, boost::compute::function<T(T, T)> fn2, T (*init)(void));
		template <typename T2>
		std::vector<T2> maptoother(boost::compute::function<T2(T)> fn);
		template <typename T2>
		T2 mapreducetoother(boost::compute::function<T2(T)> fn1, boost::compute::function<T2(T2, T2)> fn2, T2 (*init)(void));
};

template<typename T> 
boost_vector<T>::boost_vector(std::vector<T> in_vec){
	// getting size
    _size = in_vec.size();

	// declaring host vector
	std::copy(
		in_vec.begin(), in_vec.end(), back_inserter(host_vector)
	);

	// setup context, device and queue
	device = boost::compute::system::default_device();	
	ctx = boost::compute::system::default_context();
	queue = boost::compute::system::default_queue();
}

template<typename T> 
void boost_vector<T>::map(boost::compute::function<T(T)> fn){
	// create a vector on the device
    boost::compute::vector<T> device_vector(host_vector.size(), ctx);

    // transfer data from the host to the device
    boost::compute::copy(
        host_vector.begin(), host_vector.end(), device_vector.begin(), queue
    );

	// map
    boost::compute::transform(
		device_vector.begin(), device_vector.end(), device_vector.begin(), fn, queue
	);

	// copy values back to the host
    boost::compute::copy(
        device_vector.begin(), device_vector.end(), host_vector.begin(), queue
    );
}

/* (Irrelevante?)
template<typename T> 
std::vector<T> boost_vector<T>::maptoother(boost::compute::function<T(T)> fn){
	// create a vector on the device
    boost::compute::vector<T> device_vector(host_vector.size(), ctx);

    // transfer data from the host to the device
    boost::compute::copy(
        host_vector.begin(), host_vector.end(), device_vector.begin(), queue
    );

	// map
    boost::compute::transform(
		device_vector.begin(), device_vector.end(), device_vector.begin(), fn, queue
	);

	// create a new return host vector
	std::vector<T> host_vector_r;
	host_vector_r.resize(host_vector.size());

	// copy values back to the host
    boost::compute::copy(
        device_vector.begin(), device_vector.end(), host_vector_r.begin(), queue
    );
	
	return host_vector_r;
}
*/

template<typename T> 
template<typename T2>
std::vector<T2> boost_vector<T>::maptoother(boost::compute::function<T2(T)> fn){
	// create a vector on the device
    boost::compute::vector<T> device_vector(host_vector.size(), ctx);

    // transfer data from the host to the device
    boost::compute::copy(
        host_vector.begin(), host_vector.end(), device_vector.begin(), queue
    );

	// create a return vector on the device
    boost::compute::vector<T2> device_vector_r(host_vector.size(), ctx);

	// create a new return host vector
	std::vector<T2> host_vector_r;
	host_vector_r.resize(host_vector.size());

	// map
    boost::compute::transform(
		device_vector.begin(), device_vector.end(), device_vector_r.begin(), fn, queue
	);

	// copy values back to the host
    boost::compute::copy(
        device_vector_r.begin(), device_vector_r.end(), host_vector_r.begin(), queue
    );
	
	return host_vector_r;
}


template<typename T>
T boost_vector<T>::reduce(boost::compute::function<T(T, T)> fn){
	// create a vector on the device
    boost::compute::vector<T> device_vector(host_vector.size(), ctx);

    // transfer data from the host to the device
    boost::compute::copy(
        host_vector.begin(), host_vector.end(), device_vector.begin(), queue
    );

	T result = 0;

	// reduce
    boost::compute::reduce(
		device_vector.begin(), device_vector.end(), &result, fn, queue
	);

	return result;
}

template<typename T>
T boost_vector<T>::reduce(boost::compute::function<T(T, T)> fn, T (*init)(void)){
	// create a vector on the device
    boost::compute::vector<T> device_vector(host_vector.size(), ctx);

    // transfer data from the host to the device
    boost::compute::copy(
        host_vector.begin(), host_vector.end(), device_vector.begin(), queue
    );

	T result = init();

	// reduce
    boost::compute::reduce(
		device_vector.begin(), device_vector.end(), &result, fn, queue
	);

	return result;
}

template<typename T>
T boost_vector<T>::mapreduce(boost::compute::function<T(T)> fn1, boost::compute::function<T(T, T)> fn2, T (*init)(void)){
	// create a vector on the device
    boost::compute::vector<T> device_vector(host_vector.size(), ctx);

    // transfer data from the host to the device
    boost::compute::copy(
        host_vector.begin(), host_vector.end(), device_vector.begin(), queue
    );
	
	// map
    boost::compute::transform(
		device_vector.begin(), device_vector.end(), device_vector.begin(), fn1, queue
	);

	// copy values back to the host
    boost::compute::copy(
        device_vector.begin(), device_vector.end(), host_vector.begin(), queue
    );

	T result = init();

	// reduce
    boost::compute::reduce(
		device_vector.begin(), device_vector.end(), &result, fn2, queue
	);

	return result;
}

template<typename T> 
template<typename T2>
T2 boost_vector<T>::mapreducetoother(boost::compute::function<T2(T)> fn1, boost::compute::function<T2(T2, T2)> fn2, T2 (*init)(void)){
	// create a vector on the device
    boost::compute::vector<T> device_vector(host_vector.size(), ctx);

    // transfer data from the host to the device
    boost::compute::copy(
        host_vector.begin(), host_vector.end(), device_vector.begin(), queue
    );

	// create a return vector on the device
    boost::compute::vector<T2> device_vector_r(host_vector.size(), ctx);

	// map
    boost::compute::transform(
		device_vector.begin(), device_vector.end(), device_vector_r.begin(), fn1, queue
	);

	T2 result = init();

	// reduce
    boost::compute::reduce(
		device_vector_r.begin(), device_vector_r.end(), &result, fn2, queue
	);

	return result;
}

template<typename T>
T& boost_vector<T>  ::get(const int index){
    return host_vector[index];
}

template<typename T>
void boost_vector<T>::show_myself(){
    std::cout << "------------------------------------" << std::endl;
    std::cout << "Host Vector: \n";
	// print vector
    for (auto i = host_vector.begin(); i != host_vector.end(); ++i)
        std::cout << *i << ' ';
    std::cout << '\n';
    std::cout << "------------------------------------" << std::endl;
}

template<typename T>
void boost_vector<T>::show_myself(void (*f)(T)){
    std::cout << "------------------------------------" << std::endl;
    std::cout << "Host Vector: \n";
	// print vector
    for (auto i = host_vector.begin(); i != host_vector.end(); ++i)
        f(*i);
    std::cout << "------------------------------------" << std::endl;
}

template<typename T>
size_t boost_vector<T>::size(){ return this->_size; }

template<typename T>
boost_vector<T>::~boost_vector(){
    #ifndef __APPLE__
    #endif
}