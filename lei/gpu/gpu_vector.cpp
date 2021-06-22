//#define CL_HPP_ENABLE_EXCEPTIONS
//#define CL_HPP_TARGET_OPENCL_VERSION 120
//#define CL_HPP_MINIMUM_OPENCL_VERSION 120
//#define CL_HPP_CL_1_2_DEFAULT_BUILD
//#include "opencl.hpp"

//no search usar em vez das linhas anteriores
#include <CL/cl.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <functional>

using namespace cl;

namespace lei
{

template<typename T>
class gpu_vector{

    private:
        //context
        Context context;
        //queue
        CommandQueue queue;
        //length of the buffer
        int length;
        //local vector with information
        std::vector<T> vec;
        //buffer with the information
        Buffer buffer;
        
        std::string get_type();

    public:

        //class constructors
        gpu_vector(gpu_vector<T> &originalVec);
        gpu_vector(std::vector<T> vec, int size);
        gpu_vector(T* array, int size);
        
        // iterators
        typename std::vector<T>::iterator begin() { return vec.begin(); }
        typename std::vector<T>::iterator end()   { return vec.end(); }

        //instance methods
        void map(std::string func_code, const char* func_name);
        void map3vectors(std::vector<T> A, std::vector<T> B, std::string func_code, const char* func_name);
        void map3arrays(T* A, T* B, std::string func_code, const char* func_name);
        void map2vectors(std::vector<T> A, std::string func_code, const char* func_name);
        void map2arrays(T* A, std::string func_code, const char* func_name);
        T    reduce(std::string func_code, const char* func_name);
        int  size();
        void show();

};

template<typename T>
gpu_vector<T>::gpu_vector(gpu_vector<T> &originalVec){
    context = Context(CL_DEVICE_TYPE_DEFAULT);

    //create queue to which we will push commands for the device.
    queue = CommandQueue(context);

    //storing local data
    length = originalVec.size();

    for(auto i = originalVec.begin(); i != originalVec.end(); i++)
        vec.push_back(*i);

    //create buffer to store the data
    buffer = Buffer(context, vec.begin(), vec.end(), false);
}

template<typename T>
gpu_vector<T>::gpu_vector(std::vector<T> originalVec, int size){
    context = Context(CL_DEVICE_TYPE_DEFAULT);

    //create queue to which we will push commands for the device.
    queue = CommandQueue(context);

    //storing local data
    length = size;
    vec = originalVec;

    //create buffer to store the data
    buffer = Buffer(context, vec.begin(), vec.end(), false);
}

template<typename T>
gpu_vector<T>::gpu_vector(T* array, int size){
    context = Context(CL_DEVICE_TYPE_DEFAULT);

    //create queue to which we will push commands for the device.
    queue = CommandQueue(context);

    //storing local data
    length = size;

    for(auto i = 0; i < length; i++)
        vec.push_back(array[i]);

    //create buffer to store the data
    buffer = Buffer(context, vec.begin(), vec.end(), false);
}

template<typename T>
void gpu_vector<T>::map(std::string func_code, const char* func_name){
    Program::Sources sources;

    std::string name(func_name);

    std::string kernel_code =
        func_code + "\n" +
        "void kernel func_kernel(global " + get_type() + "* buffer){               "
        "    buffer[get_global_id(0)] = " + name + "(buffer[get_global_id(0)]);    "
        "}                                                                         ";

    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    Program program(context, sources);
    if(program.build() != CL_SUCCESS){
        std::cout << " Error building\n";
        exit(1);
    }

    //run the kernel (openCL 1.2)
    Kernel kernel = Kernel(program, "func_kernel");
    kernel.setArg(0, buffer);
    queue.enqueueNDRangeKernel(kernel, NullRange, NDRange(length), NullRange);
    queue.finish();

    copy(queue, buffer, vec.begin(), vec.end());
}

template<typename T>
void gpu_vector<T>::map2vectors(std::vector<T> A, std::string func_code, const char* func_name){
    Program::Sources sources;

    std::string name(func_name);

    std::string kernel_code =
        func_code + "\n" +
        "void kernel func_kernel(global " + get_type() + "* buffer, global const " + get_type() + "* A){"
        "    buffer[get_global_id(0)] = " + name + "(buffer[get_global_id(0)], A[get_global_id(0)]);    "
        "}                                                                                              ";

    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    Program program(context, sources);
    if(program.build() != CL_SUCCESS){
        std::cout << " Error building\n";
        exit(1);
    }

    Buffer bufferA(context, A.begin(), A.end(), true);

    //run the kernel (openCL 1.2)
    Kernel kernel = Kernel(program, "func_kernel");
    kernel.setArg(0, buffer);
    kernel.setArg(1, bufferA);
    queue.enqueueNDRangeKernel(kernel, NullRange, NDRange(length), NullRange);
    queue.finish();

    copy(queue, buffer, vec.begin(), vec.end());
}

template<typename T>
void gpu_vector<T>::map2arrays(T* A, std::string func_code, const char* func_name){
    Program::Sources sources;

    std::string name(func_name);

    std::string kernel_code =
        func_code + "\n" +
        "void kernel func_kernel(global " + get_type() + "* buffer, global const " + get_type() + "* A){"
        "    buffer[get_global_id(0)] = " + name + "(buffer[get_global_id(0)], A[get_global_id(0)]);    "
        "}                                                                                              ";

    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    Program program(context, sources);
    if(program.build() != CL_SUCCESS){
        std::cout << " Error building\n";
        exit(1);
    }

    cl::Buffer bufferA(context, CL_MEM_READ_ONLY, sizeof(T)*length);
    queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(T)*length, A);

    //run the kernel (openCL 1.2)
    Kernel kernel = Kernel(program, "func_kernel");
    kernel.setArg(0, buffer);
    kernel.setArg(1, bufferA);
    queue.enqueueNDRangeKernel(kernel, NullRange, NDRange(length), NullRange);
    queue.finish();

    copy(queue, buffer, vec.begin(), vec.end());
}

template<typename T>
void gpu_vector<T>::map3vectors(std::vector<T> A, std::vector<T> B, std::string func_code, const char* func_name){
    Program::Sources sources;

    std::string name(func_name);

    std::string kernel_code =
        func_code + "\n" +
        "void kernel func_kernel(global " + get_type() + "* buffer, global const " + get_type() + "* A, global const " + get_type() + "* B){"
        "    buffer[get_global_id(0)] = " + name + "(buffer[get_global_id(0)], A[get_global_id(0)], B[get_global_id(0)]);                   "
        "}                                                                                                                                  ";

    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    Program program(context, sources);
    if(program.build() != CL_SUCCESS){
        std::cout << " Error building\n";
        exit(1);
    }

    Buffer bufferA(context, A.begin(), A.end(), true);
    Buffer bufferB(context, B.begin(), B.end(), true);

    //run the kernel (openCL 1.2)
    Kernel kernel = Kernel(program, "func_kernel");
    kernel.setArg(0, buffer);
    kernel.setArg(1, bufferA);
    kernel.setArg(2, bufferB);
    queue.enqueueNDRangeKernel(kernel, NullRange, NDRange(length), NullRange);
    queue.finish();

    copy(queue, buffer, vec.begin(), vec.end());
}

template<typename T>
void gpu_vector<T>::map3arrays(T* A, T* B, std::string func_code, const char* func_name){
    Program::Sources sources;

    std::string name(func_name);

    std::string kernel_code =
        func_code + "\n" +
        "void kernel func_kernel(global " + get_type() + "* buffer, global const " + get_type() + "* A, global const " + get_type() + "* B){"
        "    buffer[get_global_id(0)] = " + name + "(buffer[get_global_id(0)], A[get_global_id(0)], B[get_global_id(0)]);                   "
        "}                                                                                                                                  ";

    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    Program program(context, sources);
    if(program.build() != CL_SUCCESS){
        std::cout << " Error building\n";
        exit(1);
    }

    cl::Buffer bufferA(context, CL_MEM_READ_ONLY, sizeof(T)*length);
    cl::Buffer bufferB(context, CL_MEM_READ_ONLY, sizeof(T)*length);
    queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(T)*length, A);
    queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, sizeof(T)*length, B);

    //run the kernel (openCL 1.2)
    Kernel kernel = Kernel(program, "func_kernel");
    kernel.setArg(0, buffer);
    kernel.setArg(1, bufferA);
    kernel.setArg(2, bufferB);
    queue.enqueueNDRangeKernel(kernel, NullRange, NDRange(length), NullRange);
    queue.finish();

    copy(queue, buffer, vec.begin(), vec.end());
}

template<typename T>
T gpu_vector<T>::reduce(std::string func_code, const char* func_name){
    Program::Sources sources;

    std::string name(func_name);

    std::string kernel_code =
        func_code + "\n" +
        "void kernel func_kernel(global " + get_type() + "* g_idata, global " + get_type() + "* g_odata, local " + get_type() + "* sdata){"
        "    const unsigned int tid = get_local_id(0);                                                                                    "
        "    const unsigned int lSize = get_local_size(0);                                                                                "
        "    const unsigned int gid = get_global_id(0);                                                                                   "
        "    sdata[tid] = g_idata[gid];                                                                                                   "
        "    barrier(CLK_LOCAL_MEM_FENCE);                                                                                                "
        "    for(unsigned int s = lSize >> 1; s > 0; s >>= 1){                                                                            "
        "        if (tid < s) sdata[tid] = " + name + "(sdata[tid], sdata[tid + s]);                                                      "
        "        barrier(CLK_LOCAL_MEM_FENCE);                                                                                            "
        "    }                                                                                                                            "
        "    if (tid == 0) g_odata[get_group_id(0)] = sdata[0];                                                                           "
        "}                                                                                                                                ";    

    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    Program program(context, sources);
    if(program.build() != CL_SUCCESS){
        std::cout << " Error building\n";
        exit(1);
    }
    
    std::vector<T> aux(length);
    Buffer obuffer(context, aux.begin(), aux.end(), false);

    //run the kernel (openCL 1.2)
    Kernel kernel = Kernel(program, "func_kernel");
    auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
    auto device = devices.front();
    int workGroupSize = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
    workGroupSize = std::min(workGroupSize, length);

    kernel.setArg(0, buffer);
    kernel.setArg(1, obuffer);
    kernel.setArg(2, sizeof(T)*workGroupSize, nullptr);
    queue.enqueueNDRangeKernel(kernel, NullRange, NDRange(length), NDRange(workGroupSize));
    queue.finish();

    copy(queue, obuffer, aux.begin(), aux.end());

    T ret = 0;

    for(T x: aux){
        ret += x;
    }

    return ret;
}

template<typename T>
void gpu_vector<T>::show(){
    for (T x: vec)
        std::cout << x << " ";
    
    std::cout << "\n";
}

template<typename T>
int gpu_vector<T>::size(){
    return length;
}

template<>
std::string gpu_vector<int>::get_type()                { return "int"; }
template<>
std::string gpu_vector<bool>::get_type()               { return "bool"; }
template<>
std::string gpu_vector<char>::get_type()               { return "char"; }
template<>
std::string gpu_vector<unsigned>::get_type()           { return "unsigned"; }
template<>
std::string gpu_vector<double>::get_type()             { return "double"; }
template<>
std::string gpu_vector<float>::get_type()              { return "float"; }
template<>
std::string gpu_vector<long>::get_type()               { return "long"; }
template<>
std::string gpu_vector<long long>::get_type()          { return "long long"; }
template<>
std::string gpu_vector<short>::get_type()              { return "short"; }
template<>
std::string gpu_vector<unsigned char>::get_type()      { return "unsigned char"; }
template<>
std::string gpu_vector<unsigned short>::get_type()     { return "unsigned short"; }
template<>
std::string gpu_vector<unsigned long>::get_type()      { return "unsigned long"; }
template<>
std::string gpu_vector<unsigned long long>::get_type() { return "unsigned long long"; }

}