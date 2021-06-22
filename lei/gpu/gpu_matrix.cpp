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
class gpu_matrix{

    private:
        //context
        Context context;
        //queue
        CommandQueue queue;
        //length of the buffer
        int length;
        //local vector with information
        std::vector<T> mat;
        //buffer with the information
        Buffer buffer;
        
        std::string get_type();

    public:

        //class constructors
        gpu_matrix(gpu_matrix<T> &originalMat);
        gpu_matrix(std::vector<T>* matrix, int size);
        gpu_matrix(T** matrix, int size);
        
        // iterators
        typename std::vector<T>::iterator begin() { return mat.begin(); }
        typename std::vector<T>::iterator end()   { return mat.end(); }

        //instance methods
        void map(std::string func_code, const char* func_name);
        void map3vectors(std::vector<T>* A, std::vector<T>* B, std::string func_code, const char* func_name);
        void map3arrays(T** A, T** B, std::string func_code, const char* func_name);
        void map2vectors(std::vector<T>* A, std::string func_code, const char* func_name);
        void map2arrays(T** A, std::string func_code, const char* func_name);
        T    reduce(std::string func_code, const char* func_name);
        int  size();
        void show();

};

template<typename T>
gpu_matrix<T>::gpu_matrix(gpu_matrix<T> &originalMat){
    context = Context(CL_DEVICE_TYPE_DEFAULT);

    //create queue to which we will push commands for the device.
    queue = CommandQueue(context);

    //storing local data
    length = originalMat.size();
    length = length * length;

    for(auto i = originalMat.begin(); i != originalMat.end(); i++)
        mat.push_back(*i);

    //create buffer to store the data
    buffer = Buffer(context, mat.begin(), mat.end(), false);
}

template<typename T>
gpu_matrix<T>::gpu_matrix(std::vector<T>* matrix, int size){
    context = Context(CL_DEVICE_TYPE_DEFAULT);

    //create queue to which we will push commands for the device.
    queue = CommandQueue(context);

    //storing local data
    length = size * size;

    for(auto i = 0; i < size; i++){
        for(auto j = matrix[i].begin(); j != matrix[i].end(); j++)
            mat.push_back(*j);
    }

    //create buffer to store the data
    buffer = Buffer(context, mat.begin(), mat.end(), false);
}

template<typename T>
gpu_matrix<T>::gpu_matrix(T** matrix, int size){
    context = Context(CL_DEVICE_TYPE_DEFAULT);

    //create queue to which we will push commands for the device.
    queue = CommandQueue(context);

    //storing local data
    length = size * size;

    for(auto i = 0; i < size; i++){
        for(auto j = 0; j < size; j++)
            mat.push_back(matrix[i][j]);
    }

    //create buffer to store the data
    buffer = Buffer(context, mat.begin(), mat.end(), false);
}

template<typename T>
void gpu_matrix<T>::map(std::string func_code, const char* func_name){
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

    //run the kernel (openCL 2.0)
    //KernelFunctor kernel(
        //Kernel(program, "func_kernel")),
        //queue, NullRange, NDRange(length), NullRange
    //);
    //kernel(buffer);

    //run the kernel (openCL 1.2)
    Kernel kernel = Kernel(program, "func_kernel");
    kernel.setArg(0, buffer);
    queue.enqueueNDRangeKernel(kernel, NullRange, NDRange(length), NullRange);
    queue.finish();

    copy(queue, buffer, mat.begin(), mat.end());
}

template<typename T>
void gpu_matrix<T>::map2vectors(std::vector<T>* A, std::string func_code, const char* func_name){
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

    std::vector<T> aux(A[0]);
    auto size = sqrt(length);

    for(auto i = 1; i < size; i++)
        aux.insert(aux.end(), A[i].begin(), A[i].end());

    Buffer bufferA(context, aux.begin(), aux.end(), true);

    //run the kernel (openCL 1.2)
    Kernel kernel = Kernel(program, "func_kernel");
    kernel.setArg(0, buffer);
    kernel.setArg(1, bufferA);
    queue.enqueueNDRangeKernel(kernel, NullRange, NDRange(length), NullRange);
    queue.finish();

    copy(queue, buffer, mat.begin(), mat.end());
}

template<typename T>
void gpu_matrix<T>::map2arrays(T** A, std::string func_code, const char* func_name){
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

    std::vector<T> aux;
    aux.insert(aux.begin(), std::begin(A[0]), std::end(A[0]));
    auto size = sqrt(length);

    for(auto i = 1; i < size; i++)
        aux.insert(aux.begin(), std::begin(A[i]), std::end(A[i]));

    Buffer bufferA(context, aux.begin(), aux.end(), true);

    //run the kernel (openCL 1.2)
    Kernel kernel = Kernel(program, "func_kernel");
    kernel.setArg(0, buffer);
    kernel.setArg(1, bufferA);
    queue.enqueueNDRangeKernel(kernel, NullRange, NDRange(length), NullRange);
    queue.finish();

    copy(queue, buffer, mat.begin(), mat.end());
}

template<typename T>
void gpu_matrix<T>::map3vectors(std::vector<T>* A, std::vector<T>* B, std::string func_code, const char* func_name){
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

    std::vector<T> aux(A[0]);
    auto size = sqrt(length);

    for(auto i = 1; i < size; i++)
        aux.insert(aux.end(), A[i].begin(), A[i].end());

    Buffer bufferA(context, aux.begin(), aux.end(), true);

    std::vector<T> aux2(B[0]);

    for(auto i = 1; i < size; i++)
        aux2.insert(aux2.end(), B[i].begin(), B[i].end());

    Buffer bufferB(context, aux2.begin(), aux2.end(), true);

    //run the kernel (openCL 1.2)
    Kernel kernel = Kernel(program, "func_kernel");
    kernel.setArg(0, buffer);
    kernel.setArg(1, bufferA);
    kernel.setArg(2, bufferB);
    queue.enqueueNDRangeKernel(kernel, NullRange, NDRange(length), NullRange);
    queue.finish();

    copy(queue, buffer, mat.begin(), mat.end());
}

template<typename T>
void gpu_matrix<T>::map3arrays(T** A, T** B, std::string func_code, const char* func_name){
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

    std::vector<T> aux;
    aux.insert(aux.begin(), std::begin(A[0]), std::end(A[0]));
    auto size = sqrt(length);

    for(auto i = 1; i < size; i++)
        aux.insert(aux.begin(), std::begin(A[i]), std::end(A[i]));

    Buffer bufferA(context, aux.begin(), aux.end(), true);

    std::vector<T> aux2;
    aux2.insert(aux2.begin(), std::begin(B[0]), std::end(B[0]));

    for(auto i = 1; i < size; i++)
        aux2.insert(aux2.begin(), std::begin(B[i]), std::end(B[i]));

    Buffer bufferB(context, aux2.begin(), aux2.end(), true);

    //run the kernel (openCL 1.2)
    Kernel kernel = Kernel(program, "func_kernel");
    kernel.setArg(0, buffer);
    kernel.setArg(1, bufferA);
    kernel.setArg(2, bufferB);
    queue.enqueueNDRangeKernel(kernel, NullRange, NDRange(length), NullRange);
    queue.finish();

    copy(queue, buffer, mat.begin(), mat.end());
}

template<typename T>
T gpu_matrix<T>::reduce(std::string func_code, const char* func_name){
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
void gpu_matrix<T>::show(){

    auto size = sqrt(length);
    auto i = 0;

    for (T x: mat){
        if(i < size){
            std::cout << x << " ";
            i++;
        }
        else{
            std::cout << "\n" << x << " ";
            i = 1;
        }
    }
    
    std::cout << "\n";
}

template<typename T>
int gpu_matrix<T>::size(){
    return sqrt(length);
}

template<>
std::string gpu_matrix<int>::get_type()                { return "int"; }
template<>
std::string gpu_matrix<bool>::get_type()               { return "bool"; }
template<>
std::string gpu_matrix<char>::get_type()               { return "char"; }
template<>
std::string gpu_matrix<unsigned>::get_type()           { return "unsigned"; }
template<>
std::string gpu_matrix<double>::get_type()             { return "double"; }
template<>
std::string gpu_matrix<float>::get_type()              { return "float"; }
template<>
std::string gpu_matrix<long>::get_type()               { return "long"; }
template<>
std::string gpu_matrix<long long>::get_type()          { return "long long"; }
template<>
std::string gpu_matrix<short>::get_type()              { return "short"; }
template<>
std::string gpu_matrix<unsigned char>::get_type()      { return "unsigned char"; }
template<>
std::string gpu_matrix<unsigned short>::get_type()     { return "unsigned short"; }
template<>
std::string gpu_matrix<unsigned long>::get_type()      { return "unsigned long"; }
template<>
std::string gpu_matrix<unsigned long long>::get_type() { return "unsigned long long"; }

}