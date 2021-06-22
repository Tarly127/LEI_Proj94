#pragma once

#ifndef __MPI_VECTOR__
#define __MPI_VECTOR__

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <vector>
#include <iostream>
#include <functional>


#ifndef __MPI_AUXILIARY__
#define __MPI_AUXILIARY__

// COPIED (source: https://www.geeksforgeeks.org/how-to-iterate-over-the-elements-of-an-stdtuple-in-c/)
// Function to iterate through all values
// I equals number of values in tuple
template <size_t I = 1, typename... Ts>
typename std::enable_if<I == sizeof...(Ts), void>::type
printTuple(std::tuple<Ts...> tup)
{
    std::cout << std::get<I-1>(tup);
    return;
}
 
template <size_t I = 1, typename... Ts>
typename std::enable_if<(I < sizeof...(Ts)), void>::type
printTuple(std::tuple<Ts...> tup)
{
    std::cout << std::get<I-1>(tup) << ", ";
 
    // Go to next element
    printTuple<I + 1>(tup);
}

template<typename T>
MPI_Datatype type()                     { return MPI_DATATYPE_NULL; }
template<>
MPI_Datatype type<int>()                { return MPI_INT; }
template<>
MPI_Datatype type<bool>()               { return MPI_C_BOOL; }
template<>
MPI_Datatype type<char>()               { return MPI_CHAR; }
template<>
MPI_Datatype type<unsigned>()           { return MPI_UNSIGNED; }
template<>
MPI_Datatype type<double>()             { return MPI_DOUBLE; }
template<>
MPI_Datatype type<float>()              { return MPI_FLOAT; }
template<>
MPI_Datatype type<long>()               { return MPI_LONG; }
template<>
MPI_Datatype type<long long>()          { return MPI_LONG_LONG; }
template<>
MPI_Datatype type<short>()              { return MPI_SHORT; }
template<>
MPI_Datatype type<unsigned char>()      { return MPI_UNSIGNED_CHAR; }
template<>
MPI_Datatype type<unsigned short>()     { return MPI_UNSIGNED_SHORT; }
template<>
MPI_Datatype type<unsigned long>()      { return MPI_UNSIGNED_LONG; }
template<>
MPI_Datatype type<unsigned long long>() { return MPI_UNSIGNED_LONG_LONG; }



#endif

namespace lei
{

typedef enum _MPIV_DIST_TYPE
{
    VCOPY  = 1001u,
    BLOCK  = 1002u
} 
MPIV_DIST_TYPE;


template<typename T>
class mpi_vector
{
    private:

        //MPI related attributes
        int _processes;
        int _my_rank;

        MPIV_DIST_TYPE _DIST;

        size_t _local_size;
        size_t _local_capacity; // may be useless
        size_t _size;

        //possibly pointless
        int* displs;
        int* sendcount;

        T* local_array;

        template<typename Fn, typename Tuple>
        auto make_for_each(Fn&& fn, Tuple &tuple);

    public:
        struct iterator
        {
            using iterator_category = std::forward_iterator_tag;
            using difference_type   = std::ptrdiff_t;
            using value_type        = T;
            using pointer           = value_type*;
            using reference         = value_type&;

            iterator(pointer ptr) : _ptr(ptr) {};
            //iterator(iterator iter) : _ptr(&iter) {};

            reference operator  *() const { return *_ptr; }
            pointer   operator  &() { return _ptr; }

            iterator& operator ++() { _ptr++; return *this; }
            iterator  operator ++(int) { iterator tmp = *this; ++(*this); return tmp; }

            friend bool operator ==(const iterator& a, const iterator& b) { return a._ptr == b._ptr; }
            friend bool operator !=(const iterator& a, const iterator& b) { return a._ptr != b._ptr; }


            private:
                pointer _ptr;
        };

        // class constructors
        mpi_vector(std::vector<T> in_vec, MPIV_DIST_TYPE DIST);
        mpi_vector(T in_arr[], size_t in_size, MPIV_DIST_TYPE DIST);
        // copy constructor
        mpi_vector(mpi_vector<T> &in_vec);
        // empty constructor
        mpi_vector(size_t size, MPIV_DIST_TYPE DIST);

        // iterators
        iterator begin() { return iterator(&local_array[0]); }
        iterator end()   { return iterator(&local_array[_local_size]);}

        // instance methods
        void show_myself();
        T& get(const int index);
        T& operator[](const int index);
        ~mpi_vector();
        size_t size();
        MPIV_DIST_TYPE getDistribution();


        template<typename... Args>
        void map(void mapper(T*, Args&...), Args&... args);

        template<typename... Args>
        T reduce(T Fn(T,T, Args&...), Args&...);

        template<typename... Args>
        void zip(mpi_vector<T>& A, mpi_vector<T>& B, T Fn(T,T,Args&...), Args&... args);

};

template<typename T> 
mpi_vector<T>::mpi_vector(std::vector<T> in_vec, MPIV_DIST_TYPE DIST)
{
    int flag;
    MPI_Initialized(&flag);

    if( !flag )
    {
        throw "<SOMETHING>";
    }

    MPI_Comm_size(MPI_COMM_WORLD, &_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &_my_rank);

    sendcount = new int[_processes];
    displs    = new int[_processes];
    _size      = in_vec.size();

    _DIST = DIST;

    switch ( _DIST )
    {
        case BLOCK:
        {
            int base_size = _size / _processes;
            int rest      = _size % _processes;

            int accum = 0;

            for(unsigned i = 0; i < _processes; i++)
            {
                sendcount[i] = i < rest ? base_size + 1 : base_size;
                displs[i]    = accum;
                accum       += sendcount[i];
            }

            local_array = new T[sendcount[_my_rank]];

            for(unsigned i = 0; i < sendcount[_my_rank] && i < _size; i++)
            {
                local_array[i] = in_vec[i + displs[_my_rank]];
            }

            _local_size = _local_capacity = sendcount[_my_rank];
            break;
        }
        case VCOPY:
        {
            for(unsigned i = 0; i < _processes; i++)
            {
                sendcount[i] = _size;
                displs[i]    = 0;
            }

            local_array = new T[sendcount[_my_rank]];

            for(unsigned i = 0; i < sendcount[_my_rank] && i < _size; i++)
            {
                local_array[i] = in_vec[i];
            }

            _local_size = _size;
            break;
        }
        default:
        {
            throw "Unrecognized Distribution!";
        }
    }
}

template<typename T> 
mpi_vector<T>::mpi_vector(T in_arr[], size_t in_size, MPIV_DIST_TYPE DIST)
{
    int flag;

    // first, check if mpi is initialized
    // useful for allowing for more than one distributed structure
    MPI_Initialized(&flag);

    if( !flag )
    {
        throw "Library has not been initialized!";
    }

    MPI_Comm_size( MPI_COMM_WORLD, &_processes );
    MPI_Comm_rank( MPI_COMM_WORLD, &_my_rank );

    sendcount  = new int[_processes];
    displs     = new int[_processes];
    _size       = in_size;

    _DIST = DIST;

    switch ( _DIST )
    {
        case VCOPY:
        {
            for(unsigned i = 0; i < _processes; i++)
            {
                sendcount[i] = _size;
                displs[i]    = 0;
            }

            local_array = new T[sendcount[_my_rank]];

            for(unsigned i = 0; i < sendcount[_my_rank]; i++)
            {
                local_array[i] = in_arr[i];
            }

            _local_size = _size; 
            break; 
        }
        case BLOCK:
        {
            int base_size = _size / _processes;
            int rest      = _size % _processes;

            int accum = 0;

            for(unsigned i = 0; i < _processes; i++)
            {
                sendcount[i] = i < rest ? base_size + 1 : base_size;
                displs[i]    = accum;
                accum       += sendcount[i];
            }

            local_array = new T[sendcount[_my_rank]];

            for(unsigned i = 0; i < sendcount[_my_rank]; i++)
            {
                local_array[i] = in_arr[i + displs[_my_rank]];
            }

            _local_size = _local_capacity = sendcount[_my_rank];

            break;
        }
        default:
        {
            throw "Unrecognized Distribution Type!";
        }
    }
}

template<typename T> 
mpi_vector<T>::mpi_vector(mpi_vector<T> &in_vec)
{

    int flag;

    // first, check if mpi is initialized
    // useful for allowing for more than one distributed structure
    MPI_Initialized(&flag);

    if( !flag )
    {
        throw "Library has not been initialized!";
    }

    MPI_Comm_size(MPI_COMM_WORLD, &_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &_my_rank);

    sendcount = new int[_processes];
    displs    = new int[_processes];
    _size      = in_vec.size();

    _DIST = in_vec.getDistribution();

    switch( _DIST )
    {
        case VCOPY:
        {
            for(unsigned i = 0; i < _processes; i++)
            {
                sendcount[i] = _size;
                displs[i]    = 0;
            }

            local_array = new T[sendcount[_my_rank]];

            unsigned index = 0;
            for(auto i = in_vec.begin(); i != in_vec.end(); i++)
            {
                local_array[index] = *(&i);
                index++;
            }

            _local_size = _size;
        }
        case BLOCK:
        {
            int base_size = _size / _processes;
            int rest      = _size % _processes;

            int accum = 0;

            for(unsigned i = 0; i < _processes; i++)
            {
                sendcount[i] = i < rest ? base_size + 1 : base_size;
                displs[i]    = accum;
                accum       += sendcount[i];
            }

            local_array = new T[sendcount[_my_rank]];

            unsigned index = 0;
            for(auto i = in_vec.begin(); i != in_vec.end(); i++)
            {
                local_array[index] = *(&i);
                index++;
            }

            _local_size = _local_capacity = sendcount[_my_rank];
        }
        default:
        {
            throw "Unrecognized Distribution Type!";
        }
    }
    
    
}

template<typename T> 
mpi_vector<T>::mpi_vector(size_t size, MPIV_DIST_TYPE DIST)
{
    int flag;

    // first, check if mpi is initialized
    // useful for allowing for more than one distributed structure
    MPI_Initialized(&flag);

    if( !flag )
    {
        throw "Library has not been initialized!";
    }

    MPI_Comm_size(MPI_COMM_WORLD, &_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &_my_rank);

    _DIST = DIST;

    sendcount = new int[_processes];
    displs    = new int[_processes];
    _size      = size;

    switch ( _DIST )
    {
        case VCOPY:
        {
            for(unsigned i = 0; i < _processes; i++)
            {
                sendcount[i] = _size;
                displs[i]    = 0;
            }

            local_array = new T[sendcount[_my_rank]];

            _local_size = _size;
            break;
        }
        case BLOCK:
        {
            int base_size = _size / _processes;
            int rest      = _size % _processes;

            int accum = 0;

            for(unsigned i = 0; i < _processes; i++)
            {
                sendcount[i] = i < rest ? base_size + 1 : base_size;
                displs[i]    = accum;
                accum       += sendcount[i];
            }

            local_array = new T[sendcount[_my_rank]];

            _local_size = _local_capacity = sendcount[_my_rank];
            break;
        }
        default:
        {
            throw "Unrecognized Distribution Type!";
        }
    }
}

template<typename T>
T& mpi_vector<T>  ::get(const int index)
{
    return local_array[index];
}

template<typename T>
MPIV_DIST_TYPE mpi_vector<T>::getDistribution() { return this->_DIST; }

template<typename T>
void mpi_vector<T>::show_myself()
{
    std::cout << "----------" << std::endl;
    std::cout << "My rank: " << _my_rank << std::endl;
    std::cout << "Displ: "  << displs[_my_rank] << std::endl;
    std::cout << "Sendcounts: " << sendcount[_my_rank] << std::endl;
    std::cout << "Local array: ";

    auto is_fundamental = std::is_fundamental<T>::value;

    #if (is_fundamental)
        for(auto i = this->begin(); i != this->end(); i++)
        {
            std::cout << (*i) << ", ";
        }
    #else
        //auto print = [](auto t){ std::cout << t << ","; };
        for(auto i = this->begin(); i != this->end(); i++)
        {
            std::cout << "(";
            printTuple(*i);
            std::cout << "),";
        }
    #endif
    std::cout << std::endl;
    std::cout << "----------" << std::endl;
}

template<typename T>
T& mpi_vector<T>  ::operator[](const int index)
{
    return get(index);
}

template<typename T>
size_t mpi_vector<T>::size(){ return this->_size; }

template<typename T>
mpi_vector<T>::~mpi_vector()
{
    #ifndef __APPLE__
        delete[] sendcount;
        delete[] displs;
        delete[] local_array;
    #endif
}


template<typename T>
template<typename Fn, typename Tuple>
auto mpi_vector<T>::make_for_each(Fn&& fn, Tuple &tuple)
{
  return std::apply([&](auto &... x) { return std::make_tuple(fn(x)...); }, tuple);
}

/* MAP */
template<class T>
template<typename... Args>
void mpi_vector<T>::map(void mapper(T*, Args&...), Args&... args)
{   
    for(auto i = begin(); i != end(); i++)
    {
        mapper(&i, args...);
    }
}

/* REDUCE */
// int
template<>
template<typename... Args>
int     mpi_vector<int>::reduce(int Fn(int, int, Args&...), Args&... args)
{
    int local_reduction = 0, total_reduction = 0; // for now

    for(auto i = this->begin(); i != this->end(); i++)
    {
        local_reduction = Fn(local_reduction, *i, args...); 
    }

    MPI_Allreduce(&local_reduction, &total_reduction, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    return total_reduction;
}

// float
template<>
template<typename... Args>
float      mpi_vector<float>::reduce(float Fn(float, float, Args&...), Args&... args)
{
    float local_reduction = 0, total_reduction = 0; // for now

    for(auto i = this->begin(); i != this->end(); i++)
    {
        local_reduction = Fn(local_reduction, *i, args...); 
    }

    MPI_Allreduce(&local_reduction, &total_reduction, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    return total_reduction;
}

// double
template<>
template<typename... Args>
double      mpi_vector<double>::reduce(double Fn(double, double, Args&...), Args&... args)
{
    double local_reduction = 0, total_reduction = 0; // for now

    for(auto i = this->begin(); i != this->end(); i++)
    {
        local_reduction = Fn(local_reduction, *i, args...); 
    }

    MPI_Allreduce(&local_reduction, &total_reduction, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return total_reduction;
}

// char
template<>
template<typename... Args>
char      mpi_vector<char>::reduce(char Fn(char, char, Args&...), Args&... args)
{
    char local_reduction = 0, total_reduction = 0; // for now

    for(auto i = this->begin(); i != this->end(); i++)
    {
        local_reduction = Fn(local_reduction, *i, args...); 
    }

    MPI_Allreduce(&local_reduction, &total_reduction, 1, MPI_CHAR, MPI_SUM, MPI_COMM_WORLD);

    return total_reduction;
}

// unsigned
template<>
template<typename... Args>
unsigned      mpi_vector<unsigned>::reduce(unsigned Fn(unsigned, unsigned, Args&...), Args&... args)
{
    unsigned local_reduction = 0, total_reduction = 0; // for now

    for(auto i = this->begin(); i != this->end(); i++)
    {
        local_reduction = Fn(local_reduction, *i, args...); 
    }

    MPI_Allreduce(&local_reduction, &total_reduction, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);

    return total_reduction;
}

// Generic Case (that works for tuples!)
template<typename T>
template<typename... Args>
T mpi_vector<T>::reduce(T Fn(T,T, Args&...), Args&... args)
{
    T pr;

    /* PARTIAL REDUCTION */
    for(auto i = this->begin(); i != this->end(); i++)
    {
        pr = Fn(pr, *i, args...);
    }

    /* FINAL REDUCTION */
    /* In Case we're dealing with a primitive type that wasn't caught by the template specializations */
    if ( std::is_fundamental<T>::value )
    {
        T tr;
        MPI_Allreduce(&pr, &tr, 1, type<T>(), MPI_SUM, MPI_COMM_WORLD);
        return tr;
    }
    /* In case we're dealing with a tuple, as expected */
    else
    {

        /* Auixiliary Lambda for FINAL REDUCTION*/
        auto coord_allreduce = [](auto pr)
        {
            using coord_t = decltype(pr);
            coord_t tr; // verify empty ctor, initializes tc correctly

            if constexpr( std::is_fundamental<coord_t>::value ) 
                throw "Tuple contains non-primitive types!";

            else
                MPI_Allreduce(&pr, &tr, 1, type<coord_t>(), MPI_SUM, MPI_COMM_WORLD);

            return tr;
        };
        try
        {
            return make_for_each(coord_allreduce, pr);
        }
        catch (const char* e)
        {
            std::cerr << e << std::endl;

            return nullptr;
        }
    }
}


/* ZIP */

template<typename T>
template<typename... Args>
void   mpi_vector<T>::zip(mpi_vector<T>& A, mpi_vector<T>& B, T Fn(T,T,Args&...), Args&... args)
{
    if (   A.size() == B.size() 
        && A.size() == this->size() 
        && A.getDistribution() == B.getDistribution() 
        && A.getDistribution() == this->getDistribution()) 
    {
        auto i_a = A.begin(); auto i_b = B.begin();
        auto k = 0;

        for(; i_a != A.end() && i_b != B.end(); i_a++, i_b++, k++)
        {
            this->set(k, Fn( *i_a, *i_b, args...));
        }
    }
    else
        throw "Input Vector sizes do not match!";
}


}

#endif