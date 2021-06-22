#pragma once

#ifndef __MPI_MATRIX__
#define __MPI_MATRIX__

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <vector>
#include <iostream>
#include <functional>
#include <typeinfo>
#include <tuple>
#include <type_traits>

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

typedef enum _MPIM_DIST_TYPE
{
    ROW_WISE        = 1001u,
    COLUMN_WISE     = 1002u,
    MCOPY           = 1003u,
    MCOPY_TRANSPOSE = 1004u,
    TRANSPOSE       = 1005u
} 
MPIM_DIST_TYPE;

template<class T>
class mpi_matrix
{
    private:
        // MPI related attributes
        int _processes;
        int _my_rank;

        MPIM_DIST_TYPE _DIST;

        size_t _local_size;
        size_t size;

        size_t _total_columns;
        size_t _local_columns;

        size_t _total_lines;
        size_t _local_lines;

        int* displs;
        int* sendcount;

        T* local_matrix;

        MPI_Datatype get_type();

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

            iterator(pointer ptr, unsigned line_size) : _ptr(ptr), _line_size(line_size), _line_counter(0) {};

            reference operator  *() const { return *_ptr; }
            pointer   operator  &() { return _ptr; }

            bool eol() { return this->_line_counter == this->_line_size - 1; }

            iterator& operator ++() 
            { 
                _line_counter = _line_counter == _line_size - 1 ? 0 : _line_counter + 1;
                _ptr++;
                return *this;
            }
            iterator  operator ++(int) { iterator tmp = *this; ++(*this); return tmp; }

            friend bool operator ==(const iterator& a, const iterator& b) { return a._ptr == b._ptr && a._line_size == b._line_size; }
            friend bool operator !=(const iterator& a, const iterator& b) { return a._ptr != b._ptr /*|| a._line_size != b._line_size*/; }

            private:
                pointer  _ptr;
                unsigned _line_size;
                unsigned _line_counter;
        };

        struct line_iterator
        {
            using iterator_category = std::forward_iterator_tag;
            using difference_type   = std::ptrdiff_t;
            using value_type        = T;
            using pointer           = value_type*;
            using reference         = value_type&;

            line_iterator(pointer ptr, size_t line_size, size_t local_lines) 
                : _ptr(ptr), 
                  _line_size(line_size), 
                  _line_counter(0),
                  _local_lines(local_lines) 
                {};

            reference operator  *() const { return *_ptr; }
            pointer   operator  &() { return _ptr; }

            bool eom() { return _ptr == nullptr || _line_counter == _local_lines; }

            line_iterator& operator ++() 
            { 
                _line_counter++;
                _ptr = _line_counter == _local_lines ? _ptr : _ptr + _line_size;
                return *this;
            }

            line_iterator  operator ++(int) { line_iterator tmp = *this; ++(*this); return tmp; }

            friend bool operator ==(const line_iterator& a, const line_iterator& b) { return a._ptr == b._ptr && a._line_counter < a._local_lines && b._line_counter < b._local_lines; }
            friend bool operator !=(const line_iterator& a, const line_iterator& b) { return a._ptr != b._ptr || a._line_counter <= a._local_lines || b._line_counter <= b._local_lines; }

            private:
                pointer  _ptr;
                size_t _line_size;
                size_t _line_counter;
                size_t _local_lines;

        };

        // class constructors
        mpi_matrix(T** in_matrix, size_t lines, size_t cols, MPIM_DIST_TYPE DIST);
        mpi_matrix(size_t lines, size_t cols, MPIM_DIST_TYPE DIST);
        mpi_matrix(mpi_matrix<T>&);

        // iterators
        iterator begin() { return iterator(&local_matrix[0], _local_columns); }
        iterator end()   { return iterator(&local_matrix[_local_size], _local_columns); }

        // line iterators
        line_iterator line_begin() { return line_iterator(local_matrix, _local_columns, _local_lines); }
        line_iterator line_end()   { return line_iterator(&local_matrix[_local_size], _local_columns,  _local_lines); }

        // instance methods
        void show_myself();
        void set(int l, int c, T new_elem);
        unsigned getDistribution();

        size_t columns();
        size_t lines();
        size_t total_lines();
        size_t total_columns();

        T* get(int, int);
        ~mpi_matrix();
        
        template<typename... Args>
        void map (void Fn(T*, Args&...), Args&...);     

        template<typename... Args>
        T reduce(T Fn(T, T, Args&...), Args&...);

        template<typename... Args>
        void zip (mpi_matrix<T>&, mpi_matrix<T>&, T Fn(T*,T*,Args&...), Args&...);
};

template<class T>
mpi_matrix<T>::mpi_matrix(T** in_m, size_t lines, size_t cols, MPIM_DIST_TYPE DIST)
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

    // since we're dealing with row-wise separation,
    // the values in sendcount ?and displs? refer to
    // the lines
    sendcount = new int[_processes];
    displs    = new int[_processes];
    size      = lines * cols;

    _total_columns = cols;
    _total_lines   = lines;

    _DIST = DIST;

    switch( _DIST )
    {
        case ROW_WISE: 
        {
            _local_columns = cols;

            int base_lines = lines / _processes;
            int rest       = lines % _processes;

            int accum = 0;

            for(unsigned i = 0; i < _processes; i++)
            {
                sendcount[i] = i < rest ? base_lines + 1 : base_lines;
                displs[i]    = accum;
                accum       += sendcount[i];
            }

            _local_lines = sendcount[_my_rank];

            local_matrix = new T[_local_lines * _local_columns];

            for(unsigned i = 0; i < _local_lines; i++)
            {
                for(unsigned j = 0; j < _local_columns; j++)
                {
                    local_matrix[i * _local_columns + j] = in_m[i + displs[_my_rank]][j];
                }
            }

            _local_size = _local_lines * _local_columns;

            break;
        }
        case COLUMN_WISE: 
        {
            _local_lines   = lines;

            int base_cols   = cols / _processes;
            int rest        = cols % _processes;

            int accum = 0;

            for(unsigned i = 0; i < _processes; i++)
            {
                sendcount[i] = i < rest ? base_cols + 1 : base_cols;
                displs[i]    = accum;
                accum       += sendcount[i];
            }

            _local_columns = sendcount[_my_rank];

            printf("%zu\n", _local_columns);

            local_matrix = new T[_local_lines * _local_columns];

            for(unsigned i = 0; i < _local_lines; i++)
            {
                for(unsigned j = displs[_my_rank], k = 0; k < _local_columns; j++, k++)
                {
                    local_matrix[i * _local_columns + k] = in_m[i][j];
                }
            }

            _local_size = _local_lines * _local_columns;
            break;
        }
        case MCOPY: 
        {
            _local_columns = cols;
            _local_lines   = lines;
            _local_size    = size;

            for(unsigned i = 0; i < _processes; i++)
            {
                sendcount[i] = _total_lines;
                displs[i]    = 0;
            }

            local_matrix = new T[_total_lines * _total_columns];

            for(unsigned i = 0; i < _total_lines; i++)
            {
                for(unsigned j = 0; j < _total_columns; j++)
                {
                    local_matrix[i * _total_lines + j] = in_m[i][j];
                }
            }
            break;
        }
        case MCOPY_TRANSPOSE:
        {
            _local_columns = cols;
            _local_lines   = lines;
            _local_size    = size;

            for(unsigned i = 0; i < _processes; i++)
            {
                sendcount[i] = _total_lines;
                displs[i]    = 0;
            }

            local_matrix = new T[_total_columns * _total_lines];

            for(unsigned i = 0; i < _total_lines; i++)
            {
                for(unsigned j = 0; j < _total_columns; j++)
                {
                    local_matrix[i * _total_columns + j] = in_m[j][i];
                }
            }
            break;   
        }
        case TRANSPOSE: 
        {
            // again, storing the matrix in rows
            _local_columns = lines;

            int base_lines = cols / _processes;
            int rest       = cols % _processes;

            int accum = 0;

            for(unsigned i = 0; i < _processes; i++)
            {
                sendcount[i] = i < rest ? base_lines + 1 : base_lines;
                displs[i]    = accum;
                accum       += sendcount[i];
            }

            _local_lines = sendcount[_my_rank];

            local_matrix = new T[_local_lines * _local_columns];

            for(unsigned i = 0; i < _local_lines; i++)
            {
                for(unsigned j = 0; j < _local_columns; j++)
                {
                    local_matrix[i * _local_lines + j] = in_m[j][i + displs[_my_rank]];
                }
            }

            _local_size = _local_lines * _local_columns;
            break;
        }
        default:
        {
            throw "Unidentified distribution type " + std::to_string(DIST);
        }
    }
}

template<class T>
mpi_matrix<T>::mpi_matrix(size_t lines, size_t cols, MPIM_DIST_TYPE DIST)
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

    // since we're dealing with row-wise separation,
    // the values in sendcount ?and displs? refer to
    // the lines
    sendcount = new int[_processes];
    displs    = new int[_processes];
    size      = lines * cols;

    _total_columns = cols;
    _total_lines   = lines;

    _DIST = DIST;

    switch( _DIST )
    {
        case ROW_WISE: 
        {
            _local_columns = cols;

            int base_lines = lines / _processes;
            int rest       = lines % _processes;

            int accum = 0;

            for(unsigned i = 0; i < _processes; i++)
            {
                sendcount[i] = i < rest ? base_lines + 1 : base_lines;
                displs[i]    = accum;
                accum       += sendcount[i];
            }

            _local_lines = sendcount[_my_rank];

            local_matrix = new T[_local_lines * _local_columns];

            _local_size = _local_lines * _local_columns;

            break;
        }
        case COLUMN_WISE: 
        {
            _local_lines   = lines;

            int base_cols   = cols / _processes;
            int rest        = cols % _processes;

            int accum = 0;

            for(unsigned i = 0; i < _processes; i++)
            {
                sendcount[i] = i < rest ? base_cols + 1 : base_cols;
                displs[i]    = accum;
                accum       += sendcount[i];
            }

            _local_columns = sendcount[_my_rank];

            local_matrix = new T[_local_lines * _local_columns];

            _local_size = _local_lines * _local_columns;
            break;
        }
        case MCOPY: 
        {
            _local_columns = cols;
            _local_lines   = lines;
            _local_size    = size;

            for(unsigned i = 0; i < _processes; i++)
            {
                sendcount[i] = _total_lines;
                displs[i]    = 0;
            }

            local_matrix = new T[_total_lines * _total_columns];

            break;
        }
        case MCOPY_TRANSPOSE:
        {
            _local_columns = cols;
            _local_lines   = lines;
            _local_size    = size;

            for(unsigned i = 0; i < _processes; i++)
            {
                sendcount[i] = _total_lines;
                displs[i]    = 0;
            }

            local_matrix = new T[_total_columns * _total_lines];

            break;   
        }
        case TRANSPOSE: 
        {
            // again, storing the matrix in rows
            _local_columns = lines;

            int base_lines = cols / _processes;
            int rest       = cols % _processes;

            int accum = 0;

            for(unsigned i = 0; i < _processes; i++)
            {
                sendcount[i] = i < rest ? base_lines + 1 : base_lines;
                displs[i]    = accum;
                accum       += sendcount[i];
            }

            _local_lines = sendcount[_my_rank];

            local_matrix = new T[_local_lines * _local_columns];

            _local_size = _local_lines * _local_columns;
            break;
        }
        default:
        {
            throw "Unidentified distribution type " + std::to_string(DIST);
        }
    }
}

template<class T>
size_t mpi_matrix<T>::columns() { return this->_local_columns; }

template<class T>
size_t mpi_matrix<T>::lines() { return this->_local_lines; }

template<class T>
size_t mpi_matrix<T>::total_lines() { return this->_total_lines; }

template<class T>
size_t mpi_matrix<T>::total_columns() { return this->_total_columns; }

template<class T>
void   mpi_matrix<T>::set(int l, int c, T new_elem) { local_matrix[l * _local_columns + c] = new_elem; }

template<class T>
T*     mpi_matrix<T>::get(int l, int c) { return &local_matrix[l * _local_columns + c]; }


template<class T>
unsigned mpi_matrix<T>::getDistribution()
{
    return _DIST;
}

template<class T>
mpi_matrix<T>::~mpi_matrix()
{
    #ifndef __APPLE__
        delete[] sendcount;
        delete[] displs;
        delete[] local_matrix;
    #endif
}



template<typename T>
template<typename Fn, typename Tuple>
auto mpi_matrix<T>::make_for_each(Fn&& fn, Tuple &tuple)
{
  return std::apply([&](auto &... x) { return std::make_tuple(fn(x)...); }, tuple);
}


/* MAP */

template<class T>
template<typename... Args>
void mpi_matrix<T>::map(void mapper(T*, Args&...), Args&... args)
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
int         mpi_matrix<int>::reduce(int Fn(int, int, Args&...), Args&... args)
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
float       mpi_matrix<float>::reduce(float Fn(float, float, Args&...), Args&... args)
{
    float local_reduction = 0.0, total_reduction = 0.0; // for now

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
double      mpi_matrix<double>::reduce(double Fn(double, double, Args&...), Args&... args)
{
    double local_reduction = 0.0, total_reduction = 0.0; // for now

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
char        mpi_matrix<char>::reduce(char Fn(char, char, Args&...), Args&... args)
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
unsigned    mpi_matrix<unsigned>::reduce(unsigned Fn(unsigned, unsigned,  Args&...), Args&... args)
{
    unsigned local_reduction = 0, total_reduction = 0; // for now

    for(auto i = this->begin(); i != this->end(); i++)
    {
        local_reduction = Fn(local_reduction, *i, args...); 
    }

    MPI_Allreduce(&local_reduction, &total_reduction, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);

    return total_reduction;
}

// Generic Tuple Case
template<typename T>
template<typename... Args>
T mpi_matrix<T>::reduce(T Fn(T,T, Args&...), Args&... args)
{
    T pr; // verify empty ctor of tuple calls empty ctor of elements of tuple  
    for(auto i = this->begin(); i != this->end(); i++)
    {
        pr = Fn(pr, *i, args...);
    }


    if ( std::is_fundamental<T>::value )
    {
        T tr;
        MPI_Allreduce(&pr, &tr, 1, type<T>(), MPI_SUM, MPI_COMM_WORLD);
        return tr;
    }
    else
    {
        auto coord_allreduce = [](auto pr)
        {
            using coord_t = decltype(pr);
            coord_t tr; // verify empty ctor, initializes tr correctly

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


template<typename T>
template<typename... Args>
void   mpi_matrix<T>::zip(mpi_matrix<T> &A, mpi_matrix<T> &B, T Fn(T*,T*,Args&...), Args&... args)
{
    if (   A.lines() == B.lines() && A.columns() == B.columns() 
        && A.columns() == this->columns && A.lines() == this->lines
        && A.getDistribution() == B.getDistribution() 
        && A.getDistribution() == this->getDistribution()) 
    {
        for(unsigned i = 0; i < this->_local_lines; i++)
            for(unsigned j = 0; j < this->_local_columns; j++)
                this->set(i,j,Fn( A.get(i,j), B.get(i,j), args...));
    }
    else
        throw "Input Matrix sizes do not match!";
}



template<class T>
void    mpi_matrix<T>::show_myself()
{
    std::cout << "----------" << std::endl;
    std::cout << "Rank: " << _my_rank << std::endl;
    std::cout << "Local Lines: " << _local_lines << std::endl;
    std::cout << "Local Columns: " << _local_columns << std::endl;
    std::cout << "Local matrix: " << std::endl;

    bool is_fundamental = std::is_fundamental<T>::value;
    #if (is_fundamental)
        for(auto i = this->begin(); i != this->end(); i++)
        {
            std::cout << (*i) << ", ";
            if( i.eol() ) std::cout << std::endl;
        }
    #else
        auto print = [](auto t){ std::cout << t << ","; };
        for(auto i = this->begin(); i != this->end(); i++)
        {
            std::cout << "(";
            printTuple(*i);
            std::cout << "),";
            if( i.eol() ) std::cout << std::endl;
        }
    #endif

    std::cout << std::endl;
    std::cout << "----------" << std::endl;
    
}

}


#endif