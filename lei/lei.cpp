#include "cpu/mpi_vector.cpp"
#include "cpu/mpi_matrix.cpp"
//#include "gpu/gpu_vector.cpp"
//#include "gpu/gpu_matrix.cpp"
//#include "boost/boost_vector.cpp"
//#include "boost/boost_matrix.cpp"

#include <cstdarg>

/*
 * @file lei.cpp 
 *
 * @author António Gonçalves
 * @author Eduardo Conceição
 * @author Gonçalo Esteves
 * @author João Fernandes
 * 
 * 
 * This file unifies the three different implementations
 * 
 **/

namespace lei
{

    void init_lei(int argc, char** argv)
    {
        int flag;

        // first, check if mpi is initialized
        // useful for allowing for more than one distributed structure
        MPI_Initialized(&flag);

        if( !flag )
        {
            MPI_Init(&argc, &argv);
        }
    }

    // Add some function to finalize
    void finish_lei()
    {
        MPI_Finalize();
    }

    template<typename T>
    int write_output(mpi_vector<T> v, const char* filename)
    {
        int flag;
        int n_processes, my_rank;

        MPI_File fh;
        MPI_Status stat;

        long written = 0;

        // first, check if mpi is initialized
        // useful for allowing for more than one distributed structure
        MPI_Initialized(&flag);

        if( !flag )
        {
            throw "<SOMETHING>";
        }

        MPI_Comm_size(MPI_COMM_WORLD, &n_processes);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

        MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh); 

        if( !fh ) return 0;

        if( my_rank == 0 )
        {
            for(auto i = v.begin(); i != v.end(); i++)
            {

                std::string aux = std::to_string(*i);
                MPI_File_write_at(fh, written, aux.c_str(), aux.length(), MPI_CHAR, MPI_STATUS_IGNORE);
                written += aux.length();
                MPI_File_write_at(fh, written, ",", 1, MPI_CHAR, MPI_STATUS_IGNORE);
                written++;
            }
            // send to process 1
            MPI_Send(&written, 1, MPI_LONG, 1, 0, MPI_COMM_WORLD);
        }
        else if ( my_rank ==  n_processes - 1 )
        {
            // receive from process (n_processes - 2) 
            MPI_Recv(&written, 1, MPI_LONG, my_rank-1, 0, MPI_COMM_WORLD, &stat);
            for(auto i = v.begin(); i != v.end(); i++)
            {

                std::string aux = std::to_string(*i);
                MPI_File_write_at(fh, written, aux.c_str(), aux.length(), MPI_CHAR, MPI_STATUS_IGNORE);
                written += aux.length();
                MPI_File_write_at(fh, written, ",", 1, MPI_CHAR, MPI_STATUS_IGNORE);
                written++;
            }
        }
        else
        {
            // receive from process (my_rank - 1)
            MPI_Recv(&written, 1, MPI_LONG, my_rank-1, 0, MPI_COMM_WORLD, &stat);
            for(auto i = v.begin(); i != v.end(); i++)
            {

                std::string aux = std::to_string(*i);
                MPI_File_write_at(fh, written, aux.c_str(), aux.length(), MPI_CHAR, MPI_STATUS_IGNORE);
                written += aux.length();
                MPI_File_write_at(fh, written, ",", 1, MPI_CHAR, MPI_STATUS_IGNORE);
                written++;
            }
            // send to process (my_rank + 1)
            MPI_Send(&written, 1, MPI_LONG, my_rank+1, 0, MPI_COMM_WORLD);
        }

        MPI_File_close(&fh);
        return 0;
    }

    template<typename T>
    int write_output(mpi_matrix<T> v, const char* filename)
    {
        int flag;
        int n_processes, my_rank;

        MPI_File fh;
        MPI_Status stat;

        long written = 0;

        // first, check if mpi is initialized
        // useful for allowing for more than one distributed structure
        MPI_Initialized(&flag);

        if( !flag )
        {
            throw "Library has not yet been initialized!\n";
        }

        MPI_Comm_size(MPI_COMM_WORLD, &n_processes);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

        MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh); 

        if( !fh ) return 0;

        if( my_rank == 0 )
        {
            for(auto i = v.begin(); i != v.end(); i++)
            {

                std::string aux = std::to_string(*i);
                MPI_File_write_at(fh, written, aux.c_str(), aux.length(), MPI_CHAR, MPI_STATUS_IGNORE);
                written += aux.length();
                MPI_File_write_at(fh, written, ",", 1, MPI_CHAR, MPI_STATUS_IGNORE);
                written++;

                if( i.eol() ) 
                { 
                    MPI_File_write_at(fh, written, "\n", 1, MPI_CHAR, MPI_STATUS_IGNORE); 
                    written++; 
                }
            }
            // send to process 1
            MPI_Send(&written, 1, MPI_LONG, 1, 0, MPI_COMM_WORLD);
        }
        else if ( my_rank ==  n_processes - 1 )
        {
            // receive from process (n_processes - 2) 
            MPI_Recv(&written, 1, MPI_LONG, my_rank-1, 0, MPI_COMM_WORLD, &stat);
            for(auto i = v.begin(); i != v.end(); i++)
            {

                std::string aux = std::to_string(*i);
                MPI_File_write_at(fh, written, aux.c_str(), aux.length(), MPI_CHAR, MPI_STATUS_IGNORE);
                written += aux.length();
                MPI_File_write_at(fh, written, ",", 1, MPI_CHAR, MPI_STATUS_IGNORE);
                written++;

                if( i.eol())
                { 
                    MPI_File_write_at(fh, written, "\n", 1, MPI_CHAR, MPI_STATUS_IGNORE); 
                    written++; 
                }
            }
        }
        else
        {
            // receive from process (my_rank - 1)
            MPI_Recv(&written, 1, MPI_LONG, my_rank-1, 0, MPI_COMM_WORLD, &stat);
            for(auto i = v.begin(); i != v.end(); i++)
            {

                std::string aux = std::to_string(*i);
                MPI_File_write_at(fh, written, aux.c_str(), aux.length(), MPI_CHAR, MPI_STATUS_IGNORE);
                written += aux.length();
                MPI_File_write_at(fh, written, ",", 1, MPI_CHAR, MPI_STATUS_IGNORE);
                written++;

                if( i.eol())
                { 
                    written += MPI_File_write_at(fh,written, "\n", 1, MPI_CHAR, MPI_STATUS_IGNORE); 
                    written++; 
                }
            }
            // send to process (my_rank + 1)
            MPI_Send(&written, 1, MPI_LONG, my_rank+1, 0, MPI_COMM_WORLD);
        }

        MPI_File_close(&fh);
        return 0;
    }

    template<size_t I = 0, typename... Args>
    typename std::enable_if<I == sizeof...(Args), int>::type
    build_tuple(std::tuple<Args...>* t, MPI_File fp, MPI_Offset offset, long elems_to_read, int uelems)
    {
        return 0;
    }

    template<size_t I = 0, typename... Args>
    typename std::enable_if<I < sizeof...(Args) && I != 0, int>::type
    build_tuple(std::tuple<Args...>* t, MPI_File fp, MPI_Offset offset, long elems_to_read, int uelems)
    {
        MPI_Status stat;
        int elems = 0;

        if (I < uelems)
        {
            for(long elems_read = 0; elems_read < elems_to_read; elems_read++)
            {
                MPI_File_read_at(fp, offset, &std::get<I>(t[elems_read]), 1, MPI_FLOAT, &stat);
                offset += sizeof(float);
                elems++;
            }
        }

        return build_tuple<I+1>(t,fp,offset, elems_to_read, uelems);
    }
    
    template<size_t I = 0, typename... Args>
    typename std::enable_if<I == 0, int>::type
    build_tuple(std::tuple<Args...>* t, MPI_File fp, MPI_Offset offset, long elems_to_read, int uelems)
    {
        MPI_Status stat;
        int elems = 0;

        auto zero = []() -> float { return 0.0; };

        if (I < uelems)
        {
            for(long elems_read = 0; elems_read < elems_to_read; elems_read++)
            {
                // Initialize the tuple
                std::apply([&](auto &... x) 
                { return std::make_tuple(zero); }, t[elems_read]);

                MPI_File_read_at(fp, offset, &std::get<I>(t[elems_read]), 1, MPI_FLOAT, &stat);
                offset += sizeof(float);
                elems++;
            }
        }

        build_tuple<I+1>(t,fp,offset, elems_to_read, uelems);

        return elems;
    }

    template<typename T>
    T* read_bin_file ( const char* filename, int* size, size_t width, size_t uwidth )
    {
        int flag;
        int n_processes, my_rank;

        int elems;
        
        MPI_File fp;
        MPI_Status stat;

        MPI_Initialized(&flag);

        if( !flag )
        {
            throw "Library has not been initialized\n";
        }

        MPI_Comm_size(MPI_COMM_WORLD, &n_processes);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

        MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fp);

        if ( !fp )
        {
            throw std::string("Cannot open file: ") + std::string(filename);
        }

        
        MPI_File_read(fp, &elems, 1, MPI_INT, &stat);

        T* v = new T[elems];

        MPI_Offset off = 0;

        (*size) = build_tuple<0>(v, fp, off, elems, uwidth);

        MPI_File_close(&fp);

        return v;
    }

}
