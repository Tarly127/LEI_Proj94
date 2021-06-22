#include "../../lei/lei.cpp"
#include <cmath>
#include <tuple>
#include <chrono>

//just so we don't use up so much space
#define Tuple_6f std::tuple<float, float, float, float, float, float>
#define Tuple_5f std::tuple<float, float, float, float, float>

#define PI   3.1415926535897932384626433832795029f
#define PIx2 6.2831853071795864769252867665590058f

// Just so people understand what's going on
#define x(T)      (std::get<0>(T))
#define kx(T)     (std::get<0>(T))
#define y(T)      (std::get<1>(T))
#define ky(T)     (std::get<1>(T))
#define z(T)      (std::get<2>(T))
#define kz(T)     (std::get<2>(T))
#define phiR(T)   (std::get<3>(T))
#define phiI(T)   (std::get<4>(T))
#define phiMag(T) (std::get<5>(T))
#define Qr(T)     (std::get<3>(T))
#define Qi(T)     (std::get<4>(T))


void ComputePhiMag(Tuple_6f* in)
{
    // mag = r^2 + i^2
    phiMag(*in) = phiR(*in) * phiR(*in) + phiI(*in) * phiI(*in);

}

void ComputeQ(Tuple_5f* in, lei::mpi_vector<Tuple_6f> &kValues)
{
    using namespace std;
    
    for(auto i = kValues.begin(); i != kValues.end(); i++)
    {
        float expArg =   PIx2 * kx(*i) * x(*in)
                       + ky(*i) * y(*in)
                       + kz(*i) * z(*in);

        float cosArg = cosf(expArg);
        float sinArg = sinf(expArg);

        float phi    = phiMag(*i);

        /*Qr*/ Qr(*in) += phi * cosArg;
        /*Qi*/ Qi(*in) += phi * sinArg;        
    }
}


int main(int argc, char** argv)
{
    auto fstart = std::chrono::high_resolution_clock::now();

    lei::init_lei(argc, argv);

    auto start = std::chrono::high_resolution_clock::now();

    int size_k = 0, size_x = 0;

    // Read the input files
    Tuple_6f* tk =  lei::read_bin_file<Tuple_6f>("in/dataset_k.in", &size_k, 6, 5);
    Tuple_5f* tx =  lei::read_bin_file<Tuple_5f>("in/dataset_x.in", &size_x, 5, 3);

    printf("%d\n", size_x);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << duration.count() << ",";

    start = std::chrono::high_resolution_clock::now();

    // Create the two structures
    lei::mpi_vector<Tuple_6f> vk(tk, size_k, lei::VCOPY);
    lei::mpi_vector<Tuple_5f> vx(tx, size_x, lei::BLOCK);

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << duration.count() << ",";

    start = std::chrono::high_resolution_clock::now();

    // Compute the Magnitude of the Complex Component, 
    // storing it in the 6th element of each tuple in vk
    vk.map(&ComputePhiMag);

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << duration.count() << ",";

    start = std::chrono::high_resolution_clock::now();

    // Compute Q and store the real component in the 3rd element
    // of each tuple and the imaginary component in the 4th
    vx.map(&ComputeQ, vk);

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << duration.count() << ",";

    delete[] tk;
    delete[] tx;

    auto fstop = std::chrono::high_resolution_clock::now();
    auto fduration = std::chrono::duration_cast<std::chrono::milliseconds>(fstop - fstart);
    std::cout << duration.count() << std::endl;

    lei::finish_lei();
}