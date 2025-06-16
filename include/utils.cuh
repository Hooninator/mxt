#ifndef UTILS_CUH
#define UTILS_CUH

#include <thrust/transform.h>
#include <thrust/device_ptr.h>

#include "common.cuh"
#include "Timer.hpp"


/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *                   MACROS  
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

#define ASSERT(cond, fmt, ...)                                                \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::fprintf(stderr, "Assertion failed: %s\n", #cond);                \
            std::fprintf(stderr, fmt "\n", ##__VA_ARGS__);            \
            std::fprintf(stderr, "%s:%d\n", __FILE__, __LINE__);     \
            std::abort();                                                         \
        }                                                                         \
    } while (0)


#define NOT_REACHABLE() \
    do { \
        std::fprintf(stderr, "Unreachable thing reached at %s:%d\n", __FILE__, __LINE__); \
        std::abort(); \
    } while (0)



#define CUDA_CHECK(call) {                                                 \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",      \
                __FILE__, __LINE__, cudaGetErrorString(err));              \
        exit(err);                                                         \
    }                                                                      \
}


#define CUSPARSE_CHECK(call) do {                                    \
    cusparseStatus_t err = call;                                     \
    if (err != CUSPARSE_STATUS_SUCCESS) {                            \
        fprintf(stderr, "cuSPARSE error in file '%s' in line %i : %s.\n", \
                __FILE__, __LINE__, cusparseGetErrorString(err));    \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} while(0)


#define CUDA_FREE(buf) do { \
    if (buf != nullptr) \
    { \
        CUDA_CHECK(cudaFree(buf)); \
    } \
} while (0)


namespace mxt
{

namespace utils
{

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *              DATA MOVEMENT
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
template <typename T>
T * d2h_cpy(T * d_arr, size_t n)
{
    T * h_arr = new T[n];
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, sizeof(T) * n, cudaMemcpyDeviceToHost));
    return h_arr;
}


template <typename T>
void d2h_cpy(T * h_arr, T * d_arr, size_t n)
{
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, sizeof(T) * n, cudaMemcpyDeviceToHost));
}


template <typename T>
T * h2d_cpy(T * h_arr, size_t n)
{
    T * d_arr;
    CUDA_CHECK(cudaMalloc(&d_arr, sizeof(T) * n));
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, sizeof(T) * n, cudaMemcpyHostToDevice));
    return d_arr;
}


template <typename T>
void h2d_cpy(T * d_arr, T * h_arr, size_t n)
{
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, sizeof(T) * n, cudaMemcpyHostToDevice));
}


template <typename T>
T * h2d_cpy(std::vector<T> h_arr)
{
    return h2d_cpy(h_arr.data(), h_arr.size());
}


/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *               PRINTING AND IO 
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

template <typename T, typename... Args>
void print_h_arr(T * h_arr, size_t n, const char * prefix, Args... args)
{
    std::printf(prefix, args...);
    std::printf("\n");
    std::for_each(std::begin(h_arr), std::begin(h_arr) + n, [](const T& x){std::cout<<x<<'\n';});
    std::flush(std::cout);
}


template <typename T, typename... Args>
void print_d_arr(T * d_arr, size_t n, const char * prefix, Args... args)
{
    T * h_arr = d2h_cpy(d_arr, n);
    print_h_arr(h_arr, n, prefix, args...);
    delete[] h_arr;
}


inline void print_separator(const char * s)
{
    std::cout<<"====================="<<s<<"====================="<<std::endl;
    sleep(1);
}


/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *               PRECISION 
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

template <typename T1, typename T2>
struct round_functor
{
    __host__ __device__ __forceinline__
    T2 operator()(T1 x) 
    {
        return T2(x);
    }
};

template <typename T1, typename T2>
T2 * d_to_u(T1 * d_in, const size_t n)
{
    T2 * d_out;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(T2) * n));

    auto d_in_ptr = thrust::device_pointer_cast<T1>(d_in);
    auto d_out_ptr = thrust::device_pointer_cast<T2>(d_out);

    thrust::transform(d_in_ptr, d_in_ptr + n, d_out_ptr, round_functor<T1, T2>{});

    return d_out;
}



}// utils
}// mxt



#endif
