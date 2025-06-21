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

#if DEBUG >= 1
#define DEBUG_PRINT(msg, ...) \
    do { \
        std::printf("%s:%d -- ", __FILE__, __LINE__); \
        std::printf(msg "\n", ##__VA_ARGS__); \
        std::flush(std::cout);\
    } while (0)

#define CHECKPOINT() \
    do { \
        std::printf("CHECKPOINT: %s:%d\n", __FILE__, __LINE__); \
        std::flush(std::cout);\
    } while (0)

#else
#define DEBUG_PRINT(msg, ...)
#define CHECKPOINT()
#endif


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


#if DEBUG >= 2
std::ofstream logfile;
#endif

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
void print_h_vec(std::vector<T>& vec, const char * prefix, Args... args)
{
    print_h_arr(vec.data(), vec.size(), prefix, args...);
}


template <typename T, typename... Args>
void print_h_arr(T * h_arr, size_t n, const char * prefix, Args... args)
{
    std::printf(prefix, args...);
    std::printf("\n");
    for (size_t i=0; i<n; i++)
    {
        std::cout<<h_arr[i]<<'\n';
    }
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
    std::cout<<std::endl;
}


template <typename T>
void write_h_arr(std::ofstream& ofs, T * h_arr, size_t n, const char * prefix)
{
    ofs<<prefix<<"\n";
    for (int i=0; i<n; i++)
    {
        ofs<<h_arr[i]<<'\n';
    }
    std::flush(ofs);
}


template <typename T>
void write_d_arr(std::ofstream& ofs, T * d_arr, size_t n, const char * prefix)
{
    T * h_arr = d2h_cpy(d_arr, n);
    write_h_arr(ofs, h_arr, n, prefix);
    delete[] h_arr;
}


inline void debug_print_separator(const char * s)
{
#if DEBUG >= 1
    print_separator(s);
#endif
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
        if constexpr(std::is_same<T1, T2>::value)
        {
            return x;
        }
        if constexpr(std::is_same<T1, double>::value)
        {
            if constexpr(std::is_same<T2, half>::value)
            {
                return __double2half(x);
            }
            if constexpr(std::is_same<T2, float>::value)
            {
                return (float)(x);
            }
        }
        if constexpr(std::is_same<T1, float>::value)
        {
            if constexpr(std::is_same<T2, half>::value)
            {
                return __float2half(x);
            }
            if constexpr(std::is_same<T2, double>::value)
            {
                return (double)(x);
            }
        }
        if constexpr(std::is_same<T1, half>::value)
        {
            if constexpr(std::is_same<T2, float>::value)
            {
                return __half2float(x);
            }
            if constexpr(std::is_same<T2, double>::value)
            {
                return (double)(__half2float(x));
            }
        }
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





/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *                   MISC
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

template <typename T, std::size_t... Is>
inline constexpr std::array<T, sizeof...(Is)> make_array_impl(std::index_sequence<Is...>) {
    return { (static_cast<void>(Is), T{})... };
}


template <typename T, std::size_t N>
inline constexpr std::array<T, N> make_array() 
{
    return make_array_impl<T>(std::make_index_sequence<N>{});
}


//template <typename T, std::size_t N, std::size_t Exclude>
//__device__ __host__ __forceinline__
//constexpr std::array<T, N - 1> remove_one(const std::array<T, N>& arr)
//{
//    std::index_sequence<(std::make_index_sequence<N>{})...> Is;
//
//    auto constexpr Excluded = []<std::size_t... Js>(const std::array<T, N> arr2, std::index_sequence<Js...>)
//    {
//        std::array<T, N - 1> result{};
//        size_t idx = 0;
//        ((Js != Exclude ? (result[idx++] = arr[Js], 0) : 0), ...);
//        return result;
//    }(arr, Is);
//
//    return Excluded;
//}


}// utils
}// mxt



#endif
