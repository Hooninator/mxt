#ifndef UTILS_CUH
#define UTILS_CUH

#include <thrust/transform.h>
#include <thrust/device_ptr.h>

#include "common.cuh"


/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *                   MACROS  
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

#define MAX_HALF 65504

#if DEBUG >= 1
#define DEBUG_PRINT(msg, ...) \
    do { \
        std::string filename = std::string(__FILE__);\
        std::string relpath = filename.substr(filename.find_last_of("/")+1, filename.size());\
        std::printf("[%s]: %d -- ", relpath.c_str(), __LINE__); \
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


#define CUBLAS_CHECK(call) do {                                    \
    cublasStatus_t err = call;                                     \
    if (err != CUBLAS_STATUS_SUCCESS) {                            \
        fprintf(stderr, "cuBLAS error in file '%s' in line %i : %d.\n", \
                __FILE__, __LINE__, err);    \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} while(0)


#define CUSPARSE_CHECK(call) do {                                    \
    cusparseStatus_t err = call;                                     \
    if (err != CUSPARSE_STATUS_SUCCESS) {                            \
        fprintf(stderr, "cuSPARSE error in file '%s' in line %i : %s.\n", \
                __FILE__, __LINE__, cusparseGetErrorString(err));    \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} while(0)


#define CUSOLVER_CHECK(call) do {                                    \
    cusolverStatus_t err = call;                                     \
    if (err != CUSOLVER_STATUS_SUCCESS) {                            \
        fprintf(stderr, "cuSolver error %d in file '%s' in line %i.\n", \
                err, __FILE__, __LINE__);    \
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


template <typename T>
T * d2d_cpy(T * d_arr, size_t n)
{
    T * d_arr2;
    CUDA_CHECK(cudaMalloc(&d_arr2, sizeof(T) * n));
    CUDA_CHECK(cudaMemcpy(d_arr2, d_arr, sizeof(T) * n, cudaMemcpyDeviceToDevice));
    return d_arr2;
}


template <typename T>
void d2d_cpy(T * d_arr, T * d_arr2, size_t n)
{
    CUDA_CHECK(cudaMemcpy(d_arr2, d_arr, sizeof(T) * n, cudaMemcpyDeviceToDevice));
}


template <typename T>
void d2d_cpy_async(T * d_arr, T * d_arr2, size_t n)
{
    CUDA_CHECK(cudaMemcpyAsync(d_arr2, d_arr, sizeof(T) * n, cudaMemcpyDeviceToDevice));
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
                return __double2float_rd(x);
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
                float x_f = __half2float(x);
                double x_d = static_cast<double>(x_f);
                return x_d;
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


template <typename T1, typename T2>
void d_to_u(T1 * d_in, T2 * d_out, const size_t n)
{
    auto d_in_ptr = thrust::device_pointer_cast<T1>(d_in);
    auto d_out_ptr = thrust::device_pointer_cast<T2>(d_out);
    thrust::transform(d_in_ptr, d_in_ptr + n, d_out_ptr, round_functor<T1, T2>{});
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
        std::cout<<h_arr[i];
        for (int j=0; j<20; j++)
        {
            std::cout<<" ";
        }
        std::cout<<"["<<prefix<<"]"<<'\n';
    }
    std::printf("\n");
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
    ofs<<"--"<<prefix<<"--\n";
    for (int i=0; i<n; i++)
    {
        ofs<<h_arr[i];
        for (int j=0; j<20; j++)
        {
            ofs<<" ";
        }
        ofs<<"["<<prefix<<"]:"<<i<<'\n';
    }
    ofs<<std::endl;
}


template <typename T>
void write_d_arr(std::ofstream& ofs, T * d_arr, size_t n, const char * prefix)
{
    T * h_arr = d2h_cpy(d_arr, n);
    write_h_arr(ofs, h_arr, n, prefix);
    delete[] h_arr;
}


// Specialize for half
void write_d_arr(std::ofstream& ofs, __half * d_arr, size_t n, const char * prefix)
{
    float * d_tmp = d_to_u<__half, float>(d_arr, n);
    float * h_arr = d2h_cpy(d_tmp, n);
    write_h_arr(ofs, h_arr, n, prefix);
    delete[] h_arr;
    CUDA_FREE(d_tmp);
}


inline void debug_print_separator(const char * s)
{
#if DEBUG >= 1
    print_separator(s);
#endif
}


/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *                   ARRAYS 
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


/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *                 CUDA TYPES
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

template <typename T>
inline constexpr cudaDataType to_cuda_dtype()
{
    if constexpr(std::is_same<T, float>::value)
    {
        return CUDA_R_32F;
    }
    else if constexpr(std::is_same<T, double>::value)
    {
        return CUDA_R_64F;
    }
    else if constexpr(std::is_same<T, __half>::value)
    {
        return CUDA_R_16F;
    }
}


template <typename In, typename Out>
cublasComputeType_t get_compute_type()
{
    if constexpr (std::is_same<In, __half>::value)
    {
        if constexpr (std::is_same<Out, __half>::value)
        {
            return CUBLAS_COMPUTE_16F;
        }
        if constexpr (std::is_same<Out, float>::value)
        {
            return CUBLAS_COMPUTE_32F;
        }
    }
    if constexpr (std::is_same<In, float>::value)
    {
        return CUBLAS_COMPUTE_32F;
    }
    if constexpr (std::is_same<In, double>::value)
    {
        return CUBLAS_COMPUTE_64F;
    }

}


template <typename I>
inline constexpr cusparseIndexType_t to_cusparse_idx()
{
    if constexpr(std::is_same<I, uint64_t>::value)
    {
        return CUSPARSE_INDEX_64I;
    }
    else if constexpr(std::is_same<I, uint32_t>::value)
    {
        return CUSPARSE_INDEX_32I;
    }
}

    
template <typename T1, typename T2>
constexpr inline bool same()
{
    return std::is_same<T1, T2>::value;
}



/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *                   MISC
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

template <typename T>
struct abs_functor
{
    __host__ __device__
    T operator()(T x) 
    {
        if constexpr(std::is_same<T, __half>::value)
        {
            return __habs(x);
        }
        else
        {
            return std::abs(x);
        }
    }

};


template <typename T>
void transpose(T * d_data, size_t m, size_t n)
{
    T * d_data_t = new T[m*n];
    for (int i=0; i<m; i++)
    {
        for (int j=0; j<n; j++)
        {
            d_data_t[j + i * n] = d_data[i + j * m];
        }
    }
    std::memcpy(d_data, d_data_t, m * n * sizeof(T));
    delete[] d_data_t;
}



size_t linear_index(const size_t * strides, const size_t * inds, size_t n)
{
    size_t ind = 0;
    size_t stride = 1;
    for (int i=0; i<n; i++)
    {
        ind += inds[i]*stride;
        stride *= strides[i];
    }
    return ind;
}



template <size_t N>
std::array<size_t, N> multidx_natural(size_t idx, const std::array<size_t, N> dims)
{
    std::array<size_t, N> multidx;
    for (int i=0; i<N; i++)
    {
        multidx[i] = idx % dims[i];
        idx /= dims[i];
    }
    return multidx;
}



template <size_t N>
std::array<size_t, N> multidx_reverse(size_t idx, std::array<size_t, N> dims)
{
    std::reverse(dims.begin(), dims.end());
    std::array<size_t, N> multidx = multidx_natural<N>(idx, dims);
    std::reverse(multidx.begin(), multidx.end());
    return multidx;
}

}// utils
}// mxt



#endif
