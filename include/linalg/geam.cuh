#ifndef GEAM_CUH
#define GEAM_CUH

namespace mxt
{
namespace linalg
{

template <typename T>
void geam(T * d_in, T * d_out, const size_t m, const size_t n)
{

    T alpha = 1.0;
    T beta = 0.0;

    if constexpr(std::is_same<T, double>::value)
    {
        CUBLAS_CHECK(cublasDgeam(globals::cublas_handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    n, m, &alpha, d_in, m, 
                                    &beta, d_out, n,
                                    d_out, n));
    }
    else if constexpr(std::is_same<T, float>::value)
    {
        CUBLAS_CHECK(cublasSgeam(globals::cublas_handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    n, m, &alpha, d_in, m, 
                                    &beta, d_out, n,
                                    d_out, n));
    }
    else
    {
        NOT_REACHABLE();
    }
}

} //linalg
} //mxt







#endif 
