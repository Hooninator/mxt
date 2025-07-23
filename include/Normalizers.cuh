#ifndef NORMALIZERS_CUH
#define NORMALIZERS_CUH

#include "common.cuh"
#include "utils.cuh"


namespace mxt
{


template <typename T>
class NormalizerNull
{

public:

    NormalizerNull(const size_t N):
        N(N)
    {
    }

    template <typename Tensor_t>
    void normalize_tensor(Tensor_t& X, const size_t mode)
    {
        //Do nothing
    }


    template <typename Matrix_t>
    void normalize_matrix(Matrix_t& U)
    {
        //Do nothing
    }

private:

    size_t N;

};


/* X_tilde = RXS --- U_tilde = DUR^-1 */
template <typename T>
class NormalizerTwoSided
{
public:

    NormalizerTwoSided(const size_t N):
        N(N), R_vec(N), S_vec(N), D_vec(N),
        R_vec_n(N), S_vec_n(N), D_vec_n(N)
    {}


    template <typename Tensor_t>
    void normalize_tensor(Tensor_t& X, const size_t mode)
    {
        size_t R_size = Tensor_t::Modes[mode];
        size_t S_size = Tensor_t::In / R_size;

        /* First, set R */
        T * d_R = kernels::unfolding_rowmax(X, mode);

        /* Apply R */

        /* Now, set S */

        /* Apply S */

        /* Add matrices to vectors */
    }


    template <typename Matrix_t>
    void normalize_matrix(Matrix_t& U, const size_t m, const size_t n)
    {
        /* First, apply R^-1 */

        /* Now, set D */

        /* Apply D */
    }


    ~NormalizerTwoSided()
    {
        for (size_t i=0; i<N; i++)
        {
            CUDA_FREE(R_vec[i]);
            CUDA_FREE(S_vec[i]);
            CUDA_FREE(D_vec[i]);
        }
    }

private:
    size_t N;
    std::vector<T*> R_vec;
    std::vector<T*> S_vec;
    std::vector<T*> D_vec;

    std::vector<size_t> R_vec_n;
    std::vector<size_t> S_vec_n;
    std::vector<size_t> D_vec_n;
};


} //mxt

#endif
