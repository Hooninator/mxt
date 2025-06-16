#ifndef SPARSE_TUCKER_CUH
#define SPARSE_TUCKER_CUH

#include "common.cuh"
#include "SparseTensor.cuh"
#include "utils.cuh"
#include "rand/rand_matrix.cuh"
#include "kernels/spttmc.cuh"

namespace mxt
{

template <typename SparseTensor_t, auto _TuckerRanks>
struct TuckerTensor
{
    using ValueType_t = SparseTensor_t::ValueType;
    using IndexType_t = SparseTensor_t::IndexType;
    using Index_t = SparseTensor_t::Index;
    static constexpr uint32_t Order = SparseTensor_t::Order;
    static constexpr Index_t TuckerRanks = _TuckerRanks;

    static_assert( TuckerRanks.size() == Order );

    TuckerTensor(Index_t& input_modes, const char * init)
    {
        input_modes = input_modes;
        init_factors(input_modes, init);
    }

    
    void init_factors(Index_t& input_modes, const char * init)
    {
        std::string init_str(init);
        if (init_str.compare("randn")==0)
        {
            init_factors_randn(input_modes);
        }
        else
        {
            NOT_REACHABLE();
        }
    }


    void init_factors_randn(Index_t& input_modes)
    {
        factors.reserve(Order);
        for (uint32_t i=0; i<Order; i++)
        {
            IndexType_t sz = input_modes[i] * TuckerRanks[i];
            rand::randn_buffer(factors[i], sz);
        }
    }


    void form_core(SparseTensor_t& X)
    {
        /* TTM chain involving X and U^T[1...N] */
    }


    SparseTensor_t core;
    std::vector<ValueType_t *> factors;
    Index_t input_modes;
};


template <typename SparseTensor_t, auto _TuckerRanks, typename Ttmc_u, typename Lra_u>
TuckerTensor<SparseTensor_t, _TuckerRanks> mixed_sparse_hooi(SparseTensor_t& X, const char * init, const size_t maxiters)
{
    using ValueType_t = SparseTensor_t::ValueType;
    using IndexType_t = SparseTensor_t::IndexType;
    using Index_t = SparseTensor_t::Index;

    static constexpr uint32_t N = SparseTensor_t::Order;
    static constexpr Index_t TuckerRanks = _TuckerRanks;

    Index_t modes = X.get_mode_sizes();

    TuckerTensor<SparseTensor_t, TuckerRanks> X_tucker(modes, init);


    /* Main Loop */
    for (size_t iter = 0; iter < maxiters; iter++)
    {

        for (uint32_t n=0; n < N; n++)
        {
            /* TTM chain with all but U[n] */

            /* Update U[n] with truncated SVD */

        }


        /* Check error/convergence */

    }

    /* Form core tensor */

    return X_tucker;
}

} //mxt 

#endif
