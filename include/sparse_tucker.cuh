#ifndef SPARSE_TUCKER_CUH
#define SPARSE_TUCKER_CUH

#include "common.cuh"
#include "SparseTensor.cuh"
#include "utils.cuh"
#include "rand/rand_matrix.cuh"
#include "kernels/spttmc.cuh"
#include "DeviceWorkspace.cuh"

namespace mxt
{

template <typename SparseTensor_t, typename TuckerShape, typename FactorValueType>
struct TuckerTensor
{
    /* This is the value type of the tensor that the TuckerTensor approximates */
    using InputValueType_t = SparseTensor_t::ValueType_t;

    /* This is the value type of the factor matrices and core tensor */
    using FactorValueType_t = FactorValueType;

    using IndexType_t = SparseTensor_t::IndexType_t;
    using Index_t = SparseTensor_t::Index_t;

    static constexpr uint32_t Order = SparseTensor_t::Order;
    static constexpr Index_t TuckerRanks = TuckerShape::dims;
    static constexpr Index_t InputModes = SparseTensor_t::Modes;

    static_assert( TuckerRanks.size() == Order );

    TuckerTensor(const char * init)
    {
        init_factors(init);
    }

    
    void init_factors(const char * init)
    {
        std::string init_str(init);
        if (init_str.compare("randn")==0)
        {
            init_factors_randn();
        }
        else
        {
            NOT_REACHABLE();
        }
    }


    void init_factors_randn()
    {
        factors.reserve(Order);
        for (uint32_t i=0; i<Order; i++)
        {
            IndexType_t sz = InputModes[i] * TuckerRanks[i];
            rand::randn_buffer(factors[i], sz);
        }
    }


    void form_core(SparseTensor_t& X)
    {
        /* TTM chain involving X and U^T[1...N] */
    }


    SparseTensor<FactorValueType_t, IndexType_t, Order, TuckerShape> core;
    std::vector<FactorValueType_t *> factors;
};


template <typename SparseTensor_t>
struct SymbolicTTMC
{

    SymbolicTTMC (SparseTensor_t& X)
    {
        using Index_t = SparseTensor_t::Index_t;
        using IndexType_t = SparseTensor_t::IndexType_t;
        using Workspace = DeviceWorkspace<size_t>;

        static constexpr uint32_t Order = SparseTensor_t::Order;
        static constexpr Index_t Modes = SparseTensor_t::Modes;

        /* Y_n_inds[n][i] -- indices of the ith mode-n slice X(:, :, ..., i_n, ..., :) */
        const size_t nnz = X.get_nnz();

        // First index determines mode unfolding, second determines row index of Y_n, then you have the acutal index list
        std::vector<std::vector<std::vector<size_t> > > h_Y_n_inds;

        static constexpr size_t ModeSum = std::reduce(Modes.begin(), Modes.end(), 0);
        std::vector<size_t> h_Y_n_offsets(ModeSum + 1, 0);

        std::vector<size_t> mode_offsets(Order, 0);
        std::exclusive_scan(Modes.begin(), Modes.end(), mode_offsets.begin(), 0);

        h_Y_n_inds.resize(Order);
        for (size_t n=0; n<Order; n++)
        {
            h_Y_n_inds[n].resize(Modes[n]);
        }

        Index_t * h_inds = utils::d2h_cpy(X.get_d_inds(), nnz);

        for (size_t i=0; i<nnz; i++)
        {
            Index_t idx = h_inds[i];
            for (uint32_t n=0; n < Order; n++)
            {
                h_Y_n_inds[n][idx[n]].push_back(i);
                size_t offset = idx[n] + mode_offsets[n];
                h_Y_n_offsets[offset + 1] += 1;
            }
        }

        std::inclusive_scan(h_Y_n_offsets.begin(), h_Y_n_offsets.end(), h_Y_n_offsets.begin());

        d_Y_n_inds.alloc(nnz * Order);
        d_Y_n_offsets.alloc(ModeSum);

        // Move to device
        d_Y_n_offsets.h2d_cpy(h_Y_n_offsets.data(), h_Y_n_offsets.size());

        // Indices to device
        size_t offset = 0;
        for (size_t n=0; n < Order; n++)
        {
            for (size_t i=0; i < Modes[n]; i++)
            {
                d_Y_n_inds.h2d_cpy(h_Y_n_inds[n][i].data(), h_Y_n_inds[n][i].size(), offset);
                offset += Modes[n];
            }
        }

        delete[] h_inds;

    }

    DeviceWorkspace<size_t> d_Y_n_inds;
    DeviceWorkspace<size_t> d_Y_n_offsets;
};


template <typename SparseTensor_t, typename TuckerShape, typename Ttmc_u, typename Lra_u>
TuckerTensor<SparseTensor_t, TuckerShape, Ttmc_u> mixed_sparse_hooi(SparseTensor_t& X, const char * init, const size_t maxiters)
{
    using ValueType_t = SparseTensor_t::ValueType_t;
    using IndexType_t = SparseTensor_t::IndexType_t;
    using Index_t = SparseTensor_t::Index_t;
    using InputShape = SparseTensor_t::ShapeType_t;

    static constexpr uint32_t N = SparseTensor_t::Order;
    static constexpr Index_t TuckerRanks = TuckerShape::dims;
    static constexpr Index_t InputModes = SparseTensor_t::Modes;

    TuckerTensor<SparseTensor_t, TuckerShape, Ttmc_u> X_tucker(init);

    Ttmc_u * d_X_vals = utils::d_to_u<ValueType_t, Ttmc_u>(X.get_d_vals(), X.get_nnz());

    Index_t * d_X_inds = X.get_d_inds();
    Ttmc_u ** d_U_list = X_tucker.factors.data();
    const size_t nnz = X.get_nnz();

    /* Workspace for storing TTMc output */
    static constexpr IndexType_t largest_mode = *(std::max_element(InputModes.begin(), InputModes.end()));
    static constexpr IndexType_t Rn_minus_1 = std::reduce(TuckerRanks.begin(), TuckerRanks.end(), 0, std::multiplies<IndexType_t>{});

    DeviceWorkspace<Ttmc_u> workspace(largest_mode * Rn_minus_1);
    Ttmc_u * d_Y_n = workspace.d_data;

    /* Symbolic TTMc -- record indices of all nonzeros that contribute to each row of the TTMc outputs 
     * Each entry of this array is a device pointer
     */
    SymbolicTTMC symbolic_ttmc(X);

    /* Main Loop */
    for (size_t iter = 0; iter < maxiters; iter++)
    {
        for (uint32_t n=0; n < N; n++)
        {
            /* TTM chain with all but U[n] */
            spttmc<Ttmc_u, IndexType_t, Index_t, InputShape, TuckerShape>(d_X_vals, d_X_inds, d_U_list, d_Y_n, nnz, n);

            /* Update U[n] with truncated SVD */

        }

        /* Check error/convergence */

        /* Clear workspace */
        workspace.zero();
    }

    /* Form core tensor */

    return X_tucker;
}

} //mxt 

#endif
