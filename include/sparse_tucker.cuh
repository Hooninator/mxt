#ifndef SPARSE_TUCKER_CUH
#define SPARSE_TUCKER_CUH

#include "common.cuh"
#include "SparseTensor.cuh"
#include "SymbolicTTMC.cuh"
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
            rand::randn_buffer(&factors[i], sz);
        }
    }


    void form_core(SparseTensor_t& X)
    {
        /* TTM chain involving X and U^T[1...N] */
    }


    ~TuckerTensor()
    {
        std::for_each(factors.begin(), factors.end(), [](FactorValueType_t * factor) {CUDA_FREE(factor);});
    }


    SparseTensor<FactorValueType_t, IndexType_t, Order, TuckerShape> core;
    std::vector<FactorValueType_t *> factors;
};


template <typename SparseTensor_t, typename TuckerShape, typename Ttmc_u, typename Lra_u>
TuckerTensor<SparseTensor_t, TuckerShape, Ttmc_u> mixed_sparse_hooi(SparseTensor_t& X, const char * init, const size_t maxiters)
{

#if DEBUG >= 2
    utils::logfile.open("logfile.out");
#endif

    using ValueType_t = SparseTensor_t::ValueType_t;
    using IndexType_t = SparseTensor_t::IndexType_t;
    using Index_t = SparseTensor_t::Index_t;
    using InputShape = SparseTensor_t::ShapeType_t;

    static constexpr uint32_t N = SparseTensor_t::Order;
    static constexpr Index_t TuckerRanks = TuckerShape::dims;
    static constexpr Index_t InputModes = SparseTensor_t::Modes;

    utils::print_separator("Making tucker tensor");
    TuckerTensor<SparseTensor_t, TuckerShape, Ttmc_u> X_tucker(init);
    utils::print_separator("Done");

    Ttmc_u * d_X_vals = utils::d_to_u<ValueType_t, Ttmc_u>(X.get_d_vals(), X.get_nnz());

    Index_t * d_X_inds = X.get_d_inds();
    Ttmc_u ** d_U_list = X_tucker.factors.data();
    const size_t nnz = X.get_nnz();

    /* Workspace for storing TTMc output */
    static constexpr IndexType_t largest_mode = *(std::max_element(InputModes.begin(), InputModes.end()));
    static constexpr IndexType_t Rn = std::reduce(TuckerRanks.begin(), TuckerRanks.end(), 1, std::multiplies<IndexType_t>{});

    DeviceWorkspace<Lra_u> workspace(largest_mode * Rn); //TODO: This is an overallocation, it should be largest_most * largest product of n-1 tucker modes
    Lra_u * d_Y_n = workspace.d_data;

    /* Symbolic TTMc -- record indices of all nonzeros that contribute to each row of the TTMc outputs 
     * Each entry of this array is a device pointer
     */
    utils::print_separator("Beginning Symbolic");
    SymbolicTTMC symbolic_ttmc(X);

#if DEBUG >= 2
    X.dump(utils::logfile);
    symbolic_ttmc.dump(utils::logfile);
#endif

    utils::print_separator("Done symbolic");

#if DEBUG >= 2
    std::ofstream ofs;
    ofs.open("ttmc_out.out");
#endif

    /* Main Loop */
    for (size_t iter = 0; iter < maxiters; iter++)
    {
        for (uint32_t n=0; n < N; n++)
        {
            /* TTM chain with all but U[n] */
            spttmc<Ttmc_u, Lra_u, IndexType_t, Index_t, N, InputShape, TuckerShape>(d_X_vals, d_X_inds, d_U_list, symbolic_ttmc, d_Y_n, nnz, n);

#if DEBUG >= 2
            utils::write_d_arr(ofs, d_Y_n, InputModes[n] * (Rn / TuckerRanks[n]), "TTMc output");
#endif

            DEBUG_PRINT("Mode %u ttmc done", n);

            /* Update U[n] with truncated SVD */

        }

        /* Check error/convergence */

        /* Clear workspace */
        workspace.zero();
    }

#if DEBUG >= 2
    ofs.close();
    utils::logfile.close();
#endif

    /* Form core tensor */

    return X_tucker;
}

} //mxt 

#endif
