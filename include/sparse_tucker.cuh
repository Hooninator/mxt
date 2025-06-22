#ifndef SPARSE_TUCKER_CUH
#define SPARSE_TUCKER_CUH

#include "common.cuh"
#include "SparseTensor.cuh"
#include "SymbolicTTMC.cuh"
#include "utils.cuh"
#include "rand/rand_matrix.cuh"
#include "linalg/svd.cuh"
#include "kernels/spttmc.cuh"
#include "kernels/transpose.cuh"
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

    using TuckerRanksShape = TuckerShape;
    using InputModesShape = SparseTensor_t::ShapeType_t;

    static constexpr uint32_t Order = SparseTensor_t::Order;
    static constexpr Index_t TuckerRanks = TuckerShape::dims;
    static constexpr Index_t InputModes = SparseTensor_t::Modes;

    static_assert( TuckerRanks.size() == Order );

    TuckerTensor(const char * init)
    {
        core_formed = false;
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
        factors_t.reserve(Order);
        for (uint32_t i=0; i<Order; i++)
        {
            IndexType_t sz = InputModes[i] * TuckerRanks[i];
            rand::randn_buffer(&factors[i], sz);
            CUDA_CHECK(cudaMalloc(&factors_t[i], sizeof(FactorValueType_t) * sz));
        }

    }


    void update_factors_t()
    {
        ([&]<std::size_t... Is>(std::index_sequence<Is...>)
        { 
             (kernels::transpose_outplace<FactorValueType_t, InputModes[Is], TuckerRanks[Is]>(factors[Is], factors_t[Is]), ...);
#if DEBUG >= 2
             (utils::write_d_arr(utils::logfile, factors_t[Is], InputModes[Is] * TuckerRanks[Is], "Factor matrix transpose"), ...);
             (utils::write_d_arr(utils::logfile, factors[Is], InputModes[Is] * TuckerRanks[Is], "Factor matrix"), ...);
#endif
        }(std::make_index_sequence<Order>{}));
        CUDA_CHECK(cudaDeviceSynchronize());
    }


    // TODO: Pass the actual tensor not pointers
    void form_core(FactorValueType_t * d_X_vals, Index_t * d_X_inds, size_t X_nnz, SymbolicTTMC& symb)
    {
        /* Set transpose factors */
        update_factors_t();

        /* Allocate core tensor */
        static constexpr IndexType_t Rn = std::reduce(TuckerRanks.begin(), TuckerRanks.end(), 1, std::multiplies<IndexType_t>{});
        CUDA_CHECK(cudaMalloc(&d_core, sizeof(FactorValueType_t) * Rn));

        /* SpTTMc involving X and U^T[1...N] */
        kernels::spttmc<FactorValueType_t, FactorValueType_t, IndexType_t, Index_t, Order, TuckerRanksShape, InputModesShape>
            (d_X_vals, d_X_inds, factors.data(), symb, d_core, X_nnz, 0);
        core_formed = true;
    }


    ~TuckerTensor()
    {
        std::for_each(factors.begin(), factors.end(), [](FactorValueType_t * factor) {CUDA_FREE(factor);});
        std::for_each(factors_t.begin(), factors_t.end(), [](FactorValueType_t * factor) {CUDA_FREE(factor);});
        CUDA_FREE(d_core);
    }

    bool core_formed;
    FactorValueType_t * d_core;
    std::vector<FactorValueType_t *> factors;
    std::vector<FactorValueType_t *> factors_t;
};


template <typename SparseTensor_t, typename TuckerShape, typename Ttmc_u, typename Lra_u>
TuckerTensor<SparseTensor_t, TuckerShape, Ttmc_u> mixed_sparse_hooi(SparseTensor_t& X, const char * init, const size_t maxiters)
{

#if DEBUG >= 2
    utils::logfile.open("logfile.out");
#endif

    //TODO: Move this
    cusolverDnHandle_t handle;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

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
            kernels::spttmc<Ttmc_u, Lra_u, IndexType_t, Index_t, N, InputShape, TuckerShape>(d_X_vals, d_X_inds, d_U_list, symbolic_ttmc, d_Y_n, nnz, n);
            DEBUG_PRINT("Mode %u ttmc done", n);

#if DEBUG >= 2
            utils::write_d_arr(ofs, d_Y_n, InputModes[n] * (Rn / TuckerRanks[n]), "TTMc output");
#endif

            /* Update U[n] with truncated SVD */
            linalg::llsv_randsvd_cusolver<Lra_u, Ttmc_u, IndexType_t, 5, 2>(handle, d_Y_n, d_U_list[n], InputModes[n], (Rn / TuckerRanks[n]), TuckerRanks[n]);
            DEBUG_PRINT("Mode %u llsv done", n);
        }

        /* Form core tensor */
        X_tucker.form_core(d_X_vals, d_X_inds, nnz, symbolic_ttmc);

#if DEBUG >= 2
        utils::write_d_arr(utils::logfile, X_tucker.d_core, Rn, "Core Tensor");
#endif

        /* Check convergence */

        /* Clear workspace */
        workspace.zero();
    }

#if DEBUG >= 2
    ofs.close();
    utils::logfile.close();
#endif

    /* Form core tensor */

    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return X_tucker;
}

} //mxt 

#endif
