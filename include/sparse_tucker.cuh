#ifndef SPARSE_TUCKER_CUH
#define SPARSE_TUCKER_CUH

#include "common.cuh"
#include "SparseTensor.cuh"
#include "SymbolicTTMC.cuh"
#include "utils.cuh"
#include "DeviceWorkspace.cuh"

#include "rand/rand_matrix.cuh"
#include "linalg/svd.cuh"
#include "linalg/gemm.cuh"
#include "kernels/spttmc.cuh"
#include "kernels/transpose.cuh"

namespace mxt
{

template <typename SparseTensor_t, typename TuckerShape, typename FactorValueType, typename CoreValueType>
struct TuckerTensor
{
    /* This is the value type of the tensor that the TuckerTensor approximates */
    using InputValueType_t = SparseTensor_t::ValueType_t;

    /* These are the value types of the factor matrices and core tensor */
    using FactorValueType_t = FactorValueType;
    using CoreValueType_t = CoreValueType;

    using IndexType_t = SparseTensor_t::IndexType_t;
    using Index_t = SparseTensor_t::Index_t;

    using TuckerRanksShape = TuckerShape;
    using InputModesShape = SparseTensor_t::ShapeType_t;

    static constexpr uint32_t Order = SparseTensor_t::Order;
    static constexpr Index_t TuckerRanks = TuckerShape::dims;
    static constexpr Index_t InputModes = SparseTensor_t::Modes;
    static constexpr IndexType_t Rn = std::reduce(TuckerRanks.begin(), TuckerRanks.end(), 1, std::multiplies<IndexType_t>{});

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

        factor_descrs.resize(Order);

        for (uint32_t i=0; i < Order; i++)
        {
            CUSPARSE_CHECK(cusparseCreateDnMat(&factor_descrs[i], InputModes[i], TuckerRanks[i], TuckerRanks[i], 
                                                factors[i], utils::to_cuda_dtype<FactorValueType_t>(),
                                                CUSPARSE_ORDER_ROW));
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


    void ttmc(SparseTensor_t& X, SymbolicTTMC& symb, FactorValueType_t * d_Y_n)
    {
        static constexpr size_t I = InputModes[0];
        static constexpr size_t R0 = TuckerRanks[0];

        /* Step 1: SpTTMC along mode 0 */
        kernels::spttmc<FactorValueType_t, FactorValueType_t, IndexType_t, Index_t, Order, InputModesShape, TuckerRanksShape, 0>
            (X.get_d_vals_low(), X.get_d_inds(), factors.data(), symb, d_Y_n, X.get_nnz());

        /* Step 2: U_0^T Y_0 = G_0 */
        linalg::gemm(d_Y_n, factors[0], d_core, I, Rn / R0, R0);

        //DeviceWorkspace<FactorValueType_t> ws(R0 * In_minus_1);
        //cusparseDnMatDescr_t d_Y_descr;
        //CUSPARSE_CHECK(cusparseCreateDnMat(&d_Y_descr, R0, In_minus_1, In_minus_1, ws.d_data, utils::to_cuda_dtype<FactorValueType_t>(), CUSPARSE_ORDER_COL));

        //size_t buf_size = 0;
        //void * d_buf = nullptr;

        ///* Step 1: SpMM to compute X_0^TU_1^T */
        //FactorValueType_t alpha = 1.0;
        //FactorValueType_t beta = 1.0;
        //CUSPARSE_CHECK(cusparseSpMM_bufferSize(globals::cusparse_handle, 
        //                                        CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        //                                        &alpha, X_descr, factor_descrs[0], &beta, d_Y_descr, 
        //                                        (( (std::is_same<FactorValueType_t, __half>::value) || 
        //                                          std::is_same<FactorValueType_t, float>::value) ? CUDA_R_32F : CUDA_R_64F),
        //                                        CUSPARSE_SPMM_ALG_DEFAULT, 
        //                                        &buf_size));
        //CUDA_CHECK(cudaMalloc(&d_buf, buf_size));
        //CUSPARSE_CHECK(cusparseSpMM(globals::cusparse_handle, 
        //                            CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        //                            &alpha, X_descr, factor_descrs[0], &beta, d_Y_descr, 
        //                            (( (std::is_same<FactorValueType_t, __half>::value) || 
        //                              std::is_same<FactorValueType_t, float>::value) ? CUDA_R_32F : CUDA_R_64F),
        //                            CUSPARSE_SPMM_ALG_DEFAULT, 
        //                            d_buf));
        //CUSPARSE_CHECK(cusparseDestroyDnMat(d_Y_descr));
        //CUDA_FREE(d_buf);

        ///* Step 2: Dense TTMs */
        //for (size_t n = 0; n < Order; n++)
        //{
        //    linalg::dense_ttm();
        //}
    }


    // TODO: Pass the actual tensor not pointers
    void form_core(SparseTensor_t& X, SymbolicTTMC& symb, FactorValueType_t * d_workspace)
    {
        /* Allocate core tensor if it hasn't been done so yet */
        if (!core_formed)
        {
            CUDA_CHECK(cudaMalloc(&d_core, sizeof(CoreValueType_t) * Rn));
        }

        /* TTM chain */
        ttmc(X, symb, d_workspace);

        core_formed = true;
    }


    double core_norm()
    {

        if (!core_formed)
        {
            std::cerr<<"Tried to compute ||G|| without first calling form_core()"<<std::endl;
            std::abort();
        }

        double * d_core_dbl = utils::d_to_u<CoreValueType_t, double>(d_core, Rn);

        double result;
        CUBLAS_CHECK(cublasNrm2Ex(globals::cublas_handle, 
                                  Rn, d_core_dbl, CUDA_R_64F,
                                  1, &result, CUDA_R_64F,
                                  CUDA_R_64F));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_FREE(d_core_dbl);

        return result*result;
    }


    ~TuckerTensor()
    {
        std::for_each(factors.begin(), factors.end(), [](FactorValueType_t * factor) {CUDA_FREE(factor);});
        std::for_each(factor_descrs.begin(), factor_descrs.end(), [](cusparseDnMatDescr_t& x) {CUSPARSE_CHECK(cusparseDestroyDnMat(x));});
        CUDA_FREE(d_core);
    }


    bool core_formed;
    CoreValueType_t * d_core;
    std::vector<FactorValueType_t *> factors;
    std::vector<cusparseDnMatDescr_t> factor_descrs;
};


template <typename SparseTensor_t, typename CoreTensor_t, typename TuckerShape, typename Lra_t>
TuckerTensor<SparseTensor_t, TuckerShape, typename SparseTensor_t::ValueTypeLow_t, CoreTensor_t> mixed_sparse_hooi(SparseTensor_t& X, const char * init, const size_t maxiters)
{

    //TODO: These needs to be renamed
    using ValueType_t = SparseTensor_t::ValueType_t;
    using IndexType_t = SparseTensor_t::IndexType_t;
    using Index_t = SparseTensor_t::Index_t;
    using InputShape = SparseTensor_t::ShapeType_t;
    using TTMc_t = SparseTensor_t::ValueTypeLow_t;

    static constexpr uint32_t N = SparseTensor_t::Order;
    static constexpr Index_t TuckerRanks = TuckerShape::dims;
    static constexpr Index_t InputModes = SparseTensor_t::Modes;

    TuckerTensor<SparseTensor_t, TuckerShape, TTMc_t, CoreTensor_t> X_tucker(init);
    std::vector<double> core_norms;
    TTMc_t * d_X_vals = X.get_d_vals_low();

    Index_t * d_X_inds = X.get_d_inds();
    IndexType_t * d_X_colinds = X.get_d_colinds();
    IndexType_t * d_X_rowptrs = X.get_d_rowptrs();
    TTMc_t ** d_U_list = X_tucker.factors.data();
    cusparseSpMatDescr_t X_cusparse_descr = X.get_cusparse_descr();
    const size_t nnz = X.get_nnz();

    /* Workspace for storing TTMc output */
    static constexpr IndexType_t largest_mode = *(std::max_element(InputModes.begin(), InputModes.end()));
    static constexpr IndexType_t Rn = std::reduce(TuckerRanks.begin(), TuckerRanks.end(), 1, std::multiplies<IndexType_t>{});

    DeviceWorkspace<char> workspace(largest_mode * Rn * sizeof(Lra_t)); //TODO: This is an overallocation, it should be largest_most * largest product of n-1 tucker modes
    Lra_t * d_Y_n = (Lra_t *)workspace.d_data;

    /* Symbolic TTMc -- record indices of all nonzeros that contribute to each row of the TTMc outputs 
     * Each entry of this array is a device pointer
     */
    utils::print_separator("Beginning Symbolic");
    SymbolicTTMC symbolic_ttmc(X);

#if DEBUG >= 2
    X.dump(globals::logfile);
    symbolic_ttmc.dump(globals::logfile);
#endif

    utils::print_separator("Done symbolic");

#if DEBUG >= 2
    std::ofstream ofs;
    ofs.open("ttmc_out.out");
#endif

    /* Main Loop */
    for (size_t iter = 0; iter < maxiters; iter++)
    {
        [&]<std::size_t... Is>(std::index_sequence<Is...>)
        {
        ((
            /* TTM chain with all but U[n] */
            utils::print_separator("SpTTMC"),
            kernels::spttmc<TTMc_t, Lra_t, IndexType_t, Index_t, N, InputShape, TuckerShape, Is>(d_X_vals, d_X_inds, d_U_list, symbolic_ttmc, d_Y_n, nnz),
#if DEBUG >= 2
            utils::write_d_arr(ofs, d_Y_n, InputModes[Is] * (Rn / TuckerRanks[Is]), "TTMc output"),
#endif
            /* Update U[n] with truncated SVD */
            utils::print_separator("SVD"),
            linalg::llsv_randsvd_cusolver<Lra_t, TTMc_t, IndexType_t, 5, 2, InputModes[Is], (Rn / TuckerRanks[Is]), TuckerRanks[Is]>(d_Y_n, d_U_list[Is])
         ), ...);
        }(std::make_index_sequence<N>{});

        /* Form core tensor */
        utils::print_separator("Forming core");
        X_tucker.form_core(X, symbolic_ttmc, (TTMc_t *)d_Y_n);

#if DEBUG >= 2
        utils::write_d_arr(globals::logfile, X_tucker.d_core, Rn, "Core Tensor");
#endif

        /* Record ||G|| */
        double core_norm = X_tucker.core_norm();
        core_norms.push_back(core_norm);

        std::cout<<"ITERATION "<<iter<<" ||G||: "<<core_norm<<'\n';

        /* Clear workspace */
        workspace.zero();
    }

#if DEBUG >= 2
    ofs.close();
#endif

    return X_tucker;
}

} //mxt 

#endif
