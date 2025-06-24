#ifndef SPARSE_TUCKER_CUH
#define SPARSE_TUCKER_CUH

#include "common.cuh"
#include "SparseTensor.cuh"
#include "SymbolicTTMC.cuh"
#include "utils.cuh"
#include "DeviceWorkspace.cuh"
#include "tensor_io.cuh"

#include "rand/rand_matrix.cuh"
#include "linalg/svd.cuh"
#include "linalg/gemm.cuh"
#include "kernels/spttmc.cuh"
#include "kernels/transpose.cuh"

namespace mxt
{

template <typename SparseTensor_t, typename CoreValueType, typename TuckerShape>
struct TuckerTensor
{
    /* This is the value type of the tensor that the TuckerTensor approximates */
    using InputValueType_t = SparseTensor_t::ValueType_t;

    /* These are the value types of the factor matrices and core tensor */
    using FactorValueType_t = SparseTensor_t::ValueTypeLow_t;
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

    TuckerTensor(const char * init, const char * factors_dir = nullptr)
    {
        core_formed = false;
        init_factors(init, factors_dir);
    }

    
    void init_factors(const char * init, const char * factors_dir)
    {
        std::string init_str(init);
        if (init_str.compare("randn")==0)
        {
            init_factors_randn();
        }
        else if (init_str.compare("file")==0)
        {
            init_factors_file(factors_dir);
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


    void init_factors_file(const char * factors_dir)
    {
        factors.reserve(Order);
        for (uint32_t i=0; i<Order; i++)
        {
            std::string path = std::string(factors_dir) + "factor_" + std::to_string(i) + ".tns";
            factors[i] = io::read_matrix_frostt<FactorValueType_t>(path.c_str(), InputModes[i], TuckerRanks[i]);
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
    }


    void form_core(CoreValueType_t * d_Y_n, CoreValueType_t * d_U_lrau)
    {
        /* Allocate core tensor if it hasn't been done so yet */
        if (!core_formed)
        {
            CUDA_CHECK(cudaMalloc(&d_core, sizeof(CoreValueType_t) * Rn));
        }

        /* d_Y_n contains X_n (U_N x ... U_1), so we only need to compute U_n^T d_Y_n
         * but since it's stored in row major order, we actually need to compute 
         * d_Y_n^T U_n = G_n^T
         * This means the core tensor will be stored in 'n-slice' major order */
        linalg::gemm(d_Y_n, d_U_lrau, d_core, InputModes[Order - 1], TuckerRanks[Order - 1], Rn / TuckerRanks[Order - 1], true, false);
        core_formed = true;
    }


    double core_norm()
    {

        if (!core_formed)
        {
            std::cerr<<"Tried to compute ||G|| without first calling form_core()"<<std::endl;
            std::abort();
        }

        DEBUG_PRINT("Rn: %lu", Rn);

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


template <typename SparseTensor_t, typename CoreTensor_t, typename TuckerShape>
TuckerTensor<SparseTensor_t, CoreTensor_t, TuckerShape> mixed_sparse_hooi(SparseTensor_t& X, const char * init, const size_t maxiters, const char * factors_dir = nullptr)
{

    //TODO: These needs to be renamed
    using ValueType_t = SparseTensor_t::ValueType_t;
    using IndexType_t = SparseTensor_t::IndexType_t;
    using Index_t = SparseTensor_t::Index_t;
    using InputShape = SparseTensor_t::ShapeType_t;
    using TTMc_t = SparseTensor_t::ValueTypeLow_t;
    using Lra_t = CoreTensor_t;

    static constexpr uint32_t N = SparseTensor_t::Order;
    static constexpr Index_t TuckerRanks = TuckerShape::dims;
    static constexpr Index_t InputModes = SparseTensor_t::Modes;

    TuckerTensor<SparseTensor_t, CoreTensor_t, TuckerShape> X_tucker(init, factors_dir);
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
    static constexpr IndexType_t largest_rank = *(std::max_element(TuckerRanks.begin(), TuckerRanks.end()));
    static constexpr IndexType_t Rn = std::reduce(TuckerRanks.begin(), TuckerRanks.end(), 1, std::multiplies<IndexType_t>{});

    DeviceWorkspace<Lra_t> spttmc_out_ws(largest_mode * Rn); //TODO: This is an overallocation, it should be largest_most * largest product of n-1 tucker modes
    Lra_t * d_Y_n = spttmc_out_ws.d_data;

    DeviceWorkspace<Lra_t> factor_lrau_ws(largest_mode * largest_rank); 
    Lra_t * d_U_lrau = factor_lrau_ws.d_data;

    /* Symbolic TTMc -- record indices of all nonzeros that contribute to each row of the TTMc outputs 
     * Each entry of this array is a device pointer
     */
    utils::print_separator("Beginning Symbolic");
    SymbolicTTMC symbolic_ttmc(X);

#if DEBUG >= 2
    symbolic_ttmc.dump(globals::logfile);
    for (int i=0; i<N; i++)
    {
        utils::write_d_arr(globals::logfile, d_U_list[i], InputModes[i] * TuckerRanks[i], "Factor matrix");
    }
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
            kernels::spttmc<TTMc_t, Lra_t, IndexType_t, Index_t, N, InputShape, TuckerShape, Is>
                           (d_X_vals, d_X_inds, d_U_list, symbolic_ttmc, d_Y_n, nnz),
#if DEBUG >= 2
            utils::write_d_arr(ofs, d_Y_n, InputModes[Is] * (Rn / TuckerRanks[Is]), "TTMc output"),
#endif
            /* Update U[n] with truncated SVD */
            utils::print_separator("SVD"),
            linalg::llsv_randsvd_cusolver<Lra_t, TTMc_t, IndexType_t, 2, InputModes[Is], (Rn / TuckerRanks[Is]), TuckerRanks[Is], (Is==N)>
                                         (d_Y_n, d_U_list[Is], d_U_lrau)
         ), ...);
        }(std::make_index_sequence<N>{});

        CUDA_CHECK(cudaDeviceSynchronize());

        /* Form core tensor */
        utils::print_separator("Forming core");
        X_tucker.form_core(d_Y_n, d_U_lrau);

#if DEBUG >= 2
        utils::write_d_arr(globals::logfile, X_tucker.d_core, Rn, "Core Tensor");
#endif

        /* Record ||G|| */
        double core_norm = X_tucker.core_norm();
        core_norms.push_back(core_norm);

        std::cout<<"ITERATION "<<iter<<" ||G||: "<<core_norm<<'\n';

        /* Clear spttmc_out_ws */
        spttmc_out_ws.zero();
    }

#if DEBUG >= 2
    ofs.close();
#endif

    return X_tucker;
}

} //mxt 

#endif
