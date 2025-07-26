#ifndef TTMC_CUH
#define TTMC_CUH

#include "common.cuh"
#include "utils.cuh"

#include "DenseTensor.cuh"
#include "MatrixCollection.cuh"
#include "Normalizers.cuh"


namespace mxt
{



template <typename InputType1, typename InputType2, typename OutputType>
void ttm_modek(InputType1 * d_X, InputType2 * d_U, OutputType * d_Y, 
                const size_t m, const size_t n, const size_t k,
                const size_t p, const size_t i,
                cublasComputeType_t compute_type)
{
    cudaDataType IT1 = utils::to_cuda_dtype<InputType1>();
    cudaDataType IT2 = utils::to_cuda_dtype<InputType2>();
    cudaDataType OT = utils::to_cuda_dtype<OutputType>();


    InputType1 alpha = 1.0;
    OutputType beta = 0.0;


    /* Setup arrays of matrices */
    void ** h_X_arr = new void * [p];
    void ** h_U_arr = new void * [p];
    void ** h_Y_arr = new void * [p];


    for (size_t i=0; i<p; i++)
    {
        h_X_arr[i] = (void *) (d_X + (i * m * n));
        h_U_arr[i] = (void *) (d_U);
        h_Y_arr[i] = (void *) (d_Y + (i * m * k));
    }

    void ** d_X_arr = utils::h2d_cpy(h_X_arr, p);
    void ** d_U_arr = utils::h2d_cpy(h_U_arr, p);
    void ** d_Y_arr = utils::h2d_cpy(h_Y_arr, p);

    CUBLAS_CHECK(cublasGemmBatchedEx(globals::cublas_handle,
                                     CUBLAS_OP_N, CUBLAS_OP_T,
                                     m, k, n,
                                     &alpha,
                                     d_X_arr, IT1, m,
                                     d_U_arr, IT2, k,
                                     &beta, d_Y_arr, OT,
                                     m, p, compute_type,
                                     CUBLAS_GEMM_DEFAULT));


    CUDA_FREE(d_X_arr);
    CUDA_FREE(d_U_arr);
    CUDA_FREE(d_Y_arr);

    delete[] h_X_arr;
    delete[] h_U_arr;
    delete[] h_Y_arr;
}


template <typename InputType1, typename InputType2, typename OutputType>
void ttm_mode1(InputType1 * d_X, InputType2 * d_U, OutputType * d_Y, 
                const size_t m, const size_t n, const size_t k,
                cublasComputeType_t compute_type)
{

    cudaDataType IT1 = utils::to_cuda_dtype<InputType1>();
    cudaDataType IT2 = utils::to_cuda_dtype<InputType2>();
    cudaDataType OT = utils::to_cuda_dtype<OutputType>();

    InputType1 alpha = 1.0;
    OutputType beta = 0.0;

    CUBLAS_CHECK(cublasGemmEx(globals::cublas_handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              m, n, k,
                              &alpha, d_U, IT2, m,
                              d_X, IT1, k,
                              &beta, d_Y, OT,
                              m, compute_type,
                              CUBLAS_GEMM_DEFAULT));

}


template <typename InputTensor_t, typename MatrixCollection_t, typename OutputTensor_t, 
          typename Normalizer, 
          typename ComputePrecision_t, typename AccumPrecision_t>
OutputTensor_t ttmc_mixed(InputTensor_t& X, MatrixCollection_t& matrices, Normalizer normalizer, cublasComputeType_t compute_type)
{

    using HighPrecision_t = InputTensor_t::ValueType_t;

    static_assert(std::is_same<HighPrecision_t, typename MatrixCollection_t::ValueType_t>::value);
    static_assert(std::is_same<HighPrecision_t, typename OutputTensor_t::ValueType_t>::value);

    static constexpr auto MatrixRows = MatrixCollection_t::Rows; 
    static constexpr auto MatrixCols = MatrixCollection_t::Cols; 
    auto TensorModes = InputTensor_t::Modes; 

    static constexpr size_t N = MatrixCollection_t::N;
    HighPrecision_t theta = 0.1;
    size_t running_size = InputTensor_t::In;

    /* What parts use mixed precision? */
    static constexpr bool compute_mixed = std::is_same<ComputePrecision_t, HighPrecision_t>::value;
    static constexpr bool accum_mixed = std::is_same<AccumPrecision_t, HighPrecision_t>::value;

    /* Normalize and convert tensor to ComputePrecision_t */
    globals::profiler->start_timer("conversion");
    HighPrecision_t * d_X = X.d_data;
    normalizer.normalize_tensor(d_X, theta, TensorModes, InputTensor_t::In, 0);
    DEBUG_PRINT("Done normalizing tensor");

    ComputePrecision_t * d_X_scaled = utils::d_to_u<HighPrecision_t, ComputePrecision_t>(d_X, InputTensor_t::In);
    globals::profiler->stop_timer("conversion");
     
    /* Set up output tensor */
    globals::profiler->start_timer("allocation");
    ComputePrecision_t * d_Y_compute;
    AccumPrecision_t * d_Y_accum;
    HighPrecision_t * d_Y_out;

    size_t d_Y_size = 0;
    size_t d_Y_size_curr = InputTensor_t::In;
    for (int i=0; i<N; i++)
    {
        d_Y_size_curr /= MatrixCols[i];
        d_Y_size_curr *= MatrixRows[i];
        d_Y_size = std::max(d_Y_size, d_Y_size_curr);
    }

    //TODO: These are overallocations
    CUDA_CHECK(cudaMalloc(&d_Y_compute, sizeof(ComputePrecision_t) * d_Y_size)); 
    CUDA_CHECK(cudaMalloc(&d_Y_accum, sizeof(AccumPrecision_t) * d_Y_size));
    CUDA_CHECK(cudaMalloc(&d_Y_out, sizeof(HighPrecision_t) * d_Y_size)); 
    globals::profiler->stop_timer("allocation");

    /* TTM chain */
    for (int i = 0; i < N; i++)
    {
        std::cout<<CYAN<<"  --Iteration "<<i<<RESET<<std::endl;

        /* Normalize tensor if not first iteration */
        if (i > 0)
        {
            DEBUG_PRINT("Normalizing tensor");
            normalizer.normalize_tensor(d_Y_out, theta, TensorModes, running_size, i);

            if (compute_mixed)
            {
                utils::d_to_u<HighPrecision_t, ComputePrecision_t>(d_Y_out, d_Y_compute, running_size);
            }
            else
            {
                std::swap(d_Y_out, d_Y_compute);
            }
        }

        std::string timername("ttmc-mode" + std::to_string(i));
        globals::profiler->start_timer(timername.c_str());

        /* Normalize and convert matrix */
        HighPrecision_t * d_U = matrices.get_matrix(i);
        normalizer.normalize_matrix(d_U, theta, MatrixRows[i], MatrixCols[i], i);
        ComputePrecision_t * d_U_scaled = utils::d_to_u<HighPrecision_t, ComputePrecision_t>(d_U, MatrixRows[i]*MatrixCols[i]);
        DEBUG_PRINT("Normalized matrix");

        /* TTM */
        size_t y_size;
        if (i == 0)
        {
            /* Special case of mode 1 */
            const size_t m = MatrixRows[0];
            const size_t n = X.unfolding_cols(0);
            const size_t k = MatrixCols[0];

            ttm_mode1(d_X_scaled, d_U_scaled, d_Y_accum, 
                      m, n, k, 
                      compute_type);

            TensorModes[0] = MatrixRows[0];
            CUDA_FREE(d_X_scaled);

            y_size = m * n;
        }
        else
        {

            size_t m = 1;
            size_t p = 1;

            for (int j=0; j<i; j++)
            {
                m *= TensorModes[j];
            }

            for (int j=i+1; j<N; j++)
            {
                p *= TensorModes[j]; 
            }

            size_t n = MatrixCols[i];
            size_t k = MatrixRows[i];

            y_size = p * m * k;


            ttm_modek(d_Y_compute, d_U_scaled, d_Y_accum, 
                      m, n, k, 
                      p, i, 
                      compute_type);


            TensorModes[i] = MatrixRows[i];

        }

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_FREE(d_U_scaled);

        /* Recover output and update size */
        if (accum_mixed)
        {
            utils::d_to_u<AccumPrecision_t, HighPrecision_t>(d_Y_accum, d_Y_out, y_size);
        }
        else
        {
            std::swap(d_Y_out, d_Y_accum);
        }


        running_size /= MatrixCols[i];
        running_size *= MatrixRows[i];

        normalizer.recover_tensor(d_Y_out, theta, TensorModes, running_size, i);

        globals::profiler->stop_timer(timername.c_str());

    }

    globals::profiler->start_timer("conversion");
    HighPrecision_t * d_Y_final = utils::d_to_u<AccumPrecision_t, HighPrecision_t>(d_Y_out, OutputTensor_t::In);
    globals::profiler->stop_timer("conversion");

    OutputTensor_t Y(d_Y_final);

    CUDA_FREE(d_Y_compute);
    CUDA_FREE(d_Y_accum);
    CUDA_FREE(d_Y_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    return Y;
}



} //mxt

#endif
