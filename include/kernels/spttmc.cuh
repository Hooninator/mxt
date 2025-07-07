#ifndef SPTTMC_CUH
#define SPTTMC_CUH

#include "common.cuh"
#include "kernel_utils.cuh"
#include "spttmc_schedule.cuh"
#include "../SymbolicTTMC.cuh"
#include "../Shape.cuh"

namespace mxt
{

namespace kernels
{

template <typename ValueTypeIn, typename ValueTypeOut, typename IndexType, auto R0, auto R1, auto R2, auto R3, int BlockStride>
__device__ void scaled_block_krpod_o5(ValueTypeIn val, IndexType * r_inds, ValueTypeIn ** d_matrices, ValueTypeIn * s_d_matrix_rows, ValueTypeOut * s_d_out, IndexType * multidx)
{
    const uint32_t tid = kernel_utils::tid_1d();
    const uint8_t wid = kernel_utils::wid();
    const uint8_t lid = kernel_utils::lid();

    ValueTypeOut valo = kernel_utils::convert<ValueTypeIn, ValueTypeOut>(val);

    if (wid == 0)
    {
        for (IndexType j = 0 + lid; j < R0; j+=warpSize)
            s_d_matrix_rows[j] = d_matrices[0][r_inds[0] * R0 + j];
    }
    else if (wid == 1)
    {
        for (IndexType j = 0 + lid; j < R1; j+=warpSize)
            s_d_matrix_rows[j + R0] = d_matrices[1][r_inds[1] * R1 + j];
    }
    else if (wid == 2)
    {
        for (IndexType j = 0 + lid; j < R2; j+=warpSize)
            s_d_matrix_rows[j + R0 + R1] = d_matrices[2][r_inds[2] * R2 + j];
    }
    else if (wid == 3)
    {
        for (IndexType j = 0 + lid; j < R3; j+=warpSize)
            s_d_matrix_rows[j + R0 + R1 + R2] = d_matrices[3][r_inds[3] * R3 + j];
    }

    __syncthreads();

    for (IndexType r0 = multidx[0]; r0 < R0; r0 += BlockStride)
    {
        ValueTypeIn mat0val = s_d_matrix_rows[r0];
        ValueTypeOut mat0valo = kernel_utils::convert<ValueTypeIn, ValueTypeOut>(mat0val);
        for (IndexType r1 = multidx[1]; r1 < R1; r1 += BlockStride)
        {
            ValueTypeIn mat1val = s_d_matrix_rows[R0 + r1];
            ValueTypeOut mat1valo = kernel_utils::convert<ValueTypeIn, ValueTypeOut>(mat1val);
            for (IndexType r2 = multidx[2]; r2 < R2; r2 += BlockStride)
            {
                ValueTypeIn mat2val = s_d_matrix_rows[R0 + R1 + r2];
                ValueTypeOut mat2valo = kernel_utils::convert<ValueTypeIn, ValueTypeOut>(mat2val);
                for (IndexType r3 = multidx[3]; r3 < R3; r3 += BlockStride)
                {
                    ValueTypeIn mat3val = s_d_matrix_rows[R0 + R1 + R2 + r3];
                    ValueTypeOut mat3valo = kernel_utils::convert<ValueTypeIn, ValueTypeOut>(mat3val);
                    s_d_out[r3 + r2 * R3 + r1 * R2 * R1 + r0 * R1 * R2 * R3 ] += (valo * mat0valo * mat1valo * mat2valo * mat3valo);
                }
            }
        }
    }

}


template <typename ValueTypeIn, typename ValueTypeOut, typename IndexType, auto R0, auto R1, auto R2, int BlockStride>
__device__ void scaled_block_krpod_o4(ValueTypeIn val, IndexType * r_inds, ValueTypeIn ** d_matrices, ValueTypeIn * s_d_matrix_rows, ValueTypeOut * s_d_out, IndexType * multidx)
{
    const uint8_t wid = kernel_utils::wid();
    const uint8_t lid = kernel_utils::lid();

    ValueTypeOut valo = kernel_utils::convert<ValueTypeIn, ValueTypeOut>(val);

    if (wid == 0)
    {
        for (IndexType j = 0 + lid; j < R0; j+=warpSize)
            s_d_matrix_rows[j] = d_matrices[0][r_inds[0] * R0 + j];
    }
    else if (wid == 1)
    {
        for (IndexType j = 0 + lid; j < R1; j+=warpSize)
            s_d_matrix_rows[j + R0] = d_matrices[1][r_inds[1] * R1 + j];
    }
    else if (wid == 2)
    {
        for (IndexType j = 0 + lid; j < R2; j+=warpSize)
            s_d_matrix_rows[j + R0 + R1] = d_matrices[2][r_inds[2] * R2 + j];
    }

    __syncthreads();


    for (IndexType r0 = multidx[0]; r0 < R0; r0 += BlockStride)
    {
        ValueTypeIn mat0val = s_d_matrix_rows[r0];
        ValueTypeOut mat0valo = kernel_utils::convert<ValueTypeIn, ValueTypeOut>(mat0val);
        for (IndexType r1 = multidx[1]; r1 < R1; r1 += BlockStride)
        {
            ValueTypeIn mat1val = s_d_matrix_rows[R0 + r1];
            ValueTypeOut mat1valo = kernel_utils::convert<ValueTypeIn, ValueTypeOut>(mat1val);
            for (IndexType r2 = multidx[2]; r2 < R2; r2 += BlockStride)
            {
                ValueTypeIn mat2val = s_d_matrix_rows[R0 + R1 + r2];
                ValueTypeOut mat2valo = kernel_utils::convert<ValueTypeIn, ValueTypeOut>(mat2val);
                s_d_out[r2 + r1 * R2 + r0 * R1 * R2 ] += valo * mat0valo * mat1valo * mat2valo;
            }
        }
    }

}


template <typename ValueTypeIn, typename ValueTypeOut, typename IndexType, auto R0, auto R1, int BlockStride>
__device__ void scaled_block_krpod_o3(ValueTypeIn val, IndexType * r_inds, ValueTypeIn ** d_matrices, ValueTypeIn * s_d_matrix_rows, ValueTypeOut * s_d_out, IndexType * multidx)
{

    const uint32_t tid = kernel_utils::tid_1d();
    const uint8_t wid = kernel_utils::wid();
    const uint8_t lid = kernel_utils::lid();

    ValueTypeOut valo = kernel_utils::convert<ValueTypeIn, ValueTypeOut>(val);

    if (wid == 0)
    {
        for (IndexType j = 0 + lid; j < R0; j+=warpSize)
            s_d_matrix_rows[j] = d_matrices[0][r_inds[0] * R0 + j];
    }
    else if (wid == 1)
    {
        for (IndexType j = 0 + lid; j < R1; j+=warpSize)
            s_d_matrix_rows[j + R0] = d_matrices[1][r_inds[1] * R1 + j];
    }

    __syncthreads();

    for (IndexType r0 = multidx[0]; r0 < R0; r0 += BlockStride)
    {
        ValueTypeIn mat0val = s_d_matrix_rows[r0];
        ValueTypeOut mat0valo = kernel_utils::convert<ValueTypeIn, ValueTypeOut>(mat0val);
        for (IndexType r1 = multidx[1]; r1 < R1; r1 += BlockStride)
        {
            ValueTypeIn mat1val = s_d_matrix_rows[R0 + r1];
            ValueTypeOut mat1valo = kernel_utils::convert<ValueTypeIn, ValueTypeOut>(mat1val);
            s_d_out[r1 + r0 * R1] += (valo * mat0valo * mat1valo);
        }
    }

}

    

template <typename ValueTypeIn, typename ValueTypeOut, typename IndexType, typename Index, uint32_t Order, typename MatNCols, size_t BlockStride>
__device__ void scaled_block_kprod(ValueTypeIn val, IndexType * r_inds, ValueTypeIn ** d_matrices, ValueTypeOut * s_d_out, ValueTypeIn * s_d_matrix_rows, IndexType * multidx)
{
    static_assert(Order <= 5 && Order >= 3);
    static constexpr auto RArray = MatNCols::dims;

    // TODO: Index sequence?
    if constexpr (Order == 3)
        scaled_block_krpod_o3<ValueTypeIn, ValueTypeOut, IndexType, RArray[0], RArray[1], BlockStride>(val, r_inds, d_matrices, s_d_matrix_rows, s_d_out, multidx);
    else if constexpr (Order == 4)
        scaled_block_krpod_o4<ValueTypeIn, ValueTypeOut, IndexType, RArray[0], RArray[1], RArray[2], BlockStride>(val, r_inds, d_matrices, s_d_matrix_rows, s_d_out, multidx);
    else if constexpr (Order == 5)
        scaled_block_krpod_o5<ValueTypeIn, ValueTypeOut, IndexType, RArray[0], RArray[1], RArray[2], RArray[3], BlockStride>(val, r_inds, d_matrices, s_d_matrix_rows, s_d_out, multidx);

    __syncthreads();

}


//TODO: Replace parameters with struct
template <typename ValueTypeIn, typename ValueTypeOut, typename IndexType, typename Index, uint32_t Order, uint32_t Mode, typename MatNCols, size_t OutputNCols, size_t ActiveColSum, size_t BlockStride>
__global__ void spttmc_kernel_v1(ValueTypeIn * d_vals, Index * d_inds, ValueTypeIn ** d_matrices, size_t * d_Y_n_inds, size_t * d_Y_n_offsets, size_t * d_Y_mode_offsets, ValueTypeOut * d_out, const size_t nnz)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = kernel_utils::tid_1d();
    const uint32_t lid = kernel_utils::lid();
    const uint32_t wid = kernel_utils::wid();

    __shared__ ValueTypeOut s_d_Y_row[OutputNCols];
    __shared__ ValueTypeIn s_d_matrix_rows[ActiveColSum];

    SPTTMC_PRINT_T0("OutputNCols: %lu", OutputNCols);

    for (size_t i=threadIdx.x; i < OutputNCols; i += blockDim.x)
    {
        if (i < OutputNCols)
        {
            s_d_Y_row[i] = ValueTypeIn(0);
        }
    }

    SPTTMC_PRINT_T0("Done with shared memory");
    __syncthreads();

    size_t start_index = d_Y_n_offsets[ d_Y_mode_offsets[Mode] + bid ];
    size_t end_index = d_Y_n_offsets[ d_Y_mode_offsets[Mode] + (bid + 1) ];

    IndexType r_inds[Order-1];
    ValueTypeIn val;
    IndexType idx;


    IndexType multidx[Order - 1];
    kernel_utils::block_multidx<IndexType, BlockStride, Order - 1>(multidx);

    //TODO: It would be better if we could just read in one index and use arithmetic to figure out what the entries of r_inds should be
    //I think we can use r for this purpose
    SPTTMC_PRINT_T0("Beginning kprod, start index %lu, end index %lu", start_index, end_index);
    for (size_t r = start_index; r < end_index; r += 1)
    {
        idx = d_Y_n_inds[r];
        val = d_vals[idx];

        uint32_t o2 = 0;
        for (uint32_t o1 = 0; o1 < Order; o1++)
        {
            if (o1 == Mode)
                continue;
            r_inds[o2++] = d_inds[idx][o1]; //TODO: warp shuffle?
        }

        /* Compute the scaled kronecker product of Order - 1 rows of the factor matrices if exclude, otherwise Order rows */
        scaled_block_kprod<ValueTypeIn, ValueTypeOut, IndexType, Index, Order, MatNCols, BlockStride>
            (val, r_inds, d_matrices, s_d_Y_row, s_d_matrix_rows, multidx);

        __syncthreads();
    }

    __syncthreads();

    /* Write the result to global memory */
    for (size_t i=threadIdx.x; i < OutputNCols; i += blockDim.x)
    {
        if (i < OutputNCols)
        {
            d_out[i + bid * OutputNCols] = s_d_Y_row[i];
        }
    }

    SPTTMC_PRINT_T0("Done with kernel");
    
}


template <typename ValueTypeIn, uint32_t Order, uint32_t Mode>
ValueTypeIn ** prune_matrices(ValueTypeIn ** d_matrices)
{
    ValueTypeIn * h_active_matrices[Order];
    ValueTypeIn ** d_active_matrices;

    CUDA_CHECK(cudaMalloc(&d_active_matrices, sizeof(ValueTypeIn *) * (Order)));

    size_t back = 0;
    for (uint32_t n = 0; n < Order; n++)
    {
        if (n != Mode)
        {
            h_active_matrices[back++] = d_matrices[n];
        }
    }

    CUDA_CHECK(cudaMemcpy(d_active_matrices, h_active_matrices, sizeof(ValueTypeIn *) * Order, cudaMemcpyHostToDevice));

    return d_active_matrices;
}


template <typename ValueTypeIn, typename ValueTypeOut, typename IndexType, typename Index, uint32_t Order, uint32_t Mode, typename MatNRowsShape, typename MatNColsShape>
void spttmc_impl(ValueTypeIn * d_vals, Index * d_inds, ValueTypeIn ** d_matrices, SymbolicTTMC& symb, ValueTypeOut * d_out, const size_t nnz)
{
    /* Remove the specified mode from the arrays of matrix columns, if Exclude==true */
    using ActiveMatNColsShape = AdjustShape<Mode, MatNColsShape>::type;
    using ActiveMatNRowsShape = AdjustShape<Mode, MatNRowsShape>::type;

    static constexpr auto ActiveMatNCols = ActiveMatNColsShape::dims;
    static constexpr auto ActiveMatNRows = ActiveMatNRowsShape::dims;

    static constexpr Index MatNRows = MatNRowsShape::dims;
    static constexpr Index MatNCols = MatNColsShape::dims;

    static constexpr size_t OutputNCols = std::reduce(MatNCols.begin(), MatNCols.end(), 1, std::multiplies<IndexType>{}) / MatNCols[Mode];
    static constexpr size_t ActiveColSum = std::reduce(ActiveMatNCols.begin(), ActiveMatNCols.end(), 0);

    const IndexType OutputNRows = MatNRows[Mode];

    /* Grid configuration */
    using Schedule = SpTTMCSchedule<Order, Mode, MatNRowsShape>; 
    uint32_t nblocks = Schedule::NBlocks;
    uint32_t tpb = Schedule::NThreads;


    /* Remove excluded matrix from the list of matrices passed to the kernel */
    ValueTypeIn ** d_active_matrices = prune_matrices<ValueTypeIn, Order, Mode>(d_matrices);

    DEBUG_PRINT("blocks: %u, threads: %u", nblocks, tpb);

    /* Call the kernel */
    spttmc_kernel_v1<ValueTypeIn, ValueTypeOut, IndexType, Index, Order, Mode, ActiveMatNColsShape, OutputNCols, ActiveColSum, Schedule::BlockStride>
        <<<nblocks, tpb>>>
        (d_vals, d_inds, 
         d_active_matrices, 
         symb.d_Y_n_inds.d_data, 
         symb.d_Y_n_offsets.d_data, 
         symb.d_Y_mode_offsets.d_data, 
         d_out, nnz);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_FREE(d_active_matrices); // Does not free the actual matrices, only the array that holds the pointers to them
}


template <typename ValueTypeIn, typename ValueTypeOut, typename ValueTypeCore, typename IndexType, typename Index, uint32_t Order, typename MatNRowsShape, typename MatNColsShape, uint32_t Mode>
void spttmc(ValueTypeIn * d_vals, Index * d_inds, ValueTypeIn ** d_matrices, SymbolicTTMC& symb, ValueTypeOut * d_out, ValueTypeCore * d_out_core, const size_t nnz)
{

    spttmc_impl<ValueTypeIn, ValueTypeOut, IndexType, Index, Order, Mode, MatNRowsShape, MatNColsShape>
        (d_vals, d_inds, d_matrices, symb, d_out, nnz);

    if constexpr (Mode == Order - 1)
    {
        static constexpr auto sz = std::reduce(MatNColsShape::dims.begin(), MatNColsShape::dims.end(), 1, std::multiplies<IndexType>{});
        utils::d_to_u<ValueTypeOut, ValueTypeCore>(d_out, d_out_core, (sz * MatNRowsShape::dims[Mode]) / MatNColsShape::dims[Mode]);
    }

}

} //kernels
} //mxt

#endif
