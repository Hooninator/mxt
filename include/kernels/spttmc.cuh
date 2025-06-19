#ifndef SPTTMC_CUH
#define SPTTMC_CUH

#include "common.cuh"
#include "kernel_utils.cuh"
#include "../SymbolicTTMC.cuh"

namespace mxt
{

template <typename ValueType, typename IndexType, auto R0, auto R1, auto R2, auto R3>
__device__ void krpod_o5(ValueType val, ValueType * s_d_matrix_rows, ValueType * s_d_out)
{
    const uint32_t tid = kernel_utils::tid_1d();
    for (IndexType r0 = 0; r0 < R0; r0++)
    {
        ValueType mat0val = s_d_matrix_rows[r0];
        for (IndexType r1 = 0; r1 < R1; r1++)
        {
            ValueType mat1val = s_d_matrix_rows[R0 + r1];
            for (IndexType r2 = 0; r2 < R2; r2++)
            {
                ValueType mat2val = s_d_matrix_rows[R0 + R1 + r2];
                for (IndexType r3 = 0 + tid; r3 < R3; r += blockDim.x)
                {
                    ValueType mat3val = s_d_matrix_rows[R0 + R1 + R2 + r3];
                    s_d_out[r3 + r2 * R3 + r1 * R2 * R1 + r0 * R1 * R2 * R3 ] += val * mat0val * mat1val * mat2val * mat3val;
                }
            }
        }
    }

}


template <typename ValueType, typename IndexType, auto R0, auto R1, auto R2>
__device__ void krpod_o4(ValueType val, ValueType * s_d_matrix_rows, ValueType * s_d_out)
{
    const uint32_t tid = kernel_utils::tid_1d();
    for (IndexType r0 = 0; r0 < R0; r0++)
    {
        ValueType mat0val = s_d_matrix_rows[r0];
        for (IndexType r1 = 0; r1 < R1; r1++)
        {
            ValueType mat1val = s_d_matrix_rows[R0 + r1];
            for (IndexType r2 = 0 + tid; r2 < R2; r2 += blockDim.x)
            {
                ValueType mat2val = s_d_matrix_rows[R0 + R1 + r2];
                s_d_out[r2 + r1 * R2 + r0 * R1 * R2 ] += val * mat0val * mat1val * mat2val;
            }
        }
    }

}


template <typename ValueType, typename IndexType, auto R0, auto R1>
__device__ void krpod_o3(ValueType val, ValueType * s_d_matrix_rows, ValueType * s_d_out)
{
    const uint32_t tid = kernel_utils::tid_1d();
    for (IndexType r0 = 0; r0 < R0; r0++)
    {
        ValueType mat0val = s_d_matrix_rows[r0];
        for (IndexType r1 = 0 + tid; r1 < R1; r1 += blockDim.x)
        {
            ValueType mat1val = s_d_matrix_rows[R0 + r1];
            s_d_out[r1 + r0 * R1] += val * mat0val * mat1val;
        }
    }

}
    

template <typename ValueType, typename IndexType, uint32_t Order, typename MatNRows, typename MatNCols, typename MatColOffsets, size_t ColProduct, size_t RowSum>
__device__ void scaled_block_kprod(ValueType val, IndexType * r_inds, ValueType ** d_matrices, ValueType * s_d_out, ValueType * s_d_matrix_rows, const int mode)
{

    static_assert(Order <= 5 && Order >= 3);

    /* First, load the specified rows into shared memory */
    const uint8_t nwarps = blockDim.x / warpSize;
    const uint8_t wid = kernel_utils::wid();
    const uint8_t lid = kernel_utils::lid();
    if (wid < Order && wid != mode)
    {
        for (IndexType j = 0 + lid; j < MatNCols::dims[wid]; j++)
        {
            //NOTE: This will have to be set very carefully
            s_d_matrix_rows[MatColOffsets::dims[wid] + j] = d_matrices[wid][r_inds[wid] * MatNCols::dims[wid] + j];
        }
    }

    __syncthreads();

    /* Now, perform the kronecker product */
    //TODO: Template this on the mode somehow
    if constexpr (Order == 3)
        krpod_o3<ValueType, IndexType, MatNCols>(val, s_d_matrix_rows, s_d_out, mode);
    else if constexpr (Order == 4)
        krpod_o4<ValueType, IndexType, MatNCols>(val, s_d_matrix_rows, s_d_out, mode);
    else if constexpr (Order == 5)
        krpod_o5<ValueType, IndexType, MatNCols>(val, s_d_matrix_rows, s_d_out, mode);



}


//TODO: Replace parameters with struct
//TODO: Make the mode a NTTP and add switch statement to the wrapper function
template <typename ValueType, typename IndexType, typename Index, 
          size_t Smem, uint32_t Order, typename MatNRows, typename MatNCols, 
          size_t ColProduct, size_t RowSum>
__global__ void spttmc_kernel_v1(ValueType * d_vals, Index * d_inds, ValueType ** d_matrices, size_t * d_Y_n_inds, size_t * d_Y_n_offsets, size_t * d_Y_mode_offsets, ValueType * d_out, const size_t nnz, const int mode)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = kernel_utils::tid_1d();
    const uint32_t lid = kernel_utils::lid();
    const uint32_t wid = kernel_utils::wid();

    __shared__ ValueType smem[SmemOut];

    ValueType * s_d_Y_row = smem;
    ValueType * s_d_matrix_rows = smem + ColProduct;

    for (size_t i=tid; i < Smem; i += blockDim.x)
    {
        if (i < Smem)
            s_d_Y_row[i] = ValueType(0);
    }

    __syncthreads();

    size_t start_index = d_Y_n_offsets[ d_Y_mode_offsets[mode] + bid ];
    size_t end_index = d_Y_n_offsets[ d_Y_mode_offsets[mode] + (bid + 1) ];

    IndexType r_inds[Order];
    ValueType val;

    for (size_t r = start_index + lid; r < end_index; r += 1)
    {
        val = d_vals[r];
        for (uint32_t o = 0; o < Order; o++)
        {
            r_inds[o] = d_inds[r][o];
        }

        /* Compute the scaled kronecker product of Order - 1 rows of the factor matrices */
        scaled_block_kprod<ValueType, IndexType, Smem, Order, MatNRows, MatNCols, ColProduct, RowSum>(val, r_inds, d_matrices, s_d_Y_row, s_d_matrix_rows, mode);
        __syncthreads();
    }

    /* Write the result to global memory */
    for (size_t i=tid; i < ColProduct; i += blockDim.x)
    {
        if (i < ColProduct)
            d_out[i] = s_d_Y_row[i];
    }

    
}


template <typename ValueType, typename IndexType, typename Index, uint32_t Order, typename MatNRowsShape, typename MatNColsShape>
void spttmc(ValueType * d_vals, Index * d_inds, ValueType ** d_matrices, SymbolicTTMC& symb, ValueType * d_out, const size_t nnz, const int mode)
{
    /* Bookkeeping */
    static constexpr Index MatNRows = MatNRowsShape::dims;
    static constexpr Index MatNCols = MatNColsShape::dims;
    static constexpr IndexType ColProduct = std::reduce(MatNCols.begin(), MatNCols.end(), 1, std::multiplies<IndexType>{}) / MatNcols[mode];
    static constexpr IndexType RowSum = std::reduce(MatNRows.begin(), MatNRows.end(), 0);

    const IndexType I = (mode == -1) ? MatNRows[0] : MatNRows[mode];

    /* Grid configuration */
    //TODO: Replace this with the scheduler thing
    const uint32_t nblocks = static_cast<uint32_t>(I);
    static constexpr uint32_t tpb = 1024;

    /* Shared memory */
    //TODO: Make this portable between different kinds of GPUs
    static constexpr size_t MaxSmem = 164 * 1024;

    static_assert( ColProduct * sizeof(ValueType) + RowSum * sizeof(ValueType) <= MaxSmem );

    // This should be enough for two blocks per SM, which will maximize occupancy
    static constexpr size_t Smem = MaxSmem / 2;

    /* Call the kernel */
    spttmc_kernel_v1<ValueType, IndexType, Index, Smem, Order, MatNRowsShape, MatNColsShape, ColProduct, RowSum>
        <<<nblocks, tpb>>>
        (d_vals, d_inds, d_matrices, symb.d_Y_n_inds.d_data, symb.d_Y_n_offsets.d_data, symb.d_Y_mode_offsets.d_data, d_out, nnz, mode);
    CUDA_CHECK(cudaDeviceSynchronize());
}


} //mxt

#endif
