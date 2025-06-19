#ifndef SPTTMC_CUH
#define SPTTMC_CUH

#include "common.cuh"
#include "kernel_utils.cuh"
#include "../SymbolicTTMC.cuh"
#include "../Shape.cuh"

namespace mxt
{

template <typename ValueType, typename IndexType, auto R0, auto R1, auto R2, auto R3>
__device__ void scaled_block_krpod_o5(ValueType val, IndexType * r_inds, ValueType ** d_matrices, ValueType * s_d_matrix_rows, ValueType * s_d_out)
{
    const uint32_t tid = kernel_utils::tid_1d();
    const uint8_t wid = kernel_utils::wid();
    const uint8_t lid = kernel_utils::lid();

    if (wid == 0)
    {
        for (IndexType j = 0 + lid; j < R0; j++)
            s_d_matrix_rows[j] = d_matrices[0][r_inds[0] * R0 + j];
    }
    else if (wid == 1)
    {
        for (IndexType j = 0 + lid; j < R1; j++)
            s_d_matrix_rows[j] = d_matrices[1][r_inds[1] * R1 + j];
    }
    else if (wid == 2)
    {
        for (IndexType j = 0 + lid; j < R2; j++)
            s_d_matrix_rows[j] = d_matrices[2][r_inds[2] * R2 + j];
    }
    else if (wid == 3)
    {
        for (IndexType j = 0 + lid; j < R3; j++)
            s_d_matrix_rows[j] = d_matrices[3][r_inds[3] * R3 + j];
    }

    __syncthreads();

    for (IndexType r0 = 0; r0 < R0; r0++)
    {
        ValueType mat0val = s_d_matrix_rows[r0];
        for (IndexType r1 = 0; r1 < R1; r1++)
        {
            ValueType mat1val = s_d_matrix_rows[R0 + r1];
            for (IndexType r2 = 0; r2 < R2; r2++)
            {
                ValueType mat2val = s_d_matrix_rows[R0 + R1 + r2];
                for (IndexType r3 = 0 + tid; r3 < R3; r3 += blockDim.x)
                {
                    ValueType mat3val = s_d_matrix_rows[R0 + R1 + R2 + r3];
                    s_d_out[r3 + r2 * R3 + r1 * R2 * R1 + r0 * R1 * R2 * R3 ] += val * mat0val * mat1val * mat2val * mat3val;
                }
            }
        }
    }

}


template <typename ValueType, typename IndexType, auto R0, auto R1, auto R2>
__device__ void scaled_block_krpod_o4(ValueType val, IndexType * r_inds, ValueType ** d_matrices, ValueType * s_d_matrix_rows, ValueType * s_d_out)
{
    const uint32_t tid = kernel_utils::tid_1d();
    const uint8_t wid = kernel_utils::wid();
    const uint8_t lid = kernel_utils::lid();

    if (wid == 0)
    {
        for (IndexType j = 0 + lid; j < R0; j++)
            s_d_matrix_rows[j] = d_matrices[0][r_inds[0] * R0 + j];
    }
    else if (wid == 1)
    {
        for (IndexType j = 0 + lid; j < R1; j++)
            s_d_matrix_rows[j] = d_matrices[1][r_inds[1] * R1 + j];
    }
    else if (wid == 2)
    {
        for (IndexType j = 0 + lid; j < R2; j++)
            s_d_matrix_rows[j] = d_matrices[2][r_inds[2] * R2 + j];
    }

    __syncthreads();

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
__device__ void scaled_block_krpod_o3(ValueType val, IndexType * r_inds, ValueType ** d_matrices, ValueType * s_d_matrix_rows, ValueType * s_d_out)
{

    const uint32_t tid = kernel_utils::tid_1d();
    const uint8_t wid = kernel_utils::wid();
    const uint8_t lid = kernel_utils::lid();

    if (wid == 0)
    {
        for (IndexType j = 0 + lid; j < R0; j++)
            s_d_matrix_rows[j] = d_matrices[0][r_inds[0] * R0 + j];
    }
    else if (wid == 1)
    {
        for (IndexType j = 0 + lid; j < R1; j++)
            s_d_matrix_rows[j] = d_matrices[1][r_inds[1] * R1 + j];
    }

    __syncthreads();

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

    

template <typename ValueType, typename IndexType, typename Index, uint32_t Order, typename MatNCols>
__device__ void scaled_block_kprod(ValueType val, IndexType * r_inds, ValueType ** d_matrices, ValueType * s_d_out, ValueType * s_d_matrix_rows)
{
    static_assert(Order <= 5 && Order >= 3);
    static constexpr auto RArray = MatNCols::dims;

    // TODO: Index sequence?
    if constexpr (Order == 3)
        scaled_block_krpod_o3<ValueType, IndexType, RArray[0], RArray[1]>(val, r_inds, d_matrices, s_d_matrix_rows, s_d_out);
    else if constexpr (Order == 4)
        scaled_block_krpod_o4<ValueType, IndexType, RArray[0], RArray[1], RArray[2]>(val, r_inds, d_matrices, s_d_matrix_rows, s_d_out);
    else if constexpr (Order == 5)
        scaled_block_krpod_o5<ValueType, IndexType, RArray[0], RArray[1], RArray[2], RArray[3]>(val, r_inds, d_matrices, s_d_matrix_rows, s_d_out);

    __syncthreads();

}


//TODO: Replace parameters with struct
template <typename ValueType, typename IndexType, typename Index, size_t Smem, uint32_t Order, uint32_t Mode, typename MatNCols, size_t ColProduct, size_t RowSum>
__global__ void spttmc_kernel_v1(ValueType * d_vals, Index * d_inds, ValueType ** d_matrices, size_t * d_Y_n_inds, size_t * d_Y_n_offsets, size_t * d_Y_mode_offsets, ValueType * d_out, const size_t nnz)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = kernel_utils::tid_1d();
    const uint32_t lid = kernel_utils::lid();
    const uint32_t wid = kernel_utils::wid();

    __shared__ ValueType smem[Smem];

    ValueType * s_d_Y_row = smem;
    ValueType * s_d_matrix_rows = smem + ColProduct;

    for (size_t i=tid; i < Smem; i += blockDim.x)
    {
        if (i < Smem)
            s_d_Y_row[i] = ValueType(0);
    }

    __syncthreads();

    size_t start_index = d_Y_n_offsets[ d_Y_mode_offsets[Mode] + bid ];
    size_t end_index = d_Y_n_offsets[ d_Y_mode_offsets[Mode] + (bid + 1) ];

    IndexType r_inds[Order - 1];
    ValueType val;

    for (size_t r = start_index + lid; r < end_index; r += 1)
    {
        val = d_vals[r];
        uint32_t o2 = 0;
        for (uint32_t o1 = 0; o1 < Order; o1++)
        {
            if (o1 == Mode)
                continue;
            r_inds[o2++] = d_inds[r][o1];
        }

        /* Compute the scaled kronecker product of Order - 1 rows of the factor matrices */
        scaled_block_kprod<ValueType, IndexType, Index, Order, MatNCols>
            (val, r_inds, d_matrices, s_d_Y_row, s_d_matrix_rows);
        __syncthreads();
    }

    /* Write the result to global memory */
    for (size_t i=tid; i < ColProduct; i += blockDim.x)
    {
        if (i < ColProduct)
            d_out[i] = s_d_Y_row[i];
    }

    
}


template <typename ValueType, typename IndexType, typename Index, uint32_t Order, uint32_t Mode, typename MatNRowsShape, typename MatNColsShape>
void spttmc_impl(ValueType * d_vals, Index * d_inds, ValueType ** d_matrices, SymbolicTTMC& symb, ValueType * d_out, const size_t nnz)
{
    /* Remove the specified mode from the arrays of matrix columns */
    using ActiveMatNColsShape = RemoveOneToShape<Mode, MatNColsShape>::type;

    static constexpr auto ActiveMatNCols = ActiveMatNColsShape::dims;

    static constexpr Index MatNRows = MatNRowsShape::dims;
    static constexpr Index MatNCols = MatNColsShape::dims;

    static constexpr IndexType ColProduct = std::reduce(ActiveMatNCols.begin(), ActiveMatNCols.end(), 1, std::multiplies<IndexType>{});
    static constexpr IndexType RowSum = std::reduce(MatNRows.begin(), MatNRows.end(), 0);

    const IndexType I = MatNRows[Mode];

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

    /* Remove excluded matrix from the list of matrices passed to the kernel */
    ValueType * h_active_matrices[Order - 1];
    size_t back = 0;
    for (uint32_t n = 0; n < Order; n++)
    {
        if (n != Mode)
        {
            h_active_matrices[back++] = d_matrices[n];
        }
    }
    ValueType ** d_active_matrices = utils::h2d_cpy(h_active_matrices, Order - 1);

    /* Call the kernel */
    spttmc_kernel_v1<ValueType, IndexType, Index, Smem, Order, Mode, ActiveMatNColsShape, ColProduct, RowSum>
        <<<nblocks, tpb>>>
        (d_vals, d_inds, d_active_matrices, symb.d_Y_n_inds.d_data, symb.d_Y_n_offsets.d_data, symb.d_Y_mode_offsets.d_data, d_out, nnz);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_FREE(d_active_matrices); // Does not free the actual matrices, only the array that holds the pointers to them
}


template <typename ValueType, typename IndexType, typename Index, uint32_t Order, typename MatNRowsShape, typename MatNColsShape>
void spttmc(ValueType * d_vals, Index * d_inds, ValueType ** d_matrices, SymbolicTTMC& symb, ValueType * d_out, const size_t nnz, const uint32_t mode)
{
    switch(mode)
    {
        case 0:
            spttmc_impl<ValueType, IndexType, Index, Order, 0, MatNRowsShape, MatNColsShape>
                (d_vals, d_inds, d_matrices, symb, d_out, nnz);
        case 1:
            spttmc_impl<ValueType, IndexType, Index, Order, 1, MatNRowsShape, MatNColsShape>
                (d_vals, d_inds, d_matrices, symb, d_out, nnz);
        case 2:
            spttmc_impl<ValueType, IndexType, Index, Order, 2, MatNRowsShape, MatNColsShape>
                (d_vals, d_inds, d_matrices, symb, d_out, nnz);
        case 3:
            spttmc_impl<ValueType, IndexType, Index, Order, 3, MatNRowsShape, MatNColsShape>
                (d_vals, d_inds, d_matrices, symb, d_out, nnz);
        case 4:
            spttmc_impl<ValueType, IndexType, Index, Order, 4, MatNRowsShape, MatNColsShape>
                (d_vals, d_inds, d_matrices, symb, d_out, nnz);
        default:
            NOT_REACHABLE();
    }
}

} //mxt

#endif
