#ifndef SPTTMC_CUH
#define SPTTMC_CUH

#include "common.cuh"
#include "kernel_utils.cuh"
#include "../SymbolicTTMC.cuh"
#include "../Shape.cuh"

namespace mxt
{

template <typename ValueTypeIn, typename ValueTypeOut, typename IndexType, auto R0, auto R1, auto R2, auto R3>
__device__ void scaled_block_krpod_o5(ValueTypeIn val, IndexType * r_inds, ValueTypeIn ** d_matrices, ValueTypeIn * s_d_matrix_rows, ValueTypeOut * s_d_out)
{
    const uint32_t tid = kernel_utils::tid_1d();
    const uint8_t wid = kernel_utils::wid();
    const uint8_t lid = kernel_utils::lid();

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

    for (IndexType r0 = 0; r0 < R0; r0++)
    {
        ValueTypeIn mat0val = s_d_matrix_rows[r0];
        for (IndexType r1 = 0; r1 < R1; r1++)
        {
            ValueTypeIn mat1val = s_d_matrix_rows[R0 + r1];
            for (IndexType r2 = 0; r2 < R2; r2++)
            {
                ValueTypeIn mat2val = s_d_matrix_rows[R0 + R1 + r2];
                for (IndexType r3 = 0 + threadIdx.x; r3 < R3; r3 += blockDim.x)
                {
                    ValueTypeIn mat3val = s_d_matrix_rows[R0 + R1 + R2 + r3];
                    s_d_out[r3 + r2 * R3 + r1 * R2 * R1 + r0 * R1 * R2 * R3 ] += kernel_utils::convert<ValueTypeIn, ValueTypeOut>(val * mat0val * mat1val * mat2val * mat3val);
                }
            }
        }
    }

}


template <typename ValueTypeIn, typename ValueTypeOut, typename IndexType, auto R0, auto R1, auto R2>
__device__ void scaled_block_krpod_o4(ValueTypeIn val, IndexType * r_inds, ValueTypeIn ** d_matrices, ValueTypeIn * s_d_matrix_rows, ValueTypeOut * s_d_out)
{
    const uint8_t wid = kernel_utils::wid();
    const uint8_t lid = kernel_utils::lid();

    SPTTMC_PRINT_T0("R0: %lu, R1: %lu, R2: %lu", R0, R1, R2);
    SPTTMC_PRINT_T0("%p", d_matrices[0]);

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

    for (IndexType r0 = 0; r0 < R0; r0++)
    {
        ValueTypeIn mat0val = s_d_matrix_rows[r0];
        for (IndexType r1 = 0; r1 < R1; r1++)
        {
            ValueTypeIn mat1val = s_d_matrix_rows[R0 + r1];
            for (IndexType r2 = 0 + threadIdx.x; r2 < R2; r2 += blockDim.x)
            {
                ValueTypeIn mat2val = s_d_matrix_rows[R0 + R1 + r2];
                s_d_out[r2 + r1 * R2 + r0 * R1 * R2 ] += kernel_utils::convert<ValueTypeIn, ValueTypeOut>(val * mat0val * mat1val * mat2val);
            }
        }
    }

}


template <typename ValueTypeIn, typename ValueTypeOut, typename IndexType, auto R0, auto R1>
__device__ void scaled_block_krpod_o3(ValueTypeIn val, IndexType * r_inds, ValueTypeIn ** d_matrices, ValueTypeIn * s_d_matrix_rows, ValueTypeOut * s_d_out)
{

    const uint32_t tid = kernel_utils::tid_1d();
    const uint8_t wid = kernel_utils::wid();
    const uint8_t lid = kernel_utils::lid();

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

    for (IndexType r0 = 0; r0 < R0; r0++)
    {
        ValueTypeIn mat0val = s_d_matrix_rows[r0];
        for (IndexType r1 = 0 + threadIdx.x; r1 < R1; r1 += blockDim.x)
        {
            ValueTypeIn mat1val = s_d_matrix_rows[R0 + r1];
            s_d_out[r1 + r0 * R1] += kernel_utils::convert<ValueTypeIn, ValueTypeOut>(val * mat0val * mat1val);
        }
    }

}

    

template <typename ValueTypeIn, typename ValueTypeOut, typename IndexType, typename Index, uint32_t Order, typename MatNCols>
__device__ void scaled_block_kprod(ValueTypeIn val, IndexType * r_inds, ValueTypeIn ** d_matrices, ValueTypeOut * s_d_out, ValueTypeIn * s_d_matrix_rows)
{
    static_assert(Order <= 5 && Order >= 3);
    static constexpr auto RArray = MatNCols::dims;

    // TODO: Index sequence?
    if constexpr (Order == 3)
        scaled_block_krpod_o3<ValueTypeIn, ValueTypeOut, IndexType, RArray[0], RArray[1]>(val, r_inds, d_matrices, s_d_matrix_rows, s_d_out);
    else if constexpr (Order == 4)
        scaled_block_krpod_o4<ValueTypeIn, ValueTypeOut, IndexType, RArray[0], RArray[1], RArray[2]>(val, r_inds, d_matrices, s_d_matrix_rows, s_d_out);
    else if constexpr (Order == 5)
        scaled_block_krpod_o5<ValueTypeIn, ValueTypeOut, IndexType, RArray[0], RArray[1], RArray[2], RArray[3]>(val, r_inds, d_matrices, s_d_matrix_rows, s_d_out);

    __syncthreads();

}


//TODO: Replace parameters with struct
template <typename ValueTypeIn, typename ValueTypeOut, typename IndexType, typename Index, size_t Smem, uint32_t Order, uint32_t Mode, typename MatNCols, size_t ActiveColProduct, size_t ActiveColSum>
__global__ void spttmc_kernel_v1(ValueTypeIn * d_vals, Index * d_inds, ValueTypeIn ** d_matrices, size_t * d_Y_n_inds, size_t * d_Y_n_offsets, size_t * d_Y_mode_offsets, ValueTypeOut * d_out, const size_t nnz)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = kernel_utils::tid_1d();
    const uint32_t lid = kernel_utils::lid();
    const uint32_t wid = kernel_utils::wid();

    //__shared__ char smem[Smem];

    __shared__ ValueTypeOut s_d_Y_row[ActiveColProduct];
    SPTTMC_PRINT_T0("ActiveColProduct: %lu", ActiveColProduct);
    __shared__ ValueTypeIn s_d_matrix_rows[ActiveColSum];

    for (size_t i=threadIdx.x; i < ActiveColProduct; i += blockDim.x)
    {
        if (i < ActiveColProduct)
            s_d_Y_row[i] = ValueTypeIn(0);
    }

    SPTTMC_PRINT_T0("Done with shared memory");
    __syncthreads();

    size_t start_index = d_Y_n_offsets[ d_Y_mode_offsets[Mode] + bid ];
    size_t end_index = d_Y_n_offsets[ d_Y_mode_offsets[Mode] + (bid + 1) ];

    IndexType r_inds[Order - 1];
    ValueTypeIn val;
    IndexType idx;

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


        /* Compute the scaled kronecker product of Order - 1 rows of the factor matrices */
        scaled_block_kprod<ValueTypeIn, ValueTypeOut, IndexType, Index, Order, MatNCols>
            (val, r_inds, d_matrices, s_d_Y_row, s_d_matrix_rows);

        __syncthreads();
    }

    __syncthreads();

    /* Write the result to global memory */
    for (size_t i=threadIdx.x; i < ActiveColProduct; i += blockDim.x)
    {
        if (i < ActiveColProduct)
        {
            d_out[i + bid * ActiveColProduct] = s_d_Y_row[i];
        }
    }

    SPTTMC_PRINT_T0("Done with kernel");
    
}


template <typename ValueTypeIn, typename ValueTypeOut, typename IndexType, typename Index, uint32_t Order, uint32_t Mode, typename MatNRowsShape, typename MatNColsShape>
void spttmc_impl(ValueTypeIn * d_vals, Index * d_inds, ValueTypeIn ** d_matrices, SymbolicTTMC& symb, ValueTypeOut * d_out, const size_t nnz)
{
    /* Remove the specified mode from the arrays of matrix columns */
    using ActiveMatNColsShape = RemoveOneToShape<Mode, MatNColsShape>::type;

    static constexpr auto ActiveMatNCols = ActiveMatNColsShape::dims;

    static constexpr Index MatNRows = MatNRowsShape::dims;
    static constexpr Index MatNCols = MatNColsShape::dims;

    utils::print_h_arr(MatNRows.data(), Order, "MatNRows");
    utils::print_h_arr(MatNCols.data(), Order, "MatNCols");

    static constexpr IndexType ActiveColProduct = std::reduce(ActiveMatNCols.begin(), ActiveMatNCols.end(), 1, std::multiplies<IndexType>{});
    static constexpr IndexType RowSum = std::reduce(MatNRows.begin(), MatNRows.end(), 0);
    static constexpr IndexType ActiveColSum = std::reduce(ActiveMatNCols.begin(), ActiveMatNCols.end(), 0);

    DEBUG_PRINT("ActiveColProduct: %lu", ActiveColProduct);

    const IndexType I = MatNRows[Mode];

    /* Grid configuration */
    //TODO: Replace this with the scheduler thing
    const uint32_t nblocks = static_cast<uint32_t>(I);
    static constexpr uint32_t tpb = 1024;

    /* Shared memory */
    //TODO: Make this portable between different kinds of GPUs
    static constexpr size_t MaxSmem = 164 * 1024;

    static_assert( ActiveColProduct * sizeof(ValueTypeIn) + RowSum * sizeof(ValueTypeIn) <= MaxSmem );

    // This should be enough for two blocks per SM, which will maximize occupancy
    static constexpr size_t Smem = MaxSmem / 2;

    /* Remove excluded matrix from the list of matrices passed to the kernel */
    ValueTypeIn * h_active_matrices[Order - 1];
    ValueTypeIn ** d_active_matrices;
    CUDA_CHECK(cudaMalloc(&d_active_matrices, sizeof(ValueTypeIn *) * (Order - 1)));
    size_t back = 0;
    for (uint32_t n = 0; n < Order; n++)
    {
        if (n != Mode)
        {
            h_active_matrices[back++] = d_matrices[n];
        }
    }
    CUDA_CHECK(cudaMemcpy(d_active_matrices, h_active_matrices, sizeof(ValueTypeIn *) * (Order - 1), cudaMemcpyHostToDevice));

    /* Call the kernel */
    spttmc_kernel_v1<ValueTypeIn, ValueTypeOut, IndexType, Index, Smem, Order, Mode, ActiveMatNColsShape, ActiveColProduct, ActiveColSum>
        <<<nblocks, tpb>>>
        (d_vals, d_inds, d_active_matrices, symb.d_Y_n_inds.d_data, symb.d_Y_n_offsets.d_data, symb.d_Y_mode_offsets.d_data, d_out, nnz);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_FREE(d_active_matrices); // Does not free the actual matrices, only the array that holds the pointers to them
}


template <typename ValueTypeIn, typename ValueTypeOut, typename IndexType, typename Index, uint32_t Order, typename MatNRowsShape, typename MatNColsShape>
void spttmc(ValueTypeIn * d_vals, Index * d_inds, ValueTypeIn ** d_matrices, SymbolicTTMC& symb, ValueTypeOut * d_out, const size_t nnz, const uint32_t mode)
{
    switch(mode)
    {
        case 0:
            spttmc_impl<ValueTypeIn, ValueTypeOut, IndexType, Index, Order, 0, MatNRowsShape, MatNColsShape>
                (d_vals, d_inds, d_matrices, symb, d_out, nnz);
            break;
        case 1:
            spttmc_impl<ValueTypeIn, ValueTypeOut, IndexType, Index, Order, 1, MatNRowsShape, MatNColsShape>
                (d_vals, d_inds, d_matrices, symb, d_out, nnz);
            break;
        case 2:
            spttmc_impl<ValueTypeIn, ValueTypeOut, IndexType, Index, Order, 2, MatNRowsShape, MatNColsShape>
                (d_vals, d_inds, d_matrices, symb, d_out, nnz);
            break;
        case 3:
            spttmc_impl<ValueTypeIn, ValueTypeOut, IndexType, Index, Order, 3, MatNRowsShape, MatNColsShape>
                (d_vals, d_inds, d_matrices, symb, d_out, nnz);
            break;
        case 4:
            spttmc_impl<ValueTypeIn, ValueTypeOut, IndexType, Index, Order, 4, MatNRowsShape, MatNColsShape>
                (d_vals, d_inds, d_matrices, symb, d_out, nnz);
            break;
        default:
            NOT_REACHABLE();
    }
}

} //mxt

#endif
