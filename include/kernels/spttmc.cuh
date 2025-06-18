#ifndef SPTTMC_CUH
#define SPTTMC_CUH

#include "common.cuh"
#include "kernel_utils.cuh"
#include "../SymbolicTTMC.cuh"

namespace mxt
{

template <typename ValueType, typename IndexType, size_t Smem, uint32_t Order, typename MatNRows, typename MatNCols, size_t TotalCols>
__device__ void scaled_block_kron_prod(ValueType val, IndexType * r_inds, ValueType ** d_matrices, ValueType * s_d_out, const int mode)
{

}


template <typename ValueType, typename IndexType, typename Index, size_t Smem, uint32_t Order, typename MatNRows, typename MatNCols, size_t TotalCols>
__global__ void spttmc_kernel(ValueType * d_vals, Index * d_inds, ValueType ** d_matrices, size_t * d_Y_n_inds, size_t * d_Y_n_offsets, size_t * d_Y_mode_offsets, ValueType * d_out, const size_t nnz, const int mode)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = kernel_utils::tid_1d();
    const uint32_t lid = kernel_utils::lid();
    const uint32_t wid = kernel_utils::wid();

    __shared__ ValueType s_d_Y_row[Smem];

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
        scaled_block_kron_prod<ValueType, IndexType, Smem, Order, MatNRows, MatNCols, TotalCols>(val, r_inds, d_matrices, s_d_Y_row, mode);
        __syncthreads();
    }

    /* Write the result to global memory */
    for (size_t i=tid; i < TotalCols; i += blockDim.x)
    {
        if (i < TotalCols)
            d_out[i] = s_d_Y_row[i];
    }

    
}


template <typename ValueType, typename IndexType, typename Index, uint32_t Order, typename MatNRowsShape, typename MatNColsShape>
void spttmc(ValueType * d_vals, Index * d_inds, ValueType ** d_matrices, SymbolicTTMC& symb, ValueType * d_out, const size_t nnz, const int mode)
{
    /* Bookkeeping */
    static constexpr Index MatNRows = MatNRowsShape::dims;
    static constexpr Index MatNCols = MatNColsShape::dims;
    const IndexType I = (mode == -1) ? MatNRows[0] : MatNRows[mode];
    static constexpr IndexType TotalCols = std::reduce(MatNCols.begin(), MatNCols.end(), 1, std::multiplies<IndexType>{});

    /* Grid configuration */
    //TODO: Replace this with the scheduler thing
    const uint32_t nblocks = static_cast<uint32_t>(I);
    static constexpr uint32_t tpb = 1024;

    /* Shared memory */
    //TODO: Make this portable between different kinds of GPUs
    static constexpr size_t MaxSmem = 164 * 1024;

    static_assert( TotalCols * sizeof(ValueType) <= MaxSmem );

    // This should be enough for two blocks per SM, which will maximize occupancy
    static constexpr size_t Smem = MaxSmem / 2;

    /* Call the kernel */
    spttmc_kernel<ValueType, IndexType, Index, Smem, Order, MatNRowsShape, MatNColsShape, TotalCols>
        <<<nblocks, tpb>>>
        (d_vals, d_inds, d_matrices, symb.d_Y_n_inds.d_data, symb.d_Y_n_offsets.d_data, symb.d_Y_mode_offsets.d_data, d_out, nnz, mode);
    CUDA_CHECK(cudaDeviceSynchronize());
}


} //mxt

#endif
