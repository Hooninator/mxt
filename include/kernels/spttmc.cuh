#ifndef SPTTMC_CUH
#define SPTTMC_CUH

#include "common.cuh"
#include "kernel_utils.cuh"

namespace mxt
{

template <typename ValueType, typename IndexType, typename Index, size_t Smem, typename MatNRows, typename MatNCols, size_t TotalCols>
__global__ void spttmc_kernel(ValueType * d_vals, Index * d_inds, ValueType ** d_matrices, ValueType * d_out, const size_t nnz, const int exclude)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = kernel_utils::tid_1d();
    const uint32_t lid = kernel_utils::lid();
    const uint32_t wid = kernel_utils::wid();

    __shared__ ValueType d_Y_row[Smem];



    
}


template <typename ValueType, typename IndexType, typename Index, typename MatNRowsShape, typename MatNColsShape>
void spttmc(ValueType * d_vals, Index * d_inds, ValueType ** d_matrices, ValueType * d_out, const size_t nnz, const int exclude)
{
    /* Bookkeeping */
    static constexpr Index MatNRows = MatNRowsShape::dims;
    static constexpr Index MatNCols = MatNColsShape::dims;
    const IndexType I = (exclude == -1) ? MatNRows[0] : MatNRows[exclude];
    static constexpr IndexType TotalCols = std::reduce(MatNCols.begin(), MatNCols.end(), 0, std::multiplies<IndexType>{});

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
    spttmc_kernel<ValueType, IndexType, Index, Smem, MatNRowsShape, MatNColsShape, TotalCols>
        <<<nblocks, tpb>>>
        (d_vals, d_inds, d_matrices, d_out, nnz, exclude);
    CUDA_CHECK(cudaDeviceSynchronize());
}


} //mxt

#endif
