#ifndef SPARSE_TUCKER_CUH
#define SPARSE_TUCKER_CUH

#include "common.cuh"
#include "SparseTensor.cuh"
#include "utils.cuh"


template <typename SparseTensor_t>
struct TuckerTensor
{
    using ValueType_t = SparseTensor_t::ValueType;
    using IndexType_t = SparseTensor_t::IndexType;

    SparseTensor_t core;

    std::vector<ValueType_t *> factors;
};


template <typename SparseTensor_t, typename ttmc_u, typename lra_u>
TuckerTensor<SparseTensor_t> mixed_sparse_hooi(SparseTensor_t& X, const size_t maxiters)
{
    const uint32_t N = SparseTensor_t::Order;
    
}

#endif
