#ifndef RAND_MATRIX_CUH
#define RAND_MATRIX_CUH

#include <random>

#include "common.cuh"
#include "utils.cuh"

namespace mxt
{

namespace rand
{

std::random_device rd;
std::mt19937 gen(rd());

template <typename ValueType, typename IndexType>
void randn_buffer(ValueType ** d_data, IndexType n)
{

    std::normal_distribution distr{0.0, 1.0};

    std::vector<ValueType> h_data(n);
    std::generate(h_data.begin(), h_data.end(), [&](){return distr(gen);});

    CUDA_CHECK(cudaMalloc(d_data, sizeof(ValueType) * n));

    utils::h2d_cpy(*d_data, h_data.data(), h_data.size());
}


template <typename ValueType>
void randn_buffer_inplace(ValueType * d_data, const size_t n)
{

    std::normal_distribution distr{0.0, 1.0};

    std::vector<ValueType> h_data(n);
    std::generate(h_data.begin(), h_data.end(), [&](){return distr(gen);});

    utils::h2d_cpy(d_data, h_data.data(), h_data.size());
}



} //rand
} //mxt




#endif
