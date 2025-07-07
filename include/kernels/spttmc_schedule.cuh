#ifndef SPTTMC_SCHEDULE_CUH
#define SPTTMC_SCHEDULE_CUH

#include <bit>

#include "common.cuh"
#include "kernel_utils.cuh"

#include "Shape.cuh"

namespace mxt
{


template <size_t N, int Mode, typename InputModes>
struct SpTTMCSchedule
{
    static constexpr uint32_t NThreads = (N - 1) % 2 == 0 ? 1024 : 512;
    static constexpr uint32_t BlockStride= 2 << (((32 - std::countl_zero(NThreads)) / (N - 1)) - 1) ;
    static constexpr uint32_t NBlocks = InputModes::dims[Mode];
};



} //mxt

#endif
