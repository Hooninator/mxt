#ifndef SPTTMC_SCHEDULE_CUH
#define SPTTMC_SCHEDULE_CUH

#include "common.cuh"
#include "kernel_utils.cuh"

namespace mxt
{

template <typename S>
struct SpTTMCSchedule
{

    std::pair<uint32_t, uint32_t> grid_config()
    {
        return static_cast<*S>(this)->grid_config_impl();
    }

};



} //mxt
