#ifndef COMMON_CUH
#define COMMON_CUH

#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <tuple>
#include <utility>
#include <type_traits>
#include <functional>
#include <unordered_set>

#include <unistd.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>


#include <omp.h>

#include "Profiler.hpp"
#include "colors.h"

#define DEBUG 1
#define DEBUG_SPTTMC_KERNEL 0


/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *              GLOBAL VARIABLES
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

namespace mxt
{

namespace globals
{

std::ofstream logfile;
cusolverDnHandle_t cusolverdn_handle;
cublasHandle_t cublas_handle;
cusparseHandle_t cusparse_handle;
Profiler * profiler;

} //globals

void mxt_init()
{

#if DEBUG >= 1
    globals::logfile.open("logfile.out");
#endif

    (cusolverDnCreate(&globals::cusolverdn_handle));
    (cublasCreate(&globals::cublas_handle));
    (cusparseCreate(&globals::cusparse_handle));

    globals::profiler = new Profiler();

}


void mxt_finalize()
{

#if DEBUG >= 1
    globals::logfile.close();
#endif

    (cusolverDnDestroy(globals::cusolverdn_handle));
    (cublasDestroy(globals::cublas_handle));
    (cusparseDestroy(globals::cusparse_handle));


    delete globals::profiler;
}

} //mxt

#endif
