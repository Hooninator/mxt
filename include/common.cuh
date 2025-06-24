#ifndef COMMON_CUH
#define COMMON_CUH

#include <vector>
#include <map>
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

#include <unistd.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>

#define DEBUG 2
//#define DEBUG_SPTTMC_KERNEL 1


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

} //globals

void mxt_init()
{

#if DEBUG >= 2
    globals::logfile.open("logfile.out");
#endif

    (cusolverDnCreate(&globals::cusolverdn_handle));
    (cublasCreate(&globals::cublas_handle));
    (cusparseCreate(&globals::cusparse_handle));
}


void mxt_finalize()
{

#if DEBUG >= 2
    globals::logfile.close();
#endif

    (cusolverDnDestroy(globals::cusolverdn_handle));
    (cublasDestroy(globals::cublas_handle));
    (cusparseDestroy(globals::cusparse_handle));
}

} //mxt

#endif
