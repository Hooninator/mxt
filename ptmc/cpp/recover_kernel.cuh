#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(expr) do { \
  cudaError_t _e = (expr); \
  if (_e != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
  } \
} while (0)
#endif

// Device-safe prefetch: no-op in device code, use builtin on host
#ifdef __CUDA_ARCH__
#define PREFETCH(ptr, rw, locality) ((void)0)
#else
#define PREFETCH(ptr, rw, locality) __builtin_prefetch(ptr, rw, locality)
#endif

template <typename InT, typename OutT, typename AccT>
struct VecIO {
  // Scalar fallback
  static __device__ inline void load_vec(const InT* p, InT &v) { v = *p; }
  static __device__ inline void store_vec(OutT* p, const OutT &v) { *p = v; }
};

template <>
struct VecIO<half, half, float> {
  static __device__ inline void load2(const half* p, half2 &v) {
    v = *reinterpret_cast<const half2*>(p);
  }
  static __device__ inline void store2(half* p, const half2 &v) {
    *reinterpret_cast<half2*>(p) = v;
  }
};

// Convert + scale helpers
template <typename T> __device__ inline float to_f32(T x) { return static_cast<float>(x); }
template <> __device__ inline float to_f32<half>(half x) { return __half2float(x); }

template <typename T> __device__ inline double to_f64(T x) { return static_cast<double>(x); }
template <> __device__ inline double to_f64<half>(half x) { return (double)__half2float(x); }

template <typename T> __device__ inline T from_f32(float x);
template <> __device__ inline float from_f32<float>(float x){ return x; }
template <> __device__ inline half  from_f32<half >(float x){ return __float2half_rn(x); }

template <typename T> __device__ inline T from_f64(double x);
template <> __device__ inline double from_f64<double>(double x){ return x; }
template <> __device__ inline float  from_f64<float >(double x){ return static_cast<float>(x); }
template <> __device__ inline half   from_f64<half  >(double x){ return __float2half_rn((float)x); }

// Kernel config - optimized tile sizes for better occupancy
#ifndef TILE_COLS
#define TILE_COLS 32
#endif
#ifndef TILE_ROWS
#define TILE_ROWS  32
#endif

#define SMEM 42 * 1024

__device__ void linear_to_2d(int linear_idx, int d1,
                              int& i0, int& i1) 
{
    i1 = linear_idx % d1;
    i0 = linear_idx / d1;
}

__device__ void linear_to_3d(int linear_idx, int d1, int d2,
                              int& i0, int& i1, int& i2) 
{
    i1 = linear_idx % d1;
    int temp = linear_idx / d1;
    i0 = temp % d2;
    i2 = temp / d2;
}

__device__ void linear_to_4d(int linear_idx, int d1, int d2, int d3,
                              int& i0, int& i1, int& i2, int& i3) 
{
    i2 = linear_idx % d2;
    int temp = linear_idx / d2;
    i1 = temp % d1;
    temp /= d1;
    i0 = temp % d3;
    i3 = temp / d3;
}

template <int N, typename InT, typename OutT, typename AccT, bool UseF64=false>
__global__
void diag_unscale_kernel2
(
    const InT* __restrict__ tY,      
    OutT* __restrict__ Y,            
    OutT theta,
    const int64_t* __restrict__ sizes,
    const int64_t sizes_prod,
    const AccT* __restrict__ r_inv[N]
)
{
    static_assert( (N==2) || (N==3) || (N==4) );

    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    int midx[N];

    static constexpr uint32_t smem_size = (SMEM / N) / sizeof(AccT);

    //__shared__ AccT smem[N][smem_size];
    //for (int i=0; i<N; i++)
    //{
    //    for (int j=threadIdx.x; j<sizes[i]; j += blockDim.x)
    //    {
    //        smem[i][j] = r_inv[i][j];
    //    }
    //}

    //__syncthreads();

    for (uint32_t i = tid; i < sizes_prod; i += gridDim.x * blockDim.x)
    {
        if constexpr(N == 2)
        {
            linear_to_2d(i, sizes[1], midx[0], midx[1]);
            Y[i] = to_f64(tY[i]) * r_inv[0][midx[0]] * r_inv[1][midx[1]];
        }
        else if constexpr(N == 3)
        {
            linear_to_3d(i, sizes[1], sizes[0], midx[0], midx[1], midx[2]);
            Y[i] = to_f64(tY[i]) * r_inv[2][midx[2]] * r_inv[0][midx[0]] * r_inv[1][midx[1]];
            //Y[i] = to_f64(tY[i]) * smem[2][midx[2]] * smem[0][midx[0]] * smem[1][midx[1]];
        }
        else if constexpr(N == 4)
        {
            linear_to_4d(i, sizes[1], sizes[2], sizes[0], midx[0], midx[1], midx[2], midx[3]);
            Y[i] = to_f64(tY[i]) * r_inv[0][midx[0]] * r_inv[1][midx[1]] * r_inv[2][midx[2]] * r_inv[3][midx[3]];
            //Y[i] = to_f64(tY[i]) * smem[0][midx[0]] * smem[1][midx[1]] * smem[2][midx[2]] * smem[3][midx[3]];
        }

        Y[i] *= theta;
    }

}




// Computes Y = tY * prod_n r_n_inv[idx_n], for an N-dim tensor laid out with sizes[] and strides[].
// We tile over the last two modes (N-1 and N) for coalesced access.
// Assumptions: tY and Y share the same layout (contiguous recommended).
template <int N, typename InT, typename OutT, typename AccT, bool UseF64=false>
__global__
void diag_unscale_kernel(
    const InT* __restrict__ tY,      // input \tilde{Y}
    OutT* __restrict__ Y,            // output Y
    double theta,
    const int64_t* __restrict__ sizes,   // length N
    const int64_t* __restrict__ strides, // length N (in elements)
    // per-mode diagonal vectors r_n_inv
    const AccT* __restrict__ r_inv[N],
    // linear block index enumerates all tiles over dims 0..N-3, and row/col tiles over N-2,N-1
    int64_t total_outer_tiles)
{
    // Map blockIdx to (outer_index, tile_row, tile_col)
    // outer index enumerates the product of sizes[0..N-3]
    const int64_t rows = sizes[N-2];
    const int64_t cols = sizes[N-1];
    const int tile_cols = TILE_COLS;
    const int tile_rows = TILE_ROWS;

    // Grid layout:
    // gridDim.x = ceil_div(cols, tile_cols)
    // gridDim.y = ceil_div(rows, tile_rows)
    // gridDim.z = product(sizes[0..N-3])  (or clamped to 2^31-1; handle via loop if needed)
    int64_t outer_linear = blockIdx.z;

    // Recover indices for dims 0..N-3 from outer_linear without div/mod on every thread:
    // Only one thread does it, then broadcast via shared memory.
    __shared__ int32_t idx_outer[N > 2 ? N-2 : 1];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int64_t rem = outer_linear;
        #pragma unroll
        for (int d = 0; d < N-2; ++d) {
            int64_t sd = sizes[d];
            int32_t ix = (int32_t)(rem % sd);
            idx_outer[d] = ix;
            rem /= sd;
        }
    }
    __syncthreads();

    // Compute the base linear offset (in elements) for the tile origin at (outer, row0, col0)
    const int row0 = blockIdx.y * tile_rows;
    const int col0 = blockIdx.x * tile_cols;

    // Precompute s_block = prod_{k=0..N-3} r_k_inv[idx_outer[k]]
    AccT s_block = AccT(1);
    #pragma unroll
    for (int d = 0; d < (N>=3 ? N-2 : 0); ++d) {
        s_block *= r_inv[d][ idx_outer[d] ];
    }

    // Shared-memory stage for the two hot diagonals on this tile
    // Use padding to avoid bank conflicts
    __shared__ AccT s_row[tile_rows + 1];
    __shared__ AccT s_col[TILE_COLS + 1];

    // Load row factors - parallelize across threads for better bandwidth utilization
    for (int rr = threadIdx.x; rr < tile_rows; rr += blockDim.x) {
        int r = row0 + rr;
        s_row[rr] = (r < rows) ? r_inv[N-2][r] : AccT(1);
    }
    // Load col factors - use all threads for better bandwidth
    for (int cc = threadIdx.x + threadIdx.y * blockDim.x; cc < tile_cols; cc += blockDim.x * blockDim.y) {
        int c = col0 + cc;
        s_col[cc] = (c < cols) ? r_inv[N-1][c] : AccT(1);
    }
    __syncthreads();

    // Compute base pointer (element index) for the (outer,row0,col0) tile
    int64_t base_elem = 0;
    #pragma unroll
    for (int d = 0; d < N-2; ++d) {
        base_elem += int64_t(idx_outer[d]) * strides[d];
    }
    base_elem += int64_t(row0) * strides[N-2];
    base_elem += int64_t(col0) * strides[N-1];

    // Strides (in elements)
    const int64_t stride_row = strides[N-2];
    const int64_t stride_col = strides[N-1];

    // Register blocking optimization: process multiple rows per thread
    const int REG_BLOCK_ROWS = 4;
    int ty = threadIdx.y;
    
    for (int rr_base = ty * REG_BLOCK_ROWS; rr_base < tile_rows; rr_base += blockDim.y * REG_BLOCK_ROWS) {
        // Precompute row scale factors for register block
        AccT s_row_reg[REG_BLOCK_ROWS];
        bool valid_row[REG_BLOCK_ROWS];
        #pragma unroll
        for (int rb = 0; rb < REG_BLOCK_ROWS; ++rb) {
            int rr = rr_base + rb;
            int r = row0 + rr;
            valid_row[rb] = (rr < tile_rows && r < rows);
            s_row_reg[rb] = valid_row[rb] ? s_block * s_row[rr] : AccT(0);
        }

        // Column processing with true vectorization for FP16
        int tx = threadIdx.x;
        if constexpr (std::is_same_v<InT, half> && std::is_same_v<OutT, float> && !UseF64) {
            // Optimized path for half->float conversion (common case)
            for (int cc = tx * 2; cc + 1 < tile_cols; cc += blockDim.x * 2) {
                int c0 = col0 + cc;
                int c1 = col0 + cc + 1;
                if (c0 >= cols || c1 >= cols) continue;

                // Process register block
                #pragma unroll
                for (int rb = 0; rb < REG_BLOCK_ROWS; ++rb) {
                    if (!valid_row[rb]) continue;
                    int rr = rr_base + rb;
                    
                    int64_t elem0 = base_elem + int64_t(rr)*stride_row + int64_t(cc)*stride_col;
                    int64_t elem1 = elem0 + stride_col;

                    // Vectorized load using half2 with prefetching
                    if (stride_col == 1) {
                        // Prefetch next cache line
                        if (rb < REG_BLOCK_ROWS - 1 && valid_row[rb + 1]) {
                            int64_t prefetch_elem = base_elem + int64_t(rr_base + rb + 1)*stride_row + int64_t(cc)*stride_col;
                            PREFETCH(tY + prefetch_elem, 0, 3);
                        }
                        
                        half2 v_in_vec = __ldg(reinterpret_cast<const half2*>(tY + elem0));
                        float2 scale_vec = make_float2(
                            (float)s_row_reg[rb] * (float)s_col[cc] * (float)theta,
                            (float)s_row_reg[rb] * (float)s_col[cc + 1] * (float)theta
                        );
                        
                        // Use built-in half2 to float2 conversion for better performance
                        float2 result_vec = __half22float2(v_in_vec);
                        result_vec.x *= scale_vec.x;
                        result_vec.y *= scale_vec.y;
                        
                        Y[elem0] = result_vec.x;
                        Y[elem1] = result_vec.y;
                    } else {
                        // Non-contiguous case with prefetching
                        if (rb < REG_BLOCK_ROWS - 1 && valid_row[rb + 1]) {
                            int64_t prefetch_elem0 = base_elem + int64_t(rr_base + rb + 1)*stride_row + int64_t(cc)*stride_col;
                            PREFETCH(tY + prefetch_elem0, 0, 3);
                        }
                        
                        half v0 = __ldg(tY + elem0);
                        half v1 = __ldg(tY + elem1);
                        
                        float scale0 = (float)s_row_reg[rb] * (float)s_col[cc] * (float)theta;
                        float scale1 = (float)s_row_reg[rb] * (float)s_col[cc + 1] * (float)theta;
                        
                        Y[elem0] = __half2float(v0) * scale0;
                        Y[elem1] = __half2float(v1) * scale1;
                    }
                }
            }
            
            // Handle remainder columns
            for (int cc = tx + ((tile_cols / (blockDim.x * 2)) * blockDim.x * 2); cc < tile_cols; cc += blockDim.x) {
                int c = col0 + cc;
                if (c >= cols) continue;
                
                #pragma unroll
                for (int rb = 0; rb < REG_BLOCK_ROWS; ++rb) {
                    if (!valid_row[rb]) continue;
                    int rr = rr_base + rb;
                    
                    int64_t elem = base_elem + int64_t(rr)*stride_row + int64_t(cc)*stride_col;
                    half v_in = __ldg(tY + elem);
                    float scale = (float)s_row_reg[rb] * (float)s_col[cc] * (float)theta;
                    Y[elem] = __half2float(v_in) * scale;
                }
            }
        } else {
            // General case for other type combinations
            for (int cc = tx; cc < tile_cols; cc += blockDim.x) {
                int c = col0 + cc;
                if (c >= cols) continue;

                #pragma unroll
                for (int rb = 0; rb < REG_BLOCK_ROWS; ++rb) {
                    if (!valid_row[rb]) continue;
                    int rr = rr_base + rb;
                    
                    int64_t elem = base_elem + int64_t(rr)*stride_row + int64_t(cc)*stride_col;
                    InT v_in = __ldg(tY + elem);

                    if constexpr (UseF64) {
                        double scale = (double)s_row_reg[rb] * (double)s_col[cc] * (double)theta;
                        double tmp = to_f64<InT>(v_in) * scale;
                        Y[elem] = from_f64<OutT>(tmp);
                    } else {
                        float scale = (float)s_row_reg[rb] * (float)s_col[cc] * (float)theta;
                        float tmp = to_f32<InT>(v_in) * scale;
                        Y[elem] = from_f32<OutT>(tmp);
                    }
                }
            }
        }
    }
}

