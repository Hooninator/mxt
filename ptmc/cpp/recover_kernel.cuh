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
template <> __device__ inline double to_f64<half>(half x) { return __half2float(x); }

template <typename T> __device__ inline T from_f32(float x);
template <> __device__ inline float from_f32<float>(float x){ return x; }
template <> __device__ inline half  from_f32<half >(float x){ return __float2half_rn(x); }

template <typename T> __device__ inline T from_f64(double x);
template <> __device__ inline double from_f64<double>(double x){ return x; }
template <> __device__ inline float  from_f64<float >(double x){ return static_cast<float>(x); }
template <> __device__ inline half   from_f64<half  >(double x){ return __float2half_rn((float)x); }

// Kernel config
#ifndef TILE_COLS
#define TILE_COLS 128
#endif
#ifndef TILE_ROWS
#define TILE_ROWS  8
#endif

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
    __shared__ AccT s_row[tile_rows];
    __shared__ AccT s_col[TILE_COLS];

    // Load row factors
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int rr = 0; rr < tile_rows; ++rr) {
            int r = row0 + rr;
            s_row[rr] = (r < rows) ? r_inv[N-2][r] : AccT(1);
        }
    }
    // Load col factors (vectorized across x)
    for (int cc = threadIdx.x; cc < tile_cols; cc += blockDim.x) {
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

    // Thread map: (ty, tx) -> (row, col) within the tile
    int ty = threadIdx.y;
    for (int rr = ty; rr < tile_rows; rr += blockDim.y) {
        int r = row0 + rr;
        if (r >= rows) continue;

        // Precompute row-product for this row: s_block * s_row[r]
        AccT s_row_total = s_block * s_row[rr];

        // Vectorize along columns if possible
        // Try float4 if OutT==float and alignment permits; otherwise scalar.
        // Here we stick to scalar for generality; you can specialize for alignment >= 16 bytes.
        int tx = threadIdx.x;
        for (int cc = tx; cc < tile_cols; cc += blockDim.x) {
            int c = col0 + cc;
            if (c >= cols) continue;

            int64_t elem = base_elem + int64_t(rr)*stride_row + int64_t(cc)*stride_col;

            // Load
            InT v_in = __ldg(tY + elem);

            // Accumulate scale in desired precision
            if constexpr (UseF64) {
                double scale = (double)s_row_total * (double)s_col[cc];
                double tmp = to_f64<InT>(v_in) * scale;
                double result = tmp * (double)theta;
                Y[elem] = from_f64<OutT>(result);
            } else {
                float scale = (float)s_row_total * (float)s_col[cc];
                float tmp = to_f32<InT>(v_in) * scale;
                float result = tmp * (float)theta;
                Y[elem] = from_f32<OutT>(result);
            }
        }
    }
}

