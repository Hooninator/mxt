#include <torch/extension.h>
#include "recover_kernel.cuh"

template <typename InT, typename OutT, typename AccT, bool UseF64=false>
void launch_diag_unscale(
    const at::Tensor& tY, at::Tensor& Y, const double theta,
    const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides_elts,
    const std::vector<at::Tensor>& r_inv_vecs)
{
    constexpr int NMAX = 16;
    const int N = (int)sizes.size();
    TORCH_CHECK(N >= 1 && N <= NMAX);
    TORCH_CHECK(N >= 2, "This fast kernel expects N>=2 (has 2D tiling).");

    // Device pointers
    auto tY_ptr = tY.data_ptr<InT>();
    auto Y_ptr  = Y.data_ptr<OutT>();

    // Pack sizes/strides to device
    at::Tensor sizes_t   = torch::empty({N}, torch::dtype(torch::kInt64).device(tY.device()));
    at::Tensor strides_t = torch::empty({N}, torch::dtype(torch::kInt64).device(tY.device()));
    CUDA_CHECK(cudaMemcpyAsync(sizes_t.data_ptr<int64_t>(), sizes.data(),   sizeof(int64_t)*N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(strides_t.data_ptr<int64_t>(), strides_elts.data(), sizeof(int64_t)*N, cudaMemcpyHostToDevice));

    // Build array of r_inv pointers 
    std::vector<const AccT*> r_inv_host(N, nullptr);
    for (int d = 0; d < N; ++d) {
        r_inv_host[d] = r_inv_vecs[d].data_ptr<AccT>();
    }
    // Copy pointer array to device
    const AccT** d_r_inv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_r_inv, sizeof(AccT*)*N));
    CUDA_CHECK(cudaMemcpyAsync(d_r_inv, r_inv_host.data(), sizeof(AccT*)*N, cudaMemcpyHostToDevice));

    // Grid
    const int64_t rows = sizes[N-2];
    const int64_t cols = sizes[N-1];
    dim3 block(256, 2, 1);  // 256 threads; good occupancy on sm80+
    dim3 grid(
        (cols + TILE_COLS - 1) / TILE_COLS,
        (rows + TILE_ROWS - 1) / TILE_ROWS,
        std::max<int64_t>(1, std::accumulate(sizes.begin(), sizes.end()-2, int64_t(1), std::multiplies<int64_t>()))
    );
    // Clamp grid.z if huge: loop over outer tiles inside the kernel (left as an exercise)
    TORCH_CHECK(grid.z <= 2147483647, "Outer grid too large; implement looping over outer tiles.");

    // Dispatch by N with a switch for full unrolling of small loops
    // Use __half for kernel template parameters, cast pointers appropriately
    using KernelInT = std::conditional_t<std::is_same_v<InT, at::Half>, half, InT>;
    using KernelOutT = std::conditional_t<std::is_same_v<OutT, at::Half>, half, OutT>;
    using KernelAccT = std::conditional_t<std::is_same_v<AccT, at::Half>, half, AccT>;
    
    // Cast pointers directly for kernel calls
    auto cast_input_ptr = [&]() -> const KernelInT* {
        if constexpr (std::is_same_v<InT, at::Half>) {
            return reinterpret_cast<const half*>(tY_ptr);
        } else {
            return tY_ptr;
        }
    };
    
    auto cast_output_ptr = [&]() -> KernelOutT* {
        if constexpr (std::is_same_v<OutT, at::Half>) {
            return reinterpret_cast<half*>(Y_ptr);
        } else {
            return Y_ptr;
        }
    };
    
    auto cast_r_inv_ptr = [&]() -> const KernelAccT** {
        if constexpr (std::is_same_v<AccT, at::Half>) {
            return reinterpret_cast<const half**>(d_r_inv);
        } else {
            return d_r_inv;
        }
    };
    
    switch (N) {
      case 2: diag_unscale_kernel<2,KernelInT,KernelOutT,KernelAccT,UseF64><<<grid,block>>>(
                  cast_input_ptr(), cast_output_ptr(), theta,
                  sizes_t.data_ptr<int64_t>(), strides_t.data_ptr<int64_t>(),
                  cast_r_inv_ptr(), grid.z); break;
      case 3: diag_unscale_kernel<3,KernelInT,KernelOutT,KernelAccT,UseF64><<<grid,block>>>(
                  cast_input_ptr(), cast_output_ptr(), theta,
                  sizes_t.data_ptr<int64_t>(), strides_t.data_ptr<int64_t>(),
                  cast_r_inv_ptr(), grid.z); break;
      case 4: diag_unscale_kernel<4,KernelInT,KernelOutT,KernelAccT,UseF64><<<grid,block>>>(
                  cast_input_ptr(), cast_output_ptr(), theta, 
                  sizes_t.data_ptr<int64_t>(), strides_t.data_ptr<int64_t>(), 
                  cast_r_inv_ptr(), grid.z); break;
      default: diag_unscale_kernel<8,KernelInT,KernelOutT,KernelAccT,UseF64><<<grid,block>>>(
                  cast_input_ptr(), cast_output_ptr(), theta, 
                  sizes_t.data_ptr<int64_t>(), strides_t.data_ptr<int64_t>(), 
                  cast_r_inv_ptr(), grid.z); break;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(d_r_inv));
}

torch::Tensor diag_unscale_forward(
    torch::Tensor tY,                 // \tilde{Y} (contiguous recommended)
    std::vector<torch::Tensor> r_inv, // length N, each 1D on device
    double theta,                     // scalar used to scale output tensor
    bool use_fp64_accum,              // Accumulate in FP64?
    c10::ScalarType out_dtype         // kFloat or kDouble, etc.
){
    const int N = tY.dim();
    TORCH_CHECK((int)r_inv.size() == N);

    // Build sizes and strides in "elements"
    std::vector<int64_t> sizes(N), strides_elts(N);
    for (int d = 0; d < N; ++d) {
        sizes[d] = tY.size(d);
        strides_elts[d] = tY.stride(d);
        TORCH_CHECK(r_inv[d].device().is_cuda());
        TORCH_CHECK(r_inv[d].numel() == sizes[d]);
    }

    // Output
    auto Y = torch::empty_like(tY, tY.options().dtype(out_dtype));

    // Dispatch combos (InT = at::Half/float/double)
    if (tY.scalar_type() == at::kHalf && out_dtype == at::kFloat) {
        if (use_fp64_accum) launch_diag_unscale<at::Half,float,double,true>(tY, Y, theta,sizes, strides_elts, r_inv);
        else                launch_diag_unscale<at::Half,float,float ,false>(tY, Y, theta, sizes, strides_elts, r_inv);
    } else if (tY.scalar_type() == at::kHalf && out_dtype == at::kHalf) {
        // Not recommended, but supported
        if (use_fp64_accum) launch_diag_unscale<at::Half,at::Half,double,true>(tY, Y, theta, sizes, strides_elts, r_inv);
        else                launch_diag_unscale<at::Half,at::Half,float ,false>(tY, Y, theta, sizes, strides_elts, r_inv);
    } else if (tY.scalar_type() == at::kFloat && out_dtype == at::kFloat) {
        if (use_fp64_accum) launch_diag_unscale<float,float,double,true>(tY, Y, theta, sizes, strides_elts, r_inv);
        else                launch_diag_unscale<float,float,float ,false>(tY, Y, theta, sizes, strides_elts, r_inv);
    } else if (tY.scalar_type() == at::kFloat && out_dtype == at::kDouble) {
        launch_diag_unscale<float,double,double,true>(tY, Y, theta, sizes, strides_elts, r_inv);
    } 
    else if (tY.scalar_type() == at::kHalf && out_dtype == at::kDouble) {
        launch_diag_unscale<at::Half,double,double,true>(tY, Y, theta, sizes, strides_elts, r_inv);
    } 
    else if (tY.scalar_type() == at::kDouble && out_dtype == at::kDouble) {
        launch_diag_unscale<double,double,double,true>(tY, Y, theta, sizes, strides_elts, r_inv);
    }
    else {
        TORCH_CHECK(false, "Unsupported dtype combination.");
    }
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("diag_unscale_forward", &diag_unscale_forward, "Materialize Y = tY * Π r_n^{-1}");
}

