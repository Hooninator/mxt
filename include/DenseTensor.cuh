
#ifndef DENSE_TENSOR_CUH
#define DENSE_TENSOR_CUH

#include "common.cuh"
#include "utils.cuh"
#include "Shape.cuh"
#include "tensor_io.cuh"


namespace mxt
{


template <typename T, typename Shape>
class DenseTensor
{
public:

    using Shape_t = Shape;
    using ValueType_t = T;

    static constexpr uint32_t Order = Shape::dims.size();
    using Index_t = std::array<size_t, Order>;

    static constexpr Index_t Modes = Shape::dims;
    static constexpr size_t In = std::reduce(Modes.begin(), Modes.end(), 1, std::multiplies<size_t>{});


    DenseTensor(T * d_data): 
        d_data(d_data)
    {}


    DenseTensor(const char * fpath)
    {
        d_data = io::read_dense_tensor_frostt<T, Shape_t>(fpath, true);
    }


    bool operator==(DenseTensor<T, Shape>& other)
    {
        static constexpr T tol = 1e-3;
        T nrm = rel_nrm(other);
        return nrm <= tol;
    }


    inline size_t unfolding_cols(const size_t k)
    {
        return In / Modes[k];
    }


    T rel_nrm(DenseTensor<T, Shape>& other)
    {
        T * d_tmp;
        CUDA_CHECK(cudaMalloc(&d_tmp, sizeof(T) * In));
        CUDA_CHECK(cudaMemcpy(d_tmp, d_data, sizeof(T) * In, cudaMemcpyDeviceToDevice));

        cudaDataType dtype = utils::to_cuda_dtype<T>();
        T alpha = -1.0;
        CUBLAS_CHECK(cublasAxpyEx(globals::cublas_handle, In, 
                                  &alpha, dtype,
                                  other.d_data, dtype, 1, 
                                  d_tmp, dtype, 1,
                                  dtype));

        T nrm_diff;
        CUBLAS_CHECK(cublasNrm2Ex(globals::cublas_handle,
                                In, d_tmp, dtype, 1, &nrm_diff,
                                dtype, dtype));
        CUDA_FREE(d_tmp);

        T nrm;
        CUBLAS_CHECK(cublasNrm2Ex(globals::cublas_handle,
                                In, d_data, dtype, 1, &nrm,
                                dtype, dtype));

        return nrm_diff / nrm;
    }


    template <typename... Inds>
    inline size_t idx(Inds... inds)
    {
        size_t i = 0;
        size_t ind = 0;
        ((ind += inds * Modes[i++]), ...);
        return ind;
    }


    void dump_logfile()
    {

    }


    ~DenseTensor()
    {
        CUDA_FREE(d_data);
    }


    T * d_data;

};

} //mxt

#endif
