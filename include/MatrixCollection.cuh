#ifndef MATRIX_COLLECTION_CUH
#define MATRIX_COLLECTION_CUH

#include "common.cuh"
#include "utils.cuh"
#include "rand/rand_matrix.cuh"


namespace mxt
{

enum MatrixGenerator_t
{
    GEN_SMALL,
    GEN_RANDN,
    GEN_BIG
};

template <typename T, typename RowsShape, typename ColsShape>
class MatrixCollection
{

    static_assert(RowsShape::dims.size() == ColsShape::dims.size());


public:
    static constexpr size_t N = RowsShape::dims.size();

    using ValueType_t = T;
    using Arr_t = std::array<size_t, N>;
    using RowShape_t = RowsShape;
    using ColShape_t = ColsShape;

    static constexpr Arr_t Rows = RowsShape::dims;
    static constexpr Arr_t Cols = ColsShape::dims;


    static constexpr Arr_t MatrixSizes = []<std::size_t... Is>(std::index_sequence<Is...>)
    {
        return std::array<size_t, N>{(Rows[Is] * Cols[Is])...};
    }(std::make_index_sequence<N>{});

    static constexpr size_t Size = []<std::size_t... Is>(std::index_sequence<Is...>)
    {
        return (MatrixSizes[Is] + ...);
    }(std::make_index_sequence<N>{});


    MatrixCollection(MatrixGenerator_t gen)
    {

        CUDA_CHECK(cudaMalloc(&d_buf, Size*sizeof(T)));
        CUDA_CHECK(cudaMemset(d_buf, 0, Size*sizeof(T)));
        matrices.reserve(N);

        size_t offset = 0;
        for (int i=0; i<N; i++)
        {
            switch(gen)
            {
                case GEN_RANDN:
                    rand::randn_buffer_inplace(d_buf + offset, Rows[i] * Cols[i]);
                    break;
                case GEN_SMALL:
                    rand::randu_buffer_inplace(d_buf + offset, Rows[i] * Cols[i]);
                    break;
                case GEN_BIG:
                    rand::randu_buffer_inplace(d_buf + offset, Rows[i] * Cols[i], 
                                                static_cast<double>(MAX_HALF));
                    break;
            }

            matrices[i] = d_buf + offset;
            offset += Rows[i] * Cols[i];
        }
    }


    MatrixCollection(const char * fpath)
    {

        CUDA_CHECK(cudaMalloc(&d_buf, Size*sizeof(T)));
        CUDA_CHECK(cudaMemset(d_buf, 0, Size*sizeof(T)));

        matrices.reserve(N);

        size_t offset = 0;
        for (int i=0; i<N; i++)
        {
            std::string fpath_i = std::string(fpath) + "matrix_" + std::to_string(i) + ".dns";

            T * d_matrix = io::read_matrix_dns<T>(fpath_i.c_str(), Rows[i], Cols[i]);

            CUDA_CHECK(cudaMemcpy(d_buf + offset, d_matrix, sizeof(T) * Rows[i] * Cols[i], cudaMemcpyDeviceToDevice));
            CUDA_FREE(d_matrix);

            matrices[i] = d_buf + offset; 

            offset += MatrixSizes[i];
        }
    }


    T * get_matrix(const size_t i)
    {
        ASSERT( (i < N), "Error: %lu not less than %lu when getting matrix", i, N);
        return matrices[i];
    }


    ~MatrixCollection()
    {
        CUDA_FREE(d_buf);
    }


    std::vector<T *> matrices;
    T * d_buf;

};



} //mxt


#endif
