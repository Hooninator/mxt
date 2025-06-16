#include "mxt.cuh"

using namespace mxt;

int main(int argc, char ** argv)
{
    static constexpr uint32_t order = 3;

    using input_modes = Shape<6,3,3> ;

    using SparseTensor_t = SparseTensor<double, uint64_t, order, input_modes>;


    SparseTensor_t X = io::read_tensor_frostt<SparseTensor_t>("$SCRATCH/tensors/matmul_6-3-3.tns");

    using tucker_ranks = Shape<2,2,2>;
    auto tucker_X = mixed_sparse_hooi<SparseTensor_t, tucker_ranks, __half, float>(X, "randn", 100);

    return 0;
}
