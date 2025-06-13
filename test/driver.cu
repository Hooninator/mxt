#include "mxt.cuh"

using namespace mxt;

int main(int argc, char ** argv)
{
    static constexpr uint32_t order = 3;
    using SparseTensor_t = SparseTensor<order, double, uint64_t>;

    SparseTensor_t X = io::read_tensor_frostt<SparseTensor_t>("$SCRATCH/tensors/matmul_6-3-3.tns");

    auto tucker_X = mixed_sparse_hooi<SparseTensor_t, __half, float>(X, 100);

    return 0;
}
