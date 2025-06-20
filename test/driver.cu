
#include "mxt.cuh"

#include <map>
#include <string>

using namespace mxt;

template <typename InputModes, typename TuckerRanks, typename HighU, typename LowU, typename LraU, typename Idx>
struct Config
{
    using InputModes_t = InputModes;
    using TuckerRanks_t = TuckerRanks;
    using HighU_t = HighU;
    using LowU_t = LowU;
    using LraU_t = LraU;
    using Idx_t = Idx;

    static constexpr uint32_t Order = InputModes_t::dims.size();
};


template <typename Conf>
void run_tensor(std::string& path)
{
    using SparseTensor_t = SparseTensor<typename Conf::HighU_t, typename Conf::Idx_t, Conf::Order, typename Conf::InputModes_t>;
    SparseTensor_t X = io::read_tensor_frostt<SparseTensor_t>(path.c_str());
    auto tucker_X = mixed_sparse_hooi<SparseTensor_t, typename Conf::TuckerRanks_t, typename Conf::LowU_t, typename Conf::LraU_t>(X, "randn", 100);
}


using NipsTns = Config<Shape<2482, 2862, 14036, 17>, 
                        Shape<10, 10, 10, 10>,
                        double, __half, float,
                        uint64_t>;


int main(int argc, char ** argv)
{
    if (argc < 3)
    {
        std::cerr<<"Usage: ./driver <tensor_name> <path_to_tensor>"<<std::endl;
        std::abort();
    }

    std::string tensor = std::string(argv[1]);
    std::string path = std::string(argv[2]);

    if (tensor.compare("nips")==0)
    {
        run_tensor<NipsTns>(path);
    }
    else
    {
        std::cerr<<"Invalid tensor: "<<tensor<<std::endl;
        std::abort();
    }
}
