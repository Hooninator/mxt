
#include "mxt.cuh"
#include "linalg/norms.cuh"

#include <map>
#include <string>
#include <sstream>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>


using namespace mxt;

template <typename InputModes, typename TuckerRanks, typename HighU, typename LowU, typename CoreTensorU, typename Idx>
struct Config
{
    using InputModes_t = InputModes;
    using TuckerRanks_t = TuckerRanks;
    using HighU_t = HighU;
    using LowU_t = LowU;
    using CoreTensorU_t = CoreTensorU;
    using Idx_t = Idx;

    static constexpr uint32_t Order = InputModes_t::dims.size();
};


static const char * base = "../test/correctness/cases/";


template <typename Conf>
void run_correctness(std::string& path, std::string& tensorname)
{
    using SparseTensor_t = SparseTensor<typename Conf::HighU_t, typename Conf::LowU_t, typename Conf::Idx_t, Conf::Order, typename Conf::InputModes_t>;

    utils::print_separator("Beginning IO");
    std::cout<<path<<std::endl;
    SparseTensor_t X = io::read_tensor_frostt<SparseTensor_t>(path.c_str());
    utils::print_separator("Done IO");

    std::string factors_dir = std::string(base);
    factors_dir.append(tensorname);
    factors_dir.append("/");

    utils::print_separator("Beginning Tucker");
    auto tucker_X = mixed_sparse_hooi<SparseTensor_t, typename Conf::CoreTensorU_t, typename Conf::TuckerRanks_t>(X, "file", 5, factors_dir.c_str());
    utils::print_separator("Done Tucker");

    /* Compare the core tensors */
    std::string core_path = std::string(factors_dir);
    core_path.append("core.tns");
    typename Conf::CoreTensorU_t * d_correct_core = io::read_dense_tensor_frostt<typename Conf::CoreTensorU_t, typename Conf::TuckerRanks_t>(core_path.c_str());

    auto Rn = std::reduce(Conf::TuckerRanks_t::dims.begin(), Conf::TuckerRanks_t::dims.end(), 1, std::multiplies<size_t>{});

    auto d_correct_ptr = thrust::device_pointer_cast(d_correct_core);
    auto d_computed_ptr = thrust::device_pointer_cast(tucker_X.d_core);

    thrust::sort(d_correct_ptr, d_correct_ptr + Rn);
    thrust::sort(d_computed_ptr, d_computed_ptr + Rn);
    
    auto err = linalg::relative_frob_norm(d_correct_core, tucker_X.d_core, Rn);

    std::cout<<"|| G_correct - G_computed||_F / ||G_correct||_F : "<<err<<std::endl;

}


using ThreeD12031 = Config<Shape<100, 80, 60>, 
                            Shape<10, 8, 6>,
                            double, double, double, 
                            uint64_t>;

using Kinetic = Config<Shape<64, 12, 10, 60>, 
                        Shape<20, 6, 5, 20>,
                        double, double, double, 
                        uint64_t>;

using Randn5 = Config<Shape<10, 20, 10, 5, 10>,
                        Shape<5, 3, 3, 2, 5>,
                        double, double, double, 
                        uint64_t>;

using Randn4Scaled = Config<Shape<50, 50, 50, 50>,
                            Shape<25, 40, 10, 5>,
                            double, double, double, 
                            uint64_t>;


int main(int argc, char ** argv)
{
    if (argc < 2)
    {
        std::cerr<<"Usage: ./correctness <tensor_name>"<<std::endl;
        std::abort();
    }

    std::string tensor = std::string(argv[1]);
    std::stringstream ss;
    ss<<"../tensors/"<<tensor<<".tns";
    std::string path = ss.str();

    mxt_init();

    if (tensor.compare("3D_12031")==0)
    {
        run_correctness<ThreeD12031>(path, tensor);
    }
    else if (tensor.compare("kinetic")==0)
    {
        run_correctness<Kinetic>(path, tensor);
    }
    else if (tensor.compare("randn5")==0)
    {
        run_correctness<Randn5>(path, tensor);
    }
    else if (tensor.compare("randn4_scaled")==0)
    {
        run_correctness<Randn4Scaled>(path, tensor);
    }
    else
    {
        std::cerr<<"Invalid tensor: "<<tensor<<std::endl;
        std::abort();
    }

    mxt_finalize();

    return 0;
}
