
#include "mxt.cuh"
#include "linalg/norms.cuh"
#include "linalg/geam.cuh"
#include "Config.hpp"

#include <map>
#include <string>
#include <sstream>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>


using namespace mxt;


static const char * base = "../test/correctness/fst_cases/";


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
    auto tucker_X = mixed_sparse_hooi<SparseTensor_t, typename Conf::CoreTensorU_t, typename Conf::LraU_t, typename Conf::TuckerRanks_t>(X, "file", 5, factors_dir.c_str());
    utils::print_separator("Done Tucker");


    /* Compare the core tensors */
    static constexpr auto Rn = std::reduce(Conf::TuckerRanks_t::dims.begin(), Conf::TuckerRanks_t::dims.end(), 1, std::multiplies<size_t>{});
    std::string core_path = std::string(factors_dir);
    core_path.append("core_final.tns");
    typename Conf::CoreTensorU_t * d_correct_core = io::read_dense_tensor_frostt<typename Conf::CoreTensorU_t, typename Conf::TuckerRanks_t>(core_path.c_str());


    typename Conf::CoreTensorU_t * d_correct_core_t;
    CUDA_CHECK(cudaMalloc(&d_correct_core_t, sizeof(typename Conf::CoreTensorU_t) * Rn));


    static constexpr auto R0 = Conf::TuckerRanks_t::dims[Conf::Order - 1];
    linalg::transpose(d_correct_core, d_correct_core_t, R0, Rn / R0);
    utils::d2d_cpy(tucker_X.d_core, d_correct_core, Rn);


    auto d_correct_core_ptr = thrust::device_pointer_cast(d_correct_core);
    auto d_correct_core_t_ptr = thrust::device_pointer_cast(d_correct_core_t);


    thrust::transform(d_correct_core_t_ptr, d_correct_core_t_ptr + Rn, d_correct_core_t_ptr, utils::abs_functor<typename Conf::CoreTensorU_t>{});
    thrust::transform(d_correct_core_ptr, d_correct_core_ptr + Rn, d_correct_core_ptr, utils::abs_functor<typename Conf::CoreTensorU_t>{});


    utils::write_d_arr(globals::logfile, d_correct_core_t, Rn, "correct_core_t");
    utils::write_d_arr(globals::logfile, d_correct_core, Rn, "computed_core");


    double err = linalg::relative_frob_norm(d_correct_core_t, d_correct_core, Rn);
    std::cout<<"|| G_correct - G_computed||_F / ||G_correct||_F : "<<err<<std::endl;


    double recon_err = tucker_X.reconstruction_error(X);
    std::cout<<"||X - X_tucker||_F / ||X||_F : "<<recon_err<<std::endl;


    CUDA_FREE(d_correct_core_t);
    CUDA_FREE(d_correct_core);
}



using ThreeD12031 = Config<Shape<100, 80, 60>, 
                            Shape<10, 8, 6>,
                            double, double, double, double,
                            uint64_t>;

using Kinetic = Config<Shape<64, 12, 10, 60>, 
                        Shape<20, 6, 5, 20>,
                        double, double, double, double,
                        uint64_t>;

using Randn5 = Config<Shape<10, 20, 10, 5, 10>,
                        Shape<5, 3, 3, 2, 5>,
                        double, double, double, double,
                        uint64_t>;

using Randn4Scaled = Config<Shape<50, 50, 50, 50>,
                            Shape<25, 40, 10, 5>,
                            double, double, double, double,
                            uint64_t>;

using Small = Config<Shape<3, 3, 3>, 
                     Shape<2,2,2>, 
                     double, double, double, double, 
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
    else if (tensor.compare("small")==0)
    {
        run_correctness<Small>(path, tensor);
    }
    else
    {
        std::cerr<<"Invalid tensor: "<<tensor<<std::endl;
        std::abort();
    }

    mxt_finalize();

    return 0;
}
