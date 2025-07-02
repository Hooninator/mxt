
#include "mxt.cuh"
#include "Config.hpp"

#include <map>
#include <string>

using namespace mxt;


template <typename Conf>
void run_trial(std::string& path)
{
    using SparseTensor_t = SparseTensor<typename Conf::HighU_t, typename Conf::LowU_t, typename Conf::Idx_t, Conf::Order, typename Conf::InputModes_t>;

    utils::print_separator("Beginning IO");
    SparseTensor_t X = io::read_tensor_frostt<SparseTensor_t>(path.c_str());
    utils::print_separator("Done IO");



    utils::print_separator("Beginning Tucker");
    globals::profiler->start_timer("hooi");
    auto tucker_X = mixed_sparse_hooi<SparseTensor_t, typename Conf::CoreTensorU_t, typename Conf::LraU_t, typename Conf::TuckerRanks_t>(X, "randn", 5);
    globals::profiler->stop_timer("hooi");
    globals::profiler->print_timer("hooi");
    utils::print_separator("Done Tucker");


    auto err = tucker_X.reconstruction_error(X);
    std::cout<<"||X - X_tucker||_F / ||X||_F : "<<err<<std::endl;

    std::ofstream core_file;
    core_file.open("core.out");
    tucker_X.dump_core(core_file);
    core_file.close();
}


template <typename Conf>
void run(std::string& path)
{
    for (uint32_t t = 0; t < 1; t++)
    {
        run_trial<Conf>(path);
    }
}


using NipsTns = Config<Shape<2482, 2862, 14036, 17>, 
                        Shape<10, 10, 10, 10>,
                        double, double, double, double,
                        uint64_t>;

using ChicagoCrime = Config<Shape<6186, 24, 77, 32>,
                            Shape<20, 20, 20, 20>,
                            double, double, double, double,
                            uint64_t>;

using Randn5Tns = Config<Shape<10, 20, 10, 5, 10>, 
                        Shape<5, 3, 3, 2, 5>,
                        double, double, double,float,
                        uint64_t>;

using Randn4Tns = Config<Shape<10, 20, 20, 10>, 
                        Shape<5, 3, 3, 2>,
                        double, __half, float, float,
                        uint64_t>;

using Randn3Tns = Config<Shape<10, 20, 10>, 
                        Shape<5, 3, 3>,
                        double, __half, float, float,
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

    mxt_init();

    if (tensor.compare("nips")==0)
    {
        run<NipsTns>(path);
    }
    else if (tensor.compare("crime")==0)
    {
        run<ChicagoCrime>(path);
    }
    else if (tensor.compare("randn3")==0)
    {
        run<Randn3Tns>(path);
    }
    else if (tensor.compare("randn4")==0)
    {
        run<Randn4Tns>(path);
    }
    else if (tensor.compare("randn5")==0)
    {
        run<Randn5Tns>(path);
    }
    else
    {
        std::cerr<<"Invalid tensor: "<<tensor<<std::endl;
        std::abort();
    }

    mxt_finalize();

    return 0;
}
