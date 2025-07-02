
#include "mxt.cuh"
#include "Config.hpp"

#include <map>
#include <string>

#define NTRIALS 5

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


template <typename... Confs>
void run(std::string& path)
{
    (Confs::print(), ...);
    for (uint32_t t = 0; t < NTRIALS; t++)
    {
        (run_trial<Confs>(path), ...);
    }
}

using Conf0 = Config<Shape<2482,2862,14036,17>,
                                                      Shape<10,10,10,10>,
                                                      double, double, double, double,
                                                      uint64_t>;

                using Conf1 = Config<Shape<2482,2862,14036,17>,
                                                      Shape<10,10,10,10>,
                                                      double, float, float, float,
                                                      uint64_t>;

                using Conf2 = Config<Shape<2482,2862,14036,17>,
                                                      Shape<10,10,10,10>,
                                                      double, __half, float, __half,
                                                      uint64_t>;

                
            int main(int argc, char ** argv)
            {
                std::string path("$SCRATCH/tensors/nips.tns");
                mxt_init();
                run<Conf0,Conf1,Conf2>(path);
                mxt_finalize();
                return 0;
            }
            