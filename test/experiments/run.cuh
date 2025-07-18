#include "mxt.cuh"
#include "Config.hpp"

#include <map>
#include <string>

#define NTRIALS 2

#define CSV_PROFILING 0

using namespace mxt;


template <typename Conf>
void run_trial(std::string& path)
{
    using SparseTensor_t = SparseTensor<typename Conf::HighU_t, typename Conf::LowU_t, typename Conf::Idx_t, Conf::Order, typename Conf::InputModes_t>;

    size_t pos = path.find_last_of("/") + 1;
    std::string tensor_name = path.substr(pos, path.size() - pos);

    Conf::print(tensor_name);

    for (uint32_t t = 0; t < NTRIALS; t++)
    {
        utils::print_separator("Beginning IO");
        SparseTensor_t X = io::read_tensor_frostt<SparseTensor_t>(path.c_str());
        utils::print_separator("Done IO");


        utils::print_separator("Beginning Tucker");
        globals::profiler->start_timer("hooi");
        auto tucker_X = mixed_sparse_hooi<SparseTensor_t, typename Conf::CoreTensorU_t, typename Conf::LraU_t, typename Conf::TuckerRanks_t>(X, "randn", 5);
        globals::profiler->stop_timer("hooi");
        globals::profiler->print_timers();
        utils::print_separator("Done Tucker");


        auto err = tucker_X.reconstruction_error(X);
        globals::profiler->add_stat("recon_err", err);

        std::cout<<"[Reconstruction Error] : "<<err<<std::endl;

        if (t == NTRIALS - 1)
        {
            std::ofstream core_file;
            core_file.open("core.out");
            tucker_X.dump_core(core_file);
            core_file.close();
        }

        globals::profiler->commit_timers();
    }


#if CSV_PROFILING

    std::string timer_csv_name("./" + tensor_name + "_" + Conf::str() + "_timings.csv");
    std::string stats_csv_name("./" + tensor_name + "_" + Conf::str() + "_stats.csv");

    globals::profiler->timers_to_csv(timer_csv_name.c_str());
    globals::profiler->stats_to_csv(stats_csv_name.c_str());

#endif

}


template <typename... Confs>
void run(std::string& path)
{
    (run_trial<Confs>(path), ...);
}

