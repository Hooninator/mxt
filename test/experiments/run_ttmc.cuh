
#include "mxt.cuh"
#include "TtmcConfig.cuh"

#include <map>
#include <string>

#define NTRIALS 100

#define SAVE 1

using namespace mxt;


template <typename Conf>
void run_trial(std::string& path)
{

    using DenseTensor_t = DenseTensor<typename Conf::HighU_t, typename Conf::MatrixCols_t>;
    using MatrixCollection_t = MatrixCollection<typename Conf::LowU_t, typename Conf::MatrixRows_t, typename Conf::MatrixCols_t>;
    using OutputDenseTensor_t = DenseTensor<typename DenseTensor_t::ValueType_t, typename Conf::MatrixRows_t>;

    static constexpr cublasComputeType_t ComputeType = Conf::ComputeType;

    size_t pos = path.find_last_of("/") + 1;
    std::string tensor_name = path.substr(pos, path.size() - pos);

    Conf::print(tensor_name);

    utils::print_separator("Beginning IO");
    DenseTensor_t X(path.c_str()); 
    utils::print_separator("Done IO");

    MatrixCollection_t matrices(Conf::Generator);

    utils::print_separator("Beginning TTMc");
    for (uint32_t t = 0; t < NTRIALS; t++)
    {
        globals::profiler->start_timer("ttmc-total");
        OutputDenseTensor_t Y = ttmc_mixed<DenseTensor_t, MatrixCollection_t, OutputDenseTensor_t>(X, matrices, ComputeType);
        globals::profiler->stop_timer("ttmc-total");

        if (t == NTRIALS - 1)
        {
            globals::profiler->print_timer("ttmc-total");
#if SAVE
            std::ofstream y_file;
            y_file.open(std::string(tensor_name + "_" + Conf::str() + "_output.dns"));
            Y.dump(y_file);
            y_file.close();
#endif
        }
    }
    utils::print_separator("Done TTMc");

    globals::profiler->commit_timers();

#if SAVE
    std::string timer_csv_name("./" + tensor_name + "_" + Conf::str() + "_timings.csv");
    globals::profiler->timers_to_csv(timer_csv_name.c_str());
#endif

}


template <typename... Confs>
void run(std::string& path)
{
    (run_trial<Confs>(path), ...);
}

