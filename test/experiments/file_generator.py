
from yaml import load, Loader


top= """
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

"""

cmake_top = """
macro(add_exp name)
    add_executable(${name} ${name}.cu)
    target_include_directories(${name} PUBLIC ../../include)
    target_compile_options(${name} PRIVATE ${COMPILER_FLAGS})
    target_link_libraries(${name} PRIVATE MPI::MPI_CXX CUDA::cudart CUDA::cusparse CUDA::cublas CUDA::cusolver OpenMP::OpenMP_CXX)
endmacro()

add_exp(driver)
"""

with open("./configs.yaml", "r") as file, open("./CMakeLists.txt", "w") as cmakefile:
    yaml_txt = file.read()
    data = load(yaml_txt, Loader=Loader)

    cmakefile.write(cmake_top)

    for tensor in data:
        filename = f"{tensor}.cu"
        shape = ','.join([str(s) for s in data[tensor]["shape"]])
        ranks = ','.join([str(s) for s in data[tensor]["ranks"]])
        configs = data[tensor]["configs"]
        with open(f"./{filename}", 'w') as file:
            file.write(top)
            conf_strs = []
            for i in range(len(configs)):
                conf_str = f"""using Conf{i} = Config<Shape<{shape}>,
                                                      Shape<{ranks}>,
                                                      double, {configs[i]},
                                                      uint64_t>;

                """
                file.write(conf_str)
                conf_strs.append(f"Conf{i}")
            bottom = f"""
            int main(int argc, char ** argv)
            {{
                std::string path("$SCRATCH/tensors/{tensor}.tns");
                mxt_init();
                run<{','.join(conf_strs)}>(path);
                mxt_finalize();
                return 0;
            }}\n
            """
            file.write(bottom)
        cmakefile.write(f"add_exp({tensor})\n")






