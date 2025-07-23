
from yaml import load, Loader
import os
from datetime import datetime


cmake_top = """
macro(add_exp name)
    add_executable(${name} ${name}.cu)
    target_include_directories(${name} PUBLIC ../../include)
    target_compile_options(${name} PRIVATE ${COMPILER_FLAGS})
    target_link_libraries(${name} PRIVATE MPI::MPI_CXX CUDA::cudart CUDA::cusparse CUDA::cublas CUDA::cusolver OpenMP::OpenMP_CXX)
endmacro()

add_exp(driver)
"""

runner_top = """#!/usr/bin/bash
#SBATCH -A m1266_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 3:00:00
#SBATCH -N 1

export OMP_NUM_THREADS=64
"""

def datetime_str():
    now = datetime.now()
    return now.strftime("%Y-%m-%d:%H")


with open("./ttmc_configs.yaml", "r") as file, open("./CMakeLists.txt", "w") as cmakefile, open("../../build/run.sh", "w") as runner:
    yaml_txt = file.read()
    data = load(yaml_txt, Loader=Loader)

    cmakefile.write(cmake_top)

    runner.write(runner_top)

    print(f"Tensors: {list(data.keys())}")

    for tensor in data:

        filename = f"{tensor}.cu"
        rows = ','.join(map(str, data[tensor]["rows"]))
        cols = data[tensor]["cols"]
        configs = data[tensor]["configs"]
        matrix_gens = data[tensor]["matrix_gens"]

        count = 0
        with open(f"./{filename}", 'w') as file:

            file.write('#include "run_ttmc.cuh"\n')
            conf_strs = []

            for k in range(len(matrix_gens)):
                matrix_gen = matrix_gens[k]
                for j in range(len(cols)):
                    col = ','.join(map(str, cols[j]))
                    for i in range(len(configs)):
                        conf_str = f"""using Conf{count} = TtmcConfig<Shape<{col}>,
                                                              Shape<{rows}>,
                                                              double, {configs[i]}, 
                                                              {matrix_gen}>;

                        """
                        file.write(conf_str)
                        conf_strs.append(f"Conf{count}")
                        count += 1

            bottom = f"""
            int main(int argc, char ** argv)
            {{
                std::string path("../tensors/{tensor}.dns");
                mxt_init();
                run<{','.join(conf_strs)}>(path);
                mxt_finalize();
                return 0;
            }}\n
            """
            file.write(bottom)

        cmakefile.write(f"add_exp({tensor})\n")

        runner.write(f"srun -n 1 -G 1 ./experiments/{tensor}\n")
        runner.write(f"mkdir ../test/experiments/data/{datetime_str()}\n")
        runner.write(f"mkdir ../test/experiments/data/{datetime_str()}/{tensor}\n")
        runner.write(f"mv *.csv ../test/experiments/data/{datetime_str()}/{tensor}\n")
        runner.write(f"mv *.dns ../test/experiments/data/{datetime_str()}/{tensor}\n")

        print(f"{count} configs generated for {tensor}")

