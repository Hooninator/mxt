cmake_minimum_required(VERSION 3.30)


project(mxt LANGUAGES CXX C CUDA)

set(CMAKE_CUDA_ARCHITECTURES "80") 

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)

set(CMAKE_CUDA_FLAGS "--extended-lambda --expt-relaxed-constexpr -ccbin=CC")
set(COMPILER_FLAGS "-O3")

add_library(mxt src/sparse_tucker.cu)
target_include_directories(mxt PUBLIC include)
target_compile_options(mxt PRIVATE ${COMPILER_FLAGS})
target_link_libraries(mxt PRIVATE MPI::MPI_CXX CUDA::cudart CUDA::cusparse CUDA::cublas)


macro(add_test name)
    add_executable(${name} test/${name}.cu)
    target_include_directories(${name} PUBLIC include)
    target_compile_options(${name} PRIVATE ${COMPILER_FLAGS})
    target_link_libraries(${name} PRIVATE MPI::MPI_CXX CUDA::cudart CUDA::cusparse CUDA::cublas)
endmacro()

add_test(driver)
