cmake_minimum_required(VERSION 3.27)

project(mxt LANGUAGES CXX C CUDA)

set(CMAKE_CUDA_ARCHITECTURES "75") 

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CUDA_FLAGS "--extended-lambda --expt-relaxed-constexpr -ccbin=g++ -fopenmp")
#set(COMPILER_FLAGS "-O3")
set(COMPILER_FLAGS "-g")

add_library(mxt src/sparse_tucker.cu)
target_include_directories(mxt PUBLIC include)
target_compile_options(mxt PRIVATE ${COMPILER_FLAGS})
target_link_libraries(mxt PRIVATE MPI::MPI_CXX CUDA::cudart CUDA::cusparse CUDA::cublas CUDA::cusolver OpenMP::OpenMP_CXX)

add_subdirectory(./fst ./fst)
add_subdirectory(./test/experiments ./experiments)
add_subdirectory(./test/correctness ./correctness)
