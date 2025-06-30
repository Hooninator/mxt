cmake_minimum_required(VERSION 3.30)
project(fst LANGUAGES Fortran)

set(CMAKE_Fortran_COMPILER="ftn")
set(CMAKE_Fortran_MODULE_DIRECTORY="./build")

find_package(OpenMP REQUIRED)

add_executable(fst 
               utils.f90 
               driver.f90 
               sptensor.f90 
               sparse_tucker.f90 
               frostt_io.f90
               matrix.f90
               kron.f90)

target_compile_options(fst PRIVATE -Wall -Wextra -O3 -fopenmp)
target_link_libraries(fst PRIVATE OpenMP::OpenMP_Fortran)


