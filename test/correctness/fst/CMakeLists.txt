cmake_minimum_required(VERSION 3.30)
project(fst LANGUAGES Fortran)

set(CMAKE_Fortran_COMPILER="ftn")
set(CMAKE_Fortran_MODULE_DIRECTORY="./build")

add_executable(fst 
               utils.f90 
               driver.f90 
               sptensor.f90 
               sparse_tucker.f90 
               frostt_io.f90
               matrix.f90
               kron.f90)
           target_compile_options(fst PUBLIC -Wall -Wextra -O3)

