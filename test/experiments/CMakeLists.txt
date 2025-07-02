
macro(add_exp name)
    add_executable(${name} ${name}.cu)
    target_include_directories(${name} PUBLIC ../../include)
    target_compile_options(${name} PRIVATE ${COMPILER_FLAGS})
    target_link_libraries(${name} PRIVATE MPI::MPI_CXX CUDA::cudart CUDA::cusparse CUDA::cublas CUDA::cusolver OpenMP::OpenMP_CXX)
endmacro()

add_exp(driver)
add_exp(nips)
add_exp(kinetic)
add_exp(uber)
add_exp(3D_12031)
