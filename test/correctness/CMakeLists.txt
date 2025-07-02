
message(${COMPILER_FLAGS})
add_executable(correctness correctness.cu)
target_include_directories(correctness PUBLIC ../../include)
target_compile_options(correctness PRIVATE ${COMPILER_FLAGS})
target_link_libraries(correctness PRIVATE MPI::MPI_CXX CUDA::cudart CUDA::cusparse CUDA::cublas CUDA::cusolver OpenMP::OpenMP_CXX)

