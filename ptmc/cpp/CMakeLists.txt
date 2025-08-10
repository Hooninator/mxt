cmake_minimum_required(VERSION 3.20)

project(recover_kron_norm LANGUAGES CXX CUDA)

# C++/CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Adjust for your GPU(s)
# e.g., "80;86;90" or leave unset to let CMake detect
set(CMAKE_CUDA_ARCHITECTURES 80)

# Find Torch and pybind11
# If CMake can't find Torch, pass -DCMAKE_PREFIX_PATH=/path/to/libtorch
find_package(Torch REQUIRED)          # provides TORCH_LIBRARIES, TORCH_INCLUDE_DIRS, TORCH_CXX_FLAGS
find_package(pybind11 REQUIRED)       # provides pybind11::module pybind11::headers

# Sources
set(SOURCES
  recover_launcher.cu
)

# Build a Python extension module
add_library(recover_kron_norm MODULE ${SOURCES})

# No "lib" prefix for Python extension
set_target_properties(recover_kron_norm PROPERTIES
  PREFIX ""
  SUFFIX ".so"
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  POSITION_INDEPENDENT_CODE ON
)

# Include dirs
target_include_directories(recover_kron_norm PRIVATE
  ${TORCH_INCLUDE_DIRS}
  # If you have your own headers:
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Apply Torch’s required compile flags (contains ABI define, etc.)
# Do NOT put these in compile_definitions.
target_compile_options(recover_kron_norm PRIVATE ${TORCH_CXX_FLAGS})

# Optional optimizations
target_compile_options(recover_kron_norm PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# Link Torch and pybind11
target_link_libraries(recover_kron_norm PRIVATE
  ${TORCH_LIBRARIES}
  pybind11::module
)

# If you need the helper on some platforms:
# target_link_libraries(recover_kron_norm PRIVATE pybind11::python_link_helper)

# Ensure CUDA arch property is set on target
set_property(TARGET recover_kron_norm PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

