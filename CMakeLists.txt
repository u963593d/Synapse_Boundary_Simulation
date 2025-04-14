cmake_minimum_required(VERSION 3.22)
project(sp_effi LANGUAGES CXX CUDA)
set(CUDA_HOST_COMPILER nvc++)

find_package(CUDA REQUIRED)

file(GLOB CUDA_FILES "*.cu")
message("CU files: " ${CUDA_FILES})


set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(CMAKE_CXX_COMPILER nvc++)
set(CMAKE_CUDA_COMPILER nvc++)
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "--expt-relaxed-constexpr; --std=c++20;-mp; -fast; -ta=tesla")


set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++20")

# nvc++コンパイラを使用するための設定


# コンパイルオプションを設定
link_directories(/tmp/nvhpc_2023_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/lib)

include_directories(/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/examples/OpenACC/SDK/include)
include_directories(/tmp/nvhpc_2023_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/include)

add_subdirectory(src/simple_model)
add_subdirectory(src/AN_network)


# ビルド実行ファイルの出力ディレクトリを設定
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)
# cuda_add_executable(effi_G_w spike_train_efficacy.cu OPTIONS "--std=c++20" "-DWAKE" "-lcudart")
# cuda_add_executable(effi_G_s spike_train_efficacy.cu OPTIONS "--std=c++20" "-DSLEEP" "-lcudart")

