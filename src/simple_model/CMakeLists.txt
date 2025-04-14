cmake_minimum_required(VERSION 3.22)

# ビルド実行ファイルの出力ディレクトリを設定
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../src/simple_model)


cuda_add_executable(effi_G_w spike_train_efficacy.cu OPTIONS "--std=c++20" "-DWAKE" "-lcudart")
cuda_add_executable(effi_G_s spike_train_efficacy.cu OPTIONS "--std=c++20" "-DSLEEP" "-lcudart")
cuda_add_executable(effi_G_w_mf spike_train_efficacy.cu OPTIONS "--std=c++20" "-DWAKE" "-DMF" "-lcudart")
cuda_add_executable(effi_G_s_mf spike_train_efficacy.cu OPTIONS "--std=c++20" "-DSLEEP" "-DMF" "-lcudart")
cuda_add_executable(effi_G_w_to3hz spike_train_efficacy.cu OPTIONS "--std=c++20" "-DWAKE" "-DTO3HZ" "-lcudart")
cuda_add_executable(effi_G_s_to3hz spike_train_efficacy.cu OPTIONS "--std=c++20" "-DSLEEP" "-DTO3HZ" "-lcudart")
cuda_add_executable(effi_G_w_high spike_train_efficacy.cu OPTIONS "--std=c++20" "-DWAKE" "-DHIGH" "-lcudart")
cuda_add_executable(effi_G_s_high spike_train_efficacy.cu OPTIONS "--std=c++20" "-DSLEEP" "-DHIGH" "-lcudart")


cuda_add_executable(effi_G_w_r spike_train_efficacy_random.cu OPTIONS "--std=c++20" "-DWAKE" "-lcudart")
cuda_add_executable(effi_G_s_r spike_train_efficacy_random.cu OPTIONS "--std=c++20" "-DSLEEP" "-lcudart")

