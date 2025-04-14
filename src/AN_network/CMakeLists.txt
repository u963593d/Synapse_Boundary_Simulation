cmake_minimum_required(VERSION 3.22)

# ビルド実行ファイルの出力ディレクトリを設定
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../src/AN_network)

cuda_add_executable(net_bifurcation_nmdar AN_network_bifurcation_nmdar_cav.cu OPTIONS "--std=c++20" "-DNMDAR" "-lcudart")
cuda_add_executable(net_bifurcation_cav AN_network_bifurcation_nmdar_cav.cu OPTIONS "--std=c++20" "-DCAV" "-lcudart")
cuda_add_executable(net_bifurcation_pre AN_network_bifurcation_pre.cu OPTIONS "--std=c++20" "-lcudart")

cuda_add_executable(AN_net_effi_nmdar AN_network_efficacy_nmdar.cu OPTIONS "--std=c++20" "-lcudart")
cuda_add_executable(AN_net_effi_cav AN_network_efficacy_cav.cu OPTIONS "--std=c++20" "-lcudart")
cuda_add_executable(AN_net_effi_pre AN_network_efficacy_pre.cu OPTIONS "--std=c++20" "-lcudart")

cuda_add_executable(AN_net_aup_nmdar AN_network_dynamics_nmdar.cu OPTIONS "--std=c++20" "-Da_UP" "-lcudart")
cuda_add_executable(AN_net_aup_cav AN_network_dynamics_cav.cu OPTIONS "--std=c++20" "-Da_UP" "-lcudart")
cuda_add_executable(AN_net_aup_pre AN_network_dynamics_pre.cu OPTIONS "--std=c++20" "-Da_UP" "-lcudart")
cuda_add_executable(AN_net_rup_nmdar AN_network_dynamics_nmdar.cu OPTIONS "--std=c++20" "-lcudart")
cuda_add_executable(AN_net_rup_cav AN_network_dynamics_cav.cu OPTIONS "--std=c++20" "-lcudart")
cuda_add_executable(AN_net_rup_pre AN_network_dynamics_pre.cu OPTIONS "--std=c++20" "-lcudart")



cuda_add_executable(AN_net_stim_cav AN_network_stim_cav.cu OPTIONS "--std=c++20" "-lcudart")

cuda_add_executable(CV_gf CV_cal_gpu.cu OPTIONS "--std=c++20" "-lcudart")


