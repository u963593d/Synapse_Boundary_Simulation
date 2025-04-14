// #define TO3HZ
// #define AUTO_HZ
// #define HIGH_HZ
// #define WAKE
// #define MF


#include "common.h"
#include <cuda_runtime.h>


#include "common.h"
#include <cuda_runtime.h>

#include <cuda.h>
#include <stdio.h>
#include <dirent.h>
//#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits.h>
#include <vector>
#include <string>
#include <array>
#include <math.h>
#include <tuple>
#include <algorithm>
#include <numeric>
#include <stdlib.h>
#include <filesystem>
#include <omp.h>
#include <random>
#include <time.h>
#include <cmath>
#include "numpy.hpp"
#include <omp.h>

//standard
//nvc++ -mp -fast -ta=tesla spike_train_efficacy.cu -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/examples/OpenACC/SDK/include/ -I/tmp/nvhpc_2023_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/include -L/tmp/nvhpc_2023_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/lib -DWAKE -lcudart -std=c++20 -o effi_G_w
//nvc++ -mp -fast -ta=tesla spike_train_efficacy.cu -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/examples/OpenACC/SDK/include/ -I/tmp/nvhpc_2023_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/include -L/tmp/nvhpc_2023_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/lib -DSLEEP -lcudart -std=c++20 -o effi_G_s

//high Hz
//nvc++ -mp -fast -ta=tesla spike_train_efficacy.cu -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/examples/OpenACC/SDK/include/ -I/tmp/nvhpc_2023_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/include -L/tmp/nvhpc_2023_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/lib -DWAKE -DHIGH -lcudart -std=c++20 -o effi_G_w_high
//nvc++ -mp -fast -ta=tesla spike_train_efficacy.cu -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/examples/OpenACC/SDK/include/ -I/tmp/nvhpc_2023_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/include -L/tmp/nvhpc_2023_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/lib -DSLEEP -DHIGH -lcudart -std=c++20 -o effi_G_s_high

//to 3Hz
//nvc++ -mp -fast -ta=tesla spike_train_efficacy.cu -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/examples/OpenACC/SDK/include/ -I/tmp/nvhpc_2023_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/include -L/tmp/nvhpc_2023_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/lib -DWAKE -DTO3HZ -lcudart -std=c++20 -o effi_G_w_to3hz
//nvc++ -mp -fast -ta=tesla spike_train_efficacy.cu -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/examples/OpenACC/SDK/include/ -I/tmp/nvhpc_2023_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/include -L/tmp/nvhpc_2023_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/lib -DSLEEP -DTO3HZ -lcudart -std=c++20 -o effi_G_s_to3hz

//mean firing rates in the up state of sleep-like firing pattern = mean firing  rates in wake-like firing pattern
//nvc++ -mp -fast -ta=tesla spike_train_efficacy.cu -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/examples/OpenACC/SDK/include/ -I/tmp/nvhpc_2023_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/include -L/tmp/nvhpc_2023_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/lib -DWAKE -DMF -lcudart -std=c++20 -o effi_G_w_mf
//nvc++ -mp -fast -ta=tesla spike_train_efficacy.cu -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/examples/OpenACC/SDK/include/ -I/tmp/nvhpc_2023_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/include -L/tmp/nvhpc_2023_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/lib -DSLEEP -DMF -lcudart -std=c++20 -o effi_G_s_mf



#define SAFE_FREE(ptr) if(ptr != NULL){free(ptr); ptr = NULL;}
#define SIZE(buff) (sizeof(buff)/sizeof(buff[s_a3[ix] 0])) 

//define neuron parameter 
#define cm 1.0
#define area 0.02
#define a_ca 0.5
#define kd_ca 30.0
#define rho_star 0.5
#define zeta 1
#define s_a_ampar 3.48
#define s_tau_ampar 2.0
#define s_a_nmdar 5
#define s_tau_nmdar 10
#define x_a 34.8
#define x_tau 0.2
#define g_nmdar 0.0138403
#define g_cav_s 0.050580
#define vL -60.95
#define vNa 55.0
#define vK -100.0
#define vCa 120.0
#define vrest -70
#define vAMPAR 0
// #define blockdim_x 32
#define blockdim_y 1




#define CHECK(call)                   \
{                                      \
  const cudaError_t error = call;      \
  if(error != cudaSuccess)             \
  {                                    \
      printf("Error: %s:%d, ", __FILE__, __LINE__);               \
      printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));    \
      exit(1);        \
  }                \
}             \


double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

__device__ double dxdt_n_cal(double x, double mp_pre, double dt){
        double fv = 1.0 / (1.0 + std::exp(-(mp_pre-20)/2.0));
        double dxdt_n = (x_a * fv- x/x_tau)*dt;
     return dxdt_n;
}

__device__ double dsdt_n_cal(double x, double s, double dt){
        double dsdt_n = (s_a_nmdar * x * (1-s) - s/s_tau_nmdar)*dt;
     return dsdt_n;
}

__device__ double dcpre_cal(double cpre, double s, double tau_ca_pre, double cpre_r, double dt){
       double i_nmda =  (g_nmdar * s * (vrest-vCa) );
       double dcpre = (- (a_ca * (i_nmda))*cpre_r - cpre/tau_ca_pre)*dt;
     return dcpre;
}

__device__ double dcpost_cal(double cpost, double mp_post, double tau_ca_post, double cpost_r, double dt){
       double fv = 1.0 / (1.0 + std::exp(-(mp_post+20)/9));
      double i_vdcc_s = (g_cav_s * std::pow(fv,2) * (mp_post-vCa));
       double dcpost = (- (a_ca * (i_vdcc_s))*cpost_r - cpost/tau_ca_post)*dt;
     return dcpost;
}

__device__ double drho_cal(double rho, double ca,double th_p, double th_d, double gp, double gd,  double sig, double tau_s, double dt, double r){
         double C_th_p;
         double C_th_d;
         
         if (ca > th_p){
            C_th_p = 1;}
        else{
            C_th_p = 0;}

        if (ca > th_d){
            C_th_d = 1;}
        else{
            C_th_d = 0;}

            double Noise = sig*std::sqrt(tau_s)*std::sqrt(C_th_d+C_th_p)*r/std::sqrt(dt)*std::sqrt(3.5);
            double drho=  ((-rho*(1-rho)*(rho_star-rho)+(gp*(1-rho)*C_th_p-gd*rho*C_th_d+Noise))/tau_s)*dt;
        // else:
        //     double drho =0;
     return drho;
}

__global__ void dt_gpu(double *x_n, double *s_n, double *cpre, double *cpost, double *rho, int N, int T, double *d_sum_rho, double dt,
    double *mp_pres, double *mp_post_t, double *rd, double cpre_r, double cpost_r, double th_p, double th_d, double gp, double gd, double tau_ca_pre, double tau_ca_post, double sig, double tau_s)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    // unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    //unsigned int idx = iy * nx + ix;
    double sum_rho = 0;  

    if (ix < N)
    {  
       for (int it = 0; it < T; it++) {
      
        double x_n_i = x_n[ix];
        double s_n_i = s_n[ix];
        double cpre_i = cpre[ix];
        double cpost_i = cpost[ix];
        double rho_i = rho[ix];
        double ca = cpre_i + cpost_i;
        double mp_pre = mp_pres[ix*T+it];
        double mp_post= mp_post_t[it];
        double r = rd[ix*T+it];

        
    // k1
       double dxdt_n = dxdt_n_cal(x_n_i, mp_pre, dt);
       double dsdt_n = dsdt_n_cal(x_n_i, s_n_i, dt);
       double dcpre = dcpre_cal(cpre_i, s_n_i, tau_ca_pre, cpre_r, dt);
       double dcpost = dcpost_cal(cpost_i, mp_post, tau_ca_post, cpost_r, dt);
       double drho = drho_cal(rho_i, ca, th_p, th_d, gp, gd, sig, tau_s, dt, r);
   
       double x_n2 = x_n_i + dxdt_n*0.5;
       double s_n2 = s_n_i + dsdt_n*0.5;
       double cpre2 = cpre_i + dcpre*0.5;
       double cpost2 = cpost_i + dcpost *0.5;
       double rho2  = rho_i + drho*0.5;
       if (rho2>1.0){
            rho2=1.0;
        }else if(rho2<0){
         rho2=0;
        }

        x_n[ix] += dxdt_n/6.0;
        s_n[ix] += dsdt_n/6.0;
        cpre[ix] += dcpre/6.0;
        cpost[ix] += dcpost/6.0;
        rho[ix] += drho/6.0;
         
         //k2
         dxdt_n = dxdt_n_cal(x_n2, mp_pre, dt );
        dsdt_n = dsdt_n_cal(x_n2, s_n2,dt);
        dcpre = dcpre_cal(cpre2, s_n2, tau_ca_pre, cpre_r, dt);
        dcpost = dcpost_cal(cpost2, mp_post, tau_ca_post, cpost_r, dt);
        drho = drho_cal(rho2, ca, th_p, th_d, gp, gd, sig, tau_s,dt, r);
   
        x_n2 = x_n_i + dxdt_n*0.5;
        s_n2 = s_n_i + dsdt_n*0.5;
        cpre2 = cpre_i + dcpre*0.5;
        cpost2 = cpost_i + dcpost*0.5;
        rho2  = rho_i + drho*0.5;
        if (rho2>1.0){
            rho2=1.0;
        }else if(rho2<0){
         rho2=0;
        }

        x_n[ix] += dxdt_n/3.0;
        s_n[ix] += dsdt_n/3.0;
        cpre[ix] += dcpre/3.0;
        cpost[ix] += dcpost/3.0;
        rho[ix] += drho/3.0;
        

       //k3
        dxdt_n = dxdt_n_cal(x_n2, mp_pre,dt);
        dsdt_n = dsdt_n_cal(x_n2, s_n2,dt);
        dcpre = dcpre_cal(cpre2, s_n2, tau_ca_pre, cpre_r, dt);
        dcpost = dcpost_cal(cpost2, mp_post, tau_ca_post, cpost_r, dt);
        drho = drho_cal(rho2, ca, th_p, th_d, gp, gd, sig, tau_s,dt, r);
   
        x_n2 = x_n_i + dxdt_n;
        s_n2 = s_n_i + dsdt_n;
        cpre2 = cpre_i + dcpre;
        cpost2 = cpost_i + dcpost;
        rho2  = rho_i + drho;
        if (rho2>1.0){
            rho2=1.0;
        }else if(rho2<0){
            rho2=0;
        }

        x_n[ix] += dxdt_n/3.0;
        s_n[ix] += dsdt_n/3.0;
        cpre[ix] += dcpre/3.0;
        cpost[ix] += dcpost/3.0;
        rho[ix] += drho/3.0;

        //k4 
        dxdt_n = dxdt_n_cal(x_n2, mp_pre,dt);
        dsdt_n = dsdt_n_cal(x_n2, s_n2,dt);
        dcpre = dcpre_cal(cpre2, s_n2, tau_ca_pre, cpre_r, dt);
        dcpost = dcpost_cal(cpost2, mp_post, tau_ca_post, cpost_r, dt);
        drho = drho_cal(rho2, ca, th_p, th_d, gp, gd, sig, tau_s, dt, r);
   
        x_n[ix] += dxdt_n/6.0;
        s_n[ix] += dsdt_n/6.0;
        cpre[ix] += dcpre/6.0;
        cpost[ix] += dcpost/6.0;
        rho[ix] += drho/6.0;
        if (rho[ix]>1.0){
            rho[ix]=1.0;
        }else if(rho[ix]<0){
            rho[ix]=0;
        }
        
        sum_rho += rho[ix];
      }
    //   printf("sum_rho: %f\n", sum_rho);
      d_sum_rho[ix] = sum_rho;
    }
}
    
void get_Hz_li(std::string sp_dir, std::vector<std::string> &Hz_li){
  DIR *dir; 
 struct dirent *diread;
    std::vector<char *> files;

    if ((dir = opendir(sp_dir.c_str())) != nullptr) {
        while ((diread = readdir(dir)) != nullptr) {
            files.push_back(diread->d_name);
        }
        closedir (dir);
    } else {
        perror ("opendir");
    }

    // for (auto file : files) std::cout << file << "| ";
    // std::cout << std::endl;
    for (auto file : files) {
        std::string file_s = file;  //char -> str
        // std::cout << file_s<< std::endl;
        if((file_s.find("post") != std::string::npos)&&(file_s.find("t0") != std::string::npos)){
                auto separator = std::string("_");         // 区切り文字
                auto separator_length = separator.length(); // 区切り文字の長さ
                auto list = std::vector<std::string>();
                if (separator_length == 0) {
                list.push_back(file_s);
                } else {
                auto offset = std::string::size_type(0);
                while (1) {
                    auto pos = file_s.find(separator, offset);
                    if (pos == std::string::npos) {
                    list.push_back(file_s.substr(offset));
                    break;
                    }
                    list.push_back(file_s.substr(offset, pos - offset));
                    offset = pos + separator_length;
                }
                }
                Hz_li.push_back(list[3]);
         }
        }
    //     for (auto Hz : Hz_li) {
    //     std::cout << Hz << std::endl;
    // }
}

std::string outF(std::string sp_dir_s, std::string dsb_s, std::string dt_str){
    DIR *dir; 
   struct dirent *diread;
    std::vector<char *> files;

    if ((dir = opendir(sp_dir_s.c_str())) != nullptr) {
        while ((diread = readdir(dir)) != nullptr) {
            files.push_back(diread->d_name);
        }
        closedir (dir);
    } else {
        perror ("opendir");
    }

    // for (auto file : files) std::cout << file << "| ";
    // std::cout << std::endl;
    std::string dsb_s2;
    for (auto file : files) {
        std::string file_s = file;
        // std::cout << file_s<< std::endl;
        if((file_s.find(dsb_s) != std::string::npos)&&(file_s.find(dt_str) != std::string::npos)){
            dsb_s2 = file_s;
            break;
        }
    }
    std::cout << dsb_s2 << std::endl;
    return dsb_s2;
}

int main(int argc, char * argv[]){
double dt = atof(argv[13]);
std::string dt_str = argv[13];
int sN = atoi(argv[11]);

#if (defined(MF) && defined(SLEEP))
    std::string sp_parent_dir = "./mp_spikes_mf/";

#else
    std::string sp_parent_dir = "./mp_spikes/";
#endif

std::cout << "sp_parent_dir: " << sp_parent_dir  << std::endl;

std::string lr_n= argv[9]; // #"STDP" 


std::string sw;// = "wake";  //"sleep"

#if defined(WAKE)
    sw="wake";
#elif defined(SLEEP)
    sw="sleep";
#else
    #error "Define WAKE or SLEEP"
#endif






 std::cout << sw  << std::endl;
std::string syn_fol = argv[14];

std::string order = argv[10];
 std::cout << "order" << order << std::endl;


std::string dsb_w = argv[15]; //"ISI_12_0.04_test";
std::string dsb_s = argv[16]; 
std::cout << "dsb_s: "<< dsb_s << std::endl;
// std::string dsb = dsb_s; //"ISI_12_0.04_test";

std::string lr_p=argv[17]; // #a0.7t70


std::string sp_dir_sw = sp_parent_dir+"/"+sw +"/";


#if defined(WAKE)
    std::string dsb = dsb_w;// outF(sp_dir_sw,  dsb_w, dt_str);
#elif defined(SLEEP)
    std::string dsb = dsb_s;//outF(sp_dir_sw,  dsb_s, dt_str);

#endif


std::cout << "dsb: " <<dsb << std::endl;


std::cout << lr_p  << std::endl;
// Np = int(argv[14])
 
std::string sp_dir = sp_parent_dir + sw+"/"+dsb+"/N_"+std::to_string(sN)+"/";

double th_p  =  atof(argv[1]);           
double th_d  = atof(argv[2]);               
double gp    = atof(argv[3]);             
double gd    = atof(argv[4]);           
double tau_ca_pre    = atof(argv[5]);             
double tau_ca_post  = atof(argv[6]);            
double sig    = atof(argv[7]);
double tau_s    = atof(argv[8]);

int seed = atoi(argv[18]);
srand(seed);
  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());
  // 平均0.0、標準偏差1.0で分布させる
  std::normal_distribution<> dist(0.0, 1.0);

int blockdim_x = atoi(argv[19]);

int ret;
std::string py_str = "python3 c_r.py "+ std::to_string(th_p)+" "+std::to_string(th_d)
           +" "+std::to_string(gp)+" "+std::to_string(gd)+" " + std::to_string(tau_ca_pre)+" "+std::to_string(tau_ca_post)
           + " " + std::to_string(sig)+" "+std::to_string(tau_s) +" "+order + " " + dt_str+" "+lr_n + " "+lr_p;
ret = std::system(py_str.c_str());
std::cout << "ret/cpp = " << ret << std::endl;
std::string c_r_f = "./c_r_"+lr_n+"_"+lr_p+"_"+order+".txt";

std::ifstream ifs(c_r_f);
    std::string str ="";
    if (ifs.fail()) {
        std::cerr << "Failed to open file." << std::endl;
    }
    int c = 0;
    double cpre_r = 0;
    double cpost_r = 0;

    while(getline(ifs,str)) {
          if (c==0){
              cpre_r = std::stof(str);
          }else{
              cpost_r = std::stof(str);
              }
        c += 1;
    }

    printf("cpre_r: %f\n", cpre_r);
    printf("cpost_r: %f\n", cpost_r);
    remove(c_r_f.c_str());



 

int dev = 0;
cudaDeviceProp deviceProp;
CHECK(cudaGetDeviceProperties(&deviceProp, dev));
printf("Using Device %d: %s\n", dev, deviceProp.name);
CHECK(cudaSetDevice(dev));
double iStart = cpuSecond();

//  int T, tnum, Tp, Lt, Ltp, avet;


#if defined(AUTO_HZ)
    std::vector<std::string> Hz_li;
    get_Hz_li(sp_dir, Hz_li);
    for (auto Hz : Hz_li) {
            std::cout << Hz << std::endl;
        }
#elif defined(TO3HZ)
        std::vector<std::string> Hz_li{"0.10", "0.30", "1.0", "3.0"};
   
#elif defined(HIGH_HZ)
    std::vector<std::string> Hz_li{"20.0", "30.0"};
#elif defined(MF)
    std::vector<std::string> Hz_li{"0.01", "0.03", "0.10", "0.30", "1.0", "3.0", "10.0"};

#else
    std::vector<std::string> Hz_li{"0.10", "0.30", "1.0", "3.0", "10.0"};
#endif


int Tp = atoi(argv[12]);


// int T; //#ms





// int Lt = T/dt;
// //Tp =60000; //ms
// int tnum = T/Tp;
int Ltp = Tp/dt;
int avet = 2;

#pragma omp parallel for
for (int m = 0; m < Hz_li.size(); m++){
    std::string fr= Hz_li[m];
    std::cout << fr << "Hz" << std::endl;

    int b_arg ;
    int b_argT;
    int b_argT2;


    int T; //#ms
    int Lt ;
    int tnum ;


    if (fr=="0.01"){
     T=2160000; //#ms

    }else if (fr=="0.03")
    {
       T=1080000;//#ms
    }else{
        T=360000; //#ms
    }
    

      Lt = T/dt;
     tnum = T/Tp;
    
     b_arg = sN* sizeof(double);
    b_argT = sN*Ltp* sizeof(double);
    b_argT2 = Ltp* sizeof(double);

    double *x_n, *s_n, *cpre, *cpost, *rho, *mp_pres_it, *mp_post_it, *rd, *sum_rho;
    // double sum_rho =0;
    x_n = (double*)malloc(b_arg);
    s_n = (double*)malloc(b_arg);
    cpre = (double*)malloc(b_arg);
    cpost = (double*)malloc(b_arg);
    rho = (double*)malloc(b_arg);
    sum_rho = (double*)malloc(b_arg);
    rd = (double*)malloc(b_argT);
    mp_pres_it =  (double*)malloc(b_argT);
    mp_post_it =  (double*)malloc(b_argT2);

    for (int i = 0; i < sN; i++) {
        // double n_r = (double)rand()/RAND_MAX;
        // double ca_r = (double)rand()/RAND_MAX*(10);
        //printf("v_r:%f \n", v_r);
        x_n[i]= 0.01;// v_r;  //:v
        s_n[i]= 0.01;
        cpre[i] = 0;//n_r;  //nk
        cpost[i] =0;
        rho[i]= 0.5; //ca_r;  //c
    }

  
    //int b_k = k1.size() * sizeof(double);
    // printf(" size_args: %d \n", b_arg);  //ok

        // malloc device global memory
    double *d_x_n, *d_s_n, *d_cpre, *d_cpost, *d_rho, *d_mp_pres, *d_mp_post, *d_rd, *d_sum_rho;
    CHECK(cudaMalloc((void **)&d_x_n, b_arg));
    CHECK(cudaMalloc((void **)&d_s_n, b_arg));
    CHECK(cudaMalloc((void **)&d_cpre, b_arg));
    CHECK(cudaMalloc((void **)&d_cpost, b_arg));
    CHECK(cudaMalloc((void **)&d_rho, b_arg));
    CHECK(cudaMalloc((void **)&d_sum_rho, b_arg));
    CHECK(cudaMalloc((void **)&d_mp_pres, b_argT));
    CHECK(cudaMalloc((void **)&d_mp_post, b_argT2));
    CHECK(cudaMalloc((void **)&d_rd, b_argT));

double rho_sums = 0;
CHECK(cudaMemcpy(d_x_n, x_n, b_arg, cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(d_s_n, s_n, b_arg, cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(d_cpre, cpre, b_arg, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_cpost, cpost, b_arg, cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(d_rho, rho, b_arg, cudaMemcpyHostToDevice));
          
for (int tc = 0; tc < tnum; tc++) {
    printf("tc: %d\n", tc);

    std::vector<int> s;
    std::vector<double> mp_pres;
    std::string mp_pre_f = sp_dir +"mp_pres_"+sw+"_"+fr+"_n0t"+std::to_string(tc)+".npy";
    // std::cout << mp_pre_f<< std::endl;
    aoba::LoadArrayFromNumpy(mp_pre_f, s, mp_pres);
    // std::cout << mp_pre_f<< std::endl; // 4 5
   // std::cout << mp_pres.size() << std::endl; // 20

    std::vector<int> s1;
    std::vector<double> mp_post;
    std::string mp_post_f = sp_dir +"mp_post_"+sw+"_"+fr+"_t"+std::to_string(tc)+".npy";
    // std::cout << mp_post_f<< std::endl;
    aoba::LoadArrayFromNumpy(mp_post_f, s1, mp_post);
    // std::cout << s1[0] << " " << s[1] << std::endl; // 4 5
   // std::cout << mp_post.size() << std::endl; // 20
    
    // printf("aa\n");

    for (int it=0; it<Ltp; it++){
    //  if (it != 0){
            // printf("it: %d\n", it);
        
        for (int n=0; n<sN; n++){
            // printf("n: %d\n", n);
            // mp_pres_it[n*Lt+it] = mp_pres[n*Lt+it];
            double r = dist(engine);
             rd[n*Ltp+it]= r;
            //   printf("random: %f\n", r);
            }
            // mp_post_it[it] = mp_post[it];
        }
    

            CHECK(cudaMemcpy(d_mp_pres, mp_pres.data(), b_argT, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_mp_post, mp_post.data(), b_argT2, cudaMemcpyHostToDevice));
           CHECK(cudaMemcpy(d_rd, rd, b_argT, cudaMemcpyHostToDevice));
            
            // printf("bb\n");

            // invoke kernel at host side
            int dimx = sN/blockdim_x;
            int dimy = 1/blockdim_y;
            dim3 block(blockdim_x, blockdim_y, 1);
            dim3 grid(dimx, dimy);
            // dim3 grid((nx + block.x - 1) / block.x,(ny + block.y - 1) / block.y, nz);

            dt_gpu<<<grid, block>>>(d_x_n, d_s_n, d_cpre, d_cpost, d_rho, sN, Ltp, d_sum_rho, dt, d_mp_pres, d_mp_post, d_rd, cpre_r, cpost_r, 
                                     th_p, th_d, gp, gd, tau_ca_pre, tau_ca_post, sig, tau_s);
            CHECK(cudaDeviceSynchronize());
        
            // printf("dt_gpu <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y);
            // check kernel error
            CHECK(cudaGetLastError());

           
            // }
            if(tc>=tnum-avet){
                CHECK(cudaMemcpy(sum_rho, d_sum_rho, b_arg, cudaMemcpyDeviceToHost));
         
                for (int i = 0; i < sN; i++){
                rho_sums += sum_rho[i];
                }
              
            }


       } //if tc ==0
    
    CHECK(cudaFree(d_x_n));
    CHECK(cudaFree(d_s_n));
    CHECK(cudaFree(d_cpre));
     CHECK(cudaFree(d_cpost));
    CHECK(cudaFree(d_rho));
    CHECK(cudaFree(d_rd));
    CHECK(cudaFree(d_mp_pres));
    CHECK(cudaFree(d_mp_post));
    CHECK(cudaFree(d_sum_rho));

    free(x_n);
    free(s_n);
    free(cpre);
    free(cpost);
    free(rho);
    free(rd);
    free(mp_pres_it);
    free(mp_post_it);
    free(sum_rho);
     

     float effi_mean =  rho_sums/(sN*2*Ltp);
     printf("mean_effi: %f\n",effi_mean);
     std::string savedir =   "./"+ syn_fol+"/" + dsb +"/" +lr_p+"/"+"N_"+std::to_string(sN) +"/";
     std::string savef = savedir +  lr_n + "_" + sw +fr+".csv";
     
     if (std::filesystem::exists(savef)){
        std::ofstream ofs;
        ofs.open(savef, std::ios::app);
   
        
            ofs << order;
            ofs <<",";
            ofs << effi_mean;
            // if(i==0){ 
            //     printf("v_: %f\n", v_[it]);
            //     }
            
            ofs << std::endl;
            std::cout << "add" << savef << std::endl;
     }else{
          std::filesystem::create_directories(savedir);
           std::ofstream ofs;
           ofs.open(savef, std::ios::out);
        
            ofs << order;
            ofs <<",";
            ofs << effi_mean;
            // if(i==0){ 
            //     printf("v_: %f\n", v_[it]);
            //     }
            
            ofs << std::endl;
           
            std::cout << "create" << savef << std::endl;
     }

}//for Hz_li
    double iElaps = cpuSecond() - iStart;
    printf("elapsed %f sec\n", iElaps);
    
    
  
}