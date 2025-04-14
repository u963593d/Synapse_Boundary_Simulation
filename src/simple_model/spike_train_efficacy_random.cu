#include "common.h"
#include <cuda_runtime.h>
// #include <cusparse_v2.h>
#include <cuda.h>
#include <stdio.h>
//#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <sstream>
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
#include <curand.h>

//nvc++ -mp -fast -ta=tesla spike_train_efficacy_random.cu -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/examples/OpenACC/SDK/include/ -I/tmp/nvhpc_2023_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/include -L/tmp/nvhpc_2023_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/lib -DWAKE -lcudart -std=c++20 -o effi_G_r_w
//nvc++ -mp -fast -ta=tesla spike_train_efficacy_random.cu -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/examples/OpenACC/SDK/include/ -I/tmp/nvhpc_2023_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/include -L/tmp/nvhpc_2023_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/lib -DSLEEP -lcudart -std=c++20 -o effi_G_r_s


// ./effi_G_r_w 1.1624237569297997 0.702631293634435 570.0727831475216 161.1929938555061 36.685471308355034 20.358295150931358 1.9876849699667516 270275.77154806757 STDP 0 10 60000 0.05 synv_r ISI_40_0.05 IBI_45_0.05 a0.7t70 0.8 6 2
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
// #define g_nmdar;// = 4.818*0.9797198*10/7
// //     double g_cav_s ;//= 0.0015355*1.0010142646448208*10

//#define g_leak ;//= 0.03022
//     double g_kvhh;//= 2.5277
//     double g_cav;//= 0.69474
//     double g_kca;//= 0.10539
//     double g_nap;//= 0.34895 
//     double g_ampar;//= 3*2
//     double tau_ca;// =62.421

//define equation



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
       double dcpost = (- (a_ca * 10*area*(i_vdcc_s))*cpost_r - cpost/tau_ca_post)*dt;
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

__global__ void dt_gpu(double *x_n, double *s_n, double *cpre, double *cpost, double *rho, int N, int T, double dt,
    double *mp_pres, double *mp_posts, double *rd, double cpre_r, double cpost_r, double th_p, double th_d, double gp, double gd, double tau_ca_pre, double tau_ca_post, double sig, double tau_s)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    // unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    //unsigned int idx = iy * nx + ix;
 
    if (ix < N)
    {  
       for (int it = 1; it < T; it++) {
      
        double x_n_i = x_n[ix];
        double s_n_i = s_n[ix];
        double cpre_i = cpre[ix];
        double cpost_i = cpost[ix];
        double rho_i = rho[ix*T+it-1];
        double ca = cpre_i + cpost_i;
        double mp_pre = mp_pres[ix*(T-1)+it-1];
        double mp_post= mp_posts[ix*(T-1)+it-1];
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
        rho[ix*T+it] = rho[ix*T+it-1]+drho/6.0;
         
         //k2
         ca = cpre2 + cpost2;
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
        rho[ix*T+it] +=drho/3.0;
        

       //k3
       ca = cpre2 + cpost2;
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
        rho[ix*T+it] += drho/3.0;

        //k4 
        ca = cpre2 + cpost2;
        dxdt_n = dxdt_n_cal(x_n2, mp_pre,dt);
        dsdt_n = dsdt_n_cal(x_n2, s_n2,dt);
        dcpre = dcpre_cal(cpre2, s_n2, tau_ca_pre, cpre_r, dt);
        dcpost = dcpost_cal(cpost2, mp_post, tau_ca_post, cpost_r, dt);
        drho = drho_cal(rho2, ca, th_p, th_d, gp, gd, sig, tau_s, dt, r);
   
        x_n[ix] += dxdt_n/6.0;
        s_n[ix] += dsdt_n/6.0;
        cpre[ix] += dcpre/6.0;
        cpost[ix] += dcpost/6.0;
        rho[ix*T+it] += drho/6.0;
        if (rho[ix*T+it]>1.0){
            rho[ix*T+it]=1.0;
        }else if(rho[ix*T+it]<0){
            rho[ix*T+it]=0;
        }
        
       
      }
    //   printf("sum_rho: %f\n", sum_rho);
      
    }
}





int main(int argc, char * argv[]){
double dt = atof(argv[13]);
std::string dt_str = argv[13];
int NE = atoi(argv[11]);

std::string sp_parent_dir = "./mp_spikes/";
std::string lr_n= argv[9]; // #"STDP" 
std::string sw;  //"sleep"


#if defined(WAKE)
    sw="wake";
#elif defined(SLEEP)
    sw="sleep";
#else
    #error "Define WAKE or SLEEP"
#endif

 std::cout << sw  << std::endl;

std::string syn_fol = argv[14];  //synv_r_0.8
std::vector<std::string> con_ss;
std::stringstream ss{syn_fol};
std::string buf;
while (std::getline(ss, buf, '_')) {
  con_ss.push_back(buf);
}
std::string con_th_str = con_ss[2];
double con_th =atof(con_th_str.c_str());
printf("con_th %f\n", con_th);

std::string order = argv[10];
 std::cout << "order" << order << std::endl;

std::string dsb_w = argv[15]; //"ISI_12_0.04_test";
std::string dsb_s = argv[16]; 

#if defined(WAKE)
    std::string dsb = dsb_w;
#elif defined(SLEEP)
    std::string dsb = dsb_s;

#endif

std::string lr_p=argv[17]; // #a0.7t70



// Np = int(argv[14])
 
std::string sp_dir = sp_parent_dir + sw+"/"+dsb+"/N_"+std::to_string(NE)+"/";

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


//make connectgion matrix
    std::vector<std::vector<int>> neuron_cons(NE, std::vector<int>(NE));
    int* neuron_syN = (int*)malloc(sizeof(int)*NE);  //array for the number of synapses per neuron
    
    if (con_th>1)
    { //homogenous con
        int con_th_i = con_th;
        for (int i = 0; i < NE; i++) 
       {
           for (int n = 0; n < con_th_i; n++) 
          { 
            while(1){

            double a = (double)rand()/RAND_MAX;
            int j = a*NE -0.001; //0~NE-1 random
            if ((i!=j) && (neuron_cons[i][j] != 1) ){
                    neuron_cons[i][j] = 1;
                    break;
                }
              }
            }
          }
        }else{//random connection
                    
            for (int i = 0; i < NE; i++) {
                for (int j = 0; j < NE; j++) {
                    if (i==j){
                        neuron_cons[i][j] = 0;
                    }
                    else{
                        double a = (double)rand()/RAND_MAX;
                        
                        if (a > con_th){
                        neuron_cons[i][j] = 1;
                        }
                    }
                }
            }
        }

    //output connection matrix
    // for (int i = 0; i < NE; i++) {
    //     for (int j = 0; j < NE; j++) {
    //          printf("neuron_cons: %d\n"  , neuron_cons[i][j]);
    //     }
    // }

    //make array for the number of synapses per neuron
    for (int i = 0; i < NE; i++) {
        int syN = 0;
        for (int j = 0; j < NE; j++) 
        {
        if (neuron_cons[i][j] ==1){
            syN += 1;
        }
    }
    neuron_syN[i]=syN;
//    printf("neuron%d, num of synapses: %d\n"  , i, syN);
    }

    //calculate total synapses
    int sN =0;
    for (int i = 0; i < NE; i++) {
        sN += neuron_syN[i];
        }
    printf("total synapses: %d\n", sN);

    
blockdim_x = sN;

int Tp = atoi(argv[12]);
int T = 360000; //#ms
int Lt = T/dt;
int tnum = T/Tp;
int Ltp = Tp/dt+1;
int avet = 2;  //

 int  b_arg = sN* sizeof(double);
   int b_argT = sN*Ltp* sizeof(double);
  int  b_argT_pre = sN*(Ltp-1)* sizeof(double);
  int  b_argT2 = (Ltp-1)* sizeof(double);

int dev = 0;
cudaDeviceProp deviceProp;
CHECK(cudaGetDeviceProperties(&deviceProp, dev));
printf("Using Device %d: %s\n", dev, deviceProp.name);
CHECK(cudaSetDevice(dev));
double iStart = cpuSecond();


std::vector<std::string> Hz_li{"0.10", "0.30", "1.0", "3.0", "10.0"};
 Tp = atoi(argv[12]);

//calc by Hz
#pragma omp parallel for
for (int m = 0; m < Hz_li.size(); m++){
    std::string fr= Hz_li[m];
    std::cout << fr << "Hz" << std::endl;
    std::string savedir_raw =   "./"+ syn_fol+"_raw"+"/" + dsb +"/" +lr_p+"/"+"N_"+std::to_string(NE) +"/"+lr_n+"/"+fr+"/"+order+"/";
    std::filesystem::create_directories(savedir_raw);
    

    double *x_n, *s_n, *cpre, *cpost, *rho, *mp_pres, *mp_posts, *rd;
    // double sum_rho =0;
    x_n = (double*)malloc(b_arg);
    s_n = (double*)malloc(b_arg);
    cpre = (double*)malloc(b_arg);
    cpost = (double*)malloc(b_arg);
    rho = (double*)malloc(b_argT);
    // sum_rho = (double*)malloc(b_arg);
    rd = (double*)malloc(b_argT);
    mp_pres =  (double*)malloc(b_argT_pre);
    mp_posts =  (double*)malloc(b_argT_pre);

    for (int i = 0; i < sN; i++) {
        // double n_r = (double)rand()/RAND_MAX;
        // double ca_r = (double)rand()/RAND_MAX*(10);
        //printf("v_r:%f \n", v_r);
        x_n[i]= 0.01;// v_r;  //:v
        s_n[i]= 0.01;
        cpre[i] = 0;//n_r;  //nk
        cpost[i] =0;
        for (int it=0; it<Ltp; it++){
           rho[i*Ltp+it]= 0.5; //ca_r;  //c
        }
    }

    double mean_nt=0;
    double CV_nt=0;
  
    //int b_k = k1.size() * sizeof(double);
    // printf(" size_args: %d \n", b_arg);  //ok

        // malloc device global memory
    double *d_x_n, *d_s_n, *d_cpre, *d_cpost, *d_rho, *d_mp_pres, *d_mp_posts, *d_rd;
    CHECK(cudaMalloc((void **)&d_x_n, b_arg));
    CHECK(cudaMalloc((void **)&d_s_n, b_arg));
    CHECK(cudaMalloc((void **)&d_cpre, b_arg));
    CHECK(cudaMalloc((void **)&d_cpost, b_arg));
    CHECK(cudaMalloc((void **)&d_rho, b_argT));
    // CHECK(cudaMalloc((void **)&d_sum_rho, b_arg));
    CHECK(cudaMalloc((void **)&d_mp_pres, b_argT_pre));
    CHECK(cudaMalloc((void **)&d_mp_posts, b_argT_pre));
    CHECK(cudaMalloc((void **)&d_rd, b_argT));


CHECK(cudaMemcpy(d_x_n, x_n, b_arg, cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(d_s_n, s_n, b_arg, cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(d_cpre, cpre, b_arg, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_cpost, cpost, b_arg, cudaMemcpyHostToDevice));

for (int tc = 0; tc < tnum; tc++) {
    printf("tc: %d\n", tc);
    
//read spike train waveform
    std::vector<int> s;
    std::vector<double> mps;
    std::string mp_f = sp_dir +"mp_pres_"+sw+"_"+fr+"_n0t"+std::to_string(tc)+".npy";
    // std::cout << mp_pre_f<< std::endl;
    aoba::LoadArrayFromNumpy(mp_f, s, mps);
    // std::cout << mp_pre_f<< std::endl; // 4 5
   // std::cout << mp_pres.size() << std::endl; // 20
    
    int sc = 0; 
    for (int i = 0; i < NE; i++) {
        std::vector<int> con_pair(NE,0);
        con_pair = neuron_cons[i];
        for (int j = 0; j < NE; j++) 
         {  
            
            if (con_pair[j]==1){
                for (int it=0; it<Ltp-1; it++){
                  mp_pres[sc*Ltp + it] = mps[i*Ltp+it];
                  mp_posts[sc*Ltp + it] = mps[j*Ltp+it];
                  double r = dist(engine);
                  rd[sc*Ltp+it]= r;
              }
              sc +=1; 
            }
         }
    }
    // printf("sc_end: %d\n",sc);

        //   printf("aa\n");
       
            // printf("it:%d\n",it);

            // transfer data from host to device
     
            CHECK(cudaMemcpy(d_rho, rho, b_argT, cudaMemcpyHostToDevice));
            // CHECK(cudaMemcpy(d_sum_rho, sum_rho, b_arg, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_mp_pres, mp_pres, b_argT_pre, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_mp_posts, mp_posts, b_argT_pre, cudaMemcpyHostToDevice));
           CHECK(cudaMemcpy(d_rd, rd, b_argT, cudaMemcpyHostToDevice));
            
            // printf("bb\n");

            // invoke kernel at host side
            int dimx = sN/blockdim_x;
            int dimy = 1/blockdim_y;
            dim3 block(blockdim_x, blockdim_y, 1);
            dim3 grid(dimx, dimy);
            // dim3 grid((nx + block.x - 1) / block.x,(ny + block.y - 1) / block.y, nz);

            dt_gpu<<<grid, block>>>(d_x_n, d_s_n, d_cpre, d_cpost, d_rho, sN, Ltp, dt, d_mp_pres, d_mp_posts, d_rd, cpre_r, cpost_r, 
                                     th_p, th_d, gp, gd, tau_ca_pre, tau_ca_post, sig, tau_s);
            CHECK(cudaDeviceSynchronize());
        
            // printf("dt_gpu <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y);
            // check kernel error
            CHECK(cudaGetLastError());

            // copy kernel result back to host side
           
            CHECK(cudaMemcpy(rho, d_rho, b_argT, cudaMemcpyDeviceToHost));
            
           
            if(tc>=tnum-avet){
                double rho_sum_t=0; 
                double CV_sum_t = 0;
                for (int it=0; it<Ltp-1; it++){
                    
                    double mean_n =0;
                    double rho_sum_n=0; 
                    double CV_n =0;
                    double var_sum_n = 0;
                    for (int i=0; i<sN; i++){
                            rho_sum_n += rho[i*Ltp+it];
                    }
                    mean_n=rho_sum_n/((double)sN);
                    // if((it<5)){
                    // printf("it %d, %f\n", it, rho_sum_n);
                    // }
                    rho_sum_t += mean_n;

                    for (int i=0; i<sN; i++){
                            var_sum_n += (mean_n-rho[i*Ltp+it])*(mean_n-rho[i*Ltp+it]);
                    }
                    CV_n = std::sqrt((var_sum_n/sN))/mean_n;
                    CV_sum_t += CV_n;
                }
                mean_nt += rho_sum_t/(Ltp-1);
                CV_nt += CV_sum_t/(Ltp-1);
            }

           

        for (int i = 0; i < sN; i++) {
            rho[i*Ltp] = rho[i*Ltp + Ltp-1];
            // if(i==0){
            // std::cout << fr << tc<<"flip:" <<  rho[i*Ltp + Ltp-1] << std::endl;
            // }
         }

       } //if tc ==0
    
    CHECK(cudaFree(d_x_n));
    CHECK(cudaFree(d_s_n));
    CHECK(cudaFree(d_cpre));
     CHECK(cudaFree(d_cpost));
    CHECK(cudaFree(d_rho));
    CHECK(cudaFree(d_rd));
    CHECK(cudaFree(d_mp_pres));
    CHECK(cudaFree(d_mp_posts));
   

    free(x_n);
    free(s_n);
    free(cpre);
    free(cpost);
    free(rho);
    free(rd);
    free(mp_pres);
    free(mp_posts);

   mean_nt=mean_nt/(double(avet));
     CV_nt=CV_nt/(double(avet));
    //  printf("mean_effi: %f\n",mean_nt);
     std::cout << fr << "mean_effi" <<mean_nt<< std::endl;
    //  printf("mean_CV: %f\n",CV_nt);
     std::cout << fr << "CV_effi" <<CV_nt<< std::endl;
     std::string savedir =   "./"+ syn_fol+"/" + dsb +"/" +lr_p+"/"+"N_"+std::to_string(NE) +"/";
     std::string savef = savedir +  lr_n + "_" + sw +fr+".csv";
     std::string savef_cv = savedir +  lr_n + "_" + sw +fr+"_cv.csv";
     
     if (std::filesystem::exists(savef)){
        std::ofstream ofs;
        ofs.open(savef, std::ios::app);
        
            ofs << order;
            ofs <<",";
            ofs << mean_nt;
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
            ofs << mean_nt;
            
            ofs << std::endl;
           
            std::cout << "create" << savef << std::endl;
     }


     if (std::filesystem::exists(savef_cv)){
        std::ofstream ofs;
        ofs.open(savef_cv, std::ios::app);
            ofs << order;
            ofs <<",";
            ofs << CV_nt;
     
            ofs << std::endl;
            std::cout << "add" << savef_cv << std::endl;
     }else{
          std::filesystem::create_directories(savedir);
           std::ofstream ofs;
           ofs.open(savef_cv, std::ios::out);
        
            ofs << order;
            ofs <<",";
            ofs << CV_nt;
            // if(i==0){ 
            //     printf("v_: %f\n", v_[it]);
            //     }
            
            ofs << std::endl;
           
            std::cout << "create" << savef_cv << std::endl;
     }

}//for Hz_li
    free(neuron_syN);
    double iElaps = cpuSecond() - iStart;
    printf("elapsed %f sec\n", iElaps);
    

}