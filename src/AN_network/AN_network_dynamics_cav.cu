#define CON_LOGNORMAL

#define NO_IN_IN
// #define TWO_N
// #define a_UP
#define RANDOM
#define G_MIN 0


#define CA_SYN_CHANGE


 
#include <cuda_runtime.h>
#include <cuda.h>    
#include "common.h"  //-I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/examples/OpenACC/SDK/include/    ~.h

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits.h>
#include <vector>
#include <string>
#include <array>
#include <math.h>
#include <tuple>
#include <time.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdlib.h>

#include <filesystem>  
#include <random>
#include <omp.h>  //OpenMP

//nvc++ -ta=tesla AN_network_dynamics_cav.cu -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/examples/OpenACC/SDK/include/ -I/tmp/nvhpc_2022_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/include -L/tmp/nvhpc_2022_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/lib -lcudart -std=c++20 -o AN_net_aup_cav
//nvc++ -ta=tesla AN_network_dynamics_cav.cu -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/examples/OpenACC/SDK/include/ -I/tmp/nvhpc_2022_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/include -L/tmp/nvhpc_2022_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/lib -lcudart -std=c++20 -o AN_net_rup_cav

#define SAFE_FREE(ptr) if(ptr != NULL){free(ptr); ptr = NULL;}
#define SIZE(buff) (sizeof(buff)/sizeof(buff))


//define neuron parameter 
#define cm 1.0
#define area 0.02
#define a_ca 0.5
#define kd_ca 30.0
#define tau_h 15.0
#define s_a_ampar 3.48
#define s_tau_ampar 2.0
#define s_a_nmdar 5 
#define s_tau_nmdar 10
 #define x_a_nmdar 34.8
 #define x_tau_nmdar 0.2



#define s_a_gabar 1
#define s_tau_gabar 10
#define vL -60.95
#define vNa 55.0
#define vK -100.0
#define vCa 120.0
#define vrest -70
#define vAMPAR 0
#define vNMDAR 0
#define vGABAR -70

#define i_leak(g_leak, v) (g_leak * (v - vL))
#define i_kvhh(g_kvhh, v, nk) (g_kvhh * std::pow(nk,4) * (v-vK))
#define m_inf_kv(v) (1.0 / (1.0 + std::exp(-(v+50.0)/20.0)))
#define i_kva(g_kva, v, h) (g_kva*(std::pow(m_inf_kv(v), 3) * h * (v - vK)))
#define m_inf_ca(v) (1.0 / (1.0 + std::exp(-(v+20.0)/9.0)))
// #define m_inf_ca2(v) (1.0 / (1.0 + std::exp(-(v)/9.0)))
#define i_cav(g_cav, v) (g_cav*(std::pow(m_inf_ca(v), 2) * (v-vCa)))
#define m_inf_k(ca) (1.0 / (1.0 + std::pow((kd_ca/ca), 3.5)))
#define i_kca(g_kca, v, ca) (g_kca*(m_inf_k(ca)* (v-vK)))
#define m_inf_na(v) (1.0 / (1.0 + std::exp(-(v+55.7)/7.7)))
#define i_nap(g_nap, v) (g_nap*(std::pow(m_inf_na(v), 3) * (v-vNa)))

#define m_tau(v) (8.0 / (std::exp(-(v+55.0)/30.0) + std::exp((v+55.0)/30.0)))
#define m_inf_kvsi(v) (1.0 / (1.0 + std::exp(-(v+34.0)/6.5)))
#define i_kvsi(g_kvsi, v, m) (g_kvsi * m * (v-vK))
#define h_inf_kir(v) (1.0/(1.0 + std::exp((v + 75.0)/4.0)))
#define i_kir(g_kir, v) (g_kir*h_inf_kir(v)*(v-vK))

#define an_e(v) (0.01 * (v+34.0) / (1.0-std::exp(-(v+34.0)/10.0)))
#define bn_e(v) (0.125 * std::exp(-(v+44.0)/25.0))
#define h_inf(v) (1.0 / (1.0 + std::exp((v+80.0)/6.0)))
#define fv(v) (1.0 / (1.0 + std::exp(-(v-20)/2.0)))
#define ah(v) (0.07 * std::exp(-(v+50.0)/10.0))
#define bh(v) (1.0 / (1.0 + std::exp(-(v+20.0)/10.0)))
#define mg_v(v) ((1/(1+std::exp(-0.062*v)*1.0/3.57)))

#define i_nmda_c(g_nmdar, g_n, s, v) (g_nmdar * g_n * s * (v-vNMDAR))
#define i_nmda_s(g_nmdar, g_n, s, v) (g_nmdar * g_n * s * (v-vCa))
#define i_vdcc(g_cav, v) (g_cav * std::pow(m_inf_ca(v),2) * (v-vCa))



//CUDA error check
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

__device__ double Nav_i_cal(double g,double v, double h){
    double am = 1.;
    if (v == -33.){
        am = 1.;
        }
    else{
        am = 0.1 * (v+33.0) / (1.0-std::exp(-(v+33.0)/10.0));
      }
    double bm = 4.0 * std::exp(-(v+53.7)/12.0);
    double m_inf = am / (am + bm);
    double  i = g * (std::pow(m_inf, 3)) * h * (v-vNa);
    return i;
}



__device__ double dvdt_cal(double v, double h_na, double nk, double h,  double ca, double dt, double i_ampar, double g_leak, double g_nav, double g_kvhh,double g_kva, double g_cav,double g_kca,double g_nap,double g_ampar,double tau_ca)
{    double i_nav = Nav_i_cal(g_nav, v, h_na);
    double dvdt = ((-10.0*area * (i_leak(g_leak, v) + i_nav +  i_kvhh(g_kvhh, v, nk)  + i_kva(g_kva, v, h) + i_cav(g_cav, v)  + i_kca(g_kca, v, ca)  + i_nap(g_nap, v))  - (i_ampar)) / (10.0*cm*area))*dt;
   return dvdt;
}

__device__ double dvdt_cal_NaK1(double v, double h_na, double nk, double h, double m, double ca, double dt, double i_ampar, double i_gabar, double g_leak, double g_nav, double g_kvhh,double g_kva,double g_kvsi, double g_cav,double g_kca,double g_nap, double g_kir, double tau_ca)
{    double i_nav = Nav_i_cal(g_nav, v, h_na);
    double dvdt = ((-10.0*area * (i_leak(g_leak, v) + i_nav +  i_kvhh(g_kvhh, v, nk)  + i_kva(g_kva, v, h) + i_kvsi(g_kvsi, v, m) + i_cav(g_cav, v)  + i_kca(g_kca, v, ca)  + i_nap(g_nap, v) + i_kir(g_kir, v))  - (i_ampar+i_gabar)) / (10.0*cm*area))*dt;
   return dvdt;
}

__device__ double dvdt_cal_NaK1_n(double v, double h_na, double nk, double h, double m, double ca, double dt, double i_ampar, double i_nmdar, double i_gabar, double g_leak, double g_nav, double g_kvhh,double g_kva,double g_kvsi, double g_cav,double g_kca,double g_nap, double g_kir, double tau_ca)
{    double i_nav = Nav_i_cal(g_nav, v, h_na);
    double dvdt = ((-10.0*area * (i_leak(g_leak, v) + i_nav +  i_kvhh(g_kvhh, v, nk)  + i_kva(g_kva, v, h) + i_kvsi(g_kvsi, v, m) + i_cav(g_cav, v)  + i_kca(g_kca, v, ca)  + i_nap(g_nap, v) + i_kir(g_kir, v))  - (i_ampar+i_nmdar+i_gabar)) / (10.0*cm*area))*dt;
   return dvdt;
}

__device__ double dvdt_cal_NaK5(double v, double h_na, double nk, double h,  double ca, double dt, double i_ampar, double i_gabar, double g_leak, double g_nav, double g_kvhh,double g_kva, double g_cav,double g_kca,double g_nap,double tau_ca)
{   double i_nav = Nav_i_cal(g_nav, v, h_na);
    double dvdt = ((-10.0*area * (i_leak(g_leak, v) + i_nav +  i_kvhh(g_kvhh, v, nk)  + i_kva(g_kva, v, h) + i_cav(g_cav, v)  + i_kca(g_kca, v, ca)  + i_nap(g_nap, v))  - (i_ampar + i_gabar)) / (10.0*cm*area))*dt;
    return dvdt;
}

__device__ double dvdt_cal_NaK7(double v, double h_na, double nk,  double ca, double dt, double i_ampar, double i_gabar, double g_leak, double g_nav, double g_kvhh, double g_cav,double g_kca,double g_nap,double g_kir,double tau_ca)
{   double i_nav = Nav_i_cal(g_nav, v, h_na);
    double dvdt = ((-10.0*area * (i_leak(g_leak, v) + i_nav +  i_kvhh(g_kvhh, v, nk)  + i_cav(g_cav, v)  + i_kca(g_kca, v, ca)  + i_nap(g_nap, v)+ i_kir(g_kir, v))  - (i_ampar + i_gabar)) / (10.0*cm*area))*dt;
    return dvdt;
}



__device__ double dcadt_cal(double v, double ca, double dt, double g_cav, double tau_ca){
    // double m_inf_ca = 1.0 / (1.0 + std::exp(-(v+20.0)/9.0));
    // double i_cav= g_cav * i_cav_e(v);
    double dcadt = (-a_ca * 10 * area * i_cav(g_cav, v) - ca/tau_ca)*dt;
    return dcadt;
}
__device__ double dcadt_cal_n(double i_vdcc, double ca, double dt, double g_cav, double tau_ca){
    // double m_inf_ca = 1.0 / (1.0 + std::exp(-(v+20.0)/9.0));
    // double i_cav= g_cav * i_cav_e(v);
    double dcadt = (-a_ca * 10 * area * i_vdcc - ca/tau_ca)*dt;
    return dcadt;
}

__device__ double dhdt_na_cal(double v,double h,double dt){
double dhdt = 4.0 * (ah(v)*(1-h) - bh(v)*h)*dt;
return dhdt;
}

__device__ double dmdt_cal(double v,double m,double dt){
double dmdt = (m_inf_kvsi(v)-m)/m_tau(v)*dt;
return dmdt;
}

__device__ double dndt_cal(double v,double nk,double dt){
    double an = 0.1;
    if (v == -34.0){
        an= 0.1;
    }
    else{
        an= an_e(v);
    }
    double bn = bn_e(v);
        double dndt =  (4.0 * (an*(1-nk)-bn*nk))*dt;
     return dndt;
}

__device__ double dhdt_cal(double v, double h, double dt){
    // double h_inf = 1.0 / (1.0 + std::exp((v+80.0)/6.0));
    double dhdt = ((h_inf(v) - h) / tau_h) * dt;
    return dhdt;
}

__device__ double dsdt_a_cal(double v, double s_a, double dt){
        // double fv = 1.0 / (1.0 + std::exp(-(v-20)/2.0));
        double dsdt_a = (s_a_ampar * fv(v) - s_a/s_tau_ampar)*dt;
     return dsdt_a;
}


__device__ double dsdt_g_cal(double v, double s_g, double dt){
        // double fv = 1.0 / (1.0 + std::exp(-(v-20)/2.0));
        double dsdt_g = (s_a_gabar * fv(v) - s_g/s_tau_gabar)*dt;
     return dsdt_g;
}


__device__ double dxdt_n_cal(double v, double x_n, double dt){
        // double fv = 1.0 / (1.0 + std::exp(-(v-20)/2.0));
        double dxdt_n = (x_a_nmdar * fv(v) - x_n/x_tau_nmdar)*dt;
     return dxdt_n;
}

__device__ double dsdt_n_cal(double x_n, double s_n, double dt){
        // double fv = 1.0 / (1.0 + std::exp(-(v-20)/2.0));
        double dsdt_n = (s_a_nmdar * x_n * (1-s_n) - s_n/s_tau_nmdar)*dt;
     return dsdt_n;
}


__device__ double dcpre_cal(double cpre, double i_nmdar, double tau_ca_pre, double cpre_r, double dt){
       
       double dcpre = (- a_ca * i_nmdar*cpre_r - cpre/tau_ca_pre)*dt;
     return dcpre;
}

__device__ double dcpost_cal(double cpost, double i_vdcc, double tau_ca_post, double cpost_r, double dt){
       double dcpost = (- a_ca * area * 10 * i_vdcc * cpost_r - cpost/tau_ca_post)*dt;
     return dcpost;
}
__device__ double drho_cal(double rho, double ca,double th_p, double th_d, double gp, double gd,  double sig, double tau_s, double dt, double r, double rho_ini){
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
        double Noise;
        if (tau_s==0){
            Noise=0;
            }else{
        Noise = sig * std::sqrt(1/tau_s) * std::sqrt(C_th_d + C_th_p) * r/std::sqrt(dt)*std::sqrt(3.5);
        }
        double drho = ((-rho * (1 - rho) * (rho_ini - rho) + (gp * (1 - rho) * C_th_p - gd * rho * C_th_d + Noise)) * tau_s) * dt;
        // else:
        //     double drho =0;
     return drho;
}




__global__ void dt_gpu_NaK1_n(double *v,  double *h_na, double *nk, double *h, double *m, double *ca, double *s_a, double *s_a2, double *s_a3, double *s_g, double *s_g2, double *s_g3,
                    double *x_n, double *x_n2, double *x_n3, double *s_n, double *s_n2, double *s_n3, double *cpre, double *cpre2,  double *cpost, double *cpost2, //double *S, double *S2, double *S3, 
                    double *Sp,   double *K, double *z,  double*r_no, double *rho, double *rho2, double *r, double max_con,double rho_ini, double smin_ampa, double smax_ampa, 
                     int *neuron_syn_ex, int *synID_ex, int *syN_cumsum_ex, int *neuron_syn_in, int *synID_in, int *syN_cumsum_in, 
                    double alpha, double beta, double adp,double bdp, double adp2,double bdp2, double tau_camk,  double tau_k,
                     double w, double b, double I, double th, double sig_n,
                     int N, int T,  double dt, 
                    double g_leak, double g_nav,double g_kvhh,double g_kva, double g_kvsi, double g_cav,double g_kca, double g_nap, double g_kir, double g_ampar, double g_nmdar ,double g_gabar, double tau_ca,
                    double cpre_r, double cpost_r, double th_p, double th_d, double gp, double gd, double tau_ca_pre, double tau_ca_post, double sig, double tau_s)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    
  for(int it=0; it<T; it++){
    if (it !=0){


    if (ix < N)  //ix neuron
    {  
        double v_n =  v[ix*T + it-1];
        double h_na_n = h_na[ix];
        double nk_n = nk[ix];
        double h_n = h[ix];
        double m_n = m[ix];
        double z_n = z[ix];
        double ca_n = ca[ix*T + it-1];
        double Sp_n = Sp[ix*T + it-1];
        double K_n = K[ix*T + it-1];
        double r_no_n  = r_no[ix*T+it-1]; 
           
        int syN_ex = neuron_syn_ex[ix];  //the number of ex synapse
        int syN_cumsum_i_ex = syN_cumsum_ex[ix] ;         //total synapses calculated before ix neuron
        
        int syN_in = neuron_syn_in[ix];  //the number of in synapse
        int syN_cumsum_i_in = syN_cumsum_in[ix] ; 

        #ifdef a_UP
            double g_n = K_n*max_con +G_MIN;
            if (g_n<1.0){
                g_n=1.0;
            }
        #else
            double g_n = Sp_n*max_con +G_MIN;
            if (g_n<1.0){
                g_n=1.0;
            }
        #endif


         double g_cav_up = g_cav*g_n;

        // k1
        double i_ampar =0;       //i_ampar in ix neuron
        double i_nmdar = 0;
        double i_vdcc = i_vdcc(g_cav_up, v_n);
        for (int i= syN_cumsum_i_ex; i<(syN_cumsum_i_ex+syN_ex); i++){
            // printf("k1 ix %d, i: %d, v_n %f\n", ix, i, v_n);
            s_a3[i] = s_a[i];
            x_n3[i] = x_n[i];
            s_n3[i] = s_n[i];
            
            // S3[i] = S[i];
            
            double cpre_n = cpre[i*T+it-1];
            double cpost_n = cpost[i*T+it-1];
            #ifdef TEST_CA 
                double ca_syn = CONSTANT_CA;
            #else
                double ca_syn = cpre_n + cpost_n;
            #endif
            // double z_n  = z[i*T+it-1]; 
            
            double rho_n = rho[i*T+it-1];
            double r_n = r[i*T+it-1];



            double dsdt_a = dsdt_a_cal(v[synID_ex[i]*T + it-1], s_a3[i], dt);
            // if (it==1){
            // printf("k1 ix %d, synID_ex[i]: %d\n", ix, synID_ex[i]);
            // }
            double dxdt_n = dxdt_n_cal(v[synID_ex[i]*T + it-1], x_n3[i], dt);
            double dsdt_n = dsdt_n_cal(x_n3[i], s_n3[i], dt);
            


            #ifdef MG_BLOCK
                double i_nmdar_i = i_nmda_s(g_nmdar, 0.1, s_n3[i], v_n) * mg_v(v_n);
            #else
                #ifdef CA_SYN_CHANGE
                    double i_nmdar_i = i_nmda_s(g_nmdar, 0.1, s_n3[i], v_n);
                #else
                    double i_nmdar_i = i_nmda_s(g_nmdar, 0.1, s_n3[i], v_n);
                #endif
            #endif
            
            double i_vdcc_i = i_vdcc(g_cav, v_n); 
            double dcpre = dcpre_cal(cpre_n,  i_nmdar_i, tau_ca_pre, cpre_r,  dt);
            double dcpost= dcpost_cal(cpost_n, i_vdcc_i, tau_ca_post, cpost_r, dt);

            double drho = drho_cal(rho_n, ca_syn, th_p,th_d, gp, gd, sig, tau_s,  dt, r_n, rho_ini);


            
            // i_ampar += g_ampar * s_a3[i]  * (v_n-vAMPAR);
            i_ampar += g_ampar * (smin_ampa + rho_n*(smax_ampa-smin_ampa)) *s_a3[i] * (v_n-vAMPAR);
            i_nmdar += i_nmda_c(g_nmdar, g_n, s_n3[i], v_n);
                                
            // i_ampar += g_ampar * s_a3[i] * std::pow(10,(rho_n*a+b)) * (v_n-vAMPAR);
            s_a2[i] =  s_a3[i] + 0.5*dsdt_a;   //next s_a
            s_a[i] =  s_a3[i] + dsdt_a/6.0;   //update s_a
            x_n2[i] =  x_n3[i] + 0.5*dxdt_n;   //next s_a
            x_n[i] =  x_n3[i] + dxdt_n/6.0;
            s_n2[i] =  s_n3[i] + 0.5*dsdt_n;   //next s_a
            s_n[i] =  s_n3[i] + dsdt_n/6.0;
           
            // S2[i] =  S3[i] + 0.5*dSdt;   //next s_a
            // S[i] =  S3[i] + dSdt/6.0;

            cpre2[i] = cpre_n + 0.5*dcpre;   //next s_a
            cpre[i*T+it] = cpre_n + dcpre/6.0;   //update s_a
            cpost2[i] = cpost_n + 0.5*dcpost;   //next s_a
            cpost[i*T+it] = cpost_n + dcpost/6.0;   //update s_a
            rho2[i] = rho_n + 0.5*drho; 
            rho[i*T+it] = rho_n + drho/6.0;
            if (rho2[i] > 1){
                rho2[i]=1;
            }else if(rho2[i]<0){
                rho2[i]=0;
            }
          
            // printf("k1 ix %d,  dSpdt %f\n", ix,dSpdt);
            // printf("k1 ix %d, synID_ex: %d, cpre %f\n", ix, synID_ex[i], cpre[i*T+it]);
        }

        double i_gabar =0;       //i_ampar in ix neuron
        for (int i= syN_cumsum_i_in; i<(syN_cumsum_i_in+syN_in); i++){
            s_g3[i] = s_g[i];
            // printf("k1 ix %d, i: %d, v_n %f\n", ix, i, v_n);
            double dsdt_g = dsdt_g_cal(v[synID_in[i]*T + it-1], s_g3[i], dt);
            // if (it==1){
            // printf("ix %d, synID_in[i]: %d\n", ix, synID_in[i]);
            // }
            
            i_gabar += g_gabar * s_g3[i] * (v_n-vGABAR);
            s_g2[i] =  s_g3[i] + 0.5*dsdt_g;   //next s_a
            s_g[i] =  s_g3[i] + dsdt_g/6.0;   //update s_a
            // printf("k1 ix %d, synID_in: %d, i_gabar %f\n", ix, synID_in[i], i_gabar);
            // printf("k1 ix %d, synID_ex: %d, rho %f\n", ix, synID_ex[i], rho[i*T+it]);
        }


        double dvdt =  dvdt_cal_NaK1_n(v_n, h_na_n, nk_n, h_n,m_n, ca_n, dt, i_ampar,i_nmdar, i_gabar, g_leak, g_nav,g_kvhh,g_kva,g_kvsi,g_cav_up,g_kca,g_nap,g_kir,tau_ca);
         double dhdt_na = dhdt_na_cal(v_n, h_na_n, dt);
        double dndt = dndt_cal(v_n, nk_n, dt);
        double dhdt = dhdt_cal(v_n, h_n, dt);
        double dmdt = dmdt_cal(v_n, m_n, dt);
        double dcadt = dcadt_cal_n(i_vdcc, ca_n, dt, g_cav_up, tau_ca);

        double dKdt = ((1/(1+std::exp(-(Sp_n -beta*ca_n -adp)*bdp))) - K_n)*dt/tau_k;
        double dSpdt = (1/(1+std::exp(-(-b*K_n + I+z_n + w*Sp_n + alpha*ca_n - adp2)*bdp2)) - Sp_n)*dt/tau_camk;
        double dzdt = (-th * z_n + sig_n * std::sqrt(2*th/dt)*r_no_n)*dt;

        double v2 = v_n+0.5*dvdt;
        double h_na2 = h_na_n+0.5*dhdt_na;
        double nk2 = nk_n+0.5*dndt;
        double h2 = h_n+0.5*dhdt;
         double m2 = m_n+0.5*dmdt;
        double ca2 = ca_n+0.5*dcadt;
        v[ix*T+it]  = v_n + dvdt/6.0;     //update v
        h_na[ix] += dhdt_na/6.0; 
        nk[ix] += dndt/6.0;   //update nk
        h[ix] += dhdt/6.0; 
        m[ix] += dmdt/6.0; 

        ca[ix*T+it] =ca_n+ dcadt/6.0;  //update ca

          double Sp2 =  Sp_n + 0.5*dSpdt;   //next s_a
            Sp[ix*T+it] =  Sp_n + dSpdt/6.0;
           double K2 =  K_n + 0.5*dKdt;   //next s_a
            K[ix*T+it] = K_n + dKdt/6.0;
            double z2 =  z_n + 0.5*dzdt;   //next s_a
            z[ix] +=  dzdt/6.0;
         
         //k2
        #ifdef a_UP
            g_n = K2*max_con +G_MIN;
            if (g_n<1.0){
                g_n=1.0;
            }
        #else
            g_n = Sp2*max_con +G_MIN;
            if (g_n<1.0){
                g_n=1.0;
            }
        #endif
        g_cav_up = g_cav*g_n;

        i_ampar =0;       //i_ampar in ix neuron
        i_nmdar = 0;
        i_vdcc = i_vdcc(g_cav_up, v2);
        for (int i= syN_cumsum_i_ex; i<(syN_cumsum_i_ex+syN_ex); i++){
            // printf("k1 ix %d, i: %d, v_n %f\n", ix, i, v_n);
            
            #ifdef TEST_CA 
                double ca_syn = CONSTANT_CA;
            #else
                 double ca_syn = cpre2[i] + cpost2[i]; 
            #endif
            
            double dsdt_a = dsdt_a_cal(v[synID_ex[i]*T + it-1], s_a2[i], dt);
            // if (it==1){
            // printf("k1 ix %d, synID_ex[i]: %d\n", ix, synID_ex[i]);
            // }
            double dxdt_n = dxdt_n_cal(v[synID_ex[i]*T + it-1], x_n2[i], dt);
            double dsdt_n = dsdt_n_cal(x_n2[i], s_n2[i], dt);

            #ifdef MG_BLOCK
                double i_nmdar_i = i_nmda_s(g_nmdar, g_n*0.1, s_n2[i], v2) * mg_v(v2);
            #else
                #ifdef CA_SYN_CHANGE
                    double i_nmdar_i = i_nmda_s(g_nmdar, g_n*0.1, s_n2[i], v2);
                #else
                    double i_nmdar_i = i_nmda_s(g_nmdar, 0.1, s_n2[i], v2);
                #endif
            #endif
            double i_vdcc_i = i_vdcc(g_cav, v2); 
            double dcpre = dcpre_cal(cpre2[i],  i_nmdar_i, tau_ca_pre, cpre_r,  dt);
            double dcpost= dcpost_cal(cpost2[i], i_vdcc_i, tau_ca_post, cpost_r, dt);
            // camk_cal(dSdt, dSpdt, S2[i], Sp2[i], ca_syn);
            double r_n = r[i*T+it-1];
            double drho = drho_cal(rho2[i], ca_syn, th_p,th_d, gp, gd, sig,  tau_s,  dt, r_n, rho_ini);
            i_ampar += g_ampar * (smin_ampa + rho2[i]*(smax_ampa-smin_ampa)) *s_a2[i] * (v2-vAMPAR);
            rho2[i] = rho[i*T+it-1] + 0.5*drho;   //next s_a
            rho[i*T+it] += drho/3.0; 
            if (rho2[i] > 1){
                rho2[i]=1;
            }else if(rho2[i]<0){
                rho2[i]=0;
            }
            

            // i_ampar += g_ampar * s_a2[i]  * (v2-vAMPAR);
            i_nmdar += i_nmda_c(g_nmdar, g_n, s_n2[i], v2);
                                
            // i_ampar += g_ampar * s_a3[i] * std::pow(10,(rho_n*a+b)) * (v_n-vAMPAR);
            s_a2[i] =  s_a3[i] + 0.5*dsdt_a;   //next s_a
            s_a[i] +=  dsdt_a/3.0;   //update s_a
            x_n2[i] =  x_n3[i] + 0.5*dxdt_n;   //next s_a
            x_n[i] +=  dxdt_n/3.0;
            s_n2[i] =  s_n3[i] + 0.5*dsdt_n;   //next s_a
            s_n[i]  += dsdt_n/3.0;
            

            cpre2[i] = cpre[i*T+it-1] + 0.5*dcpre;   //next s_a
            cpre[i*T+it] += dcpre/3.0;   //update s_a
            cpost2[i] = cpost[i*T+it-1] + 0.5*dcpost;   //next s_a
            cpost[i*T+it]  += dcpost/3.0;   //update s_

        }
        i_gabar = 0; 
         for (int i= syN_cumsum_i_in; i<(syN_cumsum_i_in + syN_in); i++){
            // printf("k2 ix %d, i: %d, v2 %f\n", ix, i, v2);
            double dsdt_g = dsdt_g_cal(v[synID_in[i]*T+it-1], s_g2[i], dt);
            // i_gabar += g_gabar * s_g2[i]*(v2-vGABAR);
            i_gabar += g_gabar * s_g2[i]*(v2-vGABAR);
            s_g2[i] = s_g3[i] + 0.5*dsdt_g;
            s_g[i] += dsdt_g/3.0;
            // printf("k2 ix %d, synID_in: %d, i_gabar %f\n", ix, synID_in[i],i_gabar);
        }


        
         dvdt =  dvdt_cal_NaK1_n(v2, h_na2, nk2, h2,m2, ca2,  dt, i_ampar, i_nmdar, i_gabar, g_leak, g_nav,g_kvhh,g_kva,g_kvsi,g_cav_up,g_kca,g_nap,g_kir,tau_ca);
          dhdt_na = dhdt_na_cal(v2, h_na2, dt);
         dndt = dndt_cal(v2, nk2, dt);
         dhdt = dhdt_cal(v2, h2, dt);
         dmdt = dmdt_cal(v2, m2, dt);
         dcadt = dcadt_cal_n(i_vdcc,  ca2, dt, g_cav_up, tau_ca);

         dKdt = ((1/(1+std::exp(-(Sp2 -beta*ca2 -adp)*bdp))) - K2)*dt/tau_k;
        dSpdt = (1/(1+std::exp(-(-b*K2 + I+z2 + w*Sp2 + alpha*ca2 -adp2)*bdp2)) - Sp2)*dt/tau_camk;
         dzdt = (-th * z2 + sig_n * std::sqrt(2*th/dt)*r_no_n)*dt;

            
        
        v2 = v_n+0.5*dvdt;
        h_na2 = h_na_n+0.5*dhdt_na;
        nk2 = nk_n+0.5*dndt;
        h2 = h_n+0.5*dhdt;
        m2 = m_n+0.5*dmdt;
        ca2 = ca_n+0.5*dcadt;
        v[ix*T+it] += dvdt/3.0;
        h_na[ix] += dhdt_na/3.0;
        nk[ix] += dndt/3.0;
        h[ix] += dhdt/3.0;
        m[ix] += dmdt/3.0;
        

         ca[ix*T+it] += dcadt/3.0;  //update ca

          Sp2 =  Sp_n + 0.5*dSpdt;   //next s_a
            Sp[ix*T+it] += dSpdt/3.0;
          K2 =  K_n + 0.5*dKdt;   //next s_a
            K[ix*T+it] += dKdt/3.0;
            z2 =  z_n + 0.5*dzdt;   //next s_a
            z[ix] +=  dzdt/3.0;
         


        //k3
        #ifdef a_UP
            g_n = K2*max_con +G_MIN;
            if (g_n<1.0){
                g_n=1.0;
            }
        #else
            g_n = Sp2*max_con +G_MIN;
            if (g_n<1.0){
                g_n=1.0;
            }
        #endif

         g_cav_up = g_cav*g_n;
          i_ampar =0;       //i_ampar in ix neuron
        i_nmdar = 0;
        i_vdcc = i_vdcc(g_cav_up, v2);
        for (int i= syN_cumsum_i_ex; i<(syN_cumsum_i_ex+syN_ex); i++){
            // printf("k1 ix %d, i: %d, v_n %f\n", ix, i, v_n);

            #ifdef TEST_CA 
                double ca_syn = CONSTANT_CA;
            #else
                 double ca_syn = cpre2[i] + cpost2[i]; 
            #endif

           
            double dsdt_a = dsdt_a_cal(v[synID_ex[i]*T + it-1], s_a2[i], dt);
            // if (it==1){
            // printf("k1 ix %d, synID_ex[i]: %d\n", ix, synID_ex[i]);
            // }
            double dxdt_n = dxdt_n_cal(v[synID_ex[i]*T + it-1], x_n2[i], dt);
            double dsdt_n = dsdt_n_cal(x_n2[i], s_n2[i], dt);

            #ifdef MG_BLOCK
                double i_nmdar_i = i_nmda_s(g_nmdar, g_n*0.1, s_n2[i], v2) * mg_v(v2);
            #else
                #ifdef CA_SYN_CHANGE
                    double i_nmdar_i = i_nmda_s(g_nmdar, g_n*0.1, s_n2[i], v2);
                #else
                    double i_nmdar_i = i_nmda_s(g_nmdar, 0.1, s_n2[i], v2);
                #endif
            #endif
            double i_vdcc_i = i_vdcc(g_cav, v2); 
            double dcpre = dcpre_cal(cpre2[i],  i_nmdar_i, tau_ca_pre, cpre_r,  dt);
            double dcpost= dcpost_cal(cpost2[i], i_vdcc_i, tau_ca_post, cpost_r, dt);
            // camk_cal(dSdt, dSpdt, S2[i], Sp2[i], ca_syn);
            double r_n = r[i*T+it-1];

            double drho = drho_cal(rho2[i], ca_syn, th_p,th_d, gp, gd, sig,  tau_s,  dt, r_n, rho_ini);
            i_ampar += g_ampar * (smin_ampa + rho2[i]*(smax_ampa-smin_ampa)) *s_a2[i] * (v2-vAMPAR);
            rho2[i] = rho[i*T+it-1] + drho;   //next s_a
            rho[i*T+it] += drho/3.0; 
            if (rho2[i] > 1){
                rho2[i]=1;
            }else if(rho2[i]<0){
                rho2[i]=0;
            }

            i_ampar += g_ampar * s_a2[i]  * (v2-vAMPAR);
            i_nmdar += i_nmda_c(g_nmdar, g_n, s_n2[i], v2);
                                
            // i_ampar += g_ampar * s_a3[i] * std::pow(10,(rho_n*a+b)) * (v_n-vAMPAR);
            s_a2[i] =  s_a3[i] + dsdt_a;   //next s_a
            s_a[i] +=  dsdt_a/3.0;   //update s_a
            x_n2[i] =  x_n3[i] + dxdt_n;   //next s_a
            x_n[i] +=  dxdt_n/3.0;
            s_n2[i] =  s_n3[i] + dsdt_n;   //next s_a
            s_n[i]  += dsdt_n/3.0;

            cpre2[i] = cpre[i*T+it-1] + dcpre;   //next s_a
            cpre[i*T+it] += dcpre/3.0;   //update s_a
            cpost2[i] = cpost[i*T+it-1] + dcpost;   //next s_a
            cpost[i*T+it]  += dcpost/3.0;   //update s_a
    
            // printf("k1 ix %d, synID_ex: %d, i_ampar %f\n", ix, synID_ex[i], i_ampar);
            // printf("k1 ix %d, synID_ex: %d, cpre %f\n", ix, synID_ex[i], cpre[i*T+it]);
        }

        i_gabar = 0; 
         for (int i= syN_cumsum_i_in; i<(syN_cumsum_i_in + syN_in); i++){
            // printf("k3 ix %d, i: %d, v2 %f\n", ix, i, v2);
            double dsdt_g = dsdt_g_cal(v[synID_in[i]*T+it-1], s_g2[i], dt);
            // i_gabar += g_gabar * s_g2[i]*(v2-vGABAR);
            i_gabar += g_gabar * s_g2[i]*(v2-vGABAR);
            s_g2[i] = s_g3[i] + dsdt_g;
            s_g[i] += dsdt_g/3.0;
            //  printf(" k3 ix %d, synID_in: %d, i_gabar %f\n", ix, synID_in[i], i_gabar);
        }
        
         
         dvdt =  dvdt_cal_NaK1_n(v2, h_na2, nk2, h2,m2, ca2,  dt, i_ampar, i_nmdar, i_gabar, g_leak, g_nav,g_kvhh,g_kva,g_kvsi,g_cav_up,g_kca,g_nap,g_kir,tau_ca);
          dhdt_na = dhdt_na_cal(v2, h_na2, dt);
         dndt = dndt_cal(v2, nk2, dt);
         dhdt = dhdt_cal(v2, h2, dt);
         dmdt = dmdt_cal(v2, m2, dt);
         dcadt = dcadt_cal_n(i_vdcc,  ca2, dt, g_cav_up, tau_ca);

         dKdt = ((1/(1+std::exp(-(Sp2-beta*ca2-adp)*bdp))) - K2)*dt/tau_k;
        dSpdt = (1/(1+std::exp(-(-b*K2 + I+z2 + w*Sp2 + alpha*ca2  -adp2)*bdp2)) - Sp2)*dt/tau_camk;
         dzdt = (-th * z2 + sig_n * std::sqrt(2*th/dt)*r_no_n)*dt;

         v2 = v_n+dvdt;
         h_na2 = h_na_n+dhdt_na;
        nk2 = nk_n+dndt;
        h2 = h_n+dhdt;
        m2 = m_n+dmdt;
        ca2 = ca_n+dcadt;
        v[ix*T+it] += dvdt/3.0;
        h_na[ix] += dhdt_na/3.0;
        nk[ix] += dndt/3.0;
        h[ix] += dhdt/3.0;
        m[ix] += dmdt/3.0;
         ca[ix*T+it] += dcadt/3.0;  //update ca

          Sp2 =  Sp_n + dSpdt;   //next s_a
            Sp[ix*T+it] += dSpdt/3.0;
          K2 =  K_n + dKdt;   //next s_a
            K[ix*T+it] += dKdt/3.0;
            z2 =  z_n + dzdt;   //next s_a
            z[ix] +=  dzdt/3.0;

        //k4 
        #ifdef a_UP
            g_n = K2*max_con +G_MIN;
            if (g_n<1.0){
                g_n=1.0;
            }
        #else
            g_n = Sp2*max_con +G_MIN;
            if (g_n<1.0){
                g_n=1.0;
            }
        #endif
         g_cav_up = g_cav*g_n;
          i_ampar =0;       //i_ampar in ix neuron
        i_nmdar = 0;
        i_vdcc = i_vdcc(g_cav_up, v2);
        for (int i= syN_cumsum_i_ex; i<(syN_cumsum_i_ex+syN_ex); i++){
            // printf("k1 ix %d, i: %d, v_n %f\n", ix, i, v_n);
            #ifdef TEST_CA 
                double ca_syn = CONSTANT_CA;
            #else
                 double ca_syn = cpre2[i] + cpost2[i]; 
            #endif
          
            double dsdt_a = dsdt_a_cal(v[synID_ex[i]*T + it-1], s_a2[i], dt);
            // if (it==1){
            // printf("k1 ix %d, synID_ex[i]: %d\n", ix, synID_ex[i]);
            // }
            double dxdt_n = dxdt_n_cal(v[synID_ex[i]*T + it-1], x_n2[i], dt);
            double dsdt_n = dsdt_n_cal(x_n2[i], s_n2[i], dt);

            #ifdef MG_BLOCK
                double i_nmdar_i = i_nmda_s(g_nmdar, g_n*0.1, s_n2[i], v2) * mg_v(v2);
            #else
                #ifdef CA_SYN_CHANGE
                    double i_nmdar_i = i_nmda_s(g_nmdar, g_n*0.1, s_n2[i], v2);
                #else
                    double i_nmdar_i = i_nmda_s(g_nmdar, 0.1, s_n2[i], v2);
                #endif
            #endif

            double i_vdcc_i = i_vdcc(g_cav, v2); 
            double dcpre = dcpre_cal(cpre2[i],  i_nmdar_i, tau_ca_pre, cpre_r,  dt);
            double dcpost= dcpost_cal(cpost2[i], i_vdcc_i, tau_ca_post, cpost_r, dt);
         
            // camk_cal(dSdt, dSpdt, S2[i], Sp2[i], ca_syn);
            double r_n = r[i*T+it-1];
            double drho = drho_cal(rho2[i], ca_syn, th_p,th_d, gp, gd, sig,  tau_s,  dt, r_n, rho_ini);
            i_ampar += g_ampar * (smin_ampa + rho2[i]*(smax_ampa-smin_ampa)) *s_a2[i] * (v2-vAMPAR);
            // rho2[i] = rho[i*T+it-1] + 0.5*drho;   //next s_a
            rho[i*T+it] += drho/6.0; 
             if (rho[i*T+it] > 1){
                rho[i*T+it]=1;
            }else if(rho[i*T+it]<0){
                rho[i*T+it]=0;
            }


            
            i_ampar += g_ampar * s_a2[i]  * (v2-vAMPAR);
            i_nmdar += i_nmda_c(g_nmdar, g_n, s_n2[i], v2);
                                
            // i_ampar += g_ampar * s_a3[i] * std::pow(10,(rho_n*a+b)) * (v_n-vAMPAR);
            s_a[i] +=  dsdt_a/6.0;   //update s_a
            x_n[i] +=  dxdt_n/6.0;
            s_n[i]  += dsdt_n/6.0;
           

            cpre[i*T+it] += dcpre/6.0;   //update s_a
            cpost[i*T+it]  += dcpost/6.0;   //update s_a
           
        }
        i_gabar = 0; 
         for (int i= syN_cumsum_i_in; i<(syN_cumsum_i_in + syN_in); i++){
            // printf("k4 ix %d, i: %d, v2 %f\n", ix, i, v2);
            double dsdt_g = dsdt_g_cal(v[synID_in[i]*T+it-1], s_g2[i], dt);
            i_gabar += g_gabar * s_g2[i]*(v2-vGABAR);
            s_g[i] += dsdt_g/6.0;
            //  printf("k4 ix %d, synID_in: %d, i_gabar %f\n", ix, synID_in[i], i_gabar);
        }

        
        
         dvdt =  dvdt_cal_NaK1_n(v2, h_na2, nk2, h2,m2, ca2,  dt, i_ampar, i_nmdar, i_gabar, g_leak, g_nav,g_kvhh,g_kva,g_kvsi,g_cav_up,g_kca,g_nap,g_kir,tau_ca);
           dhdt_na = dhdt_na_cal(v2, h_na2, dt);
         dndt = dndt_cal(v2, nk2, dt);
         dhdt = dhdt_cal(v2, h2, dt);
         dmdt = dmdt_cal(v2, m2, dt);
         dcadt = dcadt_cal_n(i_vdcc, ca2, dt, g_cav_up, tau_ca);

          dKdt = ((1/(1+std::exp(-(Sp2 -beta*ca2-adp)*bdp))) - K2)*dt/tau_k;
        dSpdt = (1/(1+std::exp(-(-b*K2 + I+z2 + w*Sp2 + alpha*ca2 -adp2)*bdp2)) - Sp2)*dt/tau_camk;
         dzdt = (-th * z2 + sig_n * std::sqrt(2*th/dt)*r_no_n)*dt;


        v[ix*T+it] += dvdt/6.0;
        h_na[ix] += dhdt_na/6.0;
        nk[ix] += dndt/6.0;
        h[ix] += dhdt/6.0;
        m[ix] += dmdt/6.0;
        ca[ix*T+it] += dcadt/6.0;
            Sp[ix*T+it] += dSpdt/6.0;
         
            K[ix*T+it] += dKdt/6.0;
          
            z[ix] +=  dzdt/6.0;


    } //ix
    __syncthreads();

        
  } // if
 } //for
}



int main(int argc, char * argv[]){
   int nav = 1;//;atoi(argv[1]);
    int kva = 1;//;atoi(argv[2]);
    int kvsi = 1;//;atoi(argv[3]);
    int kir = 1;//atoi(argv[4]);
    int nmdar = 1;//atoi(argv[5]);
    int gabar = 1;//atoi(argv[6]);
    
    double g_leak,g_nav,g_kvhh, g_kva, g_kvsi,g_cav,g_nap,g_kca, g_kir, g_ampar, g_nmdar, g_gabar, tau_ca;
    // double g_nav2, g_kva2, g_kvsi2, g_kir2, g_nmdar2, g_gabar2;
    int g_c = 1;

    // for (int i=0; i<2; ++i) {
        g_leak  = atof(argv[g_c]);
        g_c +=1;
      

        if (nav==1){
            g_nav  = atof(argv[g_c]);
            g_c +=1;
       
        } 

        g_kvhh  = atof(argv[g_c]);
        g_c += 1;
    


        if (kva==1){
            g_kva  = atof(argv[g_c]);
            g_c +=1;
           
        } 
        if (kvsi==1){
            g_kvsi  = atof(argv[g_c]);
            g_c +=1;   
           
        } 

        g_cav    = atof(argv[g_c]);
        g_c +=1; 
             
        g_kca   = atof(argv[g_c]);
        g_c +=1;
            
        g_nap    = atof(argv[g_c]);
        g_c +=1;
       
        if (kir==1){
        g_kir    = atof(argv[g_c]);
        g_c +=1;
       
        }
                    
        g_ampar  = atof(argv[g_c]);
        g_c +=1;
        

        if (nmdar==1){
            g_nmdar  = atof(argv[g_c]);
            g_c +=1; 
           
        } 
        if (gabar==1){
            g_gabar  = atof(argv[g_c]);
            g_c +=1; 
          
        } 
        tau_ca    = atof(argv[g_c]);
        g_c +=1;
        
    // }

    printf("g_gabar: %f\n", g_gabar);

    int NE = atoi(argv[g_c]);   //number of neurons
    g_c +=1;

    int Ip = atof(argv[g_c]);   //% of inhibitory
    g_c +=1;
    int NI = NE*Ip/(100-Ip);


    double con_M = atof(argv[g_c]);  
    std::string con_M_str = argv[g_c];
    g_c +=1;
    double con_M_in = atof(argv[g_c]);   
    std::string con_M_in_str = argv[g_c];
    g_c +=1;

    double con_ex_ex = atof(argv[g_c]); 
    std::string con_ex_ex_str = argv[g_c];  
    g_c +=1;
    double con_ex_in = atof(argv[g_c]);  
    std::string con_ex_in_str = argv[g_c]; 
    g_c +=1;
    double con_in_ex = atof(argv[g_c]);  
    std::string con_in_ex_str = argv[g_c]; 
    g_c +=1;
    double con_in_in = atof(argv[g_c]);  
    std::string con_in_in_str = argv[g_c]; 
    g_c +=1;

    double cpre_r    = atof(argv[g_c]);
    std::string cpre_r_str = argv[g_c];
    g_c +=1;
     
    double cpost_r    = atof(argv[g_c]);
    std::string cpost_r_str = argv[g_c];
    g_c +=1;

    double th_p    = atof(argv[g_c]);
    std::string th_p_str = argv[g_c];
    g_c +=1;
    double th_d    = atof(argv[g_c]);
    std::string th_d_str = argv[g_c];
    g_c +=1;
    double gp    = atof(argv[g_c]);
    std::string gp_str = argv[g_c];
    g_c +=1;
    
    double gd    = atof(argv[g_c]);
    std::string gd_str = argv[g_c];
    g_c +=1;
    
    double tau_ca_pre    = atof(argv[g_c]);
     std::string tau_pre_str = argv[g_c];
    g_c +=1;
    double tau_ca_post    = atof(argv[g_c]);
    std::string tau_post_str = argv[g_c];
    g_c +=1;
    double sig    = atof(argv[g_c]);
    std::string sig_str = argv[g_c];
    g_c +=1;
    double tau_s = atof(argv[g_c]);
    std::string tau_s_str = argv[g_c];
    g_c +=1;

     
    double Sp_ini    = atof(argv[g_c]);
    std::string Sp_ini_str = argv[g_c];
    g_c +=1;

    double K_ini    = atof(argv[g_c]);
    std::string K_ini_str = argv[g_c];
    g_c +=1;

    double z_ini    = atof(argv[g_c]);
    std::string z_ini_str = argv[g_c];
    g_c +=1;

  
    double alpha = atof(argv[g_c]);  //1.2
    std::string alpha_str = (argv[g_c]);
    g_c +=1;
    double beta = atof(argv[g_c]);  //1.2
    std::string beta_str = (argv[g_c]);
    g_c +=1;

 
    double adp =atof(argv[g_c]);//1.4
    std::string adp_str = (argv[g_c]);
    g_c +=1;
    double bdp =atof(argv[g_c]);  //0.1
    std::string bdp_str = (argv[g_c]);
    g_c +=1;
    double adp2 =atof(argv[g_c]);//1.4
    std::string adp2_str = (argv[g_c]);
    g_c +=1;
    double bdp2 =atof(argv[g_c]);  //0.1
    std::string bdp2_str = (argv[g_c]);
    g_c +=1;

    int tau_camk =atoi(argv[g_c]);  //1000 //ms
    std::string tau_camk_str = (argv[g_c]);
    g_c +=1;

    int tau_k =atoi(argv[g_c]);  //1000 //ms
    std::string tau_k_str = (argv[g_c]);
    g_c +=1;

    double w = atof(argv[g_c]);  //1.2
    std::string w_str = (argv[g_c]);
    g_c +=1;
    double b =atof(argv[g_c]);  //0.1
    std::string b_str = (argv[g_c]);
    g_c +=1;
 
    double I =atof(argv[g_c]);//1.4
    std::string I_str = (argv[g_c]);
    g_c +=1;

    double th =atof(argv[g_c]);//1.4
    std::string th_str = (argv[g_c]);
    g_c +=1;

    double sig_n =atof(argv[g_c]);//1.4
    std::string sig_n_str = (argv[g_c]);
    g_c +=1;

     double max_con =atof(argv[g_c]);//1.4
    std::string max_con_str = (argv[g_c]);
    g_c +=1;

   max_con = max_con - G_MIN;

   double rho_ini    = atof(argv[g_c]);
    std::string rho_ini_str = argv[g_c];
    g_c +=1;

     double smin_ampa =  atof(argv[g_c]);
    std::string smin_ampa_str = (argv[g_c]);
    g_c +=1;

    double smax_ampa =  atof(argv[g_c]);
    std::string smax_ampa_str = (argv[g_c]);
    g_c +=1;
    
   

    rho_ini= (1.0-smin_ampa)/(smax_ampa-smin_ampa);

    if(tau_s==0){
        char ampa_str[16];
	   sprintf(ampa_str,"%.1f", 1.0);
        smax_ampa_str= ampa_str;
    }
    
 

    int T = atoi(argv[g_c]);  //#ms
    g_c +=1;
    int Tp = atoi(argv[g_c]);  //#ms
    g_c +=1;

    int tbs = T/Tp;          //step size  #ms
    
    double *dt_li =  (double*)malloc(sizeof(double)*10);
    dt_li[0]=0.05;
    dt_li[1]=0.02;
    dt_li[2]=0.01;
    dt_li[3]=0.005;
    dt_li[4]=0.002;
    dt_li[5]=0.001;
    dt_li[6]=0.0005;
    dt_li[7]=0.0002;
    dt_li[8]=0.0001;
    dt_li[9]=0.00005;

    double dt_start= dt_li[0];
    
    
    
    
    double ex = atof(argv[g_c]);
    std::string ex_str= argv[g_c];
    g_c +=1;

    double ex_in = atof(argv[g_c]);
    std::string ex_in_str= argv[g_c];
    g_c +=1;
    printf("ex %f\n", ex);

    g_cav = g_cav * (std::pow(10, ex));
    


    int seed = atoi(argv[g_c]);
    g_c +=1;
    srand(seed);
    std::string init = (argv[g_c]);
    g_c +=1;

    std::string param_i = argv[g_c];
    g_c +=1;
    // int lr_c = atoi(argv[g_c]);
    // g_c +=1;
    std::string model_name = argv[g_c];
    g_c +=1;
    int blockdim_x = atoi(argv[g_c]);
    g_c +=1;

    printf("blockdim_x %d\n", blockdim_x);

  

   //make connectgion matrix
   std::vector<std::vector<int>> neuron_cons(NE+NI, std::vector<int>(NE+NI));
    int* neuron_syN_ex = (int*)malloc(sizeof(int)*(NE+NI));  //array for the number of synapses per neuron
    int* neuron_syN_in = (int*)malloc(sizeof(int)*(NE+NI));

#ifdef TWO_N
    neuron_cons[0][1]=1;
    neuron_cons[1][0]=1;
#else

    #ifndef CON_LOGNORMAL
        if (con_ex_ex>=1)
        { //homogenous con
            int con_th_i = con_ex_ex;
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
                            
                            if (a > con_ex_ex){
                            neuron_cons[i][j] = 1;
                            }
                        }
                }
                }
            for (int i = 0; i < NE; i++) {
                    for (int j = NE; j < (NE+NI); j++) {
                        if (i==j){
                            neuron_cons[i][j] = 0;
                        }
                        else{
                            double a = (double)rand()/RAND_MAX;
                            
                            if (a > con_in_ex){
                            neuron_cons[i][j] = 1;
                            }
                        }
                    }
                }
            for (int i = NE; i < (NE+NI); i++) {
                    for (int j = NE; j < (NE+NI); j++) {
                        if (i==j){
                            neuron_cons[i][j] = 0;
                        }
                        else{
                            double a = (double)rand()/RAND_MAX;
                            
                            if (a > con_in_in){
                            neuron_cons[i][j] = 1;
                            }
                        }
                    }
                }
            for (int i = NE; i < (NE+NI); i++) {
                    for (int j = 0; j < (NE); j++) {
                        if (i==j){
                            neuron_cons[i][j] = 0;
                        }
                        else{
                            double a = (double)rand()/RAND_MAX;
                            
                            if (a > con_ex_in){
                            neuron_cons[i][j] = 1;
                            }
                        }
                    }
                }
            }
    #else
            std::lognormal_distribution<double> distribution1(con_M, con_ex_ex);
            std::lognormal_distribution<double> distribution2(con_M, con_ex_in);
            std::lognormal_distribution<double> distribution3(con_M_in, con_in_ex);
            std::lognormal_distribution<double> distribution4(con_M_in, con_in_in);

            std::mt19937 mt1;
            std::mt19937 mt2;
            std::mt19937 mt3;
            std::mt19937 mt4;            // メルセンヌ・ツイスタの32ビット版

            // std::random_device rnd;     // 非決定的な乱数生成器
            mt1.seed(seed);
            mt2.seed(seed);
            mt3.seed(seed);
            mt4.seed(seed);

            
            // std::mt19937 gen(seed);
            //ex -> ex  i=post   j = pre
            for (int i=0; i<NE; ++i) {
                int con_th_i;
                while(1){
                    con_th_i = distribution1(mt1);
                    // printf("con_th_i: %d\n", con_th_i);
                if(con_th_i<NE){
                    break;
                    }
                }
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
            //in->ex   
            for (int i=0; i<NE; ++i) {
                int con_th_i;
                while(1){
                    con_th_i = distribution3(mt3);
                    // printf("con_th_i: %d\n", con_th_i);
                if(con_th_i<NI){
                    break;
                    }
                }
                for (int n = 0; n < con_th_i; n++) 
                { 
                    while(1){
                double a = (double)rand()/RAND_MAX;
                int j = NE+ a*NI -0.001; //0~NE-1 random
                if ((i!=j) && (neuron_cons[i][j] != 1) ){
                        neuron_cons[i][j] = 1;
                        break;
                    }
                }
                }
            }
            //in-> in
            for (int i=NE; i<NE+NI; ++i) {
                int con_th_i;
                while(1){
                    con_th_i = distribution4(mt4);
                    // printf("con_th_i: %d\n", con_th_i);
                if(con_th_i<NI){
                    break;
                    }
                }
                for (int n = 0; n < con_th_i; n++) 
                { 
                    while(1){
                double a = (double)rand()/RAND_MAX;
                int j = NE + a*NI -0.001; //0~NE-1 random
                if ((i!=j) && (neuron_cons[i][j] != 1) ){
                        #ifdef NO_IN_IN
                            neuron_cons[i][j] = 0;
                        #else
                            neuron_cons[i][j] = 1;
                        #endif
                        break;
                    }
                }
                }
            }
            //ex -> in  
            for (int i=NE; i<NE+NI; ++i) {
                int con_th_i;
                while(1){
                    con_th_i = distribution2(mt2);
                    // printf("con_th_i: %d\n", con_th_i);
                if(con_th_i<NE){
                    break;
                    }
                }
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
    #endif
#endif

    //make array for the number of synapses per neuron
    for (int i = 0; i < (NE+NI); i++) {
        int syN = 0;
        for (int j = 0; j < NE; j++) 
        {
        if (neuron_cons[i][j] ==1){
            syN += 1;
        }
       }
    neuron_syN_ex[i]=syN;
    // printf("neuron%d, num of excitatory synapses: %d\n"  , i, syN);  //post
    }

    for (int i = 0; i < (NE+NI); i++) {
        int syN = 0;
        for (int j = NE; j < (NE+NI); j++) 
        {
        if (neuron_cons[i][j] ==1){
            syN += 1;
        }
    }
    neuron_syN_in[i]=syN;
    // printf("neuron%d, num of inhibitory synapses: %d\n"  , i, syN);
    }

     srand(seed);
    //CON >=1
    for (int i = 0; i < (NE+NI); i++) {
        if(neuron_syN_ex[i]+neuron_syN_in[i] ==0){
            
            while(1){
                double num;
                 num = (double)rand()/RAND_MAX * (NI+NE);
                // printf("rand: %f\n", num);
                int j=num;
                // printf("rand: %f\n", num);
              
              
              if((j<NE+NI)&&(j!=i)){
                // printf("add con %d-%d\n", i, j);
                #ifdef NO_IN_IN
                    if((i<NE)||(j<NE)){
                    neuron_cons[i][j] =1;
                    // printf(" neuron_cons[j][i]: %d\n", neuron_cons[j][i]);
                    break;
                    }//if
                #else
                    neuron_cons[i][j] =1;
                    // printf(" neuron_cons[j][i]: %d\n", neuron_cons[j][i]);
                    break;
                #endif
               }//if
           }//while
        }//if

     }//for

     for (int i = 0; i < (NE+NI); i++) {
        int syN = 0;
        for (int j = 0; j < NE; j++) 
        {
        if (neuron_cons[i][j] ==1){
            syN += 1;
        }
       }
    neuron_syN_ex[i]=syN;
    printf("neuron%d, num of excitatory synapses: %d\n"  , i, syN);  //post
    }

    for (int i = 0; i < (NE+NI); i++) {
        int syN = 0;
        for (int j = NE; j < (NE+NI); j++) 
        {
        if (neuron_cons[i][j] ==1){
            syN += 1;
        }
    }
    neuron_syN_in[i]=syN;
    printf("neuron%d, num of inhibitory synapses: %d\n"  , i, syN);
    }

    //calculate total synapses
    int syN_sum_ex =0;
    for (int i = 0; i < (NE+NI); i++) {
        syN_sum_ex += neuron_syN_ex[i];
        }
    printf("total excitatory synapses: %d\n", syN_sum_ex);

    int syN_sum_in =0;
    for (int i = 0; i < (NE+NI); i++) {
        syN_sum_in += neuron_syN_in[i];
        }
    printf("total inhibitory synapses: %d\n", syN_sum_in);

     //modify synaptic conductance
    g_ampar = g_ampar/(syN_sum_ex/(NE+NI));
    if (nmdar==1){
    g_nmdar= g_nmdar/(syN_sum_ex/(NE+NI));
    }
    g_gabar= g_gabar/(syN_sum_in/(NE+NI));
   

    //make pre-synaptic index list
    int* synID_ex = (int*)malloc(sizeof(int)*syN_sum_ex);
    int* syN_cumsum_ex = (int*)malloc(sizeof(int)*(NE+NI)); 
    int syN_sum_i_ex = 0;

    for (int n=0; n<(NE+NI); n++){

            int syN = neuron_syN_ex[n];
            std::vector<int> con_in =  neuron_cons[n];
        
            int c = 0;
            for (int k=0; k<NE; k++){
                    

            if (con_in[k]==1){
                synID_ex[syN_sum_i_ex + c] =k;
                c += 1;
              }
            }

            // syN_cumsum_ex[n] = syN_sum_i_ex;
            syN_sum_i_ex += syN;
       }   

    
     int* syN_cumsum_in = (int*)malloc(sizeof(int)*(NE+NI));
    int* synID_in = (int*)malloc(sizeof(int)*syN_sum_in);
    int syN_sum_i_in = 0;
    for (int n=0; n<(NE+NI); n++){

            int syN = neuron_syN_in[n];
            std::vector<int> con_in =  neuron_cons[n];
        
            int c = 0;
            for (int k=NE; k<(NE+NI); k++){
            if (con_in[k]==1){
                synID_in[syN_sum_i_in + c] =k;
                c += 1;
               }
            }
            // syN_cumsum_in[n] = syN_sum_i_in;
            syN_sum_i_in += syN;
       }   

     for (int n=0; n<NE+NI; n++){
        int syN_sum_i_ex = 0;         //total synapses calculated before ix neuron
        for (int m=0; m<n; m++){
            syN_sum_i_ex += neuron_syN_ex[m];
          }
          syN_cumsum_ex[n] = syN_sum_i_ex;
     }
 

     for (int n=0; n<NE+NI; n++){
        int syN_sum_i_in = 0;         //total synapses calculated before ix neuron
        for (int m=0; m<n; m++){
            syN_sum_i_in += neuron_syN_in[m];
          }
          syN_cumsum_in[n] = syN_sum_i_in;
     }
      printf("aab\n");

   


    //GPU device setup
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    //start timer
    double iStart = cpuSecond();


    
    
    //bytes
   
   int b_neuron = sizeof(double)*(NE+NI);
    
    int b_syn_ex = sizeof(double)*syN_sum_ex;
  
    int b_syn_in = sizeof(double)*syN_sum_in;

    int b_neuron_i =  sizeof(int)*(NE+NI);
   
    int b_syn_i_ex = syN_sum_ex * sizeof(int);
     int b_syn_i_in = syN_sum_in * sizeof(int);

   
    //malloc host memory
    double *h_na, *nk, *h,*m, *s_a, *s_g, *x_n, *s_n,  *z;
    
    nk = (double*)malloc(b_neuron);
    s_a = (double*)malloc(b_syn_ex);
    s_g = (double*)malloc(b_syn_in);
    x_n = (double*)malloc(b_syn_ex);
    s_n = (double*)malloc(b_syn_ex);
    z = (double*)malloc(b_neuron);
   

    if (nav==1){
        h_na = (double*)malloc(b_neuron); 
    } 
    if (kva==1){
        h = (double*)malloc(b_neuron); 
    }
     if (kvsi==1){
        m = (double*)malloc(b_neuron); 
    } 

    
    // malloc device global memory
    double  *d_h_na, *d_nk, *d_h, *d_m,  *d_s_a, *d_s_a2, *d_s_a3,*d_s_g, *d_s_g2, *d_s_g3;
    double *d_x_n, *d_x_n2, *d_x_n3,*d_s_n, *d_s_n2, *d_s_n3, *d_z;
    double *d_cpre2, *d_cpost2, *d_rho2;
    int *d_neuron_syN_ex, *d_synID_ex,  *d_syN_cumsum_ex, *d_neuron_syN_in, *d_synID_in,  *d_syN_cumsum_in;
    
    if (nav==1){
    CHECK(cudaMalloc((void **)&d_h_na, b_neuron));
    }
    
    CHECK(cudaMalloc((void **)&d_nk, b_neuron));
    if (kva==1){
    CHECK(cudaMalloc((void **)&d_h, b_neuron));
    }
    if (kvsi==1){
    CHECK(cudaMalloc((void **)&d_m, b_neuron));
    }

    CHECK(cudaMalloc((void **)&d_s_a, b_syn_ex));
    CHECK(cudaMalloc((void **)&d_s_a2, b_syn_ex));
    CHECK(cudaMalloc((void **)&d_s_a3, b_syn_ex));
    CHECK(cudaMalloc((void **)&d_s_g, b_syn_in));
    CHECK(cudaMalloc((void **)&d_s_g2, b_syn_in));
    CHECK(cudaMalloc((void **)&d_s_g3, b_syn_in));

    CHECK(cudaMalloc((void **)&d_x_n, b_syn_ex));
    CHECK(cudaMalloc((void **)&d_x_n2, b_syn_ex));
    CHECK(cudaMalloc((void **)&d_x_n3, b_syn_ex));
    CHECK(cudaMalloc((void **)&d_s_n, b_syn_ex));
    CHECK(cudaMalloc((void **)&d_s_n2, b_syn_ex));
    CHECK(cudaMalloc((void **)&d_s_n3, b_syn_ex));
    
    CHECK(cudaMalloc((void **)&d_z, b_neuron));

    CHECK(cudaMalloc((void **)&d_cpre2, b_syn_ex));
  
    CHECK(cudaMalloc((void **)&d_cpost2, b_syn_ex));
 
    CHECK(cudaMalloc((void **)&d_rho2, b_syn_ex));


    CHECK(cudaMalloc((void **)&d_neuron_syN_ex, b_neuron_i));
    CHECK(cudaMalloc((void **)&d_synID_ex, b_syn_i_ex));
    CHECK(cudaMalloc((void **)&d_syN_cumsum_ex, b_neuron_i));
    CHECK(cudaMalloc((void **)&d_neuron_syN_in, b_neuron_i));
    CHECK(cudaMalloc((void **)&d_synID_in, b_syn_i_in));
    CHECK(cudaMalloc((void **)&d_syN_cumsum_in, b_neuron_i));
    
 
    //memcopy
    CHECK(cudaMemcpy(d_neuron_syN_ex, neuron_syN_ex, b_neuron_i, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_synID_ex, synID_ex, b_syn_i_ex, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_syN_cumsum_ex, syN_cumsum_ex,b_neuron_i,cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_neuron_syN_in, neuron_syN_in, b_neuron_i, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_synID_in, synID_in, b_syn_i_in, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_syN_cumsum_in, syN_cumsum_in, b_neuron_i, cudaMemcpyHostToDevice));

    

    double dt = dt_start;
        int *flag;
        flag = (int*)malloc(sizeof(int));
        flag[0]=0;
        int rep=0;

        while(1){
            if (rep>9){
        break; //while
        }

        int Ltp = Tp/dt+1 ;
        int b_neuron_t = sizeof(double)*(NE+NI)*Ltp;
        int b_syn_ex_t = sizeof(double)*syN_sum_ex*Ltp;


    double *v, *ca, *Sp, *K, *cpre, *cpost, *rho, *r, *r_no;
    v = (double*)malloc(b_neuron_t);
    ca = (double*)malloc(b_neuron_t);
    cpre = (double*)malloc(b_syn_ex_t);
    cpost = (double*)malloc(b_syn_ex_t);
    Sp = (double*)malloc(b_neuron_t);
    K = (double*)malloc(b_neuron_t);
    r_no = (double*)malloc(b_neuron_t);
    rho = (double*)malloc(b_syn_ex_t);
    r = (double*)malloc(b_syn_ex_t);

     //initialization of variables
    for (int i = 0; i < (NE+NI); i++) {

        double v_r, h_na_r, n_r, h_r,m_r, ca_r;
        if (init == "r"){
            v_r = ((double)rand()/RAND_MAX)*(-70);
            
            n_r = (double)rand()/RAND_MAX;
            h_r = (double)rand()/RAND_MAX;
            ca_r = (double)rand()/RAND_MAX*(10);

            v[i*Ltp]= v_r; 
            
            if (nav==1){
            h_na_r = (double)rand()/RAND_MAX;
            h_na[i] = h_na_r;
            }

            nk[i] = n_r;

            if (kva==1){
            h_r = (double)rand()/RAND_MAX;
            h[i] = h_r;
            }
             if (kvsi==1){
            m_r = (double)rand()/RAND_MAX;
            m[i] = m_r;
            }

            ca[i*Ltp]= ca_r;
        }
        else{

            v[i*Ltp]= -45;//v_r; 
            if (nav==1){
            h_na[i] = 0.045;//n_r;
            }
            nk[i] = 0.54;//n_r;
            if (kva==1){
            h[i] = 0.045;//n_r;
            }
             if (kvsi==1){
            m[i] = 0.34;//n_r;
            }
            ca[i*Ltp]= 1;//ca_r; 
        }
    }

      for (int i = 0; i < syN_sum_ex; i++) {
                   s_a[i] = 0.01;  
                   x_n[i] = 0.01;
                   s_n[i] = 0.01;
                   
                //    S[i] = S_ini;

                   for (int it = 0; it < Ltp; it++){
                    cpre[i*Ltp+ it] = 0.;
                    cpost[i*Ltp+ it] = 0.;
                  
                   rho[i*Ltp+ it] = rho_ini;
            }
        }

    for (int i = 0; i < (NE+NI); i++) {
     Sp[i*Ltp] = Sp_ini;
     K[i*Ltp] = K_ini;
     z[i] = z_ini;

    }

    for (int i = 0; i < syN_sum_in; i++) {
                   s_g[i] = 0.01;  
        }

    double *d_v, *d_ca;
    double *d_Sp,  *d_K,  *d_r_no;
    double *d_cpre, *d_cpost, *d_rho, *d_r;


    CHECK(cudaMalloc((void **)&d_v, b_neuron_t));
    CHECK(cudaMalloc((void **)&d_ca, b_neuron_t));
    CHECK(cudaMalloc((void **)&d_r_no, b_neuron_t));
    CHECK(cudaMalloc((void **)&d_Sp, b_neuron_t));
    CHECK(cudaMalloc((void **)&d_K, b_neuron_t));
    CHECK(cudaMalloc((void **)&d_cpre, b_syn_ex_t));
    CHECK(cudaMalloc((void **)&d_cpost, b_syn_ex_t));
    CHECK(cudaMalloc((void **)&d_rho, b_syn_ex_t));
    CHECK(cudaMalloc((void **)&d_r, b_syn_ex_t));

    if (nav==1){
     CHECK(cudaMemcpy(d_h_na, h_na, b_neuron, cudaMemcpyHostToDevice));
            }
    CHECK(cudaMemcpy(d_nk, nk, b_neuron, cudaMemcpyHostToDevice));
    if (kva==1){
    CHECK(cudaMemcpy(d_h, h, b_neuron, cudaMemcpyHostToDevice));
    }
    if (kvsi==1){
    CHECK(cudaMemcpy(d_m, m, b_neuron, cudaMemcpyHostToDevice));
    }
    
     CHECK(cudaMemcpy(d_s_a, s_a, b_syn_ex, cudaMemcpyHostToDevice));
     CHECK(cudaMemcpy(d_s_g, s_g, b_syn_in, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_x_n, x_n, b_syn_ex, cudaMemcpyHostToDevice));
     CHECK(cudaMemcpy(d_s_n, s_n, b_syn_ex, cudaMemcpyHostToDevice));
     CHECK(cudaMemcpy(d_z, z, b_neuron, cudaMemcpyHostToDevice));

      //calculte by tb
   
    std::string dir = "con_cu_i/"+model_name+"/";
    std::string savedir;
    if(tau_s!=0){
       
            #ifdef CON_LOGNORMAL
                savedir =  dir +"param_"+param_i  +"/NE"+std::to_string(NE)+"_"+ "NI"+std::to_string(NI) + "/conM"+ con_M_str +"_"+"conSD"+con_ex_ex_str +"_conMin"+ con_M_in_str +"_"+"conSDin"+con_in_in_str+"/Spini"+Sp_ini_str+ "_Kini"+K_ini_str+"_zini"+z_ini_str +"/exp"+alpha_str+"_"+beta_str+ "_"+adp_str+"_"+bdp_str+ "_"+adp2_str+"_"+bdp2_str+ "/taucamk"+tau_camk_str+"_tauk"+tau_k_str+"/w"+w_str+"_b"+b_str+"_I"+I_str+"_th"+th_str+"_sign"+sig_n_str +"/ex"+ex_str+"_exin"+ex_in_str+"/maxcon"+max_con_str+"_sminampa"+smin_ampa_str+"_smaxampa"+smax_ampa_str+"/thp"+th_p_str+"_thd"+th_d_str+"_gp"+gp_str+"_gd"+gd_str+"_taupre"+tau_pre_str+"_taupost"+tau_post_str+"_sig"+sig_str+"/tausinv_"+tau_s_str+"/seed_" +std::to_string(seed)+"/init_" +init+"/T"+std::to_string(T)+"_Tp"+std::to_string(Tp)+"/";
            #else
                savedir =  dir +"param_"+param_i  +"/thp"+th_p_str+"_thd"+th_d_str+"_gp"+gp_str+"_gd"+gd_str+"_taupre"+tau_pre_str+"_taupost"+tau_post_str+"_sig"+sig_str+"_alpha"+alpha_str+"/kmag"+a_str+ "_caMT"+caM_tot_str +"_Kcam"+Kcam_str+ "_Kg"+Kg_str+"/s00ini"+s00_ini_str+"_s11ini"+s11_ini_str+"_s01ini"+s01_ini_str+"_eini"+e_ini_str +"_dini"+d_ini_str +"/smaxcon"+smax_con_str+"_rhoini"+rho_ini_str+"_smaxampa"+smax_ampa_str+"/NE"+std::to_string(NE)+"_"+ "NI"+std::to_string(NI) + "/con_th"+con_ex_ex_str+"_"+con_ex_in_str +"_"+con_in_ex_str+"_"+con_in_in_str +"/seed_" +std::to_string(seed)+"/init_" +init+"/T"+std::to_string(T)+"_Tp"+std::to_string(Tp)+"/";
            #endif
        
    }else{

            #ifdef CON_LOGNORMAL
                savedir =  dir +"param_"+param_i  +"/NE"+std::to_string(NE)+"_"+ "NI"+std::to_string(NI) + "/conM"+ con_M_str +"_"+"conSD"+con_ex_ex_str +"_conMin"+ con_M_in_str +"_"+"conSDin"+con_in_in_str+"/Spini"+Sp_ini_str+ "_Kini"+K_ini_str+"_zini"+z_ini_str +"/exp"+alpha_str+"_"+beta_str+ "_"+adp_str+"_"+bdp_str+ "_"+adp2_str+"_"+bdp2_str+ "/taucamk"+tau_camk_str+"_tauk"+tau_k_str+"/w"+w_str+"_b"+b_str+"_I"+I_str+"_th"+th_str+"_sign"+sig_n_str+"/ex"+ex_str+"_exin"+ex_in_str+"/maxcon"+max_con_str+"_smaxampa"+smax_ampa_str+"/taupre"+tau_pre_str+"_taupost"+tau_post_str+"/seed_" +std::to_string(seed)+"/init_" +init+"/T"+std::to_string(T)+"_Tp"+std::to_string(Tp)+"/";
            #else
                savedir =  dir +"param_"+param_i  +"/NE"+std::to_string(NE)+"_"+ "NI"+std::to_string(NI) + "/conM"+ con_M_str +"_"+"conSD"+con_ex_ex_str +"_conMin"+ con_M_in_str +"_"+"conSDin"+con_in_in_str+"/taupre"+tau_pre_str+"_taupost"+tau_post_str+"/kmag"+a_str+ "_caMT"+caM_tot_str +"_Kcam"+Kcam_str+ "_Kg"+Kg_str+"/s00ini"+s00_ini_str+"_s11ini"+s11_ini_str+"_s01ini"+s01_ini_str+"_eini"+e_ini_str +"_dini"+d_ini_str +"/smaxcon"+smax_con_str+"_smaxampa"+smax_ampa_str+"/NE"+std::to_string(NE)+"_"+ "NI"+std::to_string(NI) + "/con_th"+con_ex_ex_str+"_"+con_ex_in_str +"_"+con_in_ex_str+"_"+con_in_in_str +"/seed_" +std::to_string(seed)+"/init_" +init+"/T"+std::to_string(T)+"_Tp"+std::to_string(Tp)+"/";
            #endif
   
    }

    std::string savedir_neu = savedir + "neu/";
    std::string savedir_syn = savedir + "syn/";
    std::filesystem::create_directories(savedir_neu);  //make dirs reccurently
    std::filesystem::create_directories(savedir_syn);
    std::cout << savedir << std::endl;

    std::mt19937 mt5;            // メルセンヌ・ツイスタの32ビット版
    mt5.seed(seed);
    std::normal_distribution<double> distribution5(0, 1);

     int f_flag=0;
    for (int tb=0; tb<tbs; tb++){
    //  if (it != 0){
            printf("tb: %d\n", tb);


            for (int i=0; i<NE+NI; ++i) {
                 for (int it = 0; it < Ltp; it++) {
                double number = distribution5(mt5);
                // printf("number1 %f\n",number);
            #ifdef RANDOM
                r_no[i*Ltp + it] = number;
            #else
                r_no[i*Ltp + it] =constant_num; 
                // printf("%d\n",i*Ltp+it);
            #endif
                 }
            }


              for (int i=0; i<syN_sum_ex; ++i) {
                 for (int it = 0; it < Ltp; it++) {
            double number2 = distribution5(mt5);
                // printf("number2 %f\n",number2);
            #ifdef RANDOM
                r[i*Ltp + it] = number2;
            #else
                r[i*Ltp + it] =constant_num; 
                // printf("%d\n",i*Ltp+it);
            #endif
                 }
               }

            CHECK(cudaMemcpy(d_r_no, r_no, b_neuron_t, cudaMemcpyHostToDevice));

            // transfer data from host to device
            CHECK(cudaMemcpy(d_v, v, b_neuron_t, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_ca, ca, b_neuron_t, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_cpre, cpre, b_syn_ex_t, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_cpost, cpost, b_syn_ex_t, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_Sp, Sp, b_neuron_t, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_K, K, b_neuron_t, cudaMemcpyHostToDevice));
           CHECK(cudaMemcpy(d_rho, rho, b_syn_ex_t, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_r, r, b_syn_ex_t, cudaMemcpyHostToDevice));
            
            // invoke kernel at host side
            int dimx = (NE+NI)/blockdim_x;
            int dimy = 1;
            dim3 block(blockdim_x, 1, 1);
            dim3 grid(dimx, dimy);

            dt_gpu_NaK1_n<<<grid, block>>>(d_v, d_h_na, d_nk, d_h, d_m, d_ca, d_s_a, d_s_a2, d_s_a3,d_s_g, d_s_g2, d_s_g3, 
                            d_x_n, d_x_n2, d_x_n3, d_s_n, d_s_n2, d_s_n3, d_cpre, d_cpre2, d_cpost, d_cpost2,//d_S, d_S2, d_S3, 
                            d_Sp,  d_K,  d_z, d_r_no,  d_rho, d_rho2, d_r, max_con, rho_ini, smin_ampa, smax_ampa,
                            d_neuron_syN_ex, d_synID_ex, d_syN_cumsum_ex, d_neuron_syN_in, d_synID_in, d_syN_cumsum_in,  
                            alpha,beta,   adp, bdp ,adp2,bdp2,tau_camk, tau_k, w,  b, I,  th, sig_n,
                            NE+NI, Ltp,  dt, 
                            g_leak, g_nav, g_kvhh, g_kva, g_kvsi, g_cav,g_kca, g_nap,g_kir, g_ampar, g_nmdar, g_gabar,tau_ca,
                            cpre_r, cpost_r, th_p,th_d, gp, gd,tau_ca_pre, tau_ca_post, sig,  tau_s);
              
            

            CHECK(cudaDeviceSynchronize());
        
           // check kernel error
            CHECK(cudaGetLastError());

            // copy kernel result back to host side
            CHECK(cudaMemcpy(v, d_v, b_neuron_t, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(ca, d_ca, b_neuron_t, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(cpre, d_cpre, b_syn_ex_t, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(cpost, d_cpost, b_syn_ex_t, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(Sp, d_Sp, b_neuron_t, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(K, d_K, b_neuron_t, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(rho, d_rho, b_syn_ex_t, cudaMemcpyDeviceToHost));
        

             for (int n=0; n<NE+NI; n++){
                    if(isnan(v[n*Ltp + Ltp-1]))
                    {   
                        printf("ex%f,  nan error\n",ex);
                        f_flag=1;
                        rep+=1;
                        break;
                    }
                    };
            if(f_flag==1){
                dt=dt_li[rep];
                break;  //for tbs
            }


        for (int i=0; i<NE+NI; i++){
            std::string savef = savedir_neu+"ex"+ex_str +"_"+"ex_in"+ex_in_str+"_"+"N"+std::to_string(i)+"_"+std::to_string(tb)+".bin";
            // std::cout << savef << std::endl; 
            std::ofstream ofs;
            ofs.open(savef, std::ios::out|std::ios::binary|std::ios::trunc);
            if (!ofs) {
            std::cout << "Can't open a file"<<savef<<std::endl;
            }
                
            for (int it=0; it<Ltp-1; it++){
                ofs.write(( char * ) &v[i*Ltp+it],sizeof( double ) );
                }//for
            ofs.close();

            std::string save_ca = savedir_neu+"ex"+ex_str +"_"+"ex_in"+ex_in_str+"_"+"ca_"+std::to_string(i)+"_"+std::to_string(tb)+".bin";
            std::ofstream ofs_ca;
            ofs_ca.open(save_ca, std::ios::out|std::ios::binary|std::ios::trunc);
            if (!ofs_ca) {
            std::cout << "Can't open a file"<<save_ca<<std::endl;
            }
            // std::cout <<save_cpre<<std::endl;
            for (int it=0; it<Ltp-1; it++){
                ofs_ca.write(( char * ) &ca[i*Ltp+it],sizeof( double ) );
                }//for
            ofs_ca.close();

            std::string save_Sp = savedir_neu+"ex"+ex_str +"_"+"ex_in"+ex_in_str+"_"+"Sp"+std::to_string(i)+"_"+std::to_string(tb)+".bin";
            std::string save_K = savedir_neu+"ex"+ex_str +"_"+"ex_in"+ex_in_str+"_"+"K"+std::to_string(i)+"_"+std::to_string(tb)+".bin";
              
            std::ofstream ofs_Sp;
            ofs_Sp.open(save_Sp, std::ios::out|std::ios::binary|std::ios::trunc);
            if (!ofs_Sp) {
            std::cout << "Can't open a file"<<save_Sp<<std::endl;
            }
            // std::cout <<save_Sp<<std::endl;
            for (int it=0; it<Ltp-1; it++){
                ofs_Sp.write(( char * ) &Sp[i*Ltp+it],sizeof( double ) );
                }//for
            ofs_Sp.close();

            std::ofstream ofs_K;
            ofs_K.open(save_K, std::ios::out|std::ios::binary|std::ios::trunc);
            if (!ofs_K) {
            std::cout << "Can't open a file"<<save_K<<std::endl;
            }
            // std::cout <<save_K<<std::endl;
            for (int it=0; it<Ltp-1; it++){
                ofs_K.write(( char * ) &K[i*Ltp+it],sizeof( double ) );
                }//for
            ofs_K.close();

            //save syn values
            int syN_ex = neuron_syN_ex[i];
            int syN_cumsum_ex_i = syN_cumsum_ex[i];
            for (int s=syN_cumsum_ex_i; s<syN_cumsum_ex_i+syN_ex; s++){
                int pre_synID = synID_ex[s];

                std::string save_cpre = savedir_syn+"ex"+ex_str +"_"+"ex_in"+ex_in_str+"_"+"cpre"+std::to_string(pre_synID)+"-"+std::to_string(i)+"_"+std::to_string(tb)+".bin";
                std::string save_cpost = savedir_syn+"ex"+ex_str +"_"+"ex_in"+ex_in_str+"_"+"cpost"+std::to_string(pre_synID)+"-"+std::to_string(i)+"_"+std::to_string(tb)+".bin";
                
                  std::string save_rho = savedir_syn+"ex"+ex_str +"_"+"ex_in"+ex_in_str+"_"+"rho"+std::to_string(pre_synID)+"-"+std::to_string(i)+"_"+std::to_string(tb)+".bin";
                
                // std::cout << save_rho << std::endl;

                std::ofstream ofs_cpre;
                ofs_cpre.open(save_cpre, std::ios::out|std::ios::binary|std::ios::trunc);
                if (!ofs_cpre) {
                std::cout << "Can't open a file"<<save_cpre<<std::endl;
                }
                // std::cout <<save_cpre<<std::endl;
                for (int it=0; it<Ltp-1; it++){
                    ofs_cpre.write(( char * ) &cpre[s*Ltp+it],sizeof( double ) );
                    }//for
                ofs_cpre.close();

                std::ofstream ofs_cpost;
                ofs_cpost.open(save_cpost, std::ios::out|std::ios::binary|std::ios::trunc);
                if (!ofs_cpost) {
                std::cout << "Can't open a file"<<save_cpost<<std::endl;
                }
                // std::cout <<save_cpost<<std::endl;
                for (int it=0; it<Ltp-1; it++){
                    ofs_cpost.write(( char * ) &cpost[s*Ltp+it],sizeof( double ) );
                    }//for
                ofs_cpost.close();

    

                std::ofstream ofs_rho;
                ofs_rho.open(save_rho, std::ios::out|std::ios::binary|std::ios::trunc);
                if (!ofs_rho) {
                std::cout << "Can't open a file"<<save_rho<<std::endl;
                }
                // std::cout <<save_rho<<std::endl;
                for (int it=0; it<Ltp-1; it++){
                    ofs_rho.write(( char * ) &rho[s*Ltp+it],sizeof( double ) );
                    }//for
                ofs_rho.close();
            } //for 
        }//for
        

        for (int i = 0; i < (NE+NI); i++) {
            ca[i*Ltp] = ca[i*Ltp + Ltp-1];
            v[i*Ltp] = v[i*Ltp + Ltp-1];
            Sp[i*Ltp] = Sp[i*Ltp + Ltp-1];
            K[i*Ltp] = K[i*Ltp + Ltp-1];
         }
         for (int i = 0; i < syN_sum_ex; i++) {
            cpre[i*Ltp] = cpre[i*Ltp + Ltp-1];
            cpost[i*Ltp] = cpost[i*Ltp + Ltp-1];
            
             rho[i*Ltp] = rho[i*Ltp + Ltp-1];
         }

        if(tb==tbs-1){
          
            flag[0]=1;
        }
     } //for tbs
      
    
    CHECK(cudaFree(d_v));
    CHECK(cudaFree(d_ca));
    CHECK(cudaFree(d_r_no));

    CHECK(cudaFree(d_Sp));
 
    CHECK(cudaFree(d_K));
  
    CHECK(cudaFree(d_cpre));
 
    CHECK(cudaFree(d_cpost));

        CHECK(cudaFree(d_rho));
    CHECK(cudaFree(d_r));

    free(v);
    free(Sp);
    free(K);

    free(cpre);
    free(cpost);
     free(rho);
    free(r);
      free(r_no);


      if(flag[0]==1){
            printf("ex%f  end\n",ex);
                break;  //while
            }
        

    }//while

    free(flag);
    //release host device memory
  
    if (nav==1){
    CHECK(cudaFree(d_h_na));
    }
    CHECK(cudaFree(d_nk));
    if (kva==1){
    CHECK(cudaFree(d_h));
    }
    if (kvsi==1){
    CHECK(cudaFree(d_m));
    }
    
    CHECK(cudaFree(d_s_a));
    CHECK(cudaFree(d_s_a2));
    CHECK(cudaFree(d_s_a3));
    CHECK(cudaFree(d_s_g));
    CHECK(cudaFree(d_s_g2));
    CHECK(cudaFree(d_s_g3));
    CHECK(cudaFree(d_x_n));
    CHECK(cudaFree(d_x_n2));
    CHECK(cudaFree(d_x_n3));
    CHECK(cudaFree(d_s_n));
    CHECK(cudaFree(d_s_n2));
    CHECK(cudaFree(d_s_n3));
    CHECK(cudaFree(d_z));
  
   
    CHECK(cudaFree(d_cpre2));
 
    CHECK(cudaFree(d_cpost2));
    CHECK(cudaFree(d_rho2));
  
  
    CHECK(cudaFree(d_synID_ex));
    CHECK(cudaFree(d_neuron_syN_ex));
    CHECK(cudaFree(d_syN_cumsum_ex));
    CHECK(cudaFree(d_synID_in));
    CHECK(cudaFree(d_neuron_syN_in));
    CHECK(cudaFree(d_syN_cumsum_in));

   
    if (nav==1){
    free(h_na);
    }
    free(nk);
    if (kva==1){
    free(h);
    }
    if (kvsi==1){
    free(m);
    }
 
    free(s_a);
    free(s_g);
    free(x_n);
    free(s_n);
    free(z);

    free(neuron_syN_ex);
    free(synID_ex);
    free(syN_cumsum_ex);
    free(neuron_syN_in);
    free(synID_in);
    free(syN_cumsum_in);
    free(dt_li);


    //end timer
    double iElaps = cpuSecond() - iStart;
    printf("elapsed %f sec\n", iElaps);
   
}
