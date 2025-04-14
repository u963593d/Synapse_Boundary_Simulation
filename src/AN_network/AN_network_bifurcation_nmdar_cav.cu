#define CON_LOGNORMAL

#define NMDAR


// #define CAV   //VGCC
#define NO_IN_IN  //without IN-IN connection
 
 
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


//nvc++ -mp -fast -ta=tesla AN_network_bifurcation_nmdar_cav.cu -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/examples/OpenACC/SDK/include/ -I/tmp/nvhpc_2023_223_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/include -L/tmp/nvhpc_2023_223_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/lib -lcudart -std=c++20 -o AN_net_bifurcation_nmdar
//nvc++ -mp -fast -ta=tesla AN_network_bifurcation_nmdar_cav.cu -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/examples/OpenACC/SDK/include/ -I/tmp/nvhpc_2023_223_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/include -L/tmp/nvhpc_2023_223_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/lib -lcudart -std=c++20 -o AN_net_bifurcation_cav



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

#define i_nmda(g_nmdar, v, s) (g_nmdar * s * (v-vNMDAR))



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



__device__ double dvdt_cal_NaK1_n(double v, double h_na, double nk, double h, double m, double ca, double dt, double i_ampar, double i_nmdar, double i_gabar, double g_leak, double g_nav, double g_kvhh,double g_kva,double g_kvsi, double g_cav,double g_kca,double g_nap, double g_kir, double tau_ca)
{    double i_nav = Nav_i_cal(g_nav, v, h_na);
    double dvdt = ((-10.0*area * (i_leak(g_leak, v) + i_nav +  i_kvhh(g_kvhh, v, nk)  + i_kva(g_kva, v, h) + i_kvsi(g_kvsi, v, m) + i_cav(g_cav, v)  + i_kca(g_kca, v, ca)  + i_nap(g_nap, v) + i_kir(g_kir, v))  - (i_ampar+i_nmdar+i_gabar)) / (10.0*cm*area))*dt;
   return dvdt;
}



__device__ double dcadt_cal(double v, double ca, double dt, double g_cav, double tau_ca){
    // double m_inf_ca = 1.0 / (1.0 + std::exp(-(v+20.0)/9.0));
    // double i_cav= g_cav * i_cav_e(v);
    double dcadt = (-a_ca * 10 * area * i_cav(g_cav, v) - ca/tau_ca)*dt;
    return dcadt;
}

__device__ double dcadt_cal_n(double v, double ca, double dt, double g_cav, double tau_ca){
    // double m_inf_ca = 1.0 / (1.0 + std::exp(-(v+20.0)/9.0));
    // double i_cav= g_cav * i_cav_e(v);
    double dcadt = (-a_ca * 10 * area * i_cav(g_cav, v) - ca/tau_ca)*dt;
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

__device__ double dsdt_g_cal(double v, double s_g, double dt){
        // double fv = 1.0 / (1.0 + std::exp(-(v-20)/2.0));
        double dsdt_g = (s_a_gabar * fv(v) - s_g/s_tau_gabar)*dt;
     return dsdt_g;
}



__global__ void dt_gpu_NaK1_n(double *v,  double *h_na, double *nk, double *h, double *m,double *ca, double *s_a, double *s_a2, double *s_a3, double *s_g, double *s_g2, double *s_g3, double *x_n, double *x_n2, double *x_n3, double *s_n, double *s_n2, double *s_n3, int *neuron_syn_ex, int *synID_ex, int *syN_cumsum_ex, int *neuron_syn_in, int *synID_in, int *syN_cumsum_in, int N, int T, double dt,  
                                             double g_leak, double g_nav,double g_kvhh,double g_kva, double g_kvsi, double g_cav,double g_kca, double g_nap, double g_kir, double g_ampar, double g_nmdar ,double g_gabar, double tau_ca)
{
   //v_nw: updated v
   //s_a: array for gating variables of AMPAR of each synapse
   //s_a2: next s_a
   //s_a3: updated s_a3
   //neuron_syN: array for the number of synapses per neuron
   //synID: array for index list of pre-synapse 
   
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
        double ca_n = ca[ix];
      
        
        int syN_ex = neuron_syn_ex[ix];  //the number of ex synapse
        int syN_cumsum_i_ex = syN_cumsum_ex[ix] ;         //total synapses calculated before ix neuron
        
        int syN_in = neuron_syn_in[ix];  //the number of in synapse
        int syN_cumsum_i_in = syN_cumsum_in[ix] ; 

        
        // k1
        //total current from excitatory synapses
        double i_ampar =0;       
        double i_nmdar =0;
        for (int i= syN_cumsum_i_ex; i<(syN_cumsum_i_ex+syN_ex); i++){
           
            s_a3[i] = s_a[i];
            x_n3[i] = x_n[i];
            s_n3[i] = s_n[i];
            double dsdt_a = dsdt_a_cal(v[synID_ex[i]*T + it-1], s_a3[i], dt);
            double dxdt_n = dxdt_n_cal(v[synID_ex[i]*T + it-1], x_n3[i], dt);
            double dsdt_n = dsdt_n_cal(x_n3[i], s_n3[i], dt);
            i_ampar += g_ampar * s_a3[i] * (v_n-vAMPAR);
            #ifdef MG_BLOCK
                i_nmdar += g_nmdar * s_n3[i] * (v_n-vNMDAR) * mg_v(v_n);
            #else
                i_nmdar += g_nmdar * s_n3[i] * (v_n-vNMDAR);
            #endif
            s_a2[i] =  s_a3[i] + 0.5*dsdt_a;   //next s_a
            s_a[i] =  s_a3[i] + dsdt_a/6.0;   //update s_a
            x_n2[i] =  x_n3[i] + 0.5*dxdt_n;   
            x_n[i] =  x_n3[i] + dxdt_n/6.0;
            s_n2[i] =  s_n3[i] + 0.5*dsdt_n;   
            s_n[i] =  s_n3[i] + dsdt_n/6.0;
            
        }

        //total current from inhibitory synapses
        double i_gabar =0;       
        for (int i= syN_cumsum_i_in; i<(syN_cumsum_i_in+syN_in); i++){
            s_g3[i] = s_g[i];
            double dsdt_g = dsdt_g_cal(v[synID_in[i]*T + it-1], s_g3[i], dt);
            i_gabar += g_gabar * s_g3[i] * (v_n-vGABAR);
            s_g2[i] =  s_g3[i] + 0.5*dsdt_g;   
            s_g[i] =  s_g3[i] + dsdt_g/6.0;   
        }

        double dvdt =  dvdt_cal_NaK1_n(v_n, h_na_n, nk_n, h_n,m_n, ca_n, dt, i_ampar,i_nmdar, i_gabar, g_leak, g_nav,g_kvhh,g_kva,g_kvsi,g_cav,g_kca,g_nap,g_kir,tau_ca);
        double dhdt_na = dhdt_na_cal(v_n, h_na_n, dt);
        double dndt = dndt_cal(v_n, nk_n, dt);
        double dhdt = dhdt_cal(v_n, h_n, dt);
        double dmdt = dmdt_cal(v_n, m_n, dt);
        double dcadt =  dcadt_cal_n(v_n, ca_n, dt, g_cav, tau_ca);

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
        ca[ix] += dcadt/6.0;  //update ca
         
         //k2
       i_ampar = 0; 
       i_nmdar = 0;
         for (int i= syN_cumsum_i_ex; i<(syN_cumsum_i_ex + syN_ex); i++){
            double dsdt_a = dsdt_a_cal(v[synID_ex[i]*T+it-1], s_a2[i], dt);
            double dxdt_n = dxdt_n_cal(v[synID_ex[i]*T + it-1], x_n2[i], dt);
            double dsdt_n = dsdt_n_cal(x_n2[i], s_n2[i], dt);
            i_ampar += g_ampar * s_a2[i]*(v2-vAMPAR);

            #ifdef MG_BLOCK
                i_nmdar += g_nmdar * s_n2[i] * (v2-vNMDAR) * mg_v(v2);
            #else
                i_nmdar += g_nmdar * s_n2[i] * (v2-vNMDAR);
            #endif
            s_a2[i] = s_a3[i] + 0.5*dsdt_a;
            s_a[i] += dsdt_a/3.0;
            x_n2[i] =  x_n3[i] + 0.5*dxdt_n;   
            x_n[i] += dxdt_n/3.0;
            s_n2[i] =  s_n3[i] + 0.5*dsdt_n;   
            s_n[i] +=  dsdt_n/3.0;
           
        }
        i_gabar = 0; 
         for (int i= syN_cumsum_i_in; i<(syN_cumsum_i_in + syN_in); i++){
            double dsdt_g = dsdt_g_cal(v[synID_in[i]*T+it-1], s_g2[i], dt);
            i_gabar += g_gabar * s_g2[i]*(v2-vGABAR);
            s_g2[i] = s_g3[i] + 0.5*dsdt_g;
            s_g[i] += dsdt_g/3.0;
        }
        
         dvdt =  dvdt_cal_NaK1_n(v2, h_na2, nk2, h2,m2, ca2,  dt, i_ampar, i_nmdar, i_gabar, g_leak, g_nav,g_kvhh,g_kva,g_kvsi,g_cav,g_kca,g_nap,g_kir,tau_ca);
          dhdt_na = dhdt_na_cal(v2, h_na2, dt);
         dndt = dndt_cal(v2, nk2, dt);
         dhdt = dhdt_cal(v2, h2, dt);
         dmdt = dmdt_cal(v2, m2, dt);
         dcadt = dcadt_cal_n(v2, ca2, dt, g_cav, tau_ca);   
        
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
        ca[ix] += dcadt/3.0;

        //k3
         i_ampar = 0; 
       i_nmdar = 0;
         for (int i= syN_cumsum_i_ex; i<(syN_cumsum_i_ex + syN_ex); i++){
           
            double dsdt_a = dsdt_a_cal(v[synID_ex[i]*T+it-1], s_a2[i], dt);
            double dxdt_n = dxdt_n_cal(v[synID_ex[i]*T + it-1], x_n2[i], dt);
            double dsdt_n = dsdt_n_cal(x_n2[i], s_n2[i], dt);
            i_ampar += g_ampar * s_a2[i]*(v2-vAMPAR);
            #ifdef MG_BLOCK
                i_nmdar += g_nmdar * s_n2[i] * (v2-vNMDAR) * mg_v(v2);
            #else
                i_nmdar += g_nmdar * s_n2[i] * (v2-vNMDAR);
            #endif
            s_a2[i] = s_a3[i] + dsdt_a;
            s_a[i] += dsdt_a/3.0;
            x_n2[i] =  x_n3[i] + dxdt_n;   
            x_n[i] += dxdt_n/3.0;
            s_n2[i] =  s_n3[i] + dsdt_n;  
            s_n[i] +=  dsdt_n/3.0;
           
        }
        i_gabar = 0; 
         for (int i= syN_cumsum_i_in; i<(syN_cumsum_i_in + syN_in); i++){
            double dsdt_g = dsdt_g_cal(v[synID_in[i]*T+it-1], s_g2[i], dt);
            i_gabar += g_gabar * s_g2[i]*(v2-vGABAR);
            s_g2[i] = s_g3[i] + dsdt_g;
            s_g[i] += dsdt_g/3.0;
        }
        
         dvdt =  dvdt_cal_NaK1_n(v2, h_na2, nk2, h2, m2, ca2,  dt, i_ampar, i_nmdar, i_gabar, g_leak, g_nav,g_kvhh,g_kva,g_kvsi,g_cav,g_kca,g_nap,g_kir,tau_ca);
          
        dhdt_na = dhdt_na_cal(v2, h_na2, dt);
         dndt = dndt_cal(v2, nk2, dt);
         dhdt = dhdt_cal(v2, h2, dt);
         dmdt = dmdt_cal(v2, m2, dt);
         dcadt = dcadt_cal_n(v2, ca2, dt, g_cav, tau_ca); 

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
        ca[ix] += dcadt/3.0;

        //k4 
          i_ampar = 0; 
       i_nmdar = 0;
         for (int i= syN_cumsum_i_ex; i<(syN_cumsum_i_ex + syN_ex); i++){
            double dsdt_a = dsdt_a_cal(v[synID_ex[i]*T+it-1], s_a2[i], dt);
            double dxdt_n = dxdt_n_cal(v[synID_ex[i]*T + it-1], x_n2[i], dt);
            double dsdt_n = dsdt_n_cal(x_n2[i], s_n2[i], dt);
            i_ampar += g_ampar * s_a2[i]*(v2-vAMPAR);
            #ifdef MG_BLOCK
                i_nmdar += g_nmdar * s_n2[i] * (v2-vNMDAR) * mg_v(v2);
            #else
                i_nmdar += g_nmdar * s_n2[i] * (v2-vNMDAR);
            #endif
            s_a[i] += dsdt_a/6.0;
            
            x_n[i] += dxdt_n/6.0;
          
            s_n[i] +=  dsdt_n/6.0;
        }
        i_gabar = 0; 
         for (int i= syN_cumsum_i_in; i<(syN_cumsum_i_in + syN_in); i++){
            double dsdt_g = dsdt_g_cal(v[synID_in[i]*T+it-1], s_g2[i], dt);
            i_gabar += g_gabar * s_g2[i]*(v2-vGABAR);
            s_g[i] += dsdt_g/6.0;
        }
        
         dvdt =  dvdt_cal_NaK1_n(v2, h_na2, nk2, h2,m2, ca2,  dt, i_ampar,i_nmdar, i_gabar, g_leak, g_nav,g_kvhh,g_kva,g_kvsi,g_cav,g_kca,g_nap,g_kir,tau_ca);
           dhdt_na = dhdt_na_cal(v2, h_na2, dt);
         dndt = dndt_cal(v2, nk2, dt);
         dhdt = dhdt_cal(v2, h2, dt);
         dmdt = dmdt_cal(v2, m2, dt);
         dcadt = dcadt_cal_n(v2, ca2, dt, g_cav, tau_ca); 
        v[ix*T+it] += dvdt/6.0;
        h_na[ix] += dhdt_na/6.0;
        nk[ix] += dndt/6.0;
        h[ix] += dhdt/6.0;
        m[ix] += dmdt/6.0;
        ca[ix] += dcadt/6.0;

    } //ix
    __syncthreads();

  } // if
 } //for
}


int main(int argc, char * argv[]){
    int nav = 1;
    int kva = 1;
    int kvsi = 1;
    int kir = 1;
    int nmdar = 1;
    int gabar = 1;
    
    double g_leak,g_nav,g_kvhh, g_kva, g_kvsi,g_cav,g_nap,g_kca, g_kir, g_ampar, g_nmdar, g_gabar, tau_ca;
    

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

    // printf("g_gabar_li[0]: %f\n", g_gabar_li[0]);

    int NE = atoi(argv[g_c]);   //number of excitatory neurons
    g_c +=1;
    printf("NE\n", NE);

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

    int T = atoi(argv[g_c]);  //#ms
    g_c +=1;
    int Tp = atoi(argv[g_c]);  //#ms
    g_c +=1;

    int tbs = T/Tp;         

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
    
    
    double exp_ini = atof(argv[g_c]);
    std::string exp_ini_str = argv[g_c];
     g_c +=1;
    double exp_last = atof(argv[g_c]);
    std::string exp_last_str = argv[g_c];
    // std::string ex_str= argv[g_c];
    g_c +=1;
    double exp_step = atof(argv[g_c]);
    std::string exp_step_str = argv[g_c];
     g_c +=1;
   
     int cpu_num = atoi(argv[g_c]);
     g_c +=1;


     printf("g_ampar: %f\n",g_ampar);

    int seed = atoi(argv[g_c]);
    g_c +=1;
    srand(seed);
    std::string init = (argv[g_c]);
    g_c +=1;

    std::string param_i = (argv[g_c]);
    g_c +=1;
    std::string model_name = argv[g_c];
    g_c +=1;
    int blockdim_x = atoi(argv[g_c]);
    g_c +=1;

    printf("blockdim_x %d\n", blockdim_x);

    //channel_g distribution
   



//make connectgion matrix
std::vector<std::vector<int>> neuron_cons(NE+NI, std::vector<int>(NE+NI));
int* neuron_syN_ex = (int*)malloc(sizeof(int)*(NE+NI));  //array for the number of synapses per neuron
int* neuron_syN_in = (int*)malloc(sizeof(int)*(NE+NI));


#ifndef CON_LOGNORMAL
    if (con_ex_ex>=1)  //homogenous con
    { 
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
#else    //lognormal connections
            std::lognormal_distribution<double> distribution1(con_M, con_ex_ex);
            std::lognormal_distribution<double> distribution2(con_M, con_ex_in);
            std::lognormal_distribution<double> distribution3(con_M_in, con_in_ex);
            std::lognormal_distribution<double> distribution4(con_M_in, con_in_in);

            std::mt19937 mt1;
            std::mt19937 mt2;
            std::mt19937 mt3;
            std::mt19937 mt4;            
            mt1.seed(seed);
            mt2.seed(seed);
            mt3.seed(seed);
            mt4.seed(seed);

          
            //ex -> ex 
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
    // printf("total excitatory synapses: %d\n", syN_sum_ex);

    int syN_sum_in =0;
    for (int i = 0; i < (NE+NI); i++) {
        syN_sum_in += neuron_syN_in[i];
        }
    // printf("total inhibitory synapses: %d\n", syN_sum_in);
    
    //modify synaptic conductance
     printf("g_nmdar_pre: %f\n", g_nmdar);
      double syN_perN_ex = (double)(syN_sum_ex)/(NE+NI);
     double syN_perN_in =  (double)(syN_sum_in)/(NE+NI);

    g_ampar= g_ampar/syN_perN_ex; //(syN_sum_ex/(NE+NI));
    if (nmdar==1){
    g_nmdar= g_nmdar/syN_perN_ex; //(syN_sum_ex/(NE+NI));
    }
    printf("syN_perN_ex: %f\n", syN_perN_ex);
    // printf("syN_perN_in: %f\n", syN_perN_in);

    g_gabar= g_gabar/syN_perN_in; //(syN_sum_in/(NE+NI));
 
    printf("syN_perN_in: %f\n", syN_perN_in);
    printf("g_nmdar_net: %f\n", g_nmdar);
    printf("g_gabar_net: %f\n", g_gabar);

    //make excitatory pre-synaptic index list and cumulative counts of excitatory synapses
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

     //make inhibitory pre-synaptic index list and cumulative counts of inhibitory synapses
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

     for (int i=0; i<NE+NI; i++){
        // printf("syN_cumsum_ex[%d]: %d\n", i, syN_cumsum_ex[i]);
    }

     for (int i=0; i<NE+NI; i++){
        // printf("syN_cumsum_in[%d]: %d\n", i, syN_cumsum_in[i]);
    }

    //output pre-synaptic index
    for (int i=0; i<syN_sum_ex; i++){
        // printf("synID_ex[%d]: %d\n", i, synID_ex[i]);
    }

    for (int i=0; i<syN_sum_in; i++){
        // printf("synID_in[%d]: %d\n", i, synID_in[i]);
    }

    //GPU device setup
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    //start timer
    double iStart = cpuSecond();


    
    
    //bytes

   int b_neuron2 = sizeof(double)*(NE+NI);
    int b_syn_ex2 = sizeof(double)*syN_sum_ex;
    int b_syn_in2 = sizeof(double)*syN_sum_in;
    int b_neuron_i =  sizeof(int)*(NE+NI);
    int b_neuron_i_ex = (NE)* sizeof(int);
    int b_neuron_i_in = (NI)* sizeof(int);
    int b_syn_i_ex = syN_sum_ex * sizeof(int);
     int b_syn_i_in = syN_sum_in * sizeof(int);

    int *d_neuron_syN_ex, *d_synID_ex,  *d_syN_cumsum_ex, *d_neuron_syN_in, *d_synID_in,  *d_syN_cumsum_in;


    CHECK(cudaMalloc((void **)&d_neuron_syN_ex, b_neuron_i));
    CHECK(cudaMalloc((void **)&d_synID_ex, b_syn_i_ex));
    CHECK(cudaMalloc((void **)&d_syN_cumsum_ex, b_neuron_i));
    CHECK(cudaMalloc((void **)&d_neuron_syN_in, b_neuron_i));
    CHECK(cudaMalloc((void **)&d_synID_in, b_syn_i_in));
    CHECK(cudaMalloc((void **)&d_syN_cumsum_in, b_neuron_i));
    
    CHECK(cudaMemcpy(d_neuron_syN_ex, neuron_syN_ex, b_neuron_i, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_synID_ex, synID_ex, b_syn_i_ex, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_syN_cumsum_ex, syN_cumsum_ex, b_neuron_i, cudaMemcpyHostToDevice));
     CHECK(cudaMemcpy(d_neuron_syN_in, neuron_syN_in, b_neuron_i, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_synID_in, synID_in, b_syn_i_in, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_syN_cumsum_in, syN_cumsum_in, b_neuron_i, cudaMemcpyHostToDevice));


    std::string con_log_str;
    std::string dir = "./con_cu_i/"+model_name;

    #ifndef CON_LOGNORMAL
        std::string savedir = "./"+dir+"/param_"+param_i+"/"+"NE"+std::to_string(NE)+"_"+ "NI"+std::to_string(NI) + "/con_th"+con_ex_ex_str+"_"+con_ex_in_str +"_"+con_in_ex_str+"_"+con_in_in_str +"/seed_" +std::to_string(seed)+"/init_" +init+"/T_"+std::to_string(T)+"/Tp_"+std::to_string(Tp)+"/"; 
        con_log_str = "False";
    #else
        std::string savedir = "./"+dir+"/param_"+param_i+"/"+"NE"+std::to_string(NE)+"_"+ "NI"+std::to_string(NI) + "/conM"+ con_M_str +"_"+"conSD"+con_ex_ex_str +"_conMin"+ con_M_in_str +"_"+"conSDin"+con_in_in_str+"/seed_" +std::to_string(seed)+"/init_" +init+"/T_"+std::to_string(T)+"/Tp_"+std::to_string(Tp)+"/"; 
        con_log_str = "True";
    #endif
        std::filesystem::create_directories(savedir);  //make dirs reccurently



    std::cout << savedir << std::endl;
    
    std::vector<double> ex_v;
    int c = (exp_last-exp_ini)/exp_step+1;

    for (int ic=0; ic<c; ic++){
    double ex = exp_ini + ic*exp_step; 
    if (ex < exp_last){   
    ex_v.push_back(ex);
    printf("ex: %f\n", ex);
    }
  }

  #if defined(NMDAR)
    double g_nmdar_ori = g_nmdar;
  #elif defined(CAV)
    double g_cav_ori = g_cav;
  #endif

  
 
  #pragma omp parallel for num_threads(cpu_num)
  for (int k=0; k<ex_v.size(); k++){  //ex parallel start

    printf("thread = %d, i = %2d\n", omp_get_thread_num(),k);
    double ex = ex_v[k];
    double ex_in = ex;
    char ex_str[16];
	sprintf(ex_str,"%3.2f", ex);
    char ex_in_str[16];
	sprintf(ex_in_str,"%3.2f", ex_in);
    printf("ex: %f\n", ex);

    #if defined(NMDAR)
        printf("g_nmdar_ori: %f\n", g_nmdar_ori );
    #endif



    #if defined(NMDAR)
        double g_nmdar = g_nmdar_ori * (std::pow(10, ex));
    #elif defined(CAV)
        double g_cav = g_cav_ori * (std::pow(10, ex));
    #endif
    
    #if defined(NMDAR)
        printf("g_nmdar: %f\n", g_nmdar );
    #endif


    double *h_na, *nk, *h,*m, *ca, *s_a, *x_n, *s_n, *s_g;
    nk = (double*)malloc(b_neuron2);
    ca = (double*)malloc(b_neuron2);
    s_a = (double*)malloc(b_syn_ex2);

    if (nmdar==1){
    x_n = (double*)malloc(b_syn_ex2);
    s_n = (double*)malloc(b_syn_ex2);
    }
    s_g = (double*)malloc(b_syn_in2);
    


    if (nav==1){
        h_na = (double*)malloc(b_neuron2); 
    } 
    if (kva==1){
        h = (double*)malloc(b_neuron2); 
    }
     if (kvsi==1){
        m = (double*)malloc(b_neuron2); 
    } 

    double *d_h_na, *d_nk, *d_h, *d_m, *d_ca, *d_s_a, *d_s_a2, *d_s_a3,*d_s_g, *d_s_g2, *d_s_g3;
    double *d_x_n, *d_x_n2, *d_x_n3,*d_s_n, *d_s_n2, *d_s_n3;
    if (nav==1){
    CHECK(cudaMalloc((void **)&d_h_na, b_neuron2));
    }
    CHECK(cudaMalloc((void **)&d_nk, b_neuron2));
    if (kva==1){
    CHECK(cudaMalloc((void **)&d_h, b_neuron2));
    }
    if (kvsi==1){
    CHECK(cudaMalloc((void **)&d_m, b_neuron2));
    }
    CHECK(cudaMalloc((void **)&d_ca, b_neuron2));

    CHECK(cudaMalloc((void **)&d_s_a, b_syn_ex2));
    CHECK(cudaMalloc((void **)&d_s_a2, b_syn_ex2));
    CHECK(cudaMalloc((void **)&d_s_a3, b_syn_ex2));
    if (nmdar==1){
    CHECK(cudaMalloc((void **)&d_x_n, b_syn_ex2));
    CHECK(cudaMalloc((void **)&d_x_n2, b_syn_ex2));
    CHECK(cudaMalloc((void **)&d_x_n3, b_syn_ex2));
    CHECK(cudaMalloc((void **)&d_s_n, b_syn_ex2));
    CHECK(cudaMalloc((void **)&d_s_n2, b_syn_ex2));
    CHECK(cudaMalloc((void **)&d_s_n3, b_syn_ex2));
    }
    CHECK(cudaMalloc((void **)&d_s_g, b_syn_in2));
    CHECK(cudaMalloc((void **)&d_s_g2, b_syn_in2));
    CHECK(cudaMalloc((void **)&d_s_g3, b_syn_in2));
    // CHECK(cudaMalloc((void **)&d_s_a3, b_syn));
    



    double dt = dt_start;
    int *flag;
    flag = (int*)malloc(sizeof(int));
    flag[0]=0;
    int rep=0;

    while(1){
        if (rep>9){
    break; //while
    }
    //malloc host memory
    printf("ex%f, dt %f\n", ex, dt);
    
    int Ltp=Tp/dt+1;
     int b_neuron = sizeof(double)*(NE+NI)*Ltp;
   
      int b_syn_ex = sizeof(double)*syN_sum_ex*Ltp;

    double *v;
    v = (double*)malloc(b_neuron);
    


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

            ca[i]= ca_r;
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
            ca[i]= 1;//ca_r; 
        }
    }

      for (int i = 0; i < syN_sum_ex; i++) {
                   s_a[i] = 0.01;
                   if (nmdar==1){
                   x_n[i] = 0.01;
                   s_n[i] = 0.01;
                   }
        }
    for (int i = 0; i < syN_sum_in; i++) {
                   s_g[i] = 0.01;  
        }

   
    double *d_v;
    
    CHECK(cudaMalloc((void **)&d_v, b_neuron));

    
    
    //memcopy
   
    if (nav==1){
     CHECK(cudaMemcpy(d_h_na, h_na, b_neuron2, cudaMemcpyHostToDevice));
            }
    CHECK(cudaMemcpy(d_nk, nk, b_neuron2, cudaMemcpyHostToDevice));
    if (kva==1){
    CHECK(cudaMemcpy(d_h, h, b_neuron2, cudaMemcpyHostToDevice));
    }
    if (kvsi==1){
    CHECK(cudaMemcpy(d_m, m, b_neuron2, cudaMemcpyHostToDevice));
    }
    CHECK(cudaMemcpy(d_ca, ca, b_neuron2, cudaMemcpyHostToDevice));
     CHECK(cudaMemcpy(d_s_a, s_a, b_syn_ex2, cudaMemcpyHostToDevice));
     CHECK(cudaMemcpy(d_s_g, s_g, b_syn_in2, cudaMemcpyHostToDevice));

     if (nmdar==1){
      CHECK(cudaMemcpy(d_x_n, x_n, b_syn_ex2, cudaMemcpyHostToDevice));
     CHECK(cudaMemcpy(d_s_n, s_n, b_syn_ex2, cudaMemcpyHostToDevice));
     }

    int f_flag=0;

    //calculate by block Tp
    for (int tb=0; tb<tbs; tb++){
    //  if (it != 0){
            printf("ex%f  tb: %d\n", ex, tb);
           
            // transfer data from host to device
            CHECK(cudaMemcpy(d_v, v, b_neuron, cudaMemcpyHostToDevice));
            
            
            // invoke kernel at host side
            int dimx = (NE+NI)/blockdim_x;
            int dimy = 1;
            dim3 block(blockdim_x, 1, 1);
            dim3 grid(dimx, dimy);

            if(nav==1 && kva==1 && kvsi==1 && kir==1 && nmdar ==1){
            dt_gpu_NaK1_n<<<grid, block>>>(d_v, d_h_na, d_nk, d_h, d_m, d_ca, d_s_a, d_s_a2, d_s_a3,d_s_g, d_s_g2, d_s_g3, d_x_n, d_x_n2, d_x_n3, d_s_n, d_s_n2, d_s_n3, d_neuron_syN_ex, d_synID_ex, d_syN_cumsum_ex, d_neuron_syN_in, d_synID_in, d_syN_cumsum_in, 
                                        NE+NI, Ltp,  dt, g_leak, g_nav, g_kvhh, g_kva, g_kvsi, g_cav,g_kca, g_nap,g_kir, g_ampar, g_nmdar, g_gabar,tau_ca);
            }
            
            CHECK(cudaDeviceSynchronize());
        
           // check kernel error
            CHECK(cudaGetLastError());

            // copy kernel result back to host side
            CHECK(cudaMemcpy(v, d_v, b_neuron, cudaMemcpyDeviceToHost));
            //check nan
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
        
        //save Tp block results
        for (int i=0; i<NE+NI; i++){
        std::string savef = savedir+"ex"+ex_str +"_"+"ex_in"+ex_in_str+"_"+"N"+std::to_string(i)+"_"+std::to_string(tb)+".bin";
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
           
         } //for 
           

        for (int i = 0; i < (NE+NI); i++) {
            v[i*Ltp] = v[i*Ltp + Ltp-1];
         }

        if(tb==tbs-1){
            // printf("while end flag\n");
            flag[0]=1;
            // printf("flag %d\n", flag[0]);
        }
     } //for tbs

 
    CHECK(cudaFree(d_v));
    free(v);

      if(flag[0]==1){
            printf("ex%f  end\n",ex);
                break;  //while
            }
        

    }//while

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
    CHECK(cudaFree(d_ca));
    CHECK(cudaFree(d_s_a));
    CHECK(cudaFree(d_s_a2));
    CHECK(cudaFree(d_s_a3));
    CHECK(cudaFree(d_s_g));
    CHECK(cudaFree(d_s_g2));
    CHECK(cudaFree(d_s_g3));
    if (nmdar==1){
     CHECK(cudaFree(d_x_n));
    CHECK(cudaFree(d_x_n2));
    CHECK(cudaFree(d_x_n3));
    CHECK(cudaFree(d_s_n));
    CHECK(cudaFree(d_s_n2));
    CHECK(cudaFree(d_s_n3));
    }

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
    free(ca);
    free(s_a);
    free(s_g);
    if (nmdar==1){
    free(x_n);
    free(s_n);
     }
     free(flag);


  }//for ex_v



    CHECK(cudaFree(d_synID_ex));
    CHECK(cudaFree(d_neuron_syN_ex));
    CHECK(cudaFree(d_syN_cumsum_ex));
    CHECK(cudaFree(d_synID_in));
    CHECK(cudaFree(d_neuron_syN_in));
    CHECK(cudaFree(d_syN_cumsum_in));

    free(dt_li);
    free(neuron_syN_ex);
    free(synID_ex);
    free(syN_cumsum_ex);
  
    free(neuron_syN_in);
    free(synID_in);
    free(syN_cumsum_in);

    //end timer
    double iElaps = cpuSecond() - iStart;
    printf("elapsed %f sec\n", iElaps);
   
}
