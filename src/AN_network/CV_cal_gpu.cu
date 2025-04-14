
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


//nvc++ -ta=tesla CV_cal_gpu.cu -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/examples/OpenACC/SDK/include/ -I/tmp/nvhpc_2022_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/include -L/tmp/nvhpc_2022_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/lib -lcudart -std=c++20 -o CV_gf

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


__global__ void find_peaks_gpu(double *mp, int *spc, int count, int N, int Lw, double th){

    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * N + ix;

    if (ix < count) {
      if(iy < N)
    {   
        // printf("count %d, N %d", ix, iy);
        for (int it=0; it<Lw; it++){
            if((it-1>=0)&(it+1 < Lw)){
                if((mp[ix*N*Lw+iy*Lw+it-1] < mp[ix*N*Lw+iy*Lw+it])&(mp[ix*N*Lw+iy*Lw+it]>mp[ix*N*Lw+iy*Lw+it+1])&(mp[ix*N*Lw+iy*Lw+it]>th)){
                    spc[ix*N*Lw+iy*Lw+it]=1;
                }else{
                    spc[ix*N*Lw+iy*Lw+it]=0;
                }
              }
             }
        }
    }
}


__global__ void cv_gpu(int *spc, int *sp_sum, int count, int cv_c, int N, int Lw, int Lmove, int Lcv){

    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
   // int(c*Lmove):int(c*Lmove + Lcv)
    if (ix < count){
        
        for (int c=0; c<cv_c; c++){
            int sum =0;
            for (int n=0; n<N; n++){
                for (int it=0; it<Lcv; it++){
                    
                        sum +=spc[ix*N*Lw+n*Lw+c*Lcv+it];
                    
                }
            }
            // printf("ix %d, c %d, sum %d, \n", ix, c, sum);
            sp_sum[ix*cv_c+c]=sum;
        }

   }
}



int main(int argc, char * argv[]){
int g_c=1;
//importing bin file

std::string savedir_neu= argv[g_c];
g_c +=1;

std::string outdir= argv[g_c];
g_c +=1;
std::cout << outdir << std::endl;

std::string ex_str=argv[g_c];
g_c +=1;
std::string ex_in_str=argv[g_c];
g_c +=1;
std::cout <<"ex:"<< ex_str << std::endl;

int N = atoi(argv[g_c]); 
g_c +=1;
int T = atoi(argv[g_c]); 
g_c +=1;
int Tp= atoi(argv[g_c]);
g_c +=1;
double dt = atof(argv[g_c]);
g_c +=1;

printf("N,  %d\n", N);
printf("T,  %d\n", T);
printf("Tp,  %d\n", Tp);

int Lt = T/dt;
int Ltp = Tp/dt;
int tbs=T/Tp;

double th = -20;

int move = atoi(argv[g_c]);
g_c +=1;
int w = atoi(argv[g_c]);
g_c +=1;
int move_cv = atoi(argv[g_c]);
g_c +=1;
int w_cv = atoi(argv[g_c]);
g_c +=1;

int cp_ori=atoi(argv[g_c]);
g_c +=1;

int blockdim_x =atoi(argv[g_c]); //cp
g_c +=1;
int blockdim_y =atoi(argv[g_c]);  //N
g_c +=1;

int Lw = w/dt;
int Lmove = move_cv/dt;
int Lcv = w_cv/dt;

int count = ((T-w)/move +1);
int cv_c = (w-w_cv)/move_cv+1;

printf("blockdim_y  %d\n", blockdim_y);


int cbs = count/cp_ori+1;
int c_re = count%cp_ori; 



printf("total count%d\n", count);
printf("total cv count%d\n", cv_c);

printf("total block count%d\n", cbs);

  
// printf("w_tb %d\n", w_tb);
double *cvs=(double*)malloc(sizeof(double)*count);


// int size_P = sizeof(double)*w_tb*Ltp;

// printf("a\n");
 //start timer
    double iStart = cpuSecond();
//GPU device setup
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));


// printf("aa\n");
int cp = cp_ori;
long int size_T = sizeof(double)*cp*N*Lw;
double *mp=(double*)malloc(size_T);
int *sp_sum=(int*)malloc(sizeof(int)*cp*cv_c);
// int *spc=(int*)malloc(sizeof(int)*cp*N*Lw);

double *d_mp;
int *d_spc,  *d_sp_sum;
CHECK(cudaMalloc((void **)&d_mp, size_T));
CHECK(cudaMalloc((void **)&d_spc, sizeof(int)*cp*N*Lw));
CHECK(cudaMalloc((void **)&d_sp_sum, sizeof(int)*cp*cv_c));


for (int cb=0; cb<cbs; cb++){
    printf("count block %d\n", cb);


    if(cb==cbs-1){
        free(mp);
        free(sp_sum);
        // free(spc);
        CHECK(cudaFree(d_mp));
        CHECK(cudaFree(d_spc));
        CHECK(cudaFree(d_sp_sum));

        cp=c_re;
        blockdim_x=cp;

        size_T = sizeof(double)*cp*N*Lw;
        mp=(double*)malloc(size_T);
        sp_sum=(int*)malloc(sizeof(int)*cp*cv_c);
        // spc=(int*)malloc(sizeof(int)*cp*N*Lw);
        CHECK(cudaMalloc((void **)&d_mp, size_T));
        CHECK(cudaMalloc((void **)&d_spc, sizeof(int)*cp*N*Lw));
        CHECK(cudaMalloc((void **)&d_sp_sum, sizeof(int)*cp*cv_c));

    }

    // printf("cp %d\n", cp);
    
    for (int c=0; c<cp; c++){
        int i=cp_ori*cb+c;
        int start_tb=(i*move)/Tp;
        int start_m = ((i*move)%Tp)/dt;
        int end_m = start_m + Lw;

        //   if(cb==cbs-1){
        // printf("start_tb %d\n", start_tb);
        // printf("start_m %d\n", start_m);
        // printf("end_m %d\n", end_m);
        //   }

        int w_tb;
        if ((i*move)%Tp==0){
            w_tb=(w/Tp+1)-1;
        }else{
            w_tb=((w/Tp+1));
        }

        // printf("w_tb %d\n", w_tb);
        double *mpp=(double*)malloc(sizeof(double)*w_tb*Ltp);

        // printf("mpp size %d\n", w_tb*Ltp);

        for (int n=0; n<N; n++){
            int tb_c=0;
            for (int tb=start_tb; tb<start_tb+w_tb; tb++){
                std::string v_file = savedir_neu+ "ex"+ex_str+"_"+"ex_in"+ex_in_str+"_"+"N"+std::to_string(n)+"_"+std::to_string(tb)+".bin";
                // std::cout << v_file << std::endl;

                std::ifstream ifs;
                ifs.open(v_file, std::ios::in|std::ios_base::binary);
                if (!ifs) {
                std::cout << "Can't open a file"<<v_file<<std::endl;
                        }
                // printf("tb_c %d\n", tb_c);
                for (int it=0; it<Ltp; it++){
                // printf("it %d\n", tb_c*Ltp + it);
                ifs.read( ( char* ) &mpp[tb_c*Ltp + it], sizeof( double ) );
                }
                ifs.close();
                tb_c +=1; 
            }
           
            // printf("i %d, neuron %d\n",i, n);
            // printf("mpp size %d\n", w_tb*Ltp);
            
            for (int k=start_m; k<end_m; k++){
     
                mp[c*N*Lw + n*Lw + k-start_m]=mpp[k];
            }
            // printf("post-sub\n");
        }
        free(mpp);
    }

    // printf("aaa\n");
        

    CHECK(cudaMemcpy(d_mp, mp, size_T, cudaMemcpyHostToDevice));
    
    
    

    int dimx =cp/blockdim_x;// (NE+NI)/blockdim_x;
    int dimy = N/blockdim_y;
    dim3 block(blockdim_x, blockdim_y);
    dim3 grid(dimx, dimy);

    find_peaks_gpu<<<grid, block>>>(d_mp, d_spc, cp, N, Lw, th);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    printf("end find_peaks\n");
  


    int dimx2 =cp/blockdim_x; // (NE+NI)/blockdim_x;
    int dimy2 = 1;
    dim3 block2(blockdim_x, 1);
    dim3 grid2(dimx2, dimy2);
    cv_gpu<<<grid2, block2>>>(d_spc, d_sp_sum, cp, cv_c, N, Lw, Lmove, Lcv);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

            // copy kernel result back to host side
    // CHECK(cudaMemcpy(mp, d_mp, size_T,  cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sp_sum, d_sp_sum, sizeof(int)*cp*cv_c,  cudaMemcpyDeviceToHost));





     for (int c=0; c<cp; c++){
        int i=cp_ori*cb+c;
        int sum = 0;
        for (int j=0; j<cv_c; j++){
            sum += sp_sum[c*cv_c + j];
        }
        double mean = (double)(sum)/(cv_c);
        // printf("mean %f\n", mean);

        
        if(mean==0){
        cvs[i] = 0; 
        }else{
        double var =0;
        for (int j=0; j<cv_c; j++){
            var += (double)((sp_sum[c*cv_c + j]-mean)*(sp_sum[c*cv_c + j]-mean))/cv_c;
        }
        cvs[i] = std::sqrt(var)/mean; 
        }
        // printf("count %d, cv %f\n", i, cvs[i]);
   }
}//cbs

 std::string cv_f= outdir + "_cv_ori.bin";
 std::ofstream ofs;
    ofs.open(cv_f, std::ios::out|std::ios::binary|std::ios::trunc);
    if (!ofs) {
    std::cout << "Can't open a file"<<cv_f<<std::endl;
    }
        
    for (int i=0; i<count; i++){
        ofs.write(( char * ) &cvs[i],sizeof( double ) );
        // printf("count %d, cv %f\n", i, cvs[i]);
        }//for
    ofs.close();

    



CHECK(cudaFree(d_mp));
CHECK(cudaFree(d_spc));
CHECK(cudaFree(d_sp_sum));

free(mp);
free(sp_sum);
free(cvs);
// free(spc);

//end timer
    double iElaps = cpuSecond() - iStart;
    printf("elapsed %f sec\n", iElaps);


}






// 

