// ******************************
// 
// compile with nvcc
// - remember to link against the relevant libraries
// 
// for API documentation: see docs.nvidia.com
//
// ******************************

#include <stdio.h>
#include "cufft.h"
#include "cublas.h"
#include "cuda.h"
#include "cuda_runtime_api.h"

#ifdef DOUBLE
#define Complex  cufftDoubleComplex
#define Real double
#define Transform CUFFT_Z2Z
#define TransformExec cufftExecZ2Z
#else
#define Complex  cufftComplex
#define Real float
#define Transform CUFFT_C2C
#define TransformExec cufftExecC2C
#endif

#define NX 5 
#define NY 5 
#define NZ 4
#define BATCH 1

#define NRANK 3

// initialize data on the CPU
void initData(Complex* h_data) {
 int i, j, k, b;
 for(b=0; b<BATCH; ++b)
   for(k=0; k<NZ; ++k)
    for(j=0; j<NY; ++j)
     for(i=0; i<NX; ++i){
       h_data[b * NX * NY * NZ + k*NX*NY + j*NX + i].x = 42.;
       h_data[b * NX * NY * NZ + k*NX*NY + j*NX + i].y = 0.;
     }
}


// get results back from the device and print it out
void reportGPUData(Complex *h_data, Complex* d_data) {
 int i, j, k;
 cudaMemcpy(h_data, d_data, sizeof(Complex)*NX*NY*NZ,
        cudaMemcpyDeviceToHost);
 for(k=0;k<NZ; ++k)
  for(j=0; j<NY; ++j)
    for(i=0; i<NX; ++i){
      int ind= k * NX * NY + j * NX + i;
      printf("data[%d] = (%g , %g)\n", ind, h_data[ind].x, h_data[ind].y);
    }
}


int main(int argc, char** argv)
{
 cufftHandle plan;

 Complex *h_data;
 Complex *d_data;
 Complex *d_result;


 cudaSetDevice(0);

// *******************************
// Exercise 3: get the name of the device we are running on
// *******************************


// initialize data and transfer to GPU
 h_data = (Complex*) malloc(sizeof(Complex)*NX*NY*NZ*BATCH);
 initData(h_data);

 cudaMalloc((void**)&d_data, sizeof(Complex)*NX*NY*NZ*BATCH);
 if( cudaGetLastError() != cudaSuccess)
     printf("d_data allocate error\n");

 cudaMalloc((void**)&d_result, sizeof(Complex)*NX*NY*NZ*BATCH);
 if( cudaGetLastError() != cudaSuccess)
     printf("d_result allocate error\n");

 cudaMemcpy(d_data, h_data, sizeof(Complex)*NX*NY*NZ*BATCH,
        cudaMemcpyHostToDevice);
 if( cudaGetLastError() != cudaSuccess)
     printf("transfer error\n");

// *******************************
// Exercise 1: create plan for the FFT
// *******************************


// *******************************
// Exercise 3: set up timers
// *******************************

// *******************************
// Exercise 1:  Perform the transform
// *******************************


// *******************************
// Exercise 3: report the time
// *******************************

// report result
 reportGPUData(h_data, d_result);

// *******************************
// Exercise 2: use cublas to report norm
// *******************************

// *******************************
// Exercise 1, 2, 3: cleanup
// *******************************

}

