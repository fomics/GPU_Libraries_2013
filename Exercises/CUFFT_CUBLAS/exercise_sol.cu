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

// get the name of the device we are running o
 struct cudaDeviceProp deviceProp;
 cudaGetDeviceProperties(&deviceProp, 0);
 printf("Running on %s\n", deviceProp.name);

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

// create plan for the FFT
 int n[NRANK] = { NZ, NY, NX };
 if(cufftPlanMany(&plan, NRANK, n,
             NULL, 1, NX*NY*NZ,
             NULL, 1, NX*NY*NZ, Transform, BATCH) != CUFFT_SUCCESS)
     printf("Error creating plan\n");

// set up timers
 cudaEvent_t start, stop;

 cudaEventCreate(&start);
 cudaEventCreate(&stop);

 cudaEventRecord(start, 0);

// Perform the transform
 if( TransformExec(plan, d_data, d_result, CUFFT_FORWARD)!= CUFFT_SUCCESS){
     printf("error in forward transform\n");
 }


// report the time
 cudaEventRecord(stop, 0);
 cudaEventSynchronize(stop);

 float elapsedTime;
 cudaEventElapsedTime(&elapsedTime, start, stop);
 printf("Advanced Layout 3D FFT, %d x %d x %d, size=%d: %g\n", NX, NY, NZ, sizeof(Complex),
      elapsedTime);

// report result
 reportGPUData(h_data, d_result);

 // use cublas to report norm
 float norm = cublasScnrm2(NX*NY*NZ, d_result, 1);
 printf("Result norm = %g\n", norm);

// cleanup
 cufftDestroy(plan);

 cudaEventDestroy(stop);
 cudaEventDestroy(start);
}

