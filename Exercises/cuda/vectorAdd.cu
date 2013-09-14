/**
 * @file    vectorAdd.cu
 * @brief   Cscs vectorAdd example using CUDA
 *
 * @author  Achille Peternier (achille.peternier@gmail.com), 2013
 */



//////////////
// #INCLUDE //
//////////////

   #include "cuda_runtime.h"
   #include "cscs_timer.h"
   #include <stdio.h>



/////////////
// #DEFINE //
/////////////

   #define N 128 ///< Vector size



/////////////
// KERNELS //
/////////////

/**
 * CUDA kernel for summing two vectors.
 * @param a pointer to the first vector data
 * @param b pointer to the second vector data
 * @param r pointer to the result vector data
 */
__global__ void vectorAdd(const float *a, const float *b, float *r)
{
   int i = threadIdx.x;
   r[i] = a[i] + b[i];
}



//////////
// MAIN //
//////////

/**
 * App entry point.
 * @param argc number of args
 * @param argv arguments
 * @return 0 on success, error code otherwise
 */
int main(int argc, char *argv[])
{
   // Vars:
   float *vectorA = NULL;
   float *vectorB = NULL;
   float *vectorR = NULL;
   float *temp    = NULL;
   int firstElems = min(N, 10);

   // Credits:
   printf("CUDA Vector Add - A. Peternier (C) 2013\n\n");

   // Init CUDA on the selected GPU:
   cudaError_t error;
   error = cudaSetDevice(0); // <-- change if # of GPU>1
   if (error != cudaSuccess)
   {
      printf("ERROR: unable to set device, is CUDA available?\n");
      goto Quit;
   }

   // Allocate vectors:
   error = cudaMalloc(&vectorA, N * sizeof(float));
   if (error != cudaSuccess)
   {
      printf("ERROR: unable to malloc vector A\n");
      goto Quit;
   }
	
   error = cudaMalloc(&vectorB, N * sizeof(float));
   if (error != cudaSuccess)
   {
      printf("ERROR: unable to malloc vector B\n");
      goto Quit;
   }
	
   error = cudaMalloc(&vectorR, N * sizeof(float));
   if (error != cudaSuccess)
   {
      printf("ERROR: unable to malloc vector R\n");
      goto Quit;
   }

   // Fill vectors:
   temp = (float *) malloc(N * sizeof(float));
   for (int c=0; c<N; c++)
      temp[c] = (float) c;
	
   // Copy data to input vectors:
   error = cudaMemcpy(vectorA, temp, N * sizeof(float), cudaMemcpyHostToDevice);
   if (error != cudaSuccess)
   {
      printf("ERROR: unable to copy vector A\n");
      goto Quit;
   }
	
   error = cudaMemcpy(vectorB, temp, N * sizeof(float), cudaMemcpyHostToDevice);
   if (error != cudaSuccess)
   {
      printf("ERROR: unable to copy vector B\n");
      goto Quit;
   }

   // Run kernel:
   vectorAdd<<<1, N>>>(vectorA, vectorB, vectorR);
   error = cudaGetLastError();
   if (error != cudaSuccess)
   {
      printf("ERROR: unable to launch kernel\n");
      goto Quit;
   }	

   // Wait for completion:
   error = cudaDeviceSynchronize();
   if (error != cudaSuccess)
   {
      printf("ERROR: unable to wait for synchronization\n");
      goto Quit;
   }

   // Read result back:
   error = cudaMemcpy(temp, vectorR, N * sizeof(float), cudaMemcpyDeviceToHost);
   if (error != cudaSuccess)
   {
      printf("ERROR: unable to read vector back\n");
      goto Quit;
   }

   // Show first and last elements:
   printf("Show first and last %d elements:\n", firstElems);
   for (int c=0; c<firstElems; c++)
      printf("   %d) %f\n", c, temp[c]);	
   printf("   ...\n");
   for (int c=N-firstElems; c<N; c++)
      printf("   %d) %f\n", c, temp[c]);	
   
Quit:
   // Release resources:
   if (temp)
      delete [] temp;
   cudaFree(vectorA);	
   cudaFree(vectorB);	
   cudaFree(vectorR);	

   // Done:
   return 0;
}
