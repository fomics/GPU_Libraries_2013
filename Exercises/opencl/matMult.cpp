/**
 * @file    matMult.cpp
 * @brief	Matrix multiplication example/exercise
 *
 * @author	Achille Peternier (achille.peternier@gmail.com), 2013
 */



//////////////
// #INCLUDE //
//////////////

   #include <CL/cl.h>
   #include <stdio.h>
   #include <stdlib.h>



/////////////
// #DEFINE //
/////////////

   // Matrix size:
   #define MAT_SIZE_X   4
   #define MAT_SIZE_Y   4

   // Macro for inlining kernels as C strings:
   #define STRINGIFY(x)  #x  



/////////////
// METHODS //
/////////////

/**
 * CPU matrix by matrix multiplication taking 1D float arrays as params.
 * @param matA input matrix A
 * @param matB input matrix B
 * @param matR output matrix R
 */
void cpuMultMat(float *matA, float *matB, float *matR)
{
   // For each row of matA:
   for (int i=0; i<MAT_SIZE_Y; i++)   
      // For each column of matB:
      for (int j=0; j<MAT_SIZE_X; j++)
      {
         matR[i*MAT_SIZE_X+j] = 0.0f;
         for (int k=0; k<MAT_SIZE_X; k++)       
            matR[i*MAT_SIZE_X+j] += matA[i*MAT_SIZE_X+k] * matB[k*MAT_SIZE_X+j];         
      }
}


/**
 * Display matrix content to screen.
 * @param mat input matrix (as 1D float array pointer)
 */
void printMat(float *mat)
{
   for (int i=0; i<MAT_SIZE_Y; i++)
   {
      for (int j=0; j<MAT_SIZE_X; j++)
         printf("\t%.1f", mat[i*MAT_SIZE_X+j]);
      printf("\n");
   }
}


/**
 * Fill a matrix with random values.
 * @param mat input matrix (as 1D float array pointer)
 */
void fillMat(float *mat)
{
   for (int c=0; c<MAT_SIZE_X*MAT_SIZE_Y; c++)
      mat[c] = (float) (rand()%100);
}


/**
 * Set a matrix as an identity matrix.
 * @param mat input matrix (as 1D float array pointer)
 */
void fillIdentityMat(float *mat)
{
   for (int i=0; i<MAT_SIZE_Y; i++)
      for (int j=0; j<MAT_SIZE_X; j++)
         if (j != i)
            mat[i*MAT_SIZE_X+j] = 0.0f;
         else
            mat[i*MAT_SIZE_X+j] = 1.0f;
}



//////////
// MAIN //
//////////

/**
 * App entry point.
 * @param argc argument count
 * @param argv argument value
 * @return error cde, or zero if none
 */
int main(int argc, char *argv[])
{
   // Credits:
   printf("MatrixMultiplication in OpenCL - A. Peternier (C) 2013\n\n");

   
   //////////////////////////////
   // OpenCL initialization part:
   cl_int error;

   // Use first platform found:
   cl_platform_id platform;
   error = clGetPlatformIDs(1, &platform, NULL);
   if (error != CL_SUCCESS)
   {
      printf("ERROR: unable to find any suitable platform\n");
      return 1;
   }
   char platformName[256];
   clGetPlatformInfo(platform, CL_PLATFORM_NAME, 256, platformName, NULL);
   printf("   Platform used: %s\n", platformName);

   // Use first device found:
   cl_device_id device;
   error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
   if (error != CL_SUCCESS)
   {
      printf("ERROR: unable to find any suitable device\n");
      return 1;
   }
   char deviceName[256];
   clGetDeviceInfo(device, CL_DEVICE_NAME, 256, deviceName, NULL);
   printf("   Device used  : %s\n", deviceName);

   // Create context:
   cl_context_properties contextProps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0};
   cl_context context = clCreateContext(contextProps, 1, &device, NULL, NULL, &error);
   if (error != CL_SUCCESS)
   {
      printf("ERROR: unable to create context\n");
      return 1;
   }

   // Create a command queue:
   cl_command_queue queue = clCreateCommandQueue(context, device, 0, &error);
   if (error != CL_SUCCESS)
   {
      printf("ERROR: unable to create command queue\n");
      return 1;
   }   
   
   // OpenCL initialization done.
   //////////////////////////////


   /////////////////////////////
   // Compile the OpenCL kernel:
   char *source = STRINGIFY(
      __kernel void gpuMatMult(__global float *matA, __global float *matB, __global float *matR, int sizeX, int sizeY)
      {
         int row = get_global_id(1);
         int col = get_global_id(0);

         float sum = 0.0f;
         
         for (int i=0; i<sizeX; i++)
            sum += matA[row*sizeX+i] * matB[i*sizeX+col];
         
         matR[row*sizeX+col] = sum;
      }
   );

   // Create program:
   cl_program program = clCreateProgramWithSource(context, 1, (const char **) &source, NULL, &error);
   if (error != CL_SUCCESS)
   {
      printf("ERROR: unable to build program\n");
      return 1;
   }   

   // Compile program:
   error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

   // Create kernel:
   cl_kernel kernel = clCreateKernel(program, "gpuMatMult", &error);
   if (error != CL_SUCCESS)
   {
      printf("ERROR: unable to create kernel\n");
      return 1;
   }   

   // OpenCL kernel compiled.
   //////////////////////////


   //////////////////////////
   // OpenCL allocate memory:
   cl_mem bufferMatA = clCreateBuffer(context, CL_MEM_READ_ONLY, MAT_SIZE_X*MAT_SIZE_Y*sizeof(float), NULL, &error);
   if (error != CL_SUCCESS)
   {
      printf("ERROR: unable to create buffer A\n");
      return 1;
   }   

   cl_mem bufferMatB = clCreateBuffer(context, CL_MEM_READ_ONLY, MAT_SIZE_X*MAT_SIZE_Y*sizeof(float), NULL, &error);
   if (error != CL_SUCCESS)
   {
      printf("ERROR: unable to create buffer B\n");
      return 1;
   }   

   cl_mem bufferMatR = clCreateBuffer(context, CL_MEM_WRITE_ONLY, MAT_SIZE_X*MAT_SIZE_Y*sizeof(float), NULL, &error);
   if (error != CL_SUCCESS)
   {
      printf("ERROR: unable to create buffer R\n");
      return 1;
   }   

   // OpenCL memory allocated.
   ///////////////////////////
   
   
   // Allocate three matrices:
   float matA[MAT_SIZE_X*MAT_SIZE_Y];
   float matB[MAT_SIZE_X*MAT_SIZE_Y];
   float matR[MAT_SIZE_X*MAT_SIZE_Y];
   float matRCl[MAT_SIZE_X*MAT_SIZE_Y];

   // Fill matrices:
   fillMat(matA);
   fillIdentityMat(matB);
   matB[1] = 2.0f;
   matB[2] = 3.0f;

   // Show input matrices:
   printf("\nMatrix A:\n");
      printMat(matA);
   printf("\nMatrix B:\n");
      printMat(matB);

   
   ///////
   // CPU:
   printf("\n\n*** CPU only ***\n");   

   // A*B:
   cpuMultMat(matA, matB, matR);

   // Show result:
   printf("\nMatrix A*B:\n");
      printMat(matR);


   //////////
   // OpenCL:
   printf("\n\n*** OpenCL ***\n");

   // Copy data to device:
   error = clEnqueueWriteBuffer(queue, bufferMatA, CL_TRUE, 0, MAT_SIZE_X*MAT_SIZE_Y*sizeof(float), (void *) matA, 0, NULL, NULL);
   if (error != CL_SUCCESS)
   {
      printf("ERROR: unable to copy buffer A\n");
      return 1;
   }   
   
   error = clEnqueueWriteBuffer(queue, bufferMatB, CL_TRUE, 0, MAT_SIZE_X*MAT_SIZE_Y*sizeof(float), (void *) matB, 0, NULL, NULL);
   if (error != CL_SUCCESS)
   {
      printf("ERROR: unable to copy buffer B\n");
      return 1;
   }   

   // Set kernel args:
   int sizeX = MAT_SIZE_X;
   int sizeY = MAT_SIZE_Y;
   clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &bufferMatA);
   clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &bufferMatB);
   clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &bufferMatR);
   clSetKernelArg(kernel, 3, sizeof(cl_int), (void *) &sizeX);
   clSetKernelArg(kernel, 4, sizeof(cl_int), (void *) &sizeY);

   // Set grid size:
   size_t local[2] = {4, 4}; // sizeXY must be divisible by this number
   size_t global[2] = {sizeX, sizeY};

   // Run kernel:
   error = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
   if (error != CL_SUCCESS)
   {
      printf("ERROR: unable to run kernel\n");
      return 1;
   }   
   clFinish(queue);

   // Read data back to the host:   
   error = clEnqueueReadBuffer(queue, bufferMatR, CL_TRUE, 0, MAT_SIZE_X*MAT_SIZE_Y*sizeof(float), (void *) matRCl, 0, NULL, NULL);
   
   // Show result:
   printf("\nMatrix A*B:\n");
      printMat(matRCl);

   // Compare:
   bool problem = false;
   for (int c=0; c<MAT_SIZE_X*MAT_SIZE_Y; c++)
      if (matR[c] != matRCl[c])
         problem = true;
   if (problem)
      printf("ERROR: results do not match\n");

   // Done:
   return 0;
}