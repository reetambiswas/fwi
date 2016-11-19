//#include<iostream.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cublas.h>
#include <string>
#include <math.h>
#include <fstream>
#include <time.h>

//#include<cublas_v2.h>
//#include<cuda_runtime.h>
//#include<device_launch_parameters.h>
//#include<device_launch_parameters.h>
//#include<math_functions.h>







//File Handling
extern void readASCIIFile(char *fn, int dimA[2], float *A);
int VectoFileWrite(float* A, int M, const char* fn);


//Mathematical Operation
float max( float* A , int dimA[]);
float max(float a, float b);
float* consVec(float *A, int dimA, float a);
int ind(int i, int j);
void copy1Dto1D(float* A, float* B, int dimA);




//Operations
extern void construct_source(float* field1, float* field2, float w1);
void cudaCheck(cudaError_t cudaStatus);
void matrixPadding(float* A, float* B, int m, int n, int p);
// void fdm_acoustic(float* d_velocity,float* d_field1,float* d_field2,float bc, dim3 gridDim, dim3 blockDim);



//Kernel
__global__ void calculateLaplace(float* d_laplace,float* d_wave_propagate_t, float* A /*d_laplace_temp*/, float* V,float* d_field1, float* d_field2, float dt, float dx, int m, int n, float bc, int O, float* C);

__global__ void wavefieldTransfer(float* d_laplace_temp, float* d_field2, int m, int n, int O);

__global__ void myCudaMemset(float* A, float val, int m, int n);

__global__ void add_source(float* d_wave_propagate_t, float* d_field2, float* source_grid,float w, int m, int n);

__global__ void ABC_inner(float* V,float* f1, float* f2, float* wave, float dx, float dt, float bc,int m, int n);


__global__ void ABC_outer(float* V,float* f1, float* f2, float* wave, float dx, float dt, float bc,int m, int n);

__global__ void rickerWavelet(float* w, float f, float n, float dt);

__global__ void extractCorrectRegion(float* A,float* B, int m, int n, int p);

__global__ void badBoundaryCondition(float* A, float* T, int m, int n, int p);

__global__ void calculateCerjanCoeff(int p, float* G, int I);

__global__ void cerjanMatrix(float* A, float* G, int m, int n, int p);

__global__ void cerjanBoundaryCondition(float* A, float* B, float* C, float* CM, int m, int n);