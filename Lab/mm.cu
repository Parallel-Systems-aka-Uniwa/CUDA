#include <stdio.h>

__global__ void Kernel(float *Md, float *Nd, float *Pd, int Width) {

  // Calculate the column index of the Pd element, denote by x
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  // Calculate the row index of the Pd element, denote by y
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  float Pvalue = 0;
  // each thread computes one element of the output matrix Pd.
  for (int k = 0; k < Width; ++k) {
    Pvalue += Md[y*Width + k] * Nd[k*Width + x];
  }

  // write back to the global memory
  Pd[y*Width + x] = Pvalue;
}

void MatrixMultiplication(float *M, float *N, float *P, int Width) {
    int size = Width * Width * sizeof(float);
    float *Md, *Nd, *Pd;
    // capture start time
    cudaEvent_t     start, stop;
    cudaEventCreate( &start ); cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );
    // allocate memory on the GPU
    cudaMalloc((void**)&Md, size); cudaMalloc((void**)&Nd, size); cudaMalloc((void**)&Pd, size);
    // transfer M and N to device memory
    cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice); cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);
    // kernel invocation code
    dim3 dimBlock(32, 32); dim3 dimGrid(Width/32, Width/32);
    Kernel<<<dimGrid, dimBlock>>>( Md, Nd, Pd, Width);
    // transfer P from device
    cudaMemcpy(P,Pd,size,cudaMemcpyDeviceToHost);
    // get stop time, and display the timing results
    cudaEventRecord( stop, 0 ); cudaEventSynchronize( stop );
    float   elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );
    printf( "Time to generate:  %3.1f ms\n", elapsedTime );
    // free the memory allocated on the GPU
    cudaFree(Md); cudaFree(Nd); cudaFree(Pd);
    // destroy events to free memory
    cudaEventDestroy( start ); cudaEventDestroy( stop );
}

int main(void){
    void MatrixMultiplication(float *, float *, float *, int);
    const int Width = 1024;
    int size = Width * Width * sizeof(float);
    float *M, *N, *P;

    // allocate memory on the CPU
    M = (float*)malloc(size); N = (float*)malloc(size); P = (float*)malloc(size);

    // initialize the matrices
    for (int y=0; y<Width; y++) {
        for (int x=0; x<Width; x++){
           M[y*Width + x] = x + y*Width;
           N[y*Width + x] = x + y*Width;
        }
    }

    MatrixMultiplication(M, N, P, Width);

    // free the memory allocated on the CPU
    free( M ); free( N ); free( P );

    return 0;
}