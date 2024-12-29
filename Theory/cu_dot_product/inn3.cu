#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
 

// CUDA kernel. Each thread takes care of one element of c
__global__ void innProd(float *a, float *b, float *res, int n)
{
    // Get our global thread ID
    int k, id = blockIdx.x*blockDim.x+threadIdx.x;
    float tmp = 0;
 
    // Make sure we do not go out of bounds
    for (k = id; k < n; k += blockDim.x*gridDim.x) 
        tmp += a[id] * b[id];

    atomicAdd(res,tmp);
}
 
int main( int argc, char* argv[] )
{
    // Size of vectors
    int n = 50000;

    // Size of grid 
    int blockSize = 1024;
    int gridSize = 10;

    // Host input vectors
    float *h_a;
    float *h_b;
    float *h_res;
 
    // Device input vectors
    float *d_a;
    float *d_b;
    float *d_res;
 
    // Size, in bytes, of each vector
    size_t bytes = sizeof(float);
 
    cudaEvent_t start,stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory for each vector on host
    h_a = (float*)malloc(n*bytes);
    h_b = (float*)malloc(n*bytes);
    h_res = (float*)malloc(bytes);
 
    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, n*bytes);
    cudaMalloc(&d_b, n*bytes);
    cudaMalloc(&d_res, bytes);
 
    int i;
    // Initialize vectors on host
    for( i = 0; i < n; i++ ) {
        h_a[i] = 1;
        h_b[i] = 1;
    }
 
    // Copy host vectors to device
    cudaMemcpy( d_a, h_a, n*bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, h_b, n*bytes, cudaMemcpyHostToDevice);
 
    // Execute the kernel
    printf("CUDA kernel launch with %d blocks of %d threads\n", gridSize, blockSize);
    
    cudaEventRecord(start,0);
    
    innProd<<<gridSize, blockSize>>>(d_a, d_b, d_res, n);
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
 
    // Copy result back to host
    cudaMemcpy( h_res, d_res, bytes, cudaMemcpyDeviceToHost );

    // Print result 
    printf("final result: %f\n", *h_res);
 
    cudaEventElapsedTime(&elapsedTime,start,stop);
    printf ("Time for the kernel: %f ms\n", elapsedTime);
    
    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);

 
    // Release host memory
    free(h_a);
    free(h_b);
    free(h_res);

 
    return 0;
}
