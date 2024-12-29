#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
 

// CUDA kernel 
__global__ void histo_kernel(unsigned char *buffer,
                             long size, unsigned int *histo) 
{

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // stride is total number of threads
    int stride = blockDim.x * gridDim.x;

    while (i < size) {
        histo[buffer[i]]++;
        i += stride;
    }
}
 
int main( int argc, char* argv[] )
{
    // Size of vectors
    unsigned char *buf;
    long s=1000000; 
    unsigned int *hist;

    // Size of grid 
    int blockSize = 1024;
    int gridSize = 10;

    unsigned char *d_buf;
    unsigned int *d_hist;
 
    // Size, in bytes, of each vector
    size_t bytes1 = sizeof(unsigned int);
    size_t bytes2 = sizeof(unsigned char);
 
    cudaEvent_t start,stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory for each vector on host
    buf = (unsigned char *)malloc(s*bytes2);
    hist = (unsigned int *)malloc(256*bytes1);
 
    // Allocate memory for each vector on GPU
    cudaMalloc(&d_buf, s*bytes2);
    cudaMalloc(&d_hist, 256*bytes1);
 
    int i;
    // Initialize vectors on host
    for( i = 0; i < s; i++ ) {
        buf[i] = '\0'+(i % 256);
    }
    //printf("\n BUFFER \n");
    //for( i = 0; i < s; i++ ) {
     //   printf("%c ", buf[i]);
    //}
    // Copy host vectors to device
    cudaMemcpy( d_buf, buf, s*bytes2, cudaMemcpyHostToDevice);
 
    // Execute the kernel
    printf("CUDA kernel launch with %d blocks of %d threads\n", gridSize, blockSize);
    
    cudaEventRecord(start,0);
    
    histo_kernel<<<gridSize, blockSize>>>(d_buf, s, d_hist);
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
 
    // Copy result back to host
    cudaMemcpy( hist, d_hist, 256*bytes1, cudaMemcpyDeviceToHost );

    // Print result 
    printf("\n HISTO \n");
    for( i = 0; i < 256; i++ ) {
        printf("%d ", hist[i]);
    }
 
    cudaEventElapsedTime(&elapsedTime,start,stop);
    printf ("\nTime for the kernel: %f ms\n", elapsedTime);
    
    // Release device memory
    cudaFree(d_buf);
    cudaFree(d_hist);

 
    // Release host memory
    free(buf);
    free(hist);

 
    return 0;
}
