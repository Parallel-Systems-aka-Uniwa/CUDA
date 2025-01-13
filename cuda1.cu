/*
 *  === Αρχείο: cuda1.cu ===
 *
 *  Ονοματεπώνυμο: Αθανασίου Βασίλειος Ευάγγελος
 *  Αριθμός Μητρώου: 19390005
 *  Πρόγραμμα Σπουδών: ΠΑΔΑ
 *  
 *  Μεταγλώττιση: nvcc -o cuda1 cuda1.cu
 *  Εκτέλεση: ./cuda1 A.txt OutArr.txt
 * 
 */
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define N 4
#define nThreads 2
#define nBlocks (int)ceil((float)N/nThreads)

__global__ void add(int *d_A, int *d_sum, double *d_avg) 
{
    __shared__ int cache[nThreads];  // Shared memory for each block

    // Calculate global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int totalElements = N * N;
    int tid = row * N + col; // Global index for 2D array

    if (row < N && col < N)  // Ensure within bounds
        cache[threadIdx.x + threadIdx.y * blockDim.x] = d_A[tid];
    else
        cache[threadIdx.x + threadIdx.y * blockDim.x] = 0;  // Avoid out-of-bound reads

    __syncthreads();  // Synchronize threads in the block

    // Perform parallel reduction within the block
    int i = blockDim.x * blockDim.y / 2; 
    while (i != 0) 
    {
        if (threadIdx.x + threadIdx.y * blockDim.x < i)  // Only threads with valid indices reduce
            cache[threadIdx.x + threadIdx.y * blockDim.x] += 
                cache[threadIdx.x + threadIdx.y * blockDim.x + i];
        __syncthreads();  // Synchronize threads in the block
        i /= 2;
    }

    // Atomic addition to the global sum if this thread is the first in the block
    if (threadIdx.x == 0 && threadIdx.y == 0) 
        atomicAdd(d_sum, cache[0]);

    // Calculate average after reduction
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
        *d_avg = (double)(*d_sum) / totalElements;
}


__global__ void findMax(int *d_A, int *d_amax)
{
    __shared__ int cache[nThreads];  // Shared memory for each block

    // Calculate global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int tid = row * N + col; // Global index for 2D array

    if (row < N && col < N)  // Ensure within bounds
        cache[threadIdx.x + threadIdx.y * blockDim.x] = d_A[tid];
    else
        cache[threadIdx.x + threadIdx.y * blockDim.x] = 0;  // Avoid out-of-bound reads

    __syncthreads();  // Synchronize threads in the block

    // Perform parallel reduction within the block
    int i = blockDim.x * blockDim.y / 2; 
    while (i != 0) 
    {
        if (threadIdx.x + threadIdx.y * blockDim.x < i)  // Only threads with valid indices reduce
            cache[threadIdx.x + threadIdx.y * blockDim.x] = 
                cache[threadIdx.x + threadIdx.y * blockDim.x] > cache[threadIdx.x + threadIdx.y * blockDim.x + i] ?
                cache[threadIdx.x + threadIdx.y * blockDim.x] : cache[threadIdx.x + threadIdx.y * blockDim.x + i];
        __syncthreads();  // Synchronize threads in the block
        i /= 2;
    }

    // Atomic max to the global max if this thread is the first in the block
    if (threadIdx.x == 0 && threadIdx.y == 0) 
        atomicMax(d_amax, cache[0]);
}

__device__ void atomicMin(float *address, float val)
{
    int *address_as_i = (int *) address;
    int old = *address_as_i, assumed;

    do
    {
        assumed = old;
        // Perform atomicCAS on the integer representation
        old = atomicCAS(address_as_i, assumed, 
        __float_as_int(val + __int_as_float(assumed)));
    } 
    while (assumed != old);
}


// Βij = (m–Aij)/amax
__global__ void createB(int *d_A, double *d_outArr, float *d_bmin, int *d_amax, double *d_avg)
{
    __shared__ int sharedMin[nThreads];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int totalElements = N * N;

    int cacheIndex = threadIdx.x;

    sharedMin[cacheIndex] = d_A[tid];

    __syncthreads();

    int i = blockDim.x / 2;

    while (i != 0)
    {
        if (cacheIndex < i)
            sharedMin[cacheIndex] = min(sharedMin[cacheIndex], sharedMin[cacheIndex + i]);
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        atomicMin(d_bmin, (float) sharedMin[0]);
    
    __syncthreads();

    if (tid < totalElements)
        d_outArr[tid] = (*d_avg - (double) d_A[tid]) / (double) *d_amax;
}

// Cij = (Aij+Ai(j+1)+Ai(j-1))/3
__global__ void createC(int *d_A, double *d_outArr)
{

}

int main(int argc, char *argv[])
{

    int *h_A;
    int *h_amax, *h_sum;
    int *d_A, *d_amax, *d_sum;
    double *h_OutArr;
    double *h_avg;
    double *d_OutArr, *d_avg;
    float *h_bmin, *d_bmin;

    int n, threadsPerBlock, blocksPerGrid;
    int intBytes, doubleBytes;
    int max_threads, max_block_dimX, max_block_dimY, max_block_dimZ, max_grid_dimX, max_grid_dimY, max_grid_dimZ;
    int i, j;
    FILE *fpA, *fpOutArr;
    char arr;
    float elapsedTime1, elapsedTime2, elapsedTime3, elapsedTimeAll;
    
    cudaEvent_t start, stop, startAll, stopAll;
    cudaError_t err;
    cudaDeviceProp prop;

    if (argc != 3)
    {
        printf("Usage: %s A.txt OutArr.txt\n", argv[0]);
        exit(1);
    }

    n = N;
    threadsPerBlock = nThreads;
    blocksPerGrid = nBlocks;

    cudaGetDeviceProperties(&prop, 0); // 0 is the device ID

    max_threads = prop.maxThreadsPerBlock;
    max_block_dimX = prop.maxThreadsDim[0];
    max_block_dimY = prop.maxThreadsDim[1];
    max_block_dimZ = prop.maxThreadsDim[2];
    max_grid_dimX = prop.maxGridSize[0];
    max_grid_dimY = prop.maxGridSize[1];
    max_grid_dimZ = prop.maxGridSize[2];

    printf("--------------- Device Properties ---------------\n");
    printf("Device name           : %s\n", prop.name);
    printf("Max threads per block : %d\n", max_threads);
    printf("Max block dimensions  : %d x %d x %d\n", max_block_dimX, max_block_dimY, max_block_dimZ);
    printf("Max grid dimensions   : %d x %d x %d\n", max_grid_dimX, max_grid_dimY, max_grid_dimZ);
    printf("-------------------------------------------------\n");

    if (n < 1)
    { printf("Error --> Matrix size must be at least 1\n"); exit(1); }
    if (threadsPerBlock < 1)
    { printf("Error --> Threads per block (block size) must be at least 1\n"); exit(1); }
    if (blocksPerGrid < 1)
    { printf("Error --> Blocks per grid (grid size) must be at least 1\n"); exit(1); }
    if (threadsPerBlock > max_threads)
    { printf("Error --> Threads per block (block size) exceed maximum allowed for %s\n", prop.name); exit(1); }
    if (blocksPerGrid > max_grid_dimX)
    { printf("Error --> Blocks per grid (grid size) exceed maximum allowed for %s\n", prop.name); exit(1); }

    fpA = fopen(argv[1], "w");
    if (fpA == NULL) { printf("Cannot open file %s\n", argv[1]); exit(1); }
    fpOutArr = fopen(argv[2], "w");
    if (fpOutArr == NULL) { printf("Cannot open file %s\n", argv[2]); exit(1); }

    err = cudaEventCreate(&start);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventCreate(&start) failed.\n"); exit(1); }
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventCreate(&stop) failed.\n"); exit(1); }
    err = cudaEventCreate(&startAll);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventCreate(&startAll) failed.\n"); exit(1); }
    err = cudaEventCreate(&stopAll);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventCreate(&stopAll) failed.\n"); exit(1); }

    printf("--------------- Input Parameters ---------------\n");
    printf("Matrix size        : %d x %d\n", n, n);
    printf("Blocks per Grid    : %d\n", blocksPerGrid);
    printf("Threads per Block  : %d\n", threadsPerBlock);
    printf("------------------------------------------------\n");

    intBytes = n * n * sizeof(int);
    doubleBytes = n * n * sizeof(double);

    h_A = (int *) malloc(intBytes);
    if (h_A == NULL) { printf("Error --> Memory allocation failed for A.\n"); exit(1); }
    h_OutArr = (double *) malloc(doubleBytes);
    if (h_OutArr == NULL) { printf("Error --> Memory allocation failed for OutArr.\n"); exit(1); }
    h_avg = (double *) malloc(sizeof(double));
    if (h_avg == NULL) { printf("Error --> Memory allocation failed for avg.\n"); exit(1); }
    h_amax = (int *) malloc(sizeof(int));
    if (h_amax == NULL) { printf("Error --> Memory allocation failed for max.\n"); exit(1); }
    h_bmin = (float *) malloc(sizeof(float));
    if (h_bmin == NULL) { printf("Error --> Memory allocation failed for min.\n"); exit(1); }
    h_sum = (int *) malloc(sizeof(int));
    if (h_sum == NULL) { printf("Error --> Memory allocation failed for sum.\n"); exit(1); }

    *h_sum = 0;
    *h_avg = 0.0;
    *h_amax = 0;
    *h_bmin = 0.0;

    srand(time(NULL));

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
        {
            h_A[i * n + j] = rand() % 199 - 99;                           // Τιμές στο διάστημα [-99, 99]
            h_A[i * n + j] = h_A[i * n + j] >= 0 ? h_A[i * n + j] + 10 : h_A[i * n + j] - 10;  // Τυχαία επιλογή προσήμου
            h_OutArr[i * n + j] = 0.0;
        }

/******************* ΠΑΡΑΛΛΗΛΑ ***************/
    err = cudaEventRecord(startAll, 0);

    err = cudaMalloc((void **) &d_A, intBytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_A, bytes) failed."); exit(1); }
    err = cudaMalloc((void **) &d_OutArr, doubleBytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_OutArray, bytes) failed."); exit(1); }
    err = cudaMalloc((void **) &d_sum, sizeof(int));
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_sum, sizeof(int)) failed."); exit(1); }
    err = cudaMalloc((void **) &d_avg, sizeof(double));
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_avg, sizeof(float)) failed."); exit(1); }
    err = cudaMalloc((void **) &d_amax, sizeof(int));
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_amax, sizeof(int)) failed."); exit(1); }
    err = cudaMalloc((void **) &d_bmin, sizeof(float)); 
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_bmin, sizeof(int)) failed."); exit(1); }

/* 1o kernel launch */

    err = cudaMemcpy(d_A, h_A, intBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice) failed."); exit(1); }
    err = cudaMemcpy(d_sum, h_sum, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(d_sum, h_sum, sizeof(int), cudaMemcpyHostToDevice) failed."); exit(1); }

    elapsedTimeAll = 0.0;

    err = cudaEventRecord(start, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(start, 0) failed."); exit(1); }

    dim3 dimBlock(nThreads, nThreads);
    dim3 dimGrid(nBlocks, nBlocks);

    add<<<dimGrid, dimBlock>>>(d_A, d_sum, d_avg);

    err = cudaMemcpy(h_avg, d_avg, sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(&h_avg, d_avg, sizeof(double), cudaMemcpyDeviceToHost) failed."); exit(1); }

    printf("Average: %lf\n", *h_avg);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime1, start, stop);
    printf ("Time for the kernel calcAvg<<<>>>(): %f ms\n", elapsedTime1);

/* 2o kernel launch */

    err = cudaEventRecord(start, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(start, 0) failed."); exit(1); }

    findMax<<<dimGrid, dimBlock>>>(d_A, d_amax);

    err = cudaMemcpy(h_amax, d_amax, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(&h_amax, d_amax, sizeof(int), cudaMemcpyDeviceToHost) failed."); exit(1); }

    printf("Max: %d\n", *h_amax);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime2, start, stop);
    printf ("Time for the kernel findMax<<<>>>(): %f ms\n", elapsedTime2);

/* 3o kernel launch */

    err = cudaEventRecord(start, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(start, 0) failed."); exit(1); }

    if (1)//(*h_max > N * (*h_avg))
    {
        arr = 'B';

/*
        err = cudaMemcpy(d_avg, h_avg, sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(d_avg, h_avg, sizeof(double), cudaMemcpyHostToDevice) failed."); exit(1); }
        err = cudaMemcpy(d_max, h_max, sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(d_max, h_max, sizeof(int), cudaMemcpyHostToDevice) failed."); exit(1); }
*/
        createB<<<nBlocks, nThreads>>>(d_A, d_OutArr, d_bmin, d_amax, d_avg);

        err = cudaMemcpy(h_OutArr, d_OutArr, doubleBytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(&h_OutArr, d_OutArr, doubleBytes, cudaMemcpyDeviceToHost) failed."); exit(1); }
        err = cudaMemcpy(h_bmin, d_bmin, sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(&h_bmin, d_bmin, sizeof(double), cudaMemcpyDeviceToHost) failed."); exit(1); }

        printf("Min: %f\n", *h_bmin);
    }
    else
    {
        arr = 'C';

        createC<<<nBlocks, nThreads>>>(d_A, d_OutArr);

        err = cudaMemcpy(h_OutArr, d_OutArr, doubleBytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(&h_OutArr, d_OutArr, doubleBytes, cudaMemcpyDeviceToHost) failed."); exit(1); }
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime3, start, stop);
    printf ("Time for the kernel create%c<<<>>>(): %f ms\n", arr, elapsedTime3);

/********************************************/
    err = cudaEventRecord(stopAll, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(stopAll, 0) failed."); exit(1); }
    err = cudaEventSynchronize(stopAll);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventSynchronize(stopAll) failed."); exit(1); }
    err = cudaEventElapsedTime(&elapsedTimeAll, startAll, stopAll);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventElapsedTime(&elapsedTimeAll, startAll, stopAll) failed."); exit(1); }

    printf("Time for the kernel: %f ms\n", elapsedTimeAll);

    fprintf(fpOutArr, "Array %c\n", arr);

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            fprintf(fpA, "%4d ", h_A[i * n + j]);
            fprintf(fpOutArr, "%4lf ", h_OutArr[i * n + j]);
        }

        fprintf(fpA, "\n");
        fprintf(fpOutArr, "\n");
    }

    fclose(fpA);
    fclose(fpOutArr);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(h_A);
    free(h_OutArr);
    free(h_avg);
    free(h_max);
    free(h_min);
    free(h_sum);

    cudaFree(d_A);
    cudaFree(d_OutArr);
    cudaFree(d_avg);
    cudaFree(d_max);
    cudaFree(d_min);
    cudaFree(d_sum);

    return 0;
}
