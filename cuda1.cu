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

#define N 1000000
#define nThreads 1024
#define nBlocks (int)ceil((float)N/nThreads)

__global__ void calcAvg(int *d_A, int *d_sum, double *d_avg)
{
    __shared__ int cache[nThreads];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    int temp = 0;

    while (tid < N)
    {
        temp += d_A[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;

    __syncthreads();

    int i = blockDim.x / 2;

    while (i != 0)
    {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        atomicAdd(d_sum, cache[0]);

    if (cacheIndex == 0)
        *d_avg = (double) *d_sum / N;
}

__global__ void findMax()
{

}

__global__ void createB()
{

}

__global__ void createC()
{

}

int main(int argc, char *argv[])
{

    int *h_A;
    int *h_max, *h_sum;
    int *d_A, *d_max, *d_sum;
    double *h_OutArr;
    double *h_avg, *h_min;
    double *d_OutArr, *d_avg, *d_min;

    int n, threadsPerBlock, blocksPerGrid;
    int intBytes, doubleBytes;
    int max_threads, max_block_dimX, max_block_dimY, max_block_dimZ, max_grid_dimX, max_grid_dimY, max_grid_dimZ;
    int i, j;
    FILE *fpA, *fpOutArr;
    char arr;
    float elapsedTime1, elapsedTime2, elapsedTime3, elapsedTimeAll;
    
    cudaEvent_t start, stop;
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
    h_max = (int *) malloc(sizeof(int));
    if (h_max == NULL) { printf("Error --> Memory allocation failed for max.\n"); exit(1); }
    h_min = (double *) malloc(sizeof(double));
    if (h_min == NULL) { printf("Error --> Memory allocation failed for min.\n"); exit(1); }
    h_sum = (int *) malloc(sizeof(int));
    if (h_sum == NULL) { printf("Error --> Memory allocation failed for sum.\n"); exit(1); }

    *h_sum = 0;
    *h_avg = 0.0;
    *h_max = 0;
    *h_min = 0.0;

    srand(time(NULL));

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
        {
            h_A[i * n + j] = rand() % 199 - 99;                           // Τιμές στο διάστημα [-99, 99]
            h_A[i * n + j] = h_A[i * n + j] >= 0 ? h_A[i * n + j] + 10 : h_A[i * n + j] - 10;  // Τυχαία επιλογή προσήμου
            h_OutArr[i * n + j] = 0.0;
        }

/******************* ΠΑΡΑΛΛΗΛΑ ***************/

    err = cudaMalloc((void **) &d_A, intBytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_A, bytes) failed."); exit(1); }
    err = cudaMalloc((void **) &d_OutArr, doubleBytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_OutArray, bytes) failed."); exit(1); }
    err = cudaMalloc((void **) &d_sum, sizeof(int));
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_sum, sizeof(int)) failed."); exit(1); }
    err = cudaMalloc((void **) &d_avg, sizeof(double));
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_avg, sizeof(float)) failed."); exit(1); }
    err = cudaMalloc((void **) &d_max, sizeof(int));
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_max, sizeof(int)) failed."); exit(1); }
    err = cudaMalloc((void **) &d_min, sizeof(int)); 
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_min, sizeof(int)) failed."); exit(1); }

/* 1o kernel launch */

    err = cudaMemcpy(d_A, h_A, intBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice) failed."); exit(1); }
    err = cudaMemcpy(d_sum, h_sum, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(d_sum, h_sum, sizeof(int), cudaMemcpyHostToDevice) failed."); exit(1); }

    elapsedTimeAll = 0.0;

    err = cudaEventRecord(start, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(start, 0) failed."); exit(1); }

    calcAvg<<<nBlocks, nThreads>>>(d_A, d_sum, d_avg);

    err = cudaMemcpy(&h_avg, d_avg, sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(&h_avg, d_avg, sizeof(double), cudaMemcpyDeviceToHost) failed."); exit(1); }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime1, start, stop);
    printf ("Time for the kernel calcAvg<<<>>>(): %f ms\n", elapsedTime1);
    elapsedTimeAll += elapsedTime1;

    cudaMemcpy(&h_avg, d_avg, sizeof(double), cudaMemcpyDeviceToHost);
    printf("Average: %lf\n", h_avg);

/* 2o kernel launch */

    err = cudaEventRecord(start, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(start, 0) failed."); exit(1); }


    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime2, start, stop);
    printf ("Time for the kernel findMax<<<>>>(): %f ms\n", elapsedTime2);
    elapsedTimeAll += elapsedTime2;

/* 3o kernel launch */

    err = cudaEventRecord(start, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(start, 0) failed."); exit(1); }

    arr = 'B';

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime3, start, stop);
    printf ("Time for the kernel create%c<<<>>>(): %f ms\n", arr, elapsedTime3);
    elapsedTimeAll += elapsedTime3;

/********************************************/

    printf("Time for the kernel: %f\n", elapsedTimeAll);

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            fprintf(fpA, "%4d ", h_A[i * n + j]);
          //  fprintf(fpOutArr, "%4lf ", h_OutArr[i][j]);
        }

        fprintf(fpA, "\n");
        //fprintf(fpOutArr, "\n");
    }

    fclose(fpA);
    fclose(fpOutArr);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(h_A);
    free(h_OutArr);

    cudaFree(d_A);
    cudaFree(d_OutArr);
    cudaFree(d_avg);
    cudaFree(d_max);
    cudaFree(d_min);

    return 0;
}