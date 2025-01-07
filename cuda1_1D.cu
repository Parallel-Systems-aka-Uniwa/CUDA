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
#include "book.h"
#include "lock.h"

#define N 10
#define BL 5
#define T 3

__device__ void binaryTree()
{

}

__global__ void kernel(int *d_A, double *d_OutArr, double *d_Avg, int *d_Max, int *d_Min, bool *d_checkMax)
{
    __shared__ int s_A[T][T];
    __shared__ double s_OutArr[T][T];
    
    int x, y;
    int i, j;
    int sum;
    int idx;

    x = threadIdx.x + blockIdx.x * blockDim.x;
    y = threadIdx.y + blockIdx.y * blockDim.y;

    idx = y * N + x;
    sum = 0;

    for (int i = 0; i < tX; i++)
        for (int j = 0; j < tY; j++)
            s_A[i][j] = d_A[(y + i) * N + (x + j)];
    __syncthreads();
}

int main(int argc, char *argv[])
{
    int *h_A;
    double *h_OutArr;
    double h_Avg;
    int h_Max, h_Min;
    bool h_checkMax;
    bool *d_checkMax;
    int *d_A, *d_OutArr, *d_Max, *d_Min;
    double *d_Avg;
    float elapsedTime;
    int i, j;
    int matrix_size, threadsPerBlock, blocksPerGrid;
    int max_threads, max_block_dimX, max_block_dimY, max_block_dimZ, max_grid_dimX, max_grid_dimY, max_grid_dimZ;
    int intBytes, doubleBytes;
    FILE *fpA, *fpOutArr;
    cudaEvent_t start, stop;
    cudaError_t err;
    cudaDeviceProp prop;

    if (argc != 3)
    {
        printf("Usage: %s A.txt OutArr.txt\n", argv[0]);
        exit(1);
    }

    matrix_size = N;
    grid_size = BL;
    block_size = T;

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

    if (matrix_size < 1)
    { printf("Error --> Matrix size must be at least 1\n"); exit(1); }
    if (block_size < 1)
    { printf("Error --> Threads per block (block size) must be at least 1\n"); exit(1); }
    if (grid_size < 1)
    { printf("Error --> Blocks per grid (grid size) must be at least 1\n"); exit(1); }
    if (block_size > 1024)
    { printf("Error --> Threads per block (block size) exceed maximum allowed for %s\n", prop.name); exit(1); }
    if (grid_size > 65535)
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
    printf("Matrix size : %d x %d\n", matrix_size, matrix_size);
    printf("Grid size   : %d x %d\n", grid_sizeX, grid_sizeY);
    printf("Block size  : %d x %d\n", block_sizeX, block_sizeY);
    printf("------------------------------------------------\n");

    intBytes = matrix_size * matrix_size * sizeof(int);
    doubleBytes = matrix_size * matrix_size * sizeof(double);

    h_A = (int *) malloc(intBytes);
    if (h_A == NULL) { printf("Error --> Memory allocation failed for A.\n"); exit(1); }
    h_OutArr = (double *) malloc(doubleBytes);
    if (h_OutArr == NULL) { printf("Error --> Memory allocation failed for OutArr.\n"); exit(1); }

    h_Avg = 0.0;
    h_Max = 0;
    h_Min = 0;
    h_checkMax = false;

    srand(time(NULL));

    for (i = 0; i < matrix_size; i++)
        for (j = 0; j < matrix_size; j++)
        {
            h_A[i * matrix_size + j] = rand() % 199 - 99;                           // Τιμές στο διάστημα [-99, 99]
            h_A[i * matrix_size + j] = h_A[i * matrix_size + j] >= 0 ? h_A[i * matrix_size + j] + 10 : h_A[i * matrix_size + j] - 10;  // Τυχαία επιλογή προσήμου
            h_OutArr[i * matrix_size + j] = 0.0;
        }

/******************* ΠΑΡΑΛΛΗΛΑ ***************/

    err = cudaEventRecord(start, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(start, 0) failed."); exit(1); }

    err = cudaMalloc((void **) &d_A, intBytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_A, bytes) failed."); exit(1); }
    err = cudaMalloc((void **) &d_OutArr, doubleBytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_OutArray, bytes) failed."); exit(1); }
    err = cudaMalloc((void **) &d_Avg, sizeof(double));
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_Avg, sizeof(float)) failed."); exit(1); }
    err = cudaMalloc((void **) &d_Max, sizeof(int));
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_Max, sizeof(int)) failed."); exit(1); }
    err = cudaMalloc((void **) &d_checkMax, sizeof(bool));
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_checkMax, sizeof(bool)) failed."); exit(1); }
    err = cudaMalloc((void **) &d_Min, sizeof(int)); 
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_Min, sizeof(int)) failed."); exit(1); }

    err = cudaMemcpy(d_A, h_A, intBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice) failed."); exit(1); }
    err = cudaMemcpy(d_OutArr, h_OutArr, doubleBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(d_OutArr, OutArr, bytes, cudaMemcpyHostToDevice) failed."); exit(1); }
    err = cudaMemcpy(d_Avg, &h_Avg, sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(d_Avg, &h_Avg, sizeof(float), cudaMemcpyHostToDevice) failed."); exit(1); }
    err = cudaMemcpy(d_Max, &h_Max, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(d_Max, &h_Max, sizeof(int), cudaMemcpyHostToDevice) failed."); exit(1); }
    err = cudaMemcpy(d_Min, &h_Min, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(d_Min, &h_Min, sizeof(int), cudaMemcpyHostToDevice) failed."); exit(1); }
    err = cudaMemcpy(d_checkMax, &h_checkMax, sizeof(bool), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(d_checkMax, &h_checkMax, sizeof(bool), cudaMemcpyHostToDevice) failed."); exit(1); }

    kernel<<<grid_size, block_size>>>(d_A, d_OutArr, d_Avg, d_Max, d_Min, d_checkMax);

    err = cudaEventRecord(stop, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(stop, 0) failed."); exit(1); }
    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventSynchronize(stop) failed."); exit(1); }
    err = cudaEventElapsedTime(&elapsedTime, start, stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventElapsedTime(&elapsedTime, start, stop) failed."); exit(1); }
    printf ("Time for the kernel: %f ms\n", elapsedTime);

    printf("Average: %lf\n", h_Avg);
    printf("Max: %d\n", h_Max);
    printf("Min: %d\n", h_Min);
/********************************************/

    for (i = 0; i < matrix_size; i++)
    {
        for (j = 0; j < matrix_size; j++)
        {
            fprintf(fpA, "%4d ", h_A[i][j]);
            fprintf(fpOutArr, "%4lf ", h_OutArr[i][j]);
        }

        fprintf(fpA, "\n");
        fprintf(fpOutArr, "\n");
    }

    fclose(fpA);
    fclose(fpOutArr);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    for (i = 0; i < matrix_size; i++)
    {
        free(h_A[i]);
        free(h_OutArr[i]);
    }

    free(h_A);
    free(h_OutArr);

    cudaFree(d_A);
    cudaFree(d_OutArr);
    cudaFree(d_Avg);
    cudaFree(d_Max);
    cudaFree(d_Min);

    return 0;
}