/*
 *  === Αρχείο: cuda1.cu ===
 *
 *  Ονοματεπώνυμο: Αθανασίου Βασίλειος Ευάγγελος
 *  Αριθμός Μητρώου: 19390005
 *  Πρόγραμμα Σπουδών: ΠΑΔΑ
 *  
 *  Μεταγλώττιση: nvcc -o cuda1 cuda1.cu
 *  Εκτέλεση: ./cuda1 A.txt B.txt C.txt
 * 
 */
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define N 10
#define blX 3
#define blY 2
#define tX 4
#define tY 3

__device__ void binaryTree()
{

}

__global__ void kernel(int *d_A, int *d_OutArr, double *d_Avg, int *d_Max, int *d_Min, int matrix_size)
{
    int x, y;
    int *sum;
    int idx;

    x = threadIdx.x + blockIdx.x * blockDim.x;
    y = threadIdx.y + blockIdx.y * blockDim.y;

    *d_Min = 0;

    idx = y * matrix_size + x;

    atomicAdd(d_Min, d_A[idx]);
    atomicMax(d_Max, d_A[idx]);
}

int main(int argc, char *argv[])
{
    int **h_A;
    double **h_OutArr;
    double h_Avg;
    int h_Max, h_Min;
    int *d_A, *d_OutArr, *d_Max, *d_Min;
    double *d_Avg;
    float elapsedTime;
    int i, j;
    int matrix_size, grid_sizeX, grid_sizeY, block_sizeX, block_sizeY;
    int max_threads, max_block_dimX, max_block_dimY, max_grid_dimX, max_grid_dimY;
    int total_threads;
    FILE *fpA, *fpOutArr;
    cudaEvent_t start, stop;
    cudaError_t err;
    cudaDeviceProp prop;
    size_t intBytes, doubleBytes;

    if (argc != 3)
    {
        printf("Usage: %s A.txt OutArr.txt\n", argv[0]);
        exit(1);
    }

    matrix_size = N;
    grid_sizeX = blX;
    grid_sizeY = blY;
    block_sizeX = tX;
    block_sizeY = tY;

    cudaGetDeviceProperties(&prop, 0); // 0 is the device ID

    max_threads = prop.maxThreadsPerBlock;
    max_block_dimX = prop.maxThreadsDim[0];
    max_block_dimY = prop.maxThreadsDim[1];
    max_grid_dimX = prop.maxGridSize[0];
    max_grid_dimY = prop.maxGridSize[1];

    printf("--------------- Device Properties ---------------\n");
    printf("Device name           : %s\n", prop.name);
    printf("Max threads per block : %d\n", max_threads);
    printf("Max block dimensions  : %d x %d\n", max_block_dimX, max_block_dimY);
    printf("Max grid dimensions   : %d x %d\n", max_grid_dimX, max_grid_dimY);
    printf("-------------------------------------------------\n");

    total_threads = block_sizeX * block_sizeY;

    if (block_sizeX < 1 || block_sizeY < 1)
    {
        printf("Error --> Threads per block (block size) must be at least 1\n");
        exit(1);
    }

    if (grid_sizeX < 1 || grid_sizeY < 1)
    {
        printf("Error --> Blocks per grid (grid size) must be at least 1\n");
        exit(1);
    }

    if (block_sizeX > max_block_dimX || block_sizeY > max_block_dimY)
    {
        printf("Error --> Threads per block (block size) exceed maximum allowed for %s\n", prop.name);
        exit(1);
    }

    if (total_threads > max_threads)
    {
        printf("Error --> Total threads per block exceed maximum allowed for %s\n", prop.name);
        exit(1);
    }

    if (grid_sizeX > max_grid_dimX || grid_sizeY > max_grid_dimY)
    {
        printf("Error --> Blocks per grid (grid size) exceed maximum allowed for %s\n", prop.name);
        exit(1);
    }

    fpA = fopen(argv[1], "w");
    if (fpA == NULL) 
    {
        printf("Cannot open file %s\n", argv[1]);
        exit(1);
    }

    fpOutArr = fopen(argv[2], "w");
    if (fpOutArr == NULL) 
    {
        printf("Cannot open file %s\n", argv[2]);
        exit(1);
    }

    err = cudaEventCreate(&start);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventCreate(&start) failed.\n"); exit(1); }
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventCreate(&stop) failed.\n"); exit(1); }

    printf("--------------- Input Parameters ---------------\n");
    printf("Matrix size : %d x %d\n", matrix_size, matrix_size);
    printf("Grid size   : %d x %d\n", grid_sizeX, grid_sizeY);
    printf("Block size  : %d x %d\n", block_sizeX, block_sizeY);
    printf("------------------------------------------------\n");

    h_A = (int **) malloc(matrix_size * sizeof(int *));
    h_OutArr = (double **) malloc(matrix_size * sizeof(double *));
    
    for (i = 0; i < matrix_size; i++) 
    {
        h_A[i] = (int *) malloc(matrix_size * sizeof(int));
        if (h_A[i] == NULL) 
        {
            printf("Memory allocation failed for A[%d]\n", i);
            exit(1);
        }
        h_OutArr[i] = (double *) malloc(matrix_size * sizeof(double));
        if (h_OutArr[i] == NULL) 
        {
            printf("Memory allocation failed for OutArr[%d]\n", i);
            exit(1);
        }
    }

    srand(time(NULL));

    for (i = 0; i < matrix_size; i++)
    {
        for (j = 0; j < matrix_size; j++)
        {
            h_A[i][j] = rand() % 199 - 99;                           // Τιμές στο διάστημα [-99, 99]
            h_A[i][j] = h_A[i][j] >= 0 ? h_A[i][j] + 10 : h_A[i][j] - 10;  // Τυχαία επιλογή προσήμου
            h_OutArr[i][j] = 0.0;
        }
    }

/******************* ΠΑΡΑΛΛΗΛΑ ***************/
    intBytes = matrix_size * matrix_size * sizeof(int);
    doubleBytes = matrix_size * matrix_size * sizeof(double);

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
    err = cudaMalloc((void **) &d_Min, sizeof(int)); 
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_Min, sizeof(int)) failed."); exit(1); }

    err = cudaMemcpy(d_A, h_A, intBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice) failed."); exit(1); }


    dim3 dimBlock(block_sizeX, block_sizeY);
    dim3 dimGrid(grid_sizeX, grid_sizeY);

    kernel<<<dimGrid, dimBlock>>>(d_A, d_OutArr, d_Avg, d_Max, d_Min, matrix_size);

    cudaMemcpy(h_OutArr, d_OutArr, doubleBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Avg, d_Avg, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Max, d_Max, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Min, d_Min, sizeof(int), cudaMemcpyDeviceToHost);

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