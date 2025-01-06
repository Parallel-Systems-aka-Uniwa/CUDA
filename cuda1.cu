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

__device__ void calcAvg()
{

}

__device__ void calcMin()
{

}

__device__ void findMax()
{

}

__device__ void checkMax()
{

}

__device__ void createB()
{

}

__device__ void createC()
{

}

__global__ void kernel()
{
    
}

int main(int argc, char *argv[])
{
    int **A, **B, **C;
    int *d_A, *d_N, *d_outArray, *d_max, *d_min;
    float *d_avg;
    float elapsedTime;
    int i, j;
    int matrix_size, grid_sizeX, grid_sizeY, block_sizeX, block_sizeY;
    int max_threads, max_block_dimX, max_bloc_dimY, max_grid_dimX, max_grid_dimY;
    int total_threads;
    FILE *fpA, *fpB, *fpC;
    cudaEvent_t start, stop;
    cudaError_t err;
    cudaDeviceProp prop;
    size_t bytes;

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

    printf("Max threads per block: %d\n", max_threads);
    printf("Max block dimensions: (%d, %d, %d)\n", max_block_dimX, max_block_dimY);
    printf("Max grid dimensions: (%d, %d, %d)\n", max_grid_dimX, max_grid_dimY);

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
        printf("Error --> Threads per block (block size) exceed maximum allowed for GPU Titan Rtx\n");
        exit(1);
    }

    if (total_threads > max_threads)
    {
        printf("Error --> Total threads per block exceed maximum allowed for GPU Titan Rtx\n");
        exit(1);
    }

    if (grid_sizeX > max_grid_dimX || grid_sizeY > max_grid_dimY)
    {
        printf("Error --> Blocks per grid (grid size) exceed maximum allowed for GPU Titan Rtx\n");
        exit(1);
    }

    fpA = fopen(argv[1], "w");
    if (fpA == NULL) 
    {
        printf("Cannot open file %s\n", argv[1]);
        exit(1);
    }

    fpB = fopen(argv[2], "w");
    if (fpB == NULL) 
    {
        printf("Cannot open file %s\n", argv[2]);
        exit(1);
    }

    fpC = fopen(argv[3], "w");
    if (fpC == NULL) 
    {
        printf("Cannot open file %s\n", argv[3]);
        exit(1);
    }

    err = cudaEventCreate(&start);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventCreate(&start) failed.\n"); exit(1); }
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventCreate(&stop) failed.\n"); exit(1); }


    printf("Matrix size  : %d x %d\n", matrix_size, matrix_size );
    printf("Grid size    : %d\n", grid_size);
    printf("Block size   : %d\n", block_size);
    printf("-----------------------------------\n");

    A = (int **) malloc(matrix_size * sizeof(int *));
    B = (int **) malloc(matrix_size * sizeof(int *));
    C = (int **) malloc(matrix_size * sizeof(int *));
    
    for (i = 0; i < matrix_size; i++) 
    {
        A[i] = (int *) malloc(matrix_size * sizeof(int));
        if (A[i] == NULL) 
        {
            printf("Memory allocation failed for A[%d]\n", i);
            exit(1);
        }
        B[i] = (int *) malloc(matrix_size * sizeof(int));
        if (B[i] == NULL) 
        {
            printf("Memory allocation failed for B[%d]\n", i);
            exit(1);
        }
        C[i] = (int *) malloc(matrix_size * sizeof(int));
        if (C[i] == NULL) 
        {
            printf("Memory allocation failed for C[%d]\n", i);
            exit(1);
        }
    }

    srand(time(NULL));

    for (i = 0; i < matrix_size; i++)
    {
        for (j = 0; j < matrix_size; j++)
        {
            A[i][j] = rand() % 199 - 99;                           // Τιμές στο διάστημα [-99, 99]
            A[i][j] = A[i][j] >= 0 ? A[i][j] + 10 : A[i][j] - 10;  // Τυχαία επιλογή προσήμου
            B[i][j] = 0;
            C[i][j] = 0;
        }
    }

/******************* ΠΑΡΑΛΛΗΛΑ ***************/
    bytes = matrix_size * matrix_size * sizeof(int);

    err = cudaEventRecord(start, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(start, 0) failed."); exit(1); }

    err = cudaMalloc((void **) &d_A, bytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_A, bytes) failed."); exit(1); }
    err = cudaMalloc((void **) &d_outArray, bytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_outArray, bytes) failed."); exit(1); }
    err = cudaMalloc((void **) &d_N, sizeof(int));
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_N, sizeof(int)) failed."); exit(1); }
    err = cudaMalloc((void **) &d_avg, sizeof(float));
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_avg, sizeof(float)) failed."); exit(1); }
    err = cudaMalloc((void **) &d_max, sizeof(int));
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_max, sizeof(int)) failed."); exit(1); }
    err = cudaMalloc((void **) &d_min, sizeof(int)); 
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_min, sizeof(int)) failed."); exit(1); }


    err = cudaEventRecord(stop, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(stop, 0) failed."); exit(1); }
    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventSynchronize(stop) failed."); exit(1); }
    err = cudaEventElapsedTime(&elapsedTime, start, stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventElapsedTime(&elapsedTime, start, stop) failed."); exit(1); }
    printf ("Time for the kernel: %f ms\n", elapsedTime);

/********************************************/

    for (i = 0; i < matrix_size; i++)
    {
        for (j = 0; j < matrix_size; j++)
        {
            fprintf(fpA, "%4d ", A[i][j]);
            fprintf(fpB, "%4d ", B[i][j]);
            fprintf(fpC, "%4d ", C[i][j]);
        }

        fprintf(fpA, "\n");
        fprintf(fpB, "\n");
        fprintf(fpC, "\n");
    }

    fclose(fpA);
    fclose(fpB);
    fclose(fpC);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    for (i = 0; i < matrix_size; i++)
    {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }

    free(A);
    free(B);
    free(C);

    return 0;
}