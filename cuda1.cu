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
#define T 2

__device__ calcAvg()
{

}

__device__ calcMin()
{

}

__device__ findMax()
{

}

__device__ checkMax()
{

}

__device__ createB()
{

}

__device__ createC()
{

}

__global__ kernel()
{
    
}

int main(int argc, char *argv[])
{
    int **A, **B, **C;
    int *d_A, *d_N, *d_outArray, *d_max, *d_min;
    float *d_avg;
    float elapsedTime;
    int i, j;
    int matrix_size, grid_size, block_size;
    FILE *fpA, *fpB, *fpC;
    cudaEvent_t start, stop;
    cudaError_t err;
    size_t bytes;

    matrix_size = N;
    grid_size = BL;
    block_size = T;

    cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0); // 0 is the device ID

printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
printf("Max block dimensions: (%d, %d, %d)\n",
       prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
printf("Max grid dimensions: (%d, %d, %d)\n",
       prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    
    if (block_size < 1 || block_size > 32) 
    {
        printf("Threads x Threads per block must be between 1 to 32.\n");
        exit(1);
    }

    if (grid_size < 1 || grid_size > (matrix_size / block_size))
    {
        printf("Blocks must be between 1 to 65535.\n");
        exit(1);
    }

    if (argc != 4) 
    {
        printf("Usage: %s A.txt B.txt C.txt\n", argv[0]);
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