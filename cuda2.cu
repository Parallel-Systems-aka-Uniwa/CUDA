/*
 *  === Αρχείο: cuda2.cu ===
 *
 *  Ονοματεπώνυμο: Αθανασίου Βασίλειος Ευάγγελος
 *  Αριθμός Μητρώου: 19390005
 *  Πρόγραμμα Σπουδών: ΠΑΔΑ
 *  
 *  Μεταγλώττιση: nvcc -o cuda2 cuda2.cu
 *  Εκτέλεση: ./cuda2 A.txt A_cov.txt
 * 
 */
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define N 8
#define nThreads 4
#define nBlocks (int)ceil((float)N/nThreads)

int main(int argc, char *argv[])
{
    int *h_A;
    int *d_A;

    int n, threadsPerBlock, blocksPerGrid;
    int intbytes;
    int max_threads, max_block_dimX, max_block_dimY, max_block_dimZ, max_grid_dimX, max_grid_dimY, max_grid_dimZ;
    int i, j;
    FILE *fpA, *fpAcov;
    float elapsedTime;

    cudaEvent_t start, stop;
    cudaError_t err;
    cudaDeviceProp prop;

    if (argc != 3)
    {
        printf("Usage: %s A.txt A_cov.txt\n", argv[0]);
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
    fpAcov = fopen(argv[2], "w");
    if (fpAcov == NULL) { printf("Cannot open file %s\n", argv[2]); exit(1); }

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

    h_A = (int *) malloc(intBytes);
    if (h_A == NULL) { printf("Error --> Memory allocation failed for A.\n"); exit(1); }

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
        {
            h_A[i * n + j] = rand() % 199 - 99;                           // Τιμές στο διάστημα [-99, 99]
            h_A[i * n + j] = h_A[i * n + j] >= 0 ? h_A[i * n + j] + 10 : h_A[i * n + j] - 10;  // Τυχαία επιλογή προσήμου
        }
    
    err = cudaMalloc((void **) &d_A, intBytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_A, bytes) failed."); exit(1); }

    err = cudaMemcpy(d_A, h_A, intBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice) failed."); exit(1); }

    dim3 dimBlock(nThreads, nThreads);
    dim3 dimGrid(nBlocks, nBlocks);

    cudaEventRecord(start, 0);

    // Κλήση του kernel

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time for the kernel: %f ms\n", elapsedTime);

    free(h_A);
    cudaFree(d_A);

    fclose(fpA);
    fclose(fpAcov);

    return 0;
}