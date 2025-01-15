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

__global__ void calcColMeans2D(int *d_A, float *d_Amean) 
{
    // Use a 2D grid and block
    int col = blockIdx.x;              // Each block processes one column
    int row = threadIdx.x + threadIdx.y * blockDim.x; // Flatten thread indices
    int stride = blockDim.x * blockDim.y; // Total threads in the block

    __shared__ float cache[nThreads]; // Dynamically allocated shared memory

    float sum = 0.0f;

    // Accumulate sum for this thread's portion of the column
    int i = row;
    while (i < N) {
        sum += d_A[i * N + col];
        i += stride; // Move to the next row assigned to this thread
    }

    // Store the partial sum in shared memory
    cache[row] = sum;
    __syncthreads();

    // Perform parallel reduction in shared memory
    for (int s = stride / 2; s > 0; s /= 2) {
        if (row < s) {
            cache[row] += cache[row + s];
        }
        __syncthreads();
    }

    // Write the final mean for the column
    if (row == 0) {
        d_Amean[col] = cache[0] / N;
    }
}



__global__ void subMeans(int *d_A, float *d_Amean)
{

}

__global__ void calcCov(int *d_A, float *d_Acov)
{
}

int main(int argc, char *argv[])
{
    int *h_A;
    float *h_Acov, *h_Amean;
    int *d_A;
    float *d_Acov, *d_Amean;

    int n, threadsPerBlock, blocksPerGrid;
    int intBytes, floatBytes;
    int max_threads, max_block_dimX, max_block_dimY, max_block_dimZ, max_grid_dimX, max_grid_dimY, max_grid_dimZ;
    int i, j;
    FILE *fpA, *fpAcov, *fpAmean;
    float elapsedTime;

    cudaEvent_t start, stop;
    cudaError_t err;
    cudaDeviceProp prop;

    if (argc != 4)
    {
        printf("Usage: %s A.txt A_means.txt A_cov.txt\n", argv[0]);
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
    fpAmean = fopen(argv[2], "w");
    if (fpAmean == NULL) { printf("Cannot open file %s\n", argv[2]); exit(1); }
    fpAcov = fopen(argv[3], "w");
    if (fpAcov == NULL) { printf("Cannot open file %s\n", argv[3]); exit(1); }

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
    floatBytes = n * n * sizeof(float);

    h_A = (int *) malloc(intBytes);
    if (h_A == NULL) { printf("Error --> Memory allocation failed for A.\n"); exit(1); }
    h_Acov = (float *) malloc(floatBytes);
    if (h_Acov == NULL) { printf("Error --> Memory allocation failed for A_cov.\n"); exit(1); }
    h_Amean = (float *) malloc(n * sizeof(float));
    if (h_Amean == NULL) { printf("Error --> Memory allocation failed for A_mean.\n"); exit(1); }

    //srand(time(NULL));

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            h_A[i * n + j] = rand() % 199 - 99;                           // Τιμές στο διάστημα [-99, 99]
            h_A[i * n + j] = h_A[i * n + j] >= 0 ? h_A[i * n + j] + 10 : h_A[i * n + j] - 10;  // Τυχαία επιλογή προσήμου
            h_Acov[i * n + j] = 0.0;
        }
        h_Amean[i] = 0.0;
    }
    
    err = cudaMalloc((void **) &d_A, intBytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_A, bytes) failed."); exit(1); }
    err = cudaMalloc((void **) &d_Acov, floatBytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_Acov, bytes) failed."); exit(1); }
    err = cudaMalloc((void **) &d_Amean, n * sizeof(float));
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_Amean, bytes) failed."); exit(1); }

    err = cudaMemcpy(d_A, h_A, intBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice) failed."); exit(1); }

    dim3 dimBlock(nThreads, nThreads);
    dim3 dimGrid(nBlocks, nBlocks);

    cudaEventRecord(start, 0);

    // Κλήση του kernel
    calcColMeans<<<dimGrid, dimBlock>>>(d_A, d_Amean);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    err = cudaMemcpy(h_Amean, d_Amean, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(h_Amean, d_Amean, bytes, cudaMemcpyDeviceToHost) failed."); exit(1); }

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            fprintf(fpA, "%4d ", h_A[i * n + j]);
        }
        fprintf(fpAmean, "%10.2f\n", h_Amean[i]);
        fprintf(fpA, "\n");
    }


    printf("Time for the kernel: %f ms\n", elapsedTime);

    free(h_A);
    free(h_Acov);
    free(h_Amean);
    cudaFree(d_A);
    cudaFree(d_Acov);
    cudaFree(d_Amean);

    fclose(fpA);
    fclose(fpAmean);
    fclose(fpAcov);

    return 0;
}