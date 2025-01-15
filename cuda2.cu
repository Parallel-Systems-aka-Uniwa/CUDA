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

__global__ void calcColMeans(int *d_A, float *d_Amean) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Global column index

    if (col >= N) return; // Ensure we don't go out of bounds

    float sum = 0.0f;

    // Compute the sum of the column
    for (int row = 0; row < N; ++row) {
        sum += d_A[row * N + col];
    }

    // Store the result in the output array
    d_Amean[col] = sum / N; // Calculate the mean
}


__global__ void subMeansT(int *d_A, float *d_Amean, float *d_Asubmeans, float *d_ATsubmeans)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Global row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Global column index

    if (row < N && col < N) {
        // Subtract the mean from each element and store it
        d_Asubmeans[row * N + col] = d_A[row * N + col] - d_Amean[col];
        // Transpose the matrix for the inverted operation
        d_ATsubmeans[col * N + row] = d_Asubmeans[row * N + col];
    }
}


__global__ void calcCov(float *d_Asubmeans, float *d_ATsubmeans, float *d_Acov)
{
    __shared__ float cache[nThreads]; // Dynamically allocated shared memory
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float   temp = 0;
    while (tid < N) {
        temp += d_Asubmeans[tid] * d_ATsubmeans[tid];
        tid += blockDim.x * gridDim.x;
    }
    
    // set the cache values
    cache[cacheIndex] = temp;
    
    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        d_Acov[blockIdx.x] = cache[0];
}

int main(int argc, char *argv[])
{
    int *h_A;
    float *h_Acov, *h_Amean, *h_Asubmeans, *h_ATsubmeans;
    int *d_A;
    float *d_Acov, *d_Amean, *d_Asubmeans, *d_ATsubmeans;

    int n, threadsPerBlock, blocksPerGrid;
    int intBytes, floatBytes;
    int max_threads, max_block_dimX, max_block_dimY, max_block_dimZ, max_grid_dimX, max_grid_dimY, max_grid_dimZ;
    int i, j;
    FILE *fpA, *fpAcov, *fpAsubmeans, *fpATsubmeans, *fpAmean;
    float elapsedTime1, elapsedTime2, elapsedTime3;

    cudaEvent_t start, stop;
    cudaError_t err;
    cudaDeviceProp prop;

    if (argc != 6)
    {
        printf("Usage: %s A.txt A_means.txt A_submeans.txt AT_submeans.txt A_cov.txt\n", argv[0]);
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
    fpAsubmeans = fopen(argv[3], "w");
    if (fpAsubmeans == NULL) { printf("Cannot open file %s\n", argv[3]); exit(1); }
    fpATsubmeans = fopen(argv[4], "w");
    if (fpATsubmeans == NULL) { printf("Cannot open file %s\n", argv[4]); exit(1); }
    fpAcov = fopen(argv[5], "w");
    if (fpAcov == NULL) { printf("Cannot open file %s\n", argv[5]); exit(1); }

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
    h_Amean = (float *) malloc(n * sizeof(float));
    if (h_Amean == NULL) { printf("Error --> Memory allocation failed for A_mean.\n"); exit(1); }
    h_Asubmeans = (float *) malloc(floatBytes);
    if (h_Asubmeans == NULL) { printf("Error --> Memory allocation failed for A_submeans.\n"); exit(1); }
    h_ATsubmeans = (float *) malloc(floatBytes);
    if (h_ATsubmeans == NULL) { printf("Error --> Memory allocation failed for AT_submeans.\n"); exit(1); }
    h_Acov = (float *) malloc(floatBytes);
    if (h_Acov == NULL) { printf("Error --> Memory allocation failed for A_cov.\n"); exit(1); }

    //srand(time(NULL));

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            h_A[i * n + j] = rand() % 199 - 99;                           // Τιμές στο διάστημα [-99, 99]
            h_A[i * n + j] = h_A[i * n + j] >= 0 ? h_A[i * n + j] + 10 : h_A[i * n + j] - 10;  // Τυχαία επιλογή προσήμου
            h_Asubmeans[i * n + j] = 0.0;
            h_ATsubmeans[i * n + j] = 0.0;
            h_Acov[i * n + j] = 0.0;
        }
        h_Amean[i] = 0.0;
    }
    
    err = cudaMalloc((void **) &d_A, intBytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_A, bytes) failed."); exit(1); }
    err = cudaMalloc((void **) &d_Amean, n * sizeof(float));
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_Amean, bytes) failed."); exit(1); }
    err = cudaMalloc((void **) &d_Asubmeans, floatBytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_Asubmeans, bytes) failed."); exit(1); }
    err = cudaMalloc((void **) &d_ATsubmeans, floatBytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_ATsubmeans, bytes) failed."); exit(1); }
    err = cudaMalloc((void **) &d_Acov, floatBytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_Acov, bytes) failed."); exit(1); }

    err = cudaMemcpy(d_A, h_A, intBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice) failed."); exit(1); }

    dim3 dimBlock(nThreads, nThreads);
    dim3 dimGrid(nBlocks, nBlocks);

/* 1st kernel launch */

    cudaEventRecord(start, 0);

    calcColMeans<<<nBlocks, nThreads>>>(d_A, d_Amean);

    cudaEventRecord(stop, 0);


    err = cudaMemcpy(h_Amean, d_Amean, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(h_Amean, d_Amean, bytes, cudaMemcpyDeviceToHost) failed."); exit(1); }

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime1, start, stop);
    printf("Time for the kernel calcColMeans<<<>>>(): %f ms\n", elapsedTime1);

/* 2nd kernel launch */

    cudaEventRecord(start, 0);

    subMeansT<<<dimGrid, dimBlock>>>(d_A, d_Amean, d_Asubmeans, d_ATsubmeans);

    cudaEventRecord(stop, 0);

    err = cudaMemcpy(h_Asubmeans, d_Asubmeans, floatBytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(h_Asubmeans, d_Asubmeans, bytes, cudaMemcpyDeviceToHost) failed."); exit(1); }
    err = cudaMemcpy(h_ATsubmeans, d_ATsubmeans, floatBytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(h_ATsubmeans, d_ATsubmeans, bytes, cudaMemcpyDeviceToHost) failed."); exit(1); }

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime2, start, stop);
    printf("Time for the kernel subMeans<<<>>>(): %f ms\n", elapsedTime2);

/* 3rd kernel launch */

    cudaEventRecord(start, 0);

    calcCov<<<nBlocks, nThreads>>>(d_Asubmeans, d_ATsubmeans, d_Acov);

    cudaEventRecord(stop, 0);

    err = cudaMemcpy(h_Acov, d_Acov, floatBytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(h_Acov, d_Acov, bytes, cudaMemcpyDeviceToHost) failed."); exit(1); }

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime3, start, stop);
    printf("Time for the kernel calcCov<<<>>>(): %f ms\n", elapsedTime3);

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            fprintf(fpA, "%4d ", h_A[i * n + j]);
            fprintf(fpAsubmeans, "%4.2f ", h_Asubmeans[i * n + j]);
            fprintf(fpATsubmeans, "%4.2f ", h_ATsubmeans[i * n + j]);
            fprintf(fpAcov, "%4.2f ", h_Acov[i * n + j]);
        }
        fprintf(fpAmean, "%4.2f\n", h_Amean[i]);
        fprintf(fpA, "\n");
        fprintf(fpAsubmeans, "\n");
        fprintf(fpATsubmeans, "\n");
        fprintf(fpAcov, "\n");
    }


    free(h_A);
    free(h_Acov);
    free(h_Amean);
    free(h_Asubmeans);
    free(h_ATsubmeans);
    
    cudaFree(d_A);
    cudaFree(d_Acov);
    cudaFree(d_Amean);
    cudaFree(d_Asubmeans);
    cudaFree(d_ATsubmeans);

    fclose(fpA);
    fclose(fpAmean);
    fclose(fpAcov);
    fclose(fpAsubmeans);
    fclose(fpATsubmeans);

    return 0;
}