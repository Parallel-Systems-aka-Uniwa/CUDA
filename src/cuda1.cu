/*
 *  === File: cuda1.cu ===
 *
 *  Full Name: Athanasiou Vasileios Evangelos
 *  Student ID: 19390005
 *  Degree Program: PADA
 *  
 *  Compilation: nvcc -o cuda1 cuda1.cu
 *  Execution: ./cuda1 A.txt OutArr.txt
 * 
 */
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define N 10
#define nThreads 2
#define nBlocks (int)ceil((float)N/nThreads)

/*
 *  === Kernel Function: calcAvg ===
 *  Parameters: 
 *      - d_A: Input array (Device).
 *      - d_sum: Sum of the elements of the array (Device).
 *      - d_avg: Average of the elements of the array (Device).
 *  Returns: Nothing.
 * 
 *  Description:
 *      Computes the sum and average of all elements of array A using reduction and atomic operations.
 */
__global__ void calcAvg(int *d_A, int *d_sum, float *d_avg) 
{
    __shared__ int cache[nThreads * nThreads];  // Use shared memory for fast access by threads within the same block

    // Compute the global ID of each thread in the x and y dimensions
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    int totalElements = N * N;
    int tid = i * N + j; // Global index for the 2D array

    if (i < N && j < N)  // Ensure we are within bounds
        cache[threadIdx.x + threadIdx.y * blockDim.x] = d_A[tid];
    else
        cache[threadIdx.x + threadIdx.y * blockDim.x] = 0;  // Avoid out-of-bound reads

    __syncthreads();  // Synchronize threads within the block

    // Perform parallel reduction using a tree-based algorithm within the block
    int k = blockDim.x * blockDim.y / 2; 
    while (k != 0) 
    {
        if (threadIdx.x + threadIdx.y * blockDim.x < k)  // Only threads with valid IDs participate in reduction
            cache[threadIdx.x + threadIdx.y * blockDim.x] += 
                cache[threadIdx.x + threadIdx.y * blockDim.x + k];
        __syncthreads();  // Synchronize threads within the block
        k /= 2; 
    }

    // Use an atomic addition to update the global sum if this thread is the first in the block
    if (threadIdx.x == 0 && threadIdx.y == 0) 
        atomicAdd(d_sum, cache[0]);

    // Compute the average after the reduction (only in block (0,0))
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
        *d_avg = (float)(*d_sum) / totalElements;
}

/*
 *  === Kernel Function: findMax ===
 *  Parameters: 
 *      - d_A: Input array (Device).
 *      - d_amax: The maximum element of array A (Device).
 *  Returns: Nothing.
 * 
 *  Description:
 *      Computes the maximum element of array A using reduction and atomic operations.
 */
__global__ void findMax(int *d_A, int *d_amax)
{
    __shared__ int cache[nThreads * nThreads];  // Use shared memory for fast access by threads within the same block

    // Compute the global ID of each thread in the x and y dimensions
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    int tid = i * N + j; // Global index for the 2D array

    if (i < N && j < N)  // Ensure we are within bounds
        cache[threadIdx.x + threadIdx.y * blockDim.x] = d_A[tid];
    else
        cache[threadIdx.x + threadIdx.y * blockDim.x] = 0;  // Avoid out-of-bound reads

    __syncthreads();  // Synchronize threads within the block

    // Perform parallel reduction using a tree-based algorithm within the block
    int k = blockDim.x * blockDim.y / 2; 
    while (k != 0) 
    {
        if (threadIdx.x + threadIdx.y * blockDim.x < k)  // Only threads with valid IDs participate in reduction
            cache[threadIdx.x + threadIdx.y * blockDim.x] = 
                cache[threadIdx.x + threadIdx.y * blockDim.x] > cache[threadIdx.x + threadIdx.y * blockDim.x + k] ?
                cache[threadIdx.x + threadIdx.y * blockDim.x] : cache[threadIdx.x + threadIdx.y * blockDim.x + k];
        __syncthreads();  // Synchronize threads within the block
        k /= 2;
    }

    // Use atomic operation to update the global maximum if this thread is the first in the block
    if (threadIdx.x == 0 && threadIdx.y == 0) 
        atomicMax(d_amax, cache[0]);
}

// Custom atomicMin for floating-point numbers using atomicCAS
__device__ void atomicMin(float *address, float val)
{
    int *address_as_i = (int *) address;  // Convert address to an integer pointer
    int old = *address_as_i, assumed;

    // Convert the floating-point value to its integer representation
    int val_as_int = __float_as_int(val);

    do
    {
        assumed = old;
        // Perform atomicCAS on the integer representation
        old = atomicCAS(address_as_i, assumed, min(val_as_int, assumed));
    } 
    while (assumed != old);  // Repeat until no change occurs
}

/*
 *  === Kernel Function: createB ===
 *  Parameters: 
 *      - d_A: Input array (Device).
 *      - d_outArr: Output array B (Device).
 *      - d_bmin: The minimum element of array B (Device).
 *      - d_amax: The maximum element of array A (Device).
 *      - d_avg: The average of the elements of array A (Device).
 *  Returns: Nothing.
 * 
 *  Description:
 *      Computes array B with elements Bij = (m - Aij) / amax, where m is the average of A,
 *      and finds the minimum element using reduction and atomic operations.
 */
__global__ void createB(int *d_A, float *d_outArr, float *d_bmin, int *d_amax, float *d_avg)
{
    __shared__ float cache[nThreads * nThreads];  // Use shared memory for fast access by threads within the same block

    // Compute the global ID of each thread in the x and y dimensions
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute the corresponding value for Bij
    if (i < N && j < N)
    {
        if (*d_amax != 0)
            d_outArr[i * N + j] = (*d_avg - (float) d_A[i * N + j]) / (float) *d_amax;
        else
            d_outArr[i * N + j] = 0.0; // Handle division by zero
    }

    int tid = i * N + j; // Global index for the 2D array

    // Load data into shared memory (cache)
    if (i < N && j < N)  
        cache[threadIdx.x + threadIdx.y * blockDim.x] = d_outArr[tid];
    else
        cache[threadIdx.x + threadIdx.y * blockDim.x] = 10000000000000.0;  // Use a very large value for threads out of bounds

    __syncthreads();  // Synchronize threads within the block

    // Perform parallel reduction using a tree-based algorithm within the block
    int k = blockDim.x * blockDim.y / 2;  // Half of the total threads
    while (k != 0) 
    {
        if (threadIdx.x + threadIdx.y * blockDim.x < k) 
        {
            cache[threadIdx.x + threadIdx.y * blockDim.x] = 
                cache[threadIdx.x + threadIdx.y * blockDim.x] < cache[threadIdx.x + threadIdx.y * blockDim.x + k] ?
                cache[threadIdx.x + threadIdx.y * blockDim.x] : cache[threadIdx.x + threadIdx.y * blockDim.x + k];
        }
        __syncthreads();  // Synchronize threads
        k /= 2;  // Halve the step in each iteration
    }

    // Use the custom atomicMin operation to update the global minimum of B
    if (threadIdx.x == 0 && threadIdx.y == 0) 
        atomicMin(d_bmin, cache[0]);
}

/*
 *  === Kernel Function: createC ===
 *  Parameters: 
 *      - d_A: Input array (Device).
 *      - d_outArr: Output array C (Device).
 *  Returns: Nothing.
 * 
 *  Description:
 *      Computes array C with elements 
 *      Cij = {Aij + Ai(j+1) + Ai(j-1)} / 3, taking into account
 *      that if j+1 = N then Ai(j+1)=Ai0, and if j-1 = -1 then Ai(j-1)=Ai(N-1).
 */
__global__ void createC(int *d_A, float *d_outArr)
{
    // Compute the global ID of each thread in the x and y dimensions
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    int left, right, current;

    // Check if we are within bounds
    if (i < N && j < N)
    {
        left = (j - 1 != -1) ? d_A[i * N + (j - 1)] : d_A[i * N + (N - 1)];  // Left element (boundary handling)
        right = (j + 1 != N) ? d_A[i * N + (j + 1)] : d_A[i * N + 0];  // Right element (boundary handling)
        current = d_A[i * N + j];  // Current element
        
        // Compute Cij
        d_outArr[i * N + j] = (float) (current + left + right) / 3.0;
    }
}

/*
 *  === Function: create2DArray ===
 *  Parameters: 
 *      - Array: Pointer to a one-dimensional array (treated as 2D).
 *  Returns: Nothing.
 * 
 *  Description:
 *      Creates a random NxN array with values in the range [1, 100].
 *      Ensures that the maximum element of the array is greater than (or less than) N times the 
 *      average of the array.
 */
void create2DArray(int *Array)
{
    int sum = 0;  // Sum of the elements of the array
    int amax = 0; // Maximum value in the array
    int i, j, m;

    // Fill the array with random values and compute the sum and maximum value
    for (i = 0; i < N; ++i) 
    {
        for (j = 0; j < N; ++j) 
        {
            Array[i * N + j] = rand() % 100 + 1; // Random value in [1, 100]
            sum += Array[i * N + j]; // Add to total sum
            if (Array[i * N + j] > amax) 
            {
                amax = Array[i * N + j]; // Update maximum value
            }
        }
    }

    m = sum / (N * N); // Compute the average
    while (amax <= N * m) // Ensure that the maximum value is greater than N * m
    {
        i = rand() % N; // Random row selection
        j = rand() % N; // Random column selection
        Array[i * N + j] += (N * m - amax + 1); // Increase the element so the condition is met
        amax = Array[i * N + j]; // Update maximum value
    }
}

/*
 *  === Second Part: Host Code using CUDA Kernels ===
 */
 
// Declaration of host main function
int main(int argc, char *argv[])
{
    int *h_A;                                                        // [Host] Array A 
    int *h_amax, *h_sum;                                             // [Host] Maximum element of A and the sum of the elements
    int *d_A, *d_amax, *d_sum;                                       // [Device] Array A, maximum element, and sum of elements
    float *h_OutArr;                                                 // [Host] Output array B or C
    float *h_avg;                                                    // [Host] Average of the elements of A
    float *d_OutArr, *d_avg;                                         // [Device] Array B or C and the average
    float *h_bmin, *d_bmin;                                          // [Host] Minimum element of B and [Device] minimum element
    int n, threadsPerBlock, blocksPerGrid;                           // Array size, threads per block, and blocks per grid
    int intBytes, floatBytes;                                        // Array sizes in bytes
    int max_threads, max_block_dimX, max_block_dimY, max_block_dimZ; // Maximum values for threads and block dimensions
    int max_grid_dimX, max_grid_dimY, max_grid_dimZ;                 // Maximum values for grid dimensions
    int i, j;                                                        // Loop indices
    FILE *fpA, *fpOutArr;                                            // File pointers for storing arrays A and B or C
    char arr;                                                        // Name of the output array (B or C)
    float elapsedTime1, elapsedTime2, elapsedTime3;                  // Kernel execution times
    cudaEvent_t start, stop;                                         // CUDA events for timing
    cudaError_t err;                                                 // CUDA error code
    cudaDeviceProp prop;                                             // Device properties

    // Check command-line parameter count
    if (argc != 3)
    {
        printf("Usage: %s A.txt OutArr.txt\n", argv[0]);
        exit(1);
    }

    // Initialize parameters
    n = N;
    threadsPerBlock = nThreads;
    blocksPerGrid = nBlocks;

    // Get CUDA device properties
    err = cudaGetDeviceProperties(&prop, 0); 
    if (err != cudaSuccess) { printf("CUDA Error --> cudaGetDeviceProperties failed.\n"); exit(1); }

    // Assign maximum values from device properties
    max_threads = prop.maxThreadsPerBlock;
    max_block_dimX = prop.maxThreadsDim[0];
    max_block_dimY = prop.maxThreadsDim[1];
    max_block_dimZ = prop.maxThreadsDim[2];
    max_grid_dimX = prop.maxGridSize[0];
    max_grid_dimY = prop.maxGridSize[1];
    max_grid_dimZ = prop.maxGridSize[2];

    // Print device properties
    printf("--------------- Device Properties ---------------\n");
    printf("Device name           : %s\n", prop.name);
    printf("Max threads per block : %d\n", max_threads);
    printf("Max block dimensions  : %d x %d x %d\n", max_block_dimX, max_block_dimY, max_block_dimZ);
    printf("Max grid dimensions   : %d x %d x %d\n", max_grid_dimX, max_grid_dimY, max_grid_dimZ);
    printf("-------------------------------------------------\n");

    // Validate parameter values
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

    // Open files to store arrays
    fpA = fopen(argv[1], "w");
    if (fpA == NULL) { printf("Cannot open file %s\n", argv[1]); exit(1); }
    fpOutArr = fopen(argv[2], "w");
    if (fpOutArr == NULL) { printf("Cannot open file %s\n", argv[2]); exit(1); }

    // Print input parameters
    printf("--------------- Input Parameters ---------------\n");
    printf("Matrix size        : %d x %d\n", n, n);
    printf("Blocks per Grid    : %d\n", blocksPerGrid);
    printf("Threads per Block  : %d\n", threadsPerBlock);
    printf("------------------------------------------------\n");

    // Calculate array sizes in bytes
    intBytes = n * n * sizeof(int);
    floatBytes = n * n * sizeof(float);

    // Allocate host memory for arrays
    h_A = (int *) malloc(intBytes);
    if (h_A == NULL) { printf("Error --> Memory allocation failed for A.\n"); exit(1); }
    h_OutArr = (float *) malloc(floatBytes);
    if (h_OutArr == NULL) { printf("Error --> Memory allocation failed for OutArr.\n"); exit(1); }
    h_avg = (float *) malloc(sizeof(float));
    if (h_avg == NULL) { printf("Error --> Memory allocation failed for avg.\n"); exit(1); }
    h_amax = (int *) malloc(sizeof(int));
    if (h_amax == NULL) { printf("Error --> Memory allocation failed for max.\n"); exit(1); }
    h_bmin = (float *) malloc(sizeof(float));
    if (h_bmin == NULL) { printf("Error --> Memory allocation failed for min.\n"); exit(1); }
    h_sum = (int *) malloc(sizeof(int));
    if (h_sum == NULL) { printf("Error --> Memory allocation failed for sum.\n"); exit(1); }

    // Initialize variables
    *h_sum = 0;
    *h_avg = 0.0;
    *h_amax = 0;
    *h_bmin = 0.0;

    srand(time(NULL));

    // Create array A using a random number generator program
    create2DArray(h_A);
    printf("The array A has been stored in file %s\n", argv[1]);

// ============== Start of Parallel Computation ==============

    // Create CUDA events for timing
    err = cudaEventCreate(&start);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventCreate(&start) failed.\n"); exit(1); }
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventCreate(&stop) failed.\n"); exit(1); }

    // Allocate device memory
    err = cudaMalloc((void **) &d_A, intBytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_A, bytes) failed."); exit(1); }
    err = cudaMalloc((void **) &d_OutArr, floatBytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_OutArray, bytes) failed."); exit(1); }
    err = cudaMalloc((void **) &d_sum, sizeof(int));
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_sum, sizeof(int)) failed."); exit(1); }
    err = cudaMalloc((void **) &d_avg, sizeof(float));
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_avg, sizeof(float)) failed."); exit(1); }
    err = cudaMalloc((void **) &d_amax, sizeof(int));
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_amax, sizeof(int)) failed."); exit(1); }
    err = cudaMalloc((void **) &d_bmin, sizeof(float)); 
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_bmin, sizeof(int)) failed."); exit(1); }

    // Create 2D grid with 2D blocks
    dim3 dimBlock(nThreads, nThreads);
    dim3 dimGrid(N/nBlocks, N/nBlocks);

    /* 
     * === Execute Kernel: calcAvg<<<dimGrid, dimBlock>>> ===
     * Purpose:
     *   - Computes the sum of all elements of d_A and calculates their average.
     *
     * Grid and Block Configuration:
     *   - dimGrid  : Number of blocks in the grid (X: nBlocks, Y: nBlocks, Z: 1).
     *   - dimBlock : Number of threads per block (X: nThreads, Y: nThreads, Z: 1).
     *
     * Memory Transfers:
     *   - cudaMemcpy(d_A, h_A, intBytes, cudaMemcpyHostToDevice) transfers the input array from host to device.
     *   - cudaMemcpy(d_sum, h_sum, sizeof(int), cudaMemcpyHostToDevice) transfers the initial sum variable to the device.
     *   - cudaMemcpy(h_avg, d_avg, sizeof(float), cudaMemcpyDeviceToHost) retrieves the computed average from device to host.
     *
     * Performance Measurement:
     *   - Uses CUDA events to measure kernel execution time.
     *
     * Error Handling:
     *   - Each CUDA call is followed by an error check (if not cudaSuccess).
     */
    err = cudaMemcpy(d_A, h_A, intBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice) failed."); exit(1); }

    err = cudaEventRecord(start,0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(start,0) failed."); exit(1); }

    calcAvg<<<dimGrid, dimBlock>>>(d_A, d_sum, d_avg);
    
    err = cudaEventRecord(stop,0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(stop,0) failed."); exit(1); }

    err = cudaMemcpy(h_avg, d_avg, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(h_avg, d_avg, sizeof(float), cudaMemcpyDeviceToHost) failed."); exit(1); }

    printf("Average: %4.2f\n", *h_avg);

    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventSynchronize(stop) failed."); exit(1); }
    err = cudaEventElapsedTime(&elapsedTime1, start, stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventElapsedTime(&elapsedTime1, start, stop) failed."); exit(1); }
    
    printf ("Time for the kernel calcAvg<<<>>>(): %f ms\n", elapsedTime1);

    /* 
     * === Execute Kernel: findMax<<<dimGrid, dimBlock>>> ===
     * Purpose:
     *   - Computes the maximum element of array d_A and stores it in d_amax.
     *
     * Grid and Block Configuration:
     *   - dimGrid  : Number of blocks in the grid (X: nBlocks, Y: nBlocks, Z: 1).
     *   - dimBlock : Number of threads per block (X: nThreads, Y: nThreads, Z: 1).
     *
     * Memory Transfers:
     *   - (Previous call) cudaMemcpy(d_A, h_A, intBytes, cudaMemcpyHostToDevice): Transfers input array to device.
     *   - cudaMemcpy(h_amax, d_amax, sizeof(int), cudaMemcpyDeviceToHost): Retrieves the maximum element from device to host.
     *
     * Performance Measurement:
     *   - Uses CUDA events to measure kernel execution time.
     *
     * Error Handling:
     *   - Each CUDA call is followed by an error check.
     */
    err = cudaEventRecord(start, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(start,0) failed."); exit(1); }
    
    findMax<<<dimGrid, dimBlock>>>(d_A, d_amax);
    
    err = cudaEventRecord(stop, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(stop,0) failed."); exit(1); }

    err = cudaMemcpy(h_amax, d_amax, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(&h_amax, d_amax, sizeof(int), cudaMemcpyDeviceToHost) failed."); exit(1); }

    printf("Max: %d\n", *h_amax);

    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventSynchronize(stop) failed."); exit(1); }
    err = cudaEventElapsedTime(&elapsedTime2, start, stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventElapsedTime(&elapsedTime2, start, stop) failed."); exit(1); }

    printf ("Time for the kernel findMax<<<>>>(): %f ms\n", elapsedTime2);

    /* 
     * === Execute Kernel: createB or createC ===
     * Purpose:
     *   - If the condition (amax > N * m) is satisfied:
     *       - Execute kernel createB to compute array B with values
     *         Bij = (m - Aij) / amax.
     *       - Also find the minimum element of array B.
     *   - If the condition is not satisfied:
     *       - Execute kernel createC to compute array C with values
     *         Cij = {Aij + Ai(j+1) + Ai(j-1)} / 3.
     *
     * Grid and Block Configuration:
     *   - dimGrid  : Number of blocks in the grid (X: nBlocks, Y: nBlocks, Z: 1).
     *   - dimBlock : Number of threads per block (X: nThreads, Y: nThreads, Z: 1).
     *
     * Memory Transfers:
     *   - cudaMemcpy(h_OutArr, d_OutArr, floatBytes, cudaMemcpyDeviceToHost): Retrieves the output array (B or C) from device to host.
     *   - If kernel createB is executed:
     *       - cudaMemcpy(h_bmin, d_bmin, sizeof(float), cudaMemcpyDeviceToHost): Retrieves the minimum element of array B.
     *
     * Performance Measurement:
     *   - Uses CUDA events to measure kernel execution time.
     *
     * Error Handling:
     *   - Each CUDA call is followed by an error check.
     */
    err = cudaEventRecord(start, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(start, 0) failed."); exit(1); }

    if (*h_amax > N * (*h_avg))
    {
        arr = 'B';

        createB<<<dimGrid, dimBlock>>>(d_A, d_OutArr, d_bmin, d_amax, d_avg);
        
        err = cudaEventRecord(stop, 0);
        if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(stop,0) failed."); exit(1); }

        err = cudaMemcpy(h_OutArr, d_OutArr, floatBytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(&h_OutArr, d_OutArr, doubleBytes, cudaMemcpyDeviceToHost) failed."); exit(1); }
        err = cudaMemcpy(h_bmin, d_bmin, sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(&h_bmin, d_bmin, sizeof(float), cudaMemcpyDeviceToHost) failed."); exit(1); }

        printf("The array %c has been stored in file %s\n", arr,  argv[2]);
        printf("Min: %4.4f\n", *h_bmin);
    }
    else
    {
        arr = 'C';

        createC<<<dimGrid, dimBlock>>>(d_A, d_OutArr);
        
        err = cudaEventRecord(stop, 0);
        if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(stop,0) failed."); exit(1); }

        err = cudaMemcpy(h_OutArr, d_OutArr, floatBytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(&h_OutArr, d_OutArr, doubleBytes, cudaMemcpyDeviceToHost) failed."); exit(1); }

        printf("The array %c has been stored in file %s\n", arr,  argv[2]);
    }

    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventSynchronize(stop) failed."); exit(1); }
    err = cudaEventElapsedTime(&elapsedTime3, start, stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventElapsedTime(&elapsedTime3, start, stop) failed."); exit(1); }

    printf ("Time for the kernel create%c<<<>>>(): %f ms\n", arr, elapsedTime3);

// ============== End of Parallel Computation ==============

    fprintf(fpOutArr, "Array %c\n", arr);

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            fprintf(fpA, "%4d ", h_A[i * n + j]);
            fprintf(fpOutArr, "%4.4f ", h_OutArr[i * n + j]);
        }

        fprintf(fpA, "\n");
        fprintf(fpOutArr, "\n");
    }

    fclose(fpA);
    fclose(fpOutArr);

    err = cudaEventDestroy(start);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventDestroy(start) failed."); exit(1); }
    err = cudaEventDestroy(stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventDestroy(stop) failed."); exit(1); }

    free(h_A);
    free(h_OutArr);
    free(h_avg);
    free(h_amax);
    free(h_bmin);
    free(h_sum);

    err = cudaFree(d_A);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaFree(d_A) failed."); exit(1); }
    err = cudaFree(d_OutArr);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaFree(d_OutArr) failed."); exit(1); }
    err = cudaFree(d_avg);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaFree(d_avg) failed."); exit(1); }
    err = cudaFree(d_amax);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaFree(d_amax) failed."); exit(1); }
    err = cudaFree(d_bmin);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaFree(d_bmin) failed."); exit(1); }
    err = cudaFree(d_sum);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaFree(d_sum) failed."); exit(1); }

    return 0;
}

/*
 *  === Function: create2DArray ===
 *  Parameters: 
 *      - Array: Pointer to a one-dimensional array (handled as 2D).
 *  Returns: Nothing.
 * 
 *  Description:
 *      Creates a random NxN array with values in the range [1, 100].
 *      Ensures that the maximum element of the array is either greater or less than N times the
 *      average of the array.
 */
void create2DArray(int *Array)
{
    int sum = 0;  // Sum of the array elements
    int amax = 0; // Maximum value in the array
    int i, j, m;

    // Fill the array with random values and compute the sum and maximum value
    for (i = 0; i < N; ++i) 
    {
        for (j = 0; j < N; ++j) 
        {
            Array[i * N + j] = rand() % 100 + 1; // Random value in [1, 100]
            sum += Array[i * N + j]; // Add to total sum
            if (Array[i * N + j] > amax) 
            {
                amax = Array[i * N + j]; // Update maximum value
            }
        }
    }

    m = sum / (N * N); // Compute the average
    while (amax <= N * m) // Verify that the maximum value is greater than N * m
    {
        i = rand() % N; // Randomly select a row
        j = rand() % N; // Randomly select a column
        Array[i * N + j] += (N * m - amax + 1); // Increase the element to satisfy the condition
        amax = Array[i * N + j]; // Update maximum value
    }
}
