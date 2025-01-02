#include <stdio.h>
#include <stdlib.h>

__global__ void multiply(int *matrix, int *input, int *output, int N)
{
        int     i = blockIdx.x * blockDim.x + threadIdx.x, j, sum = 0;

        if (i < N) {
                for (j = 0; j < N; j++) {
                        sum += (matrix[i * N + j] * input[j]);
                }
                output[i] = sum;
        }
}

int main(int argc, char *argv[])
{
        int             N, i, j, fill, blocks, threads;
        int             *input_h, *output_h, *matrix_h; /* Pointers for vectors on the host. */
        int             *input_d, *output_d, *matrix_d; /* Pointer for vector on the device. */

        if (argc != 3) { printf("Usage:\n");  printf("./mv <Size (1 to 20000)> <Threads per block (1 to 1024)>\n"); exit(1); }
        N = atoi(argv[1]);
        threads = atoi(argv[2]);
        matrix_h = (int *)malloc(N * N * sizeof(int));
        input_h = (int *)malloc(N * sizeof(int));
        output_h = (int *)malloc(N * sizeof(int));
        for (i = 0; i < N; i++) { input_h[i] = 1; output_h[i] = 0; }
        for (i = 0; i < N; i++) {
                fill = i % 100;
                for (j = 0; j < N; j++) {
                        matrix_h[i * N + j] = fill;
                }
        }
        cudaMalloc(&input_d, N * sizeof(int));
        cudaMalloc(&output_d, N * sizeof(int));
        cudaMalloc(&matrix_d, N * N * sizeof(int));
        cudaMemcpy(input_d, input_h, N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(matrix_d, matrix_h, N * N * sizeof(int), cudaMemcpyHostToDevice);
        blocks = (N - 1) / threads + 1;
        multiply<<<blocks, threads>>>(matrix_d, input_d, output_d, N);
        cudaMemcpy(output_h, output_d, N * sizeof(int), cudaMemcpyDeviceToHost);

        printf("output_h[%4d] = %d\n", 0, output_h[0]);
        printf("output_h[%4d] = %d\n", (N - 1) / 2, output_h[(N - 1) / 2]);
        printf("output_h[%4d] = %d\n", N - 1, output_h[N - 1]);
}