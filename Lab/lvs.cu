#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void multiply(int *vector, int num, int N)
{
        int     i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < N) {
                vector[i] *= num;
        }
}

int main(int argc, char *argv[])
{
        int             N, num, i, blocks, threads;
        int             *input_h, *output_h;
        int             *vector_d;

        if (argc != 4) {
                printf("Usage:\n");
                printf("./lvs <Size of vectors (1 to 1 million)> <Scalar (1 integer)> <Threads per block (1 to 1024)>\n");
                exit(1);
        }

        N = atoi(argv[1]);
        num = atoi(argv[2]);
        threads = atoi(argv[3]);

        input_h = (int *)malloc(N * sizeof(int));
        output_h = (int *)malloc(N * sizeof(int));
        for (i = 0; i < N; i++) {
                input_h[i] = 1;
                output_h[i] = 0;
        }
        cudaMalloc(&vector_d, N * sizeof(int));
        cudaMemcpy(vector_d, input_h, N * sizeof(int), cudaMemcpyHostToDevice);
        blocks = (N - 1) / threads + 1;
        multiply<<<blocks, threads>>>(vector_d, num, N);
        cudaMemcpy(output_h, vector_d, N * sizeof(int), cudaMemcpyDeviceToHost);
        printf("output_h[%4d] = %d\n", 0, output_h[0]);
        printf("output_h[%4d] = %d\n", (N - 1) / 2, output_h[(N - 1) / 2]);
        printf("output_h[%4d] = %d\n", N - 1, output_h[N - 1]);
}