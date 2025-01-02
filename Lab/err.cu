#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
        int N, i;
        int *input_h, *output_h;
        int *vector_d;
        cudaError_t  err;

        if (argc != 2) {
                printf("Usage: ./err <Size of vectors upto 1024>\n");
                exit(1);
        }

        N = atoi(argv[1]);
        if (N < 1 || N > 1024) { printf("Error: Size of vectors between 1 and 1024!\n"); exit(1); }
        input_h = (int *)malloc(N * sizeof(int));
        if (input_h == NULL) { printf("Could not allocate memory for input vector on host.\n"); exit(1); }
        output_h = (int *)malloc(N * sizeof(int));
        if (output_h == NULL) { printf("Could not allocate memory for output vector on host.\n"); exit(1); }

        for (i = 0; i < N; i++) {
                input_h[i] = 1;
                output_h[i] = 0;
        }
        err = cudaMalloc(&vector_d, N * sizeof(int));
        if (err != cudaSuccess) { printf("Could not allocate memory for vector on the device.\n"); exit(1); }
        err = cudaMemcpy(vector_d, input_h, N * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { printf("Could not copy input vector to device.\n"); exit(1); }
        err = cudaMemcpy(output_h, vector_d, N * sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { printf("Could not copy vector from device to output vector on host.\n"); exit(1); }
        printf("output_h[%4d] = %d\n", 0, output_h[0]);
        printf("output_h[%4d] = %d\n", (N - 1) / 2, output_h[(N - 1) / 2]);
        printf("output_h[%4d] = %d\n", N - 1, output_h[N - 1]);
}