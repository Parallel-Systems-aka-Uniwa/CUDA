#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
        int N, i;
        int *input_h, *output_h;
        int *vector_d;

        if (argc != 2) {
                printf("Usage: ./arg <Size of vectors upto 1024>\n");
                exit(1);
        }

        N = atoi(argv[1]);
        if (N < 1 || N > 1024) exit(1);

        input_h = (int *)malloc(N * sizeof(int));
        output_h = (int *)malloc(N * sizeof(int));
        for (i = 0; i < N; i++) {
                input_h[i] = 1;
                output_h[i] = 0;
        }
        cudaMalloc(&vector_d, N * sizeof(int));
        cudaMemcpy(vector_d, input_h, N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(output_h, vector_d, N * sizeof(int), cudaMemcpyDeviceToHost);

        printf("output_h[%4d] = %d\n", 0, output_h[0]);
        printf("output_h[%4d] = %d\n", (N - 1) / 2, output_h[(N - 1) / 2]);
        printf("output_h[%4d] = %d\n", N - 1, output_h[N - 1]);
}