#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define PADDING		(128)

/*
 * Here you will have to create a kernel that will multiply a matrix with
 * a vector.
 *
 * When writing the kernel, assume that the size of the vector (hence each
 * dimension of the matrix too) might be > 1024.
 */
__global__ void multiply(int *matrix, int *input, int *output, int N, int ld)
{
	int	i = blockIdx.x * blockDim.x + threadIdx.x, j, sum = 0;

	if (i < N) {
		for (j = 0; j < N; j++) {
			sum += (matrix[j * ld + i] * input[j]);
		}
		output[i] = sum;
	}
}

int main(int argc, char *argv[])
{
	int		N, ld, i, j, fill, blocks, threads;
	int		*input_h, *output_h, *matrix_h;	/* Pointers for vectors on the host. */
	int		*input_d, *output_d, *matrix_d;	/* Pointer for vector on the device. */
	cudaError_t	err;

	if (argc != 3) {
		printf("Usage:\n");
		printf("04-matrix-vector <Size> <Threads per block>\n");
		exit(1);
	}

	N = atoi(argv[1]);
	threads = atoi(argv[2]);

	if (N < 1) {
		printf("Size should be at least 1.\n");
		exit(1);
	}

	if (N > 20000) {
		printf("Maximum size is 20000.\n");
		exit(1);
	}

	if (threads < 1) {
		printf("Number of threads per block must be at least 1.\n");
		exit(1);
	}

	if (threads > 1024) {
		printf("Number of threads per block must be at most 1024.\n");
		exit(1);
	}

	ld = (((N - 1) / PADDING) + 1 ) * PADDING;

	/*
	 * Allocate memory for matrix, input and output vector on host.
	 */

	matrix_h = (int *)malloc(N * ld * sizeof(int));
	if (matrix_h == NULL) {
		printf("Could not allocate memory for matrix on host.\n");
		exit(1);
	}

	input_h = (int *)malloc(N * sizeof(int));
	if (input_h == NULL) {
		printf("Could not allocate memory for input vector on host.\n");
		exit(1);
	}

	output_h = (int *)malloc(N * sizeof(int));
	if (output_h == NULL) {
		printf("Could not allocate memory for output vector on host.\n");
		exit(1);
	}

	/*
	 * Initialize input and output vector on host.
	 * Notice that all elements of the output vector are initialized to zero.
	 */
	for (i = 0; i < N; i++) {
		input_h[i] = 1;
		output_h[i] = 0;
	}

	/*
	 * Initialize matrix on host.
	 */
	for (i = 0; i < N; i++) {
		fill = i % 100;
		for (j = 0; j < N; j++) {
			matrix_h[j * ld + i] = fill;
		}
	}

	/*
	 * Here you will have to:
	 *  - Allocate memory for the input and output vectors on the device.
	 *  - Allocate memory for a matrix on the device (use padding!).
	 *  - Copy the input vector from the host to the device.
	 *  - Copy the input matrix from the host to the device.
	 *  - Call the kernel you will write to multiply the matrix with the vector.
	 *  - Copy the output vector from the device to the output vector on the host.
	 */
	err = cudaMalloc(&input_d, N * sizeof(int));
	if (err != cudaSuccess) {
		printf("Could not allocate memory for the input vector on the device.\n");
                exit(1);
        }

	err = cudaMalloc(&output_d, N * sizeof(int));
	if (err != cudaSuccess) {
		printf("Could not allocate memory for the output vector on the device.\n");
                exit(1);
        }

	err = cudaMalloc(&matrix_d, N * ld * sizeof(int));
	if (err != cudaSuccess) {
		printf("Could not allocate memory for matrix on the device.\n");
                exit(1);
        }

	err = cudaMemcpy(input_d, input_h, N * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
		printf("Could not copy input vector to device.\n");
		exit(1);
        }

	err = cudaMemcpy(matrix_d, matrix_h, N * ld * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
		printf("Could not copy input vector to device.\n");
		exit(1);
        }

	blocks = (N - 1) / threads + 1;
	multiply<<<blocks, threads>>>(matrix_d, input_d, output_d, N, ld);

	err = cudaMemcpy(output_h, output_d, N * sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
		printf("Could not copy vector from device to output vector on host.\n");
		exit(1);
        }

	/*
	 * We print a few results to make certain that everything went fine.
	 */

	printf("output_h[%4d] = %d\n", 0, output_h[0]);
	printf("output_h[%4d] = %d\n", (N - 1) / 2, output_h[(N - 1) / 2]);
	printf("output_h[%4d] = %d\n", N - 1, output_h[N - 1]);
}

