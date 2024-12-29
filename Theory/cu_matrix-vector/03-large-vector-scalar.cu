#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

/*
 * Here you will have to create a kernel that will multiply each element
 * of a vector with a scalar number. The latter will be passed as a parameter
 * to the kernel.
 *
 * When writing the kernel, assume that the size of the vector might be > 1024.
 */
__global__ void multiply(int *vector, int num, int N)
{
	int	i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N) {
		vector[i] *= num;
	}
}

int main(int argc, char *argv[])
{
	int		N, num, i, blocks, threads;
	int		*input_h, *output_h;	/* Pointers for vectors on the host. */
	int		*vector_d;		/* Pointer for vector on the device. */
	cudaError_t	err;

	if (argc != 4) {
		printf("Usage:\n");
		printf("03-large-vector-scalar <Size of vectors> <Scalar> <Threads per block>\n");
		exit(1);
	}

	N = atoi(argv[1]);
	num = atoi(argv[2]);
	threads = atoi(argv[3]);

	if (N < 1) {
		printf("Size of vectors should be at least 1.\n");
		exit(1);
	}

	if (N > 1000000) {
		printf("Maximum size of vectors is 1000000.\n");
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

	/*
	 * Allocate memory for input and output vector on host.
	 */
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
	 * Here you will have to:
	 *  - Allocate memory for a vector on the device.
	 *  - Copy the input vector from the host to the device.
	 *  - Call the kernel you will write to multiply each element of the vector with the scalar 'num'.
	 *  - Copy the vector from the device to the output vector on the host.
	 */
	err = cudaMalloc(&vector_d, N * sizeof(int));
	if (err != cudaSuccess) {
		printf("Could not allocate memory for vector on the device.\n");
                exit(1);
        }

	err = cudaMemcpy(vector_d, input_h, N * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
		printf("Could not copy input vector to device.\n");
		exit(1);
        }

	blocks = (N - 1) / threads + 1;
	multiply<<<blocks, threads>>>(vector_d, num, N);

	err = cudaMemcpy(output_h, vector_d, N * sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
		printf("Could not copy vector from device to output vector on host.\n");
		exit(1);
        }

	/*
	 * If you completed the previous tasks correctly,
	 * all elements of the output vector should now contain 'num'.
	 * We print a few of them to make certain that everything went fine.
	 */

	printf("output_h[%4d] = %d\n", 0, output_h[0]);
	printf("output_h[%4d] = %d\n", (N - 1) / 2, output_h[(N - 1) / 2]);
	printf("output_h[%4d] = %d\n", N - 1, output_h[N - 1]);
}


