#include <stdio.h>
#include <stdlib.h>

__global__ void kernel(int *a_d) {
	*a_d += 1;
}

int main() {
        int a=0, *a_d;
        cudaMalloc((void**) &a_d, sizeof(int));
        cudaMemcpy(a_d, &a, sizeof(int), cudaMemcpyHostToDevice);
        kernel<<<1000,1000>>>(a_d);
        cudaMemcpy(&a, a_d, sizeof(int), cudaMemcpyDeviceToHost);
        printf("a = %d\n", a);
        cudaFree(a_d);
}
