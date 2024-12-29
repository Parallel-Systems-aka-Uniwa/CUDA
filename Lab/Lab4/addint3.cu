#include <stdio.h>

__global__ void add(int *a, int *b, int *size_a)
{
        int i;
        for (i=0; i<*size_a; i++,a++)
        {
                *a = *a + *b;
        }
}


int main(void) {
        int *a, b, size_a;
        int *d_a, *d_b, *d_size_a;
        int size = sizeof(int);
        int i;

        printf("Size of array a:");
        scanf("%d", &size_a);
        a = (int*) malloc(size_a * size);
        for(i=0; i<size_a; i++) {
                printf("Element %d=", i);
                scanf("%d", a + i);
        }
        printf("Give integer b:");
        scanf("%d", &b);
        cudaMalloc((void **)&d_a, size_a * size);
        cudaMalloc((void **)&d_b, size);
        cudaMalloc((void **)&d_size_a, size);
        cudaMemcpy(d_a, a, size_a * size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_size_a, &size_a, size, cudaMemcpyHostToDevice);
        add<<<1,1>>>(d_a, d_b, d_size_a);
        cudaMemcpy(a, d_a, size_a * size, cudaMemcpyDeviceToHost);
        for (i=0; i<size_a; i++,a++) {
                printf("A[%d]=%d\n", i, *a);
        }
        cudaFree(d_a); cudaFree(d_b);
        return 0;
}