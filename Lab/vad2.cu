#include <stdio.h>

__global__ void add(int *a, int *b)
{
        int i = threadIdx.x;
        a=a+i;
        b=b+i;
        *a = *a + *b;
}


int main(void) {
        int *a, *b, size_ab;
        int *d_a, *d_b;
        int size = sizeof(int);
        int i;

        printf("Size of arrays a and b:");
        scanf("%d", &size_ab);
        a = (int*) malloc(size_ab * size);
        for(i=0; i<size_ab; i++)
        {
                printf("A Element %d=", i);
                scanf("%d", a + i);
        }
        b = (int*) malloc(size_ab * size);
        for(i=0; i<size_ab; i++)
        {
                printf("B Element %d=", i);
                scanf("%d", b + i);
        }

        cudaMalloc((void **)&d_a, size_ab * size); cudaMalloc((void **)&d_b, size_ab * size);
        cudaMemcpy(d_a, a, size_ab * size, cudaMemcpyHostToDevice); cudaMemcpy(d_b, b, size_ab * size, cudaMemcpyHostToDevice);
        add<<<1,size_ab>>>(d_a, d_b);
        cudaMemcpy(a, d_a, size_ab * size, cudaMemcpyDeviceToHost);

        for (i=0; i<size_ab; i++,a++) { printf("A[%d]=%d\n", i, *a); }
        cudaFree(d_a); cudaFree(d_b);
        return 0;
}