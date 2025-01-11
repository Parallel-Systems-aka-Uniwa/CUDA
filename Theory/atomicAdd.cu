/* Εύρεση του αθροίσματος ενός μεγάλου πλήθους αριθμών με χρήση δεντρικού
αλγόριθμου και χρήση της ατομικής εντολής atomicAdd() */
#include <stdio.h>
#include <cuda.h>

#define N 1000000
#define nThreads 1024
#define nBlocks (int)ceil((float)N/nThreads)

__global__ void add1( float *R, float *C) 
{
    __shared__ float smarray[nThreads];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int smindex = threadIdx.x;
    // Μεταφορά στην κοινή μνήμη των αριθμών που αντιστοιχούν στο μπλοκ
    smarray[smindex] = R[tid];
    // Αναμονή των νημάτων του μπλοκ μέχρι να ολοκληρωθεί η μεταφορά
    __syncthreads();
    // Εφαρμογή δεντρικού αλγόριθμου για τον υπολογισμό του αθροίσματος
    int i = blockDim.x/2;
    while (i != 0) 
    {
        if (smindex < i)
            smarray[smindex] += smarray[smindex + i];
        __syncthreads();
        i /= 2;
    }
    
    // Ενημέρωση του καθολικού αθροίσματος με χρήση ατομικής εντολής
    if (smindex == 0)
        atomicAdd(C, smarray[0]);
    }

int main(void) 
{
    float *R, *C;
    float *Rd, *Cd;
    
    // Εκχώρηση μνήμης στις δομές και μεταβλητές του host
    R = (float*)malloc( N*sizeof(float) );
    C = (float*)malloc( sizeof(float) );
    
    cudaEvent_t start,stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Εκχώρηση μνήμης στις δομές και μεταβλητές της συσκευής
    cudaMalloc( (void**)&Rd, N*sizeof(float) );
    cudaMalloc( (void**)&Cd, sizeof(float) );

    // Αρχικοποίηση του πίνακα εισόδου στον host
    for (int i=0; i<N; i++) 
        R[i] = 1.0;

    // Μεταφορά των δεδομένων του πίνακα εισόδου από τον host στη συσκευή
    cudaMemcpy(Rd, R, N*sizeof(float), cudaMemcpyHostToDevice );
    cudaEventRecord(start,0);
    
    add1<<<nBlocks,nThreads>>>(Rd, Cd);
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    
    // Μεταφορά του αποτελέσματος από τη συσκευή στον host
    cudaMemcpy(C, Cd, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Final sum is equal to %f \n", C[0]);
    cudaEventElapsedTime(&elapsedTime,start,stop);
    printf ("Time for the kernel: %f ms\n", elapsedTime);

    // Αποδέσμευση μνήμης στη συσκευή
    cudaFree(Rd );
    cudaFree(Cd);

    // Αποδέσμευση μνήμης στον host
    free(R);
    free(C);
}