/*
 *  === Αρχείο: cuda1.cu ===
 *
 *  Ονοματεπώνυμο: Αθανασίου Βασίλειος Ευάγγελος
 *  Αριθμός Μητρώου: 19390005
 *  Πρόγραμμα Σπουδών: ΠΑΔΑ
 *  
 *  Μεταγλώττιση: nvcc -o cuda1 cuda1.cu
 *  Εκτέλεση: ./cuda1 A.txt OutArr.txt
 * 
 */
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define N 8
#define nThreads 4
#define nBlocks (int)ceil((float)N/nThreads)

/*
 *  === Συνάρτηση Πυρήνα: calcAvg ===
 *  Παράμετροι: 
 *      - d_A: Πίνακας εισόδου (Device).
 *      - d_sum: Το άθροισμα των στοιχείων του πίνακα (Device).
 *      - d_avg: Ο μέσος όρος των στοιχείων του πίνακα (Device).
 *  Επιστρέφει: Τίποτα.
 * 
 *  Περιγραφή:
 *      Υπολογίζει το άθροισμα και τον μέσο όρο όλων των στοιχείων του πίνακα A
 *      με χρήση reduction και atomic εντολών.
 */
__global__ void calcAvg(int *d_A, int *d_sum, float *d_avg) 
{
    __shared__ int cache[nThreads * N + nThreads];  // Χρήση shared memory για γρήγορη προσπέλαση από threads ίδιου block

    // Υπολογισμός global Id κάθε thread στην x και στην y διάσταση
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    int totalElements = N * N;
    int tid = i * N + j; // Global index για τον 2D πίνακα

    if (i < N && j < N)  // Διασφάλιση ότι βρισκόμαστε εντός ορίων
        cache[threadIdx.x + threadIdx.y * blockDim.x] = d_A[tid];
    else
        cache[threadIdx.x + threadIdx.y * blockDim.x] = 0;  // Αποφυγή ανάγνωσης εκτός ορίων

    __syncthreads();  // Συγχρονισμός νημάτων στο block

    // Εκτέλεση παράλληλου reduction με χρήση δεντρικού αλγορίθμου εντός του block
    int k = blockDim.x * blockDim.y / 2; 
    while (k != 0) 
    {
        if (threadIdx.x + threadIdx.y * blockDim.x < k)  // Μόνο threads με έγκυρα Id εκτελούν reduction
            cache[threadIdx.x + threadIdx.y * blockDim.x] += 
                cache[threadIdx.x + threadIdx.y * blockDim.x + k];
        __syncthreads();  // Συγχρονισμός νημάτων στο block
        k /= 2; 
    }

    // Χρήση atomic λειτουργίας πρόσθεσης στο συνολικό άθροισμα εάν το νήμα είναι το πρώτο στο block
    if (threadIdx.x == 0 && threadIdx.y == 0) 
        atomicAdd(d_sum, cache[0]);

    // Υπολογισμός του μέσου όρου μετά το reduction
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
        *d_avg = (float)(*d_sum) / totalElements;
}


/*
 *  === Συνάρτηση Πυρήνα: findMax ===
 *  Παράμετροι: 
 *      - d_A: Πίνακας εισόδου (Device).
 *      - d_amax: Το μέγιστο στοιχείο του πίνακα A (Device).
 *  Επιστρέφει: Τίποτα.
 * 
 *  Περιγραφή:
 *      Υπολογίζει το μέγιστο στοιχείο του πίνακα A χρησιμοποιώντας reduction και atomic εντολές.
 */
__global__ void findMax(int *d_A, int *d_amax)
{
    __shared__ int cache[nThreads * N + nThreads];  // Χρήση shared memory για γρήγορη προσπέλαση από threads ίδιου block

    // Υπολογισμός global Id κάθε thread στην x και στην y διάσταση
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    int tid = i * N + j; // Global index για τον 2D πίνακα

    if (i < N && j < N)  // Διασφάλιση ότι βρισκόμαστε εντός ορίων
        cache[threadIdx.x + threadIdx.y * blockDim.x] = d_A[tid];
    else
        cache[threadIdx.x + threadIdx.y * blockDim.x] = 0;  // Αποφυγή ανάγνωσης εκτός ορίων

    __syncthreads();  // Συγχρονισμός νημάτων στο block

    // Εκτέλεση παράλληλου reduction με χρήση δεντρικού αλγορίθμου εντός του block
    int k = blockDim.x * blockDim.y / 2; 
    while (k != 0) 
    {
        if (threadIdx.x + threadIdx.y * blockDim.x < k)  // Μόνο threads με έγκυρα Id εκτελούν reduction
            cache[threadIdx.x + threadIdx.y * blockDim.x] = 
                cache[threadIdx.x + threadIdx.y * blockDim.x] > cache[threadIdx.x + threadIdx.y * blockDim.x + k] ?
                cache[threadIdx.x + threadIdx.y * blockDim.x] : cache[threadIdx.x + threadIdx.y * blockDim.x + k];
        __syncthreads();  // Συγχρονισμός νημάτων στο block
        k /= 2;
    }

    // Χρήση atomic λειτουργίας εύρεσης μέγιστου στοιχείου στο συνολικό μέγιστο αν αυτό το νήμα είναι το πρώτο στο block
    if (threadIdx.x == 0 && threadIdx.y == 0) 
        atomicMax(d_amax, cache[0]);
}


// Custom atomicMin για αριθμούς κινητής υποδιαστολής με χρήση atomicCAS
__device__ void atomicMin(float *address, float val)
{
    int *address_as_i = (int *) address;  // Μετατροπή της διεύθυνσης σε ακέραιο δείκτη
    int old = *address_as_i, assumed;

    // Μετατροπή της τιμής κινητής υποδιαστολής σε ακέραια αναπαράσταση
    int val_as_int = __float_as_int(val);

    do
    {
        assumed = old;
        // Εκτέλεση atomicCAS στη δεκαδική αναπαράσταση
        old = atomicCAS(address_as_i, assumed, min(val_as_int, assumed));
    } 
    while (assumed != old);  // Επανάληψη μέχρι να μην αλλάξει η τιμή
}


/*
 *  === Συνάρτηση: createB ===
 *  Παράμετροι: 
 *      - d_A: Πίνακας εισόδου (Device).
 *      - d_outArr: Πίνακας εξόδου B (Device).
 *      - d_bmin: Το ελάχιστο στοιχείο του πίνακα B (Device).
 *      - d_amax: Το μέγιστο στοιχείο του πίνακα A (Device).
 *      - d_avg: Ο μέσος όρος των στοιχείων του πίνακα A (Device).
 *  Επιστρέφει: Τίποτα.
 * 
 *  Περιγραφή:
 *      Υπολογίζει τον πίνακα B με στοιχεία  Bij = (m - Aij) / amax, όπου m είναι ο μέσος όρος των στοιχείων του Α 
 *      και βρίσκει το ελάχιστο στοιχείο του με χρήση reduction και atomic εντολών.
 */
__global__ void createB(int *d_A, float *d_outArr, float *d_bmin, int *d_amax, float *d_avg)
{
    __shared__ float cache[nThreads * N + nThreads];  // Χρήση shared memory για γρήγορη προσπέλαση από threads ίδιου block

    // Υπολογισμός global Id κάθε thread στην x και στην y διάσταση
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Υπολογισμός της αντίστοιχης τιμής για το Bij
    if (i < N && j < N)
    {
        if (*d_amax != 0)
            d_outArr[i * N + j] = (*d_avg - (float) d_A[i * N + j]) / (float) *d_amax;
        else
            d_outArr[i * N + j] = 0.0; // Διαχείριση της διαίρεσης με το μηδέν
    }

    int tid = i * N + j; // Global index για τον 2D πίνακα

    // Φόρτωση δεδομένων στην κοινή μνήμη (cache)
    if (i < N && j < N)  
        cache[threadIdx.x + threadIdx.y * blockDim.x] = d_outArr[tid];
    else
        cache[threadIdx.x + threadIdx.y * blockDim.x] = 10000000000000.0;  // Χρήση μεγάλης τιμής ως προσωρινή τιμή για νήματα εκτός ορίων

    __syncthreads();  // Συγχρονισμός νημάτων στο block

    // Εκτέλεση παράλληλου reduction με χρήση δεντρικού αλγορίθμου εντός του block
    int k = blockDim.x * blockDim.y / 2;  // Μείωση κατά το μισό των συνολικών νημάτων
    while (k != 0) 
    {
        if (threadIdx.x + threadIdx.y * blockDim.x < k) 
        {
            cache[threadIdx.x + threadIdx.y * blockDim.x] = 
                cache[threadIdx.x + threadIdx.y * blockDim.x] < cache[threadIdx.x + threadIdx.y * blockDim.x + k] ?
                cache[threadIdx.x + threadIdx.y * blockDim.x] : cache[threadIdx.x + threadIdx.y * blockDim.x + k];
        }
        __syncthreads();  // Συγχρονισμός νημάτων
        k /= 2;  // Μείωση του βήματος κατά το μισό σε κάθε επανάληψη
    }

    // Χρήση atomic λειτουργίας για την ελάχιστη τιμή του πίνακα B χρησιμοποιώντας την custom atomicMin
    if (threadIdx.x == 0 && threadIdx.y == 0) 
        atomicMin(d_bmin, cache[0]);  // Χρήση της custom atomicMin για αριθμούς κινητής υποδιαστολής
}

/*
 *  === Συνάρτηση: createC ===
 *  Παράμετροι: 
 *      - d_A: Πίνακας εισόδου (Device).
 *      - d_outArr: Πίνακας εξόδου C (Device).
 *  Επιστρέφει: Τίποτα.
 * 
 *  Περιγραφή:
 *      Υπολογίζει τον πίνακα C με στοιχεία 
 *      Cij = {Aij + Ai(j+1) + Ai(j-1)} / 3, λαμβάνοντας υπόψη
 *      εάν j+1=N τότε Ai(j+1)=Ai0, ενώ εάν j-1=-1 τότε Ai(j-1)=Ai(Ν-1).
 */
__global__ void createC(int *d_A, float *d_outArr)
{
    // Υπολογισμός global Id κάθε thread στην x και στην y διάσταση
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    int left, right, current;

    // Έλεγχος αν βρισκόμαστε εντός ορίων
    if (i < N && j < N)
    {
        left = (j - 1 != -1) ? d_A[i * N + (j - 1)] : d_A[i * N + (N - 1)];  // Αριστερό στοιχείο(διαχείριση ορίων)
        right = (j + 1 != N) ? d_A[i * N + (j + 1)] : d_A[i * N + 0];  // Δεξιό στοιχείο (διαχείριση ορίων)
        current = d_A[i * N + j];  // Τρέχον στοιχείο
        
        // Υπολογισμός του Cij
        d_outArr[i * N + j] = (float) (current + left + right) / 3.0;
    }
}


void create2DArray(int *Array);

int main(int argc, char *argv[])
{
    int *h_A;                                                        // [Host] Ο πίνακας Α 
    int *h_amax, *h_sum;                                             // [Host] Το μέγιστο στοιχείο του Α και το άθροισμα των στοιχείων
    int *d_A, *d_amax, *d_sum;                                       // [Device] Ο πίνακας Α, το μέγιστο στοιχείο και το άθροισμα των στοιχείων
    float *h_OutArr;                                                 // [Host] Ο πίνακας B ή C
    float *h_avg;                                                    // [Host] Ο μέσος όρος των στοιχείων του Α
    float *d_OutArr, *d_avg;                                         // [Device] Ο πίνακας B ή C και ο μέσος όρος
    float *h_bmin, *d_bmin;                                          // [Host] Το ελάχιστο στοιχείο του B και [Device] το ελάχιστο στοιχείο
    int n, threadsPerBlock, blocksPerGrid;                           // Το μέγεθος του πίνακα, τα νήματα ανά μπλοκ και τα μπλοκ ανά πλέγμα
    int intBytes, floatBytes;                                        // Το μέγεθος των πινάκων σε bytes
    int max_threads, max_block_dimX, max_block_dimY, max_block_dimZ; // Οι μέγιστες τιμές για τα νήματα και τις διαστάσεις των μπλοκ
    int max_grid_dimX, max_grid_dimY, max_grid_dimZ;                 // Οι μέγιστες τιμές για τις διαστάσεις των πλεγμάτων
    int i, j;                                                        // Δείκτες επανάληψης
    FILE *fpA, *fpOutArr;                                            // Δείκτες αρχείων για την αποθήκευση των πινάκων Α και Β ή C
    char arr;                                                        // Το όνομα του πίνακα Β ή C
    float elapsedTime1, elapsedTime2, elapsedTime3;                  // Ο χρόνος εκτέλεσης των kernels
    cudaEvent_t start, stop;                                         // Τα σημεία αναφοράς του χρόνου εκτέλεσης
    cudaError_t err;                                                 // Κωδικός σφάλματος CUDA
    cudaDeviceProp prop;                                             // Τα χαρακτηριστικά της συσκευής

    // Έλεγχος αριθμού παραμέτρων γραμμής εντολών
    if (argc != 3)
    {
        printf("Usage: %s A.txt OutArr.txt\n", argv[0]);
        exit(1);
    }

    // Αρχικοποίηση παραμέτρων
    n = N;
    threadsPerBlock = nThreads;
    blocksPerGrid = nBlocks;

    // Λήψη ιδιοτήτων συσκευής CUDA
    err = cudaGetDeviceProperties(&prop, 0); 
    if (err != cudaSuccess) { printf("CUDA Error --> cudaGetDeviceProperties failed.\n"); exit(1); }

    // Ανάθεση μέγιστων τιμών από τις ιδιότητες της συσκευής
    max_threads = prop.maxThreadsPerBlock;
    max_block_dimX = prop.maxThreadsDim[0];
    max_block_dimY = prop.maxThreadsDim[1];
    max_block_dimZ = prop.maxThreadsDim[2];
    max_grid_dimX = prop.maxGridSize[0];
    max_grid_dimY = prop.maxGridSize[1];
    max_grid_dimZ = prop.maxGridSize[2];

    // Εκτύπωση χαρακτηριστικών συσκευής
    printf("--------------- Device Properties ---------------\n");
    printf("Device name           : %s\n", prop.name);
    printf("Max threads per block : %d\n", max_threads);
    printf("Max block dimensions  : %d x %d x %d\n", max_block_dimX, max_block_dimY, max_block_dimZ);
    printf("Max grid dimensions   : %d x %d x %d\n", max_grid_dimX, max_grid_dimY, max_grid_dimZ);
    printf("-------------------------------------------------\n");

    // Έλεγχος έγκυρων τιμών για τις παραμέτρους
    if (n < 1)
    { printf("Error --> Matrix size must be at least 1\n"); exit(1); }
    if (threadsPerBlock < 1)
    { printf("Error --> Threads per block (block size) must be at least 1\n"); exit(1); }
    if (blocksPerGrid < 1)
    { printf("Error --> Blocks per grid (grid size) must be at least 1\n"); exit(1); }
    if (threadsPerBlock > max_threads)
    { printf("Error --> Threads per block (block size) exceed maximum allowed for %s\n", prop.name); exit(1); }
    if (blocksPerGrid > max_grid_dimX)
    { printf("Error --> Blocks per grid (grid size) exceed maximum allowed for %s\n", prop.name); exit(1); }

    // Άνοιγμα αρχείων για αποθήκευση των πινάκων
    fpA = fopen(argv[1], "w");
    if (fpA == NULL) { printf("Cannot open file %s\n", argv[1]); exit(1); }
    fpOutArr = fopen(argv[2], "w");
    if (fpOutArr == NULL) { printf("Cannot open file %s\n", argv[2]); exit(1); }

    // Εκτύπωση παραμέτρων εισόδου
    printf("--------------- Input Parameters ---------------\n");
    printf("Matrix size        : %d x %d\n", n, n);
    printf("Blocks per Grid    : %d\n", blocksPerGrid);
    printf("Threads per Block  : %d\n", threadsPerBlock);
    printf("------------------------------------------------\n");

    // Υπολογισμός μεγέθους πινάκων σε bytes
    intBytes = n * n * sizeof(int);
    floatBytes = n * n * sizeof(float);

    // Δέσμευση μνήμης για τους πίνακες στη μνήμη του Host
    h_A = (int *) malloc(intBytes);
    if (h_A == NULL) { printf("Error --> Memory allocation failed for A.\n"); exit(1); }
    h_OutArr = (float *) malloc(floatBytes);
    if (h_OutArr == NULL) { printf("Error --> Memory allocation failed for OutArr.\n"); exit(1); }
    h_avg = (float *) malloc(sizeof(float));
    if (h_avg == NULL) { printf("Error --> Memory allocation failed for avg.\n"); exit(1); }
    h_amax = (int *) malloc(sizeof(int));
    if (h_amax == NULL) { printf("Error --> Memory allocation failed for max.\n"); exit(1); }
    h_bmin = (float *) malloc(sizeof(float));
    if (h_bmin == NULL) { printf("Error --> Memory allocation failed for min.\n"); exit(1); }
    h_sum = (int *) malloc(sizeof(int));
    if (h_sum == NULL) { printf("Error --> Memory allocation failed for sum.\n"); exit(1); }

    // Αρχικοποίηση μεταβλητών
    *h_sum = 0;
    *h_avg = 0.0;
    *h_amax = 0;
    *h_bmin = 0.0;

    srand(time(NULL));

    // Δημιουργία πίνακα Α με πρόγραμμα γεννήτριας τυχαίων αριθμών
    create2DArray(h_A);
    printf("The array A has been stored in file %s\n", argv[1]);


// ============== Έναρξη Παράλληλου Υπολογισμού ==============

    // Δημιουργία CUDA events για μέτρηση χρόνου
    err = cudaEventCreate(&start);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventCreate(&start) failed.\n"); exit(1); }
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventCreate(&stop) failed.\n"); exit(1); }

    // Δέσμευση μνήμης στη συσκευή (Device)
    err = cudaMalloc((void **) &d_A, intBytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_A, bytes) failed."); exit(1); }
    err = cudaMalloc((void **) &d_OutArr, floatBytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_OutArray, bytes) failed."); exit(1); }
    err = cudaMalloc((void **) &d_sum, sizeof(int));
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_sum, sizeof(int)) failed."); exit(1); }
    err = cudaMalloc((void **) &d_avg, sizeof(float));
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_avg, sizeof(float)) failed."); exit(1); }
    err = cudaMalloc((void **) &d_amax, sizeof(int));
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_amax, sizeof(int)) failed."); exit(1); }
    err = cudaMalloc((void **) &d_bmin, sizeof(float)); 
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_bmin, sizeof(int)) failed."); exit(1); }

    // Δημιουργία 2D grid με 2D blocks
    dim3 dimBlock(nThreads, nThreads);
    dim3 dimGrid(N/nBlocks, N/nBlocks);

/* 
 * === Εκτέλεση Kernel: calcAvg<<<dimGrid, dimBlock>>> ===
 * Σκοπός:
 *   - Υπολογίζει το άθροισμα όλων των στοιχείων του πίνακα d_A και υπολογίζει τον μέσο όρο τους.
 *
 * Διαμόρφωση Πλέγματος και Μπλοκ:
 *   - dimGrid  : Αντιπροσωπεύει τον αριθμό των μπλοκ στο πλέγμα (X: nBlocks, Y: nBlocks, Z: 1).
 *   - dimBlock : Αντιπροσωπεύει τον αριθμό νημάτων σε κάθε μπλοκ (X: nThreads, Y: nThreads, Z: 1).
 * 
 * Μεταφορές Μνήμης:
 *   - cudaMemcpy(d_A, h_A, intBytes, cudaMemcpyHostToDevice) μεταφέρει τον πίνακα εισόδου από τη μνήμη του host στη μνήμη της συσκευής.
 *   - cudaMemcpy(d_sum, h_sum, sizeof(int), cudaMemcpyHostToDevice) μεταφέρει τη μεταβλητή αρχικοποίησης του αθροίσματος στη συσκευή.
 *   - cudaMemcpy(h_avg, d_avg, sizeof(float), cudaMemcpyDeviceToHost) ανακτά τον υπολογισμένο μέσο όρο από τη συσκευή στη μνήμη του host.
 *
 * Μέτρηση απόδοσης:
 *   - Χρησιμοποιεί cudaEvent δομή για τη μέτρηση του χρόνου εκτέλεσης του kernel.
 *
 * Διαχείριση Σφαλμάτων:
 *   - Κάθε κλήση της CUDA ρουτίνας ακολουθείται από έλεγχο σφάλματος (αν επιστρέφεται cudaSuccess).
 */
    err = cudaMemcpy(d_A, h_A, intBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice) failed."); exit(1); }

    err = cudaEventRecord(start,0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(start,0) failed."); exit(1); }

    calcAvg<<<dimGrid, dimBlock>>>(d_A, d_sum, d_avg);
    
    err = cudaEventRecord(stop,0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(stop,0) failed."); exit(1); }

    err = cudaMemcpy(h_avg, d_avg, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(h_avg, d_avg, sizeof(float), cudaMemcpyDeviceToHost) failed."); exit(1); }

    printf("Average: %4.2f\n", *h_avg);

    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventSynchronize(stop) failed."); exit(1); }
    err = cudaEventElapsedTime(&elapsedTime1, start, stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventElapsedTime(&elapsedTime1, start, stop) failed."); exit(1); }
    
    printf ("Time for the kernel calcAvg<<<>>>(): %f ms\n", elapsedTime1);

/* 
 * === Εκτέλεση Kernel: findMax<<<dimGrid, dimBlock>>> ===
 * Σκοπός:
 *   - Υπολογίζει το μέγιστο στοιχείο του πίνακα d_A και το αποθηκεύει στη μεταβλητή d_amax.
 *
 * Διαμόρφωση Πλέγματος και Μπλοκ:
 *   - dimGrid  : Αντιπροσωπεύει τον αριθμό των μπλοκ στο πλέγμα (X: nBlocks, Y: nBlocks, Z: 1).
 *   - dimBlock : Αντιπροσωπεύει τον αριθμό νημάτων σε κάθε μπλοκ (X: nThreads, Y: nThreads, Z: 1).
 * 
 * Μεταφορές Μνήμης:
 *   - cudaMemcpy(d_A, h_A, intBytes, cudaMemcpyHostToDevice) (προηγούμενη κλήση): Μεταφέρει τον πίνακα εισόδου στη μνήμη της συσκευής.
 *   - cudaMemcpy(h_amax, d_amax, sizeof(int), cudaMemcpyDeviceToHost): Ανακτά το μέγιστο στοιχείο από τη συσκευή στη μνήμη του host.
 *
 * Μέτρηση απόδοσης:
 *   - Χρησιμοποιεί cudaEvent δομή για τη μέτρηση του χρόνου εκτέλεσης του kernel.
 *
 * Διαχείριση Σφαλμάτων:
 *   - Κάθε κλήση της CUDA ρουτίνας ακολουθείται από έλεγχο σφάλματος (αν επιστρέφεται cudaSuccess).
 */
    err = cudaEventRecord(start, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(start,0) failed."); exit(1); }
    
    findMax<<<dimGrid, dimBlock>>>(d_A, d_amax);
    
    err = cudaEventRecord(stop, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(stop,0) failed."); exit(1); }

    err = cudaMemcpy(h_amax, d_amax, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(&h_amax, d_amax, sizeof(int), cudaMemcpyDeviceToHost) failed."); exit(1); }

    printf("Max: %d\n", *h_amax);

    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventSynchronize(stop) failed."); exit(1); }
    err = cudaEventElapsedTime(&elapsedTime2, start, stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventElapsedTime(&elapsedTime2, start, stop) failed."); exit(1); }

    printf ("Time for the kernel findMax<<<>>>(): %f ms\n", elapsedTime2);

/* 
 * === Εκτέλεση Kernel: createB ή createC ===
 * Σκοπός:
 *   - Αν η συνθήκη (amax > N * m) ικανοποιείται:
 *       - Εκτελείται η συνάρτηση kernel createB για τον υπολογισμό του πίνακα B με τιμές
 *         Bij = (m - Aij) / amax.
 *       - Βρίσκεται επίσης το ελάχιστο στοιχείο του πίνακα B.
 *   - Αν η συνθήκη δεν ικανοποιείται:
 *       - Εκτελείται η συνάρτηση kernel createC για τον υπολογισμό του πίνακα C με τιμές
 *         Cij = {Aij + Ai(j+1) + Ai(j-1)} / 3.
 *
 * Διαμόρφωση Πλέγματος και Μπλοκ:
 *   - dimGrid  : Αντιπροσωπεύει τον αριθμό των μπλοκ στο πλέγμα (X: nBlocks, Y: nBlocks, Z: 1).
 *   - dimBlock : Αντιπροσωπεύει τον αριθμό νημάτων σε κάθε μπλοκ (X: nThreads, Y: nThreads, Z: 1).
 * 
 * Μεταφορές Μνήμης:
 *   - cudaMemcpy(h_OutArr, d_OutArr, floatBytes, cudaMemcpyDeviceToHost): Ανακτά τον πίνακα εξόδου (B ή C) από τη μνήμη της συσκευής στη μνήμη του host.
 *   - Αν εκτελείται η συνάρτηση kernel createB:
 *       - cudaMemcpy(h_bmin, d_bmin, sizeof(float), cudaMemcpyDeviceToHost): Ανακτά το ελάχιστο στοιχείο του πίνακα B.
 *
 * Μέτρηση απόδοσης:
 *   - Χρησιμοποιεί cudaEvent δομή για τη μέτρηση του χρόνου εκτέλεσης του kernel.
 *
 * Διαχείριση Σφαλμάτων:
 *   - Κάθε κλήση της CUDA ρουτίνας ακολουθείται από έλεγχο σφάλματος (αν επιστρέφεται cudaSuccess).
 */
    err = cudaEventRecord(start, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(start, 0) failed."); exit(1); }

    if (*h_amax > N * (*h_avg))
    {
        arr = 'B';

        createB<<<dimGrid, dimBlock>>>(d_A, d_OutArr, d_bmin, d_amax, d_avg);
        
        err = cudaEventRecord(stop, 0);
        if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(stop,0) failed."); exit(1); }

        err = cudaMemcpy(h_OutArr, d_OutArr, floatBytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(&h_OutArr, d_OutArr, doubleBytes, cudaMemcpyDeviceToHost) failed."); exit(1); }
        err = cudaMemcpy(h_bmin, d_bmin, sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(&h_bmin, d_bmin, sizeof(float), cudaMemcpyDeviceToHost) failed."); exit(1); }

        printf("The array %c has been stored in file %s\n", arr,  argv[2]);
        printf("Min: %4.4f\n", *h_bmin);
    }
    else
    {
        arr = 'C';

        createC<<<dimGrid, dimBlock>>>(d_A, d_OutArr);
        
        err = cudaEventRecord(stop, 0);
        if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(stop,0) failed."); exit(1); }

        err = cudaMemcpy(h_OutArr, d_OutArr, floatBytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(&h_OutArr, d_OutArr, doubleBytes, cudaMemcpyDeviceToHost) failed."); exit(1); }

        printf("The array %c has been stored in file %s\n", arr,  argv[2]);
    }

    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventSynchronize(stop) failed."); exit(1); }
    err = cudaEventElapsedTime(&elapsedTime3, start, stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventElapsedTime(&elapsedTime3, start, stop) failed."); exit(1); }

    printf ("Time for the kernel create%c<<<>>>(): %f ms\n", arr, elapsedTime3);

// ============== Λήξη Παράλληλου Υπολογισμού ==============

    fprintf(fpOutArr, "Array %c\n", arr);

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            fprintf(fpA, "%4d ", h_A[i * n + j]);
            fprintf(fpOutArr, "%4.4f ", h_OutArr[i * n + j]);
        }

        fprintf(fpA, "\n");
        fprintf(fpOutArr, "\n");
    }

    fclose(fpA);
    fclose(fpOutArr);

    err = cudaEventDestroy(start);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventDestroy(start) failed."); exit(1); }
    err = cudaEventDestroy(stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventDestroy(stop) failed."); exit(1); }

    free(h_A);
    free(h_OutArr);
    free(h_avg);
    free(h_amax);
    free(h_bmin);
    free(h_sum);

    err = cudaFree(d_A);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaFree(d_A) failed."); exit(1); }
    err = cudaFree(d_OutArr);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaFree(d_OutArr) failed."); exit(1); }
    err = cudaFree(d_avg);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaFree(d_avg) failed."); exit(1); }
    err = cudaFree(d_amax);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaFree(d_amax) failed."); exit(1); }
    err = cudaFree(d_bmin);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaFree(d_bmin) failed."); exit(1); }
    err = cudaFree(d_sum);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaFree(d_sum) failed."); exit(1); }

    return 0;
}

/*
 *  === Συνάρτηση: create2DArray ===
 *  Παράμετροι: 
 *      - Array: Δείκτης σε πίνακα μονοδιάστατο (μεταχείριση ως δισδιάστατου).
 *  Επιστρέφει: Τίποτα.
 * 
 *  Περιγραφή:
 *      Δημιουργεί έναν τυχαίο πίνακα διαστάσεων NxN με τιμές στο διάστημα [1, 100].
 *      Εξασφαλίζει ότι το μέγιστο στοιχείο του πίνακα είναι μεγαλύτερο ή μικρότερο από το Ν επί τον 
 *      μέσο όρο του πίνακα.
 */
void create2DArray(int *Array)
{
    int sum = 0;  // Άθροισμα των στοιχείων του πίνακα
    int amax = 0; // Μέγιστη τιμή στον πίνακα
    int i, j, m;

    // Γέμισμα του πίνακα με τυχαίες τιμές και υπολογισμός του αθροίσματος και της μέγιστης τιμής
    for (i = 0; i < N; ++i) 
    {
        for (j = 0; j < N; ++j) 
        {
            Array[i * N + j] = rand() % 100 + 1; // Τυχαία τιμή στο διάστημα [1, 100]
            sum += Array[i * N + j]; // Προσθήκη στο συνολικό άθροισμα
            if (Array[i * N + j] > amax) 
            {
                amax = Array[i * N + j]; // Ενημέρωση της μέγιστης τιμής
            }
        }
    }

    m = sum / (N * N); // Υπολογισμός του μέσου όρου
    while (amax <= N * m) // Επαλήθευση ότι η μέγιστη τιμή είναι μεγαλύτερη από N * m
    {
        i = rand() % N; // Τυχαία επιλογή γραμμής
        j = rand() % N; // Τυχαία επιλογή στήλης
        Array[i * N + j] += (N * m - amax + 1); // Αύξηση του στοιχείου ώστε να ικανοποιηθεί η συνθήκη
        amax = Array[i * N + j]; // Ενημέρωση της μέγιστης τιμής
    }
}