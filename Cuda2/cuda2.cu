/*
 *  === Αρχείο: cuda2.cu ===
 *
 *  Ονοματεπώνυμο: Αθανασίου Βασίλειος Ευάγγελος
 *  Αριθμός Μητρώου: 19390005
 *  Πρόγραμμα Σπουδών: ΠΑΔΑ
 *  
 *  Μεταγλώττιση: nvcc -o cuda2 cuda2.cu
 *  Εκτέλεση: ./cuda2 A.txt A_means.txt A_submeans.txt AT_submeans.txt A_cov.txt
 * 
 */
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define N 10
#define nThreads 2
#define nBlocks (int)ceil((float)N/nThreads)

/*
 * === Συνάρτηση: calcColMeans ===
 * Παράμετροι: 
 *   - d_A: Ο πίνακας εισόδου (Device) με στοιχεία ακέραιου τύπου.
 *   - d_Amean: Ο πίνακας εξόδου (Device) που αποθηκεύει τον μέσο όρο κάθε στήλης.
 * Επιστρέφει: Τίποτα.
 * 
 * Περιγραφή:
 *   Η συνάρτηση αυτή υπολογίζει τον μέσο όρο κάθε στήλης του πίνακα d_A και αποθηκεύει
 *   τα αποτελέσματα στον πίνακα d_Amean. Κάθε νήμα είναι υπεύθυνο για μία στήλη του πίνακα.
 */
__global__ void calcColMeans(int *d_A, float *d_Amean) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Υπολογισμός του δείκτη της στήλης (global)
    int row; // Δείκτης γραμμής

    if (col >= N) return; // Εξασφαλίζουμε ότι δε ξεπερνάμε τα όρια του πίνακα

    float sum = 0.0f;

    // Υπολογισμός του αθροίσματος της στήλης
    for (row = 0; row < N; ++row) 
        sum += d_A[row * N + col];

    // Αποθήκευση του αποτελέσματος στον πίνακα εξόδου
    d_Amean[col] = sum / N; // Υπολογισμός του μέσου όρου
}


/*
 * === Συνάρτηση: subMeansT ===
 * Παράμετροι: 
 *   - d_A: Ο πίνακας εισόδου (Device) με ακέραια στοιχεία.
 *   - d_Amean: Ο πίνακας εισόδου (Device) που περιέχει τους μέσους όρους των στηλών του d_A.
 *   - d_Asubmeans: Ο πίνακας εξόδου (Device) που αποθηκεύει τα στοιχεία του d_A αφού αφαιρεθεί ο μέσος όρος.
 *   - d_ATsubmeans: Ο ανάστροφος πίνακας εξόδου (Device) που περιέχει τη μεταφορά του d_Asubmeans.
 * Επιστρέφει: Τίποτα.
 * 
 * Περιγραφή:
 *   Η συνάρτηση αυτή αφαιρεί τον μέσο όρο κάθε στήλης του πίνακα d_A από τα στοιχεία της
 *   και αποθηκεύει τα αποτελέσματα στον πίνακα d_Asubmeans. Επιπλέον, υπολογίζει και αποθηκεύει
 *   τη μεταφορά του πίνακα d_Asubmeans στον πίνακα `d_ATsubmeans`.
 */
__global__ void subMeansT(int *d_A, float *d_Amean, float *d_Asubmeans, float *d_ATsubmeans)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Υπολογισμός του δείκτη της γραμμής (global)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Υπολογισμός του δείκτη της στήλης (global)

    if (row < N && col < N) 
    {
        // Αφαίρεση του μέσου όρου από κάθε στοιχείο και αποθήκευση
        d_Asubmeans[row * N + col] = d_A[row * N + col] - d_Amean[col];
        // Υπολογισμός και αποθήκευση της μεταφοράς του πίνακα
        d_ATsubmeans[col * N + row] = d_Asubmeans[row * N + col];
    }
}

/*
 * === Συνάρτηση: calcCov ===
 * Παράμετροι: 
 *   - d_Asubmeans: Ο πίνακας εισόδου (Device) που περιέχει τα στοιχεία του πίνακα αφού αφαιρεθεί ο μέσος όρος.
 *   - d_Acov: Ο πίνακας εξόδου (Device) που περιέχει τον υπολογισμένο πίνακα συνδιακύμανσης.
 * Επιστρέφει: Τίποτα.
 * 
 * Περιγραφή:
 *   Η συνάρτηση αυτή υπολογίζει τον πίνακα συνδιακύμανσης για τα δεδομένα του d_Asubmeans.
 *   Υπολογίζονται μόνο τα στοιχεία του άνω τριγωνικού μέρους του πίνακα συνδιακύμανσης,
 *   καθώς ο πίνακας είναι συμμετρικός.
 */
__global__ void calcCov(float *d_Asubmeans, float *d_Acov)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Υπολογισμός δείκτη γραμμής στον πίνακα συνδιακύμανσης
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Υπολογισμός δείκτη στήλης στον πίνακα συνδιακύμανσης
    float sum; // Αθροιστής
    int k; // Δείκτης επανάληψης

    if (row < N && col < N && row <= col) // Υπολογισμός μόνο για το άνω τριγωνικό μέρος του πίνακα
    { 
        sum = 0.0f;

        // Υπολογισμός του εσωτερικού γινομένου για τη γραμμή `row` και τη στήλη `col`
        for (k = 0; k < N; ++k) 
            sum += d_Asubmeans[row * N + k] * d_Asubmeans[col * N + k];

        // Αποθήκευση του αποτελέσματος στον πίνακα συνδιακύμανσης
        d_Acov[row * N + col] = sum;
    }
}


int main(int argc, char *argv[])
{
    int *h_A;                                                        // [Host] Ο πίνακας εισόδου A
    float *h_Acov, *h_Amean, *h_Asubmeans, *h_ATsubmeans;            // [Host] Ο πίνακας συνδιακύμανσης, ο πίνακας μέσων όρων, ο πίνακας διαφορών και ο μεταθετικός πίνακας διαφορών
    int *d_A;                                                        // [Device] Ο πίνακας εισόδου A
    float *d_Acov, *d_Amean, *d_Asubmeans, *d_ATsubmeans;            // [Device] Ο πίνακας συνδιακύμανσης, ο πίνακας μέσων όρων, ο πίνακας διαφορών και ο μεταθετικός πίνακας διαφορών
    int n, threadsPerBlock, blocksPerGrid;                           // Το μέγεθος του πίνακα, τα νήματα ανά μπλοκ και τα μπλοκ ανά πλέγμα
    int intBytes, floatBytes;                                        // Το μέγεθος των πινάκων σε bytes για ακέραιους και κινητής υποδιαστολής
    int max_threads, max_block_dimX, max_block_dimY, max_block_dimZ; // Οι μέγιστες τιμές για τα νήματα και τις διαστάσεις των μπλοκ
    int max_grid_dimX, max_grid_dimY, max_grid_dimZ;                 // Οι μέγιστες τιμές για τις διαστάσεις των πλέγματων
    int i, j;                                                        // Δείκτες επανάληψης για εντολές for
    FILE *fpA, *fpAcov, *fpAsubmeans, *fpATsubmeans, *fpAmean;       // Δείκτες αρχείων για την αποθήκευση των πινάκων A, συνδιακύμανσης, διαφορών, μεταθετικού και μέσων όρων
    float elapsedTime1, elapsedTime2, elapsedTime3;                  // Οι χρόνοι εκτέλεσης για κάθε kernel
    cudaEvent_t start, stop;                                         // CUDA events για μέτρηση χρόνου εκτέλεσης
    cudaError_t err;                                                 // Κωδικός σφάλματος CUDA
    cudaDeviceProp prop;                                             // Τα χαρακτηριστικά της συσκευής CUDA

    // Έλεγχος αριθμού παραμέτρων γραμμής εντολών
    if (argc != 6)
    {
        printf("Usage: %s A.txt A_means.txt A_submeans.txt AT_submeans.txt A_cov.txt\n", argv[0]);
        exit(1);
    }

    // Αρχικοποίηση παραμέτρων
    n = N;
    threadsPerBlock = nThreads;
    blocksPerGrid = nBlocks;

    // Λήψη ιδιοτήτων συσκευής CUDA
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaGetDeviceProperties() failed.\n"); exit(1); } 

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
    fpAmean = fopen(argv[2], "w");
    if (fpAmean == NULL) { printf("Cannot open file %s\n", argv[2]); exit(1); }
    fpAsubmeans = fopen(argv[3], "w");
    if (fpAsubmeans == NULL) { printf("Cannot open file %s\n", argv[3]); exit(1); }
    fpATsubmeans = fopen(argv[4], "w");
    if (fpATsubmeans == NULL) { printf("Cannot open file %s\n", argv[4]); exit(1); }
    fpAcov = fopen(argv[5], "w");
    if (fpAcov == NULL) { printf("Cannot open file %s\n", argv[5]); exit(1); }
  
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
    h_Amean = (float *) malloc(n * sizeof(float));
    if (h_Amean == NULL) { printf("Error --> Memory allocation failed for A_mean.\n"); exit(1); }
    h_Asubmeans = (float *) malloc(floatBytes);
    if (h_Asubmeans == NULL) { printf("Error --> Memory allocation failed for A_submeans.\n"); exit(1); }
    h_ATsubmeans = (float *) malloc(floatBytes);
    if (h_ATsubmeans == NULL) { printf("Error --> Memory allocation failed for AT_submeans.\n"); exit(1); }
    h_Acov = (float *) malloc(floatBytes);
    if (h_Acov == NULL) { printf("Error --> Memory allocation failed for A_cov.\n"); exit(1); }

    srand(time(NULL));

    // Δημιουργία τυχαίου πίνακα Α και αρχικοποίηση των πινάκων με μηδενικές τιμές
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            h_A[i * n + j] = rand() % 199 - 99;                           // Τιμές στο διάστημα [-99, 99]
            h_A[i * n + j] = h_A[i * n + j] >= 0 ? h_A[i * n + j] + 10 : h_A[i * n + j] - 10;  // Τυχαία επιλογή προσήμου
            h_Asubmeans[i * n + j] = 0.0;
            h_ATsubmeans[i * n + j] = 0.0;
            h_Acov[i * n + j] = 0.0;
        }
        h_Amean[i] = 0.0;
    }
    printf("The array A has been stored in file %s\n", argv[1]);

// ============== Έναρξη Παράλληλου Υπολογισμού ============== 

    // Δημιουργία CUDA events για μέτρηση χρόνου
    err = cudaEventCreate(&start);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventCreate(&start) failed.\n"); exit(1); }
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventCreate(&stop) failed.\n"); exit(1); }

    // Δέσμευση μνήμης στη συσκευή
    err = cudaMalloc((void **) &d_A, intBytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_A, bytes) failed."); exit(1); }
    err = cudaMalloc((void **) &d_Amean, n * sizeof(float));
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_Amean, bytes) failed."); exit(1); }
    err = cudaMalloc((void **) &d_Asubmeans, floatBytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_Asubmeans, bytes) failed."); exit(1); }
    err = cudaMalloc((void **) &d_ATsubmeans, floatBytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_ATsubmeans, bytes) failed."); exit(1); }
    err = cudaMalloc((void **) &d_Acov, floatBytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_Acov, bytes) failed."); exit(1); }

    err = cudaMemcpy(d_A, h_A, intBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice) failed."); exit(1); }

    // Δημιουργία 2D grid με 2D block
    dim3 dimBlock(nThreads, nThreads);
    dim3 dimGrid(nBlocks, nBlocks);

/* 
 * === Εκτέλεση Kernel: calcColMeans<<<nBlocks, nThreads>>> ===
 * Σκοπός:
 *   - Υπολογίζει τον μέσο όρο κάθε στήλης του πίνακα d_A και αποθηκεύει τα αποτελέσματα στον πίνακα d_Amean.
 *
 * Διαμόρφωση Πλέγματος και Μπλοκ:
 *   - dimGrid  : Αντιπροσωπεύει τον αριθμό των μπλοκ στο πλέγμα (X: nBlocks, Y: 1, Z: 1).
 *   - dimBlock : Αντιπροσωπεύει τον αριθμό νημάτων σε κάθε μπλοκ (X: nThreads, Y: 1, Z: 1).
 * 
 * Μεταφορές Μνήμης:
 *   - cudaMemcpy(d_A, h_A, intBytes, cudaMemcpyHostToDevice) μεταφέρει τον πίνακα εισόδου από τη μνήμη του host στη μνήμη της συσκευής.
 *   - cudaMemcpy(h_Amean, d_Amean, n * sizeof(float), cudaMemcpyDeviceToHost) ανακτά τον υπολογισμένο πίνακα μέσων όρων από τη συσκευή στη μνήμη του host.
 *
 * Μέτρηση απόδοσης:
 *   - Χρησιμοποιεί cudaEvent δομή για τη μέτρηση του χρόνου εκτέλεσης του kernel.
 *
 * Διαχείριση Σφαλμάτων:
 *   - Κάθε κλήση της CUDA ρουτίνας ακολουθείται από έλεγχο σφάλματος (αν επιστρέφεται cudaSuccess).
 */
    err = cudaEventRecord(start, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(start, 0) failed."); exit(1); }

    calcColMeans<<<nBlocks, nThreads>>>(d_A, d_Amean);

    err = cudaEventRecord(stop, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(stop, 0) failed."); exit(1); }

    err = cudaMemcpy(h_Amean, d_Amean, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(h_Amean, d_Amean, bytes, cudaMemcpyDeviceToHost) failed."); exit(1); }

    printf("The array A_means has been stored in file %s\n", argv[2]);

    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventSynchronize(stop) failed."); exit(1); }
    err = cudaEventElapsedTime(&elapsedTime1, start, stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventElapsedTime(&elapsedTime1, start, stop) failed."); exit(1); }
    
    printf("Time for the kernel calcColMeans<<<>>>(): %f ms\n", elapsedTime1);

/* 
 * === Εκτέλεση Kernel: subMeansT<<<dimGrid, dimBlock>>> ===
 * Σκοπός:
 *   - Υπολογίζει τη διαφορά κάθε στοιχείου του πίνακα d_A από τον μέσο όρο της στήλης του 
 *     και αποθηκεύει τα αποτελέσματα στον πίνακα d_Asubmeans.
 *   - Δημιουργεί τον ανάστροφο πίνακα d_ATsubmeans βασισμένο στον πίνακα d_Asubmeans.
 *
 * Διαμόρφωση Πλέγματος και Μπλοκ:
 *   - dimGrid  : Αντιπροσωπεύει τον αριθμό των μπλοκ στο πλέγμα (X: nBlocks, Y: nBlocks, Z: 1).
 *   - dimBlock : Αντιπροσωπεύει τον αριθμό νημάτων σε κάθε μπλοκ (X: nThreads, Y: nThreads, Z: 1).
 * 
 * Μεταφορές Μνήμης:
 *   - cudaMemcpy(d_A, h_A, intBytes, cudaMemcpyHostToDevice) μεταφέρει τον πίνακα εισόδου από τη μνήμη του host στη μνήμη της συσκευής.
 *   - cudaMemcpy(h_Asubmeans, d_Asubmeans, floatBytes, cudaMemcpyDeviceToHost) ανακτά τον πίνακα d_Asubmeans από τη μνήμη της συσκευής στη μνήμη του host.
 *   - cudaMemcpy(h_ATsubmeans, d_ATsubmeans, floatBytes, cudaMemcpyDeviceToHost) ανακτά τον πίνακα d_ATsubmeans από τη μνήμη της συσκευής στη μνήμη του host.
 *
 * Μέτρηση απόδοσης:
 *   - Χρησιμοποιεί cudaEvent δομή για τη μέτρηση του χρόνου εκτέλεσης του kernel.
 *
 * Διαχείριση Σφαλμάτων:
 *   - Κάθε κλήση της CUDA ρουτίνας ακολουθείται από έλεγχο σφάλματος (αν επιστρέφεται cudaSuccess).
 */
    err = cudaEventRecord(start, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(start, 0) failed."); exit(1); }

    subMeansT<<<dimGrid, dimBlock>>>(d_A, d_Amean, d_Asubmeans, d_ATsubmeans);

    err = cudaEventRecord(stop, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(stop, 0) failed."); exit(1); }

    err = cudaMemcpy(h_Asubmeans, d_Asubmeans, floatBytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(h_Asubmeans, d_Asubmeans, bytes, cudaMemcpyDeviceToHost) failed."); exit(1); }
    err = cudaMemcpy(h_ATsubmeans, d_ATsubmeans, floatBytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(h_ATsubmeans, d_ATsubmeans, bytes, cudaMemcpyDeviceToHost) failed."); exit(1); }

    printf("The array A_submeans has been stored in file %s\n", argv[3]);
    printf("The array AT_submeans has been stored in file %s\n", argv[4]);

    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventSynchronize(stop) failed."); exit(1); }
    err = cudaEventElapsedTime(&elapsedTime2, start, stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventElapsedTime(&elapsedTime2, start, stop) failed."); exit(1); }
    
    printf("Time for the kernel subMeansT<<<>>>(): %f ms\n", elapsedTime2);

/* 
 * === Εκτέλεση Kernel: calcCov<<<dimGrid, dimBlock>>> ===
 * Σκοπός:
 *   - Υπολογίζει τον πίνακα συνδιακύμανσης d_Acov από τον πίνακα τιμών d_Asubmeans, 
 *     ο οποίος περιέχει τις τιμές του αρχικού πίνακα με αφαιρημένους τον μέσο όρο κάθε στήλης.
 *
 * Διαμόρφωση Πλέγματος και Μπλοκ:
 *   - dimGrid  : Αντιπροσωπεύει τον αριθμό των μπλοκ στο πλέγμα (X: nBlocks, Y: nBlocks, Z: 1).
 *   - dimBlock : Αντιπροσωπεύει τον αριθμό νημάτων σε κάθε μπλοκ (X: nThreads, Y: nThreads, Z: 1).
 * 
 * Μεταφορές Μνήμης:
 *   - cudaMemcpy(h_Acov, d_Acov, floatBytes, cudaMemcpyDeviceToHost) ανακτά τον υπολογισμένο πίνακα συνδιακύμανσης από τη μνήμη της συσκευής στη μνήμη του host.
 *
 * Μέτρηση απόδοσης:
 *   - Χρησιμοποιεί cudaEvent δομή για τη μέτρηση του χρόνου εκτέλεσης του kernel.
 *
 * Διαχείριση Σφαλμάτων:
 *   - Κάθε κλήση της CUDA ρουτίνας ακολουθείται από έλεγχο σφάλματος (αν επιστρέφεται cudaSuccess).
 */
    err = cudaEventRecord(start, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(start, 0) failed."); exit(1); }

    calcCov<<<dimGrid, dimBlock>>>(d_Asubmeans, d_Acov);

    err = cudaEventRecord(stop, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(stop, 0) failed."); exit(1); }

    err = cudaMemcpy(h_Acov, d_Acov, floatBytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(h_Acov, d_Acov, bytes, cudaMemcpyDeviceToHost) failed."); exit(1); }

    printf("The array A_cov has been stored in file %s\n", argv[5]);

    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventSynchronize(stop) failed."); exit(1); }
    err = cudaEventElapsedTime(&elapsedTime3, start, stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventElapsedTime(&elapsedTime3, start, stop) failed."); exit(1); }
    
    printf("Time for the kernel calcCov<<<>>>(): %f ms\n", elapsedTime3);

// ============== Λήξη Παράλληλου Υπολογισμού ==============

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            fprintf(fpA, "%4d ", h_A[i * n + j]);
            fprintf(fpAsubmeans, "%4.4f ", h_Asubmeans[i * n + j]);
            fprintf(fpATsubmeans, "%4.4f ", h_ATsubmeans[i * n + j]);
            fprintf(fpAcov, "%4.4f ", h_Acov[i * n + j]);
        }
        fprintf(fpAmean, "%4.4f\n", h_Amean[i]);
        fprintf(fpA, "\n");
        fprintf(fpAsubmeans, "\n");
        fprintf(fpATsubmeans, "\n");
        fprintf(fpAcov, "\n");
    }

    fclose(fpA);
    fclose(fpAmean);
    fclose(fpAcov);
    fclose(fpAsubmeans);
    fclose(fpATsubmeans);

    free(h_A);
    free(h_Acov);
    free(h_Amean);
    free(h_Asubmeans);
    free(h_ATsubmeans);
    
    err = cudaFree(d_A);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaFree(d_A) failed."); exit(1); }
    err = cudaFree(d_Acov);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaFree(d_Acov) failed."); exit(1); }
    err = cudaFree(d_Amean);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaFree(d_Amean) failed."); exit(1); }
    err = cudaFree(d_Asubmeans);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaFree(d_Asubmeans) failed."); exit(1); }
    err = cudaFree(d_ATsubmeans);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaFree(d_ATsubmeans) failed."); exit(1); }

    return 0;
}
