/*
 *  === Αρχείο: cuda2.cu ===
 *
 *  Ονοματεπώνυμο: Αθανασίου Βασίλειος Ευάγγελος
 *  Αριθμός Μητρώου: 19390005
 *  Πρόγραμμα Σπουδών: ΠΑΔΑ
 *  
 *  Μεταγλώττιση: nvcc -o cuda2 cuda2.cu
 *  Εκτέλεση: ./cuda2 A.txt A_cov.txt
 * 
 */
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define N 10
#define BL 5
#define T 2

int main(int argc, char *argv[])
{
    int **A, **A_cov;
    int *d_A;
    FILE *fp_A, *fp_A_cov;
    int i, j;
    int matrix_size, grid_size, block_size;

    matrix_size = N;
    grid_size = BL;
    block_size = T;
    
    if (block_size < 1 || block_size > 1024)
    {
        printf("Threads per block must be between 1 to 1024.\n");
        exit(1);
    }

    if (grid_size < 1 || grid_size > 65535)
    {
        printf("Blocks must be between 1 to 65535.\n");
        exit(1);
    }

    if (argc != 3) 
    {
        printf("Usage: %s A.txt A_cov.txt\n", argv[0]);
        exit(1);
    }

    fpA = fopen(argv[1], "w");
    if (fpA == NULL) 
    {
        printf("Cannot open file %s\n", argv[1]);
        exit(1);
    }

    fpA_cov = fopen(argv[2], "w");
    if (fpA_cov == NULL) 
    {
        printf("Cannot open file %s\n", argv[2]);
        exit(1);
    }

    printf("Matrix size  : %d x %d\n", matrix_size, matrix_size );
    printf("Grid size    : %d\n", grid_size);
    printf("Block size   : %d\n", block_size);
    printf("-----------------------------------\n");

    A = (int **) malloc(matrix_size * sizeof(int *));
    A_cov = (int **) malloc(matrix_size * sizeof(int *));
    for (i = 0; i < matrix_size; i++) 
    {
        A[i] = (int *) malloc(matrix_size * sizeof(int));
        if (A[i] == NULL) 
        {
            printf("Memory allocation failed for A[%d]\n", i);
            exit(1);
        }
        A_cov[i] = (int *) malloc(matrix_size * sizeof(int));
        if (A_cov[i] == NULL) 
        {
            printf("Memory allocation failed for A_cov[%d]\n", i);
            exit(1);
        }
    }

    srand(time(NULL));

    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            A[i][j] = rand() % 199 - 99;                           // Τιμές στο διάστημα [-99, 99]
            A[i][j] = A[i][j] >= 0 ? A[i][j] + 10 : A[i][j] - 10;  // Τυχαία επιλογή προσήμου
        }
    }


    return 0;
}