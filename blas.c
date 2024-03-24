#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <lapacke.h>
#include <cblas.h>

#define N 1000 // Size of the matrices

int main() {
    // Seed the random number generator
    srand(time(NULL));

    // Define matrices A, B, and C
    double (*A)[N] = malloc(N * sizeof(*A));
    double (*B)[N] = malloc(N * sizeof(*B));
    double (*C)[N] = malloc(N * sizeof(*C));

    // Check if memory allocation succeeded
    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Fill matrices A and B with random numbers
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            A[i][j] = rand() % 100; // Fill A with random values (0-99)
            B[i][j] = rand() % 100; // Fill B with random values (0-99)
            // A[i][j] = (double)rand() / RAND_MAX; // Random number between 0 and 1
            // B[i][j] = (double)rand() / RAND_MAX; // Random number between 0 and 1
        }
    }
    clock_t start = clock(); // Start measuring time

    // Perform matrix multiplication using LAPACKe
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, (const double*)A, N, (const double*)B, N, 0.0, (double*)C, N);

    clock_t end = clock(); // Stop measuring time
    // Print the result
    printf("Resultant Matrix:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%lf ", C[i][j]);
        }
        printf("\n");
    }

    // Calculate elapsed time in milliseconds
    double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

    printf("Computation time: %f milliseconds\n", elapsed_time);

    // Free allocated memory
    free(A);
    free(B);
    free(C);

    return 0;
}
