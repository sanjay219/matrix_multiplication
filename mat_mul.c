#include <stdio.h>
#include <stdlib.h>
#include <time.h>  // we are planning to measure the computation time

#define N 1000 // Size of matrices (N x N)


//matrix multiply function
void matrix_multiply(int **A, int **B, int **C, int size) {
    // Initialize result matrix C to zeros
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i][j] = 0; // Initialize each element to zero
        }
    }

    // Perform matrix multiplication
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}


int main() {
    int **A, **B, **C;

    // Allocate memory for matrices
    A = (int **)malloc(N * sizeof(int *));
    B = (int **)malloc(N * sizeof(int *));
    C = (int **)malloc(N * sizeof(int *));

    for (int i = 0; i < N; i++) {
        A[i] = (int *)malloc(N * sizeof(int));
        B[i] = (int *)malloc(N * sizeof(int));
        C[i] = (int *)calloc(N, sizeof(int)); // Initialize C with zeros
    }

    // Initialize matrices A and B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 100; // Fill A with random values (0-99)
            B[i][j] = rand() % 100; // Fill B with random values (0-99)
        }
    }
    clock_t start = clock(); // Start measuring time
    // Perform matrix multiplication
    matrix_multiply(A, B, C, N);

    clock_t end = clock(); // Stop measuring time

    // Print result matrix C (optional)
    printf("Result matrix C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }

    
    // Calculate elapsed time in milliseconds
    double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

    printf("Computation time: %f milliseconds\n", elapsed_time);

    // Free allocated memory
    for (int i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);

    return 0;
}
