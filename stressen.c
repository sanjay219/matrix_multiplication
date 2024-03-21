#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1000 // Size of matrices (N x N)

void matrix_multiply(int **A, int **B, int **C, int size); // Standard matrix multiplication
void matrix_multiply_strassen(int n, int **A, int **B, int **C); // Strassen's algorithm
void add_matrices(int n, int **A, int **B, int **C);
void subtract_matrices(int n, int **A, int **B, int **C);
void split_matrix(int n, int **A, int **A11, int **A12, int **A21, int **A22);
void join_matrices(int n, int **A11, int **A12, int **A21, int **A22, int **A);
void free_matrix(int n, int **A); // Improved memory management

// Declaring matrix_multiply function
void matrix_multiply(int **A, int **B, int **C, int size) {
  // Initialize result matrix C to zeros
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      C[i][j] = 0; // Initialize each element to zero
    }
  }

  // Perform standard matrix multiplication
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      for (int k = 0; k < size; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void matrix_multiply_strassen(int n, int **A, int **B, int **C) {
    if (n <= 64) { // Base case threshold
        matrix_multiply(A, B, C, n); // Fallback to normal matrix multiplication
        return;
    }

    // Size for submatrices
    int newSize = n / 2;

    // Allocate memory for submatrices
    int **A11, **A12, **A21, **A22;
    int **B11, **B12, **B21, **B22;
    int **C11, **C12, **C21, **C22;

    A11 = (int **)malloc(newSize * sizeof(int *));
    A12 = (int **)malloc(newSize * sizeof(int *));
    A21 = (int **)malloc(newSize * sizeof(int *));
    A22 = (int **)malloc(newSize * sizeof(int *));

    B11 = (int **)malloc(newSize * sizeof(int *));
    B12 = (int **)malloc(newSize * sizeof(int *));
    B21 = (int **)malloc(newSize * sizeof(int *));
    B22 = (int **)malloc(newSize * sizeof(int *));

    C11 = (int **)malloc(newSize * sizeof(int *));
    C12 = (int **)malloc(newSize * sizeof(int *));
    C21 = (int **)malloc(newSize * sizeof(int *));
    C22 = (int **)malloc(newSize * sizeof(int *));

    for (int i = 0; i < newSize; i++) {
        A11[i] = (int *)malloc(newSize * sizeof(int));
        A12[i] = (int *)malloc(newSize * sizeof(int));
        A21[i] = (int *)malloc(newSize * sizeof(int));
        A22[i] = (int *)malloc(newSize * sizeof(int));

        B11[i] = (int *)malloc(newSize * sizeof(int));
        B12[i] = (int *)malloc(newSize * sizeof(int));
        B21[i] = (int *)malloc(newSize * sizeof(int));
        B22[i] = (int *)malloc(newSize * sizeof(int));

        C11[i] = (int *)malloc(newSize * sizeof(int));
        C12[i] = (int *)malloc(newSize * sizeof(int));
        C21[i] = (int *)malloc(newSize * sizeof(int));
        C22[i] = (int *)malloc(newSize * sizeof(int));
    }

    // Split matrices into submatrices
    split_matrix(n, A, A11, A12, A21, A22);
    split_matrix(n, B, B11, B12, B21, B22);
    split_matrix(n, C, C11, C12, C21, C22);

    // Strassen algorithm
    int **P1, **P2, **P3, **P4, **P5, **P6, **P7;
    P1 = (int **)malloc(newSize * sizeof(int *));
    P2 = (int **)malloc(newSize * sizeof(int *));
    P3 = (int **)malloc(newSize * sizeof(int *));
    P4 = (int **)malloc(newSize * sizeof(int *));
    P5 = (int **)malloc(newSize * sizeof(int *));
    P6 = (int **)malloc(newSize * sizeof(int *));
    P7 = (int **)malloc(newSize * sizeof(int *));

    for (int i = 0; i < newSize; i++) {
        P1[i] = (int *)malloc(newSize * sizeof(int));
        P2[i] = (int *)malloc(newSize * sizeof(int));
        P3[i] = (int *)malloc(newSize * sizeof(int));
        P4[i] = (int *)malloc(newSize * sizeof(int));
        P5[i] = (int *)malloc(newSize * sizeof(int));
        P6[i] = (int *)malloc(newSize * sizeof(int));
        P7[i] = (int *)malloc(newSize * sizeof(int));
    }

    // Calculating intermediate matrices using Strassen algorithm
    subtract_matrices(newSize, B12, B22, P1);
    matrix_multiply_strassen(newSize, A11, P1, P2);
    add_matrices(newSize, A11, A12, P1);
    matrix_multiply_strassen(newSize, P1, B22, P3);
    add_matrices(newSize, A21, A22, P1);
    matrix_multiply_strassen(newSize, P1, B11, P4);
    subtract_matrices(newSize, B21, B11, P5);
    matrix_multiply_strassen(newSize, A22, P5, P6);
    subtract_matrices(newSize, B12, B22, P5);
    add_matrices(newSize, A11, A22, P1);
    add_matrices(newSize, B11, B22, P2);
    matrix_multiply_strassen(newSize, P1, P2, P7);

    // Calculating resulting submatrices
    add_matrices(newSize, P7, P4, C11);
    subtract_matrices(newSize, P5, P3, C12);
    add_matrices(newSize, P1, P2, C21);
    add_matrices(newSize, P6, P3, C22);

    // Joining resulting submatrices into the result matrix C
    join_matrices(newSize, C11, C12, C21, C22, C);

    // Free memory for submatrices
    free_matrix(newSize, A11);
    free_matrix(newSize, A12);
    free_matrix(newSize, A21);
    free_matrix(newSize, A22);
    free_matrix(newSize, B11);
    free_matrix(newSize, B12);
    free_matrix(newSize, B21);
    free_matrix(newSize, B22);
    free_matrix(newSize, C11);
    free_matrix(newSize, C12);
    free_matrix(newSize, C21);
    free_matrix(newSize, C22);
    free_matrix(newSize, P1);
    free_matrix(newSize, P2);
    free_matrix(newSize, P3);
    free_matrix(newSize, P4);
    free_matrix(newSize, P5);
    free_matrix(newSize, P6);
    free_matrix(newSize, P7);
}

void add_matrices(int n, int **A, int **B, int **C) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

void subtract_matrices(int n, int **A, int **B, int **C) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}

void split_matrix(int n, int **A, int **A11, int **A12, int **A21, int **A22) {
    int newSize = n / 2;
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + newSize];
            A21[i][j] = A[i + newSize][j];
            A22[i][j] = A[i + newSize][j + newSize];
        }
    }
}

void join_matrices(int n, int **A11, int **A12, int **A21, int **A22, int **A) {
    int newSize = n / 2;
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            A[i][j] = A11[i][j];
            A[i][j + newSize] = A12[i][j];
            A[i + newSize][j] = A21[i][j];
            A[i + newSize][j + newSize] = A22[i][j];
        }
    }
}

void free_matrix(int n, int **A) {
    for (int i = 0; i < n; i++) {
        free(A[i]);
    }
    free(A);
}

int main() {
    int **A, **B, **C;

    // Allocate memory for matrices
    A = (int **)malloc(N * sizeof(int *));
    B = (int **)malloc(N * sizeof(int *));
    C = (int **)malloc(N * sizeof(int *));
    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < N; i++) {
        A[i] = (int *)malloc(N * sizeof(int));
        B[i] = (int *)malloc(N * sizeof(int));
        C[i] = (int *)malloc(N * sizeof(int));

        if (A[i] == NULL || B[i] == NULL || C[i] == NULL)
        {
            fprintf(stderr, "Memory allocation failed.\n");
            exit(EXIT_FAILURE);
        }
    }

    // Initialize matrices A and B
    srand(time(NULL)); // Seed random number generator
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 100; // Fill A with random values (0-99)
            B[i][j] = rand() % 100; // Fill B with random values (0-99)
        }
    }

    clock_t start = clock(); // Start measuring time

    // Perform matrix multiplication using Strassen algorithm
    matrix_multiply_strassen(N, A, B, C);

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