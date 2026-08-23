#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>
#include <stdbool.h>

#define ALIGNMENT_BYTES 64
#define DEFAULT_TOLERANCE 1e-9

typedef enum {
    SCHEDULE_STATIC = 0,
    SCHEDULE_DYNAMIC = 1
} omp_sched_type_t;

typedef enum {
    VARIANT_NAIVE_IJK = 0,
    VARIANT_IKJ = 1,
    VARIANT_BLOCKED_SERIAL = 2,
    VARIANT_OMP_IKJ = 3,
    VARIANT_OMP_BLOCKED = 4,
    VARIANT_OMP_BLOCKED_COLLAPSE = 5
} kernel_variant_t;

/* Aligned dynamic memory allocation and release */
double* allocate_matrix(int n);
void free_matrix(double* mat);

/* Deterministic numerical initialization */
void init_matrix_deterministic(double* mat, int n, double seed);
void zero_matrix(double* mat, int n);

/* Matrix multiplication computational kernels */
void matmul_naive_ijk(const double* A, const double* B, double* C, int n);
void matmul_ikj(const double* A, const double* B, double* C, int n);
void matmul_blocked_serial(const double* A, const double* B, double* C, int n, int block_size);
void matmul_omp_ikj(const double* A, const double* B, double* C, int n, int num_threads, omp_sched_type_t sched);
void matmul_omp_blocked(const double* A, const double* B, double* C, int n, int block_size, int num_threads, omp_sched_type_t sched);
void matmul_omp_blocked_collapse(const double* A, const double* B, double* C, int n, int block_size, int num_threads, omp_sched_type_t sched);

/* Correctness verification against reference */
bool verify_matrices(const double* ref, const double* test, int n, double tolerance, double* max_diff_out, double* max_rel_diff_out);

/* Utility string helpers */
const char* variant_to_string(kernel_variant_t variant);
const char* schedule_to_string(omp_sched_type_t sched);

#endif /* MATRIX_H */
