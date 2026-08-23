#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

/*
 * Memory allocation with 64-byte alignment.
 * Aligns memory to CPU cache-line boundaries to prevent cache-line splits
 * and enable compiler SIMD vectorization (AVX2 / AVX-512).
 */
double* allocate_matrix(int n) {
    size_t size = (size_t)n * (size_t)n * sizeof(double);
#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
    double* ptr = (double*)_aligned_malloc(size, ALIGNMENT_BYTES);
#elif defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
    double* ptr = NULL;
    if (posix_memalign((void**)&ptr, ALIGNMENT_BYTES, size) != 0) {
        ptr = NULL;
    }
#else
    double* ptr = (double*)malloc(size);
#endif
    return ptr;
}

void free_matrix(double* mat) {
    if (mat == NULL) return;
#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
    _aligned_free(mat);
#else
    free(mat);
#endif
}

/*
 * Deterministic pseudo-random / trigonometric initialization.
 * Generates bounded non-zero values in [-1.0, 1.0] avoiding arithmetic overflow.
 */
void init_matrix_deterministic(double* mat, int n, double seed) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double val = sin((double)(i + 1) * 0.125 + seed) * cos((double)(j + 1) * 0.25);
            mat[i * n + j] = val;
        }
    }
}

void zero_matrix(double* mat, int n) {
    memset(mat, 0, (size_t)n * (size_t)n * sizeof(double));
}

/*
 * 1. Naive Serial Matrix Multiplication (ijk loop order)
 * Matrix A: row-major traversal A[i*n + k] (stride 1 across k)
 * Matrix B: column-wise strided traversal B[k*n + j] (stride N across k)
 * Poor spatial locality on B for large N because consecutive k iterations jump by N elements.
 */
void matmul_naive_ijk(const double* A, const double* B, double* C, int n) {
    for (int i = 0; i < n; i++) {
        int i_offset = i * n;
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i_offset + k] * B[k * n + j];
            }
            C[i_offset + j] = sum;
        }
    }
}

/*
 * 2. Loop-Order Optimized Matrix Multiplication (ikj loop order)
 * Matrix A: A[i*n + k] is scalar in the innermost loop (kept in register).
 * Matrix B: B[k*n + j] is accessed with unit stride in j (continuous row streaming).
 * Matrix C: C[i*n + j] is accumulated with unit stride in j.
 * Maximizes spatial locality: each 64-byte cache line brings 8 consecutive doubles.
 */
void matmul_ikj(const double* A, const double* B, double* C, int n) {
    zero_matrix(C, n);
    for (int i = 0; i < n; i++) {
        int i_offset = i * n;
        for (int k = 0; k < n; k++) {
            double a_ik = A[i_offset + k];
            int k_offset = k * n;
            for (int j = 0; j < n; j++) {
                C[i_offset + j] += a_ik * B[k_offset + j];
            }
        }
    }
}

/*
 * 3. Cache-Blocked Serial Matrix Multiplication (ii, kk, jj, i, k, j)
 * Partitions matrices into BxB sub-blocks to fit within CPU cache hierarchy.
 * Uses MIN(..., n) boundary clamping to support any matrix size n without padding.
 */
void matmul_blocked_serial(const double* A, const double* B, double* C, int n, int block_size) {
    zero_matrix(C, n);
    for (int ii = 0; ii < n; ii += block_size) {
        int i_max = MIN(ii + block_size, n);
        for (int kk = 0; kk < n; kk += block_size) {
            int k_max = MIN(kk + block_size, n);
            for (int jj = 0; jj < n; jj += block_size) {
                int j_max = MIN(jj + block_size, n);

                for (int i = ii; i < i_max; i++) {
                    int i_offset = i * n;
                    for (int k = kk; k < k_max; k++) {
                        double a_ik = A[i_offset + k];
                        int k_offset = k * n;
                        for (int j = jj; j < j_max; j++) {
                            C[i_offset + j] += a_ik * B[k_offset + j];
                        }
                    }
                }
            }
        }
    }
}

/*
 * 4. OpenMP Parallel IKJ Matrix Multiplication
 * Distributes outermost loop i among available threads.
 */
void matmul_omp_ikj(const double* A, const double* B, double* C, int n, int num_threads, omp_sched_type_t sched) {
    zero_matrix(C, n);
    if (sched == SCHEDULE_STATIC) {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int i = 0; i < n; i++) {
            int i_offset = i * n;
            for (int k = 0; k < n; k++) {
                double a_ik = A[i_offset + k];
                int k_offset = k * n;
                for (int j = 0; j < n; j++) {
                    C[i_offset + j] += a_ik * B[k_offset + j];
                }
            }
        }
    } else {
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
        for (int i = 0; i < n; i++) {
            int i_offset = i * n;
            for (int k = 0; k < n; k++) {
                double a_ik = A[i_offset + k];
                int k_offset = k * n;
                for (int j = 0; j < n; j++) {
                    C[i_offset + j] += a_ik * B[k_offset + j];
                }
            }
        }
    }
}

/*
 * 5. OpenMP Blocked Matrix Multiplication (Parallel on ii tile loop)
 * Parallelizes the outer tile loop ii.
 * Iteration space for thread distribution: ceil(n / block_size).
 * Each thread handles a horizontal strip of tiles of height <= block_size.
 */
void matmul_omp_blocked(const double* A, const double* B, double* C, int n, int block_size, int num_threads, omp_sched_type_t sched) {
    zero_matrix(C, n);
    if (sched == SCHEDULE_STATIC) {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int ii = 0; ii < n; ii += block_size) {
            int i_max = MIN(ii + block_size, n);
            for (int kk = 0; kk < n; kk += block_size) {
                int k_max = MIN(kk + block_size, n);
                for (int jj = 0; jj < n; jj += block_size) {
                    int j_max = MIN(jj + block_size, n);
                    for (int i = ii; i < i_max; i++) {
                        int i_offset = i * n;
                        for (int k = kk; k < k_max; k++) {
                            double a_ik = A[i_offset + k];
                            int k_offset = k * n;
                            for (int j = jj; j < j_max; j++) {
                                C[i_offset + j] += a_ik * B[k_offset + j];
                            }
                        }
                    }
                }
            }
        }
    } else {
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
        for (int ii = 0; ii < n; ii += block_size) {
            int i_max = MIN(ii + block_size, n);
            for (int kk = 0; kk < n; kk += block_size) {
                int k_max = MIN(kk + block_size, n);
                for (int jj = 0; jj < n; jj += block_size) {
                    int j_max = MIN(jj + block_size, n);
                    for (int i = ii; i < i_max; i++) {
                        int i_offset = i * n;
                        for (int k = kk; k < k_max; k++) {
                            double a_ik = A[i_offset + k];
                            int k_offset = k * n;
                            for (int j = jj; j < j_max; j++) {
                                C[i_offset + j] += a_ik * B[k_offset + j];
                            }
                        }
                    }
                }
            }
        }
    }
}

/*
 * 6. OpenMP Blocked Matrix Multiplication with collapse(2)
 *
 * Concurrency & Data-Race Semantics:
 * By collapsing the two outer tile loops (ii and jj), OpenMP flattens the 2D grid of
 * output tiles (ii, jj) into a single iteration space of size ceil(n/B) * ceil(n/B).
 * Each iteration exclusively updates a distinct sub-matrix C[ii..i_max, jj..j_max].
 * Because no two threads write to the same element of C, there are NO direct write data races.
 * The accumulation along kk runs serially per thread block.
 *
 * Parallelism & Locality Trade-off:
 * - Available work chunks increase from ceil(n/B) to ceil(n/B)^2, which prevents thread starvation
 *   when thread count exceeds ceil(n/B).
 * - However, schedule chunks distribute (ii, jj) pairs across threads, potentially altering
 *   temporal reuse of A[ii, kk] or B[kk, jj] tiles across cores compared to sequential row-strip allocation.
 * - Cache-line sharing: While distinct tiles own separate elements, tile boundaries sharing
 *   a 64-byte cache line (8 doubles) can theoretically experience transient cache-line contention.
 */
void matmul_omp_blocked_collapse(const double* A, const double* B, double* C, int n, int block_size, int num_threads, omp_sched_type_t sched) {
    zero_matrix(C, n);
    if (sched == SCHEDULE_STATIC) {
        #pragma omp parallel for collapse(2) num_threads(num_threads) schedule(static)
        for (int ii = 0; ii < n; ii += block_size) {
            for (int jj = 0; jj < n; jj += block_size) {
                int i_max = MIN(ii + block_size, n);
                int j_max = MIN(jj + block_size, n);
                for (int kk = 0; kk < n; kk += block_size) {
                    int k_max = MIN(kk + block_size, n);
                    for (int i = ii; i < i_max; i++) {
                        int i_offset = i * n;
                        for (int k = kk; k < k_max; k++) {
                            double a_ik = A[i_offset + k];
                            int k_offset = k * n;
                            for (int j = jj; j < j_max; j++) {
                                C[i_offset + j] += a_ik * B[k_offset + j];
                            }
                        }
                    }
                }
            }
        }
    } else {
        #pragma omp parallel for collapse(2) num_threads(num_threads) schedule(dynamic)
        for (int ii = 0; ii < n; ii += block_size) {
            for (int jj = 0; jj < n; jj += block_size) {
                int i_max = MIN(ii + block_size, n);
                int j_max = MIN(jj + block_size, n);
                for (int kk = 0; kk < n; kk += block_size) {
                    int k_max = MIN(kk + block_size, n);
                    for (int i = ii; i < i_max; i++) {
                        int i_offset = i * n;
                        for (int k = kk; k < k_max; k++) {
                            double a_ik = A[i_offset + k];
                            int k_offset = k * n;
                            for (int j = jj; j < j_max; j++) {
                                C[i_offset + j] += a_ik * B[k_offset + j];
                            }
                        }
                    }
                }
            }
        }
    }
}

/*
 * Correctness verification: computes absolute and relative maximum differences
 * against the serial reference output.
 */
bool verify_matrices(const double* ref, const double* test, int n, double tolerance, double* max_diff_out, double* max_rel_diff_out) {
    double max_diff = 0.0;
    double max_rel = 0.0;
    bool passed = true;

    for (int i = 0; i < n * n; i++) {
        double r = ref[i];
        double t = test[i];
        double diff = fabs(r - t);
        if (diff > max_diff) {
            max_diff = diff;
        }
        double denom = fabs(r);
        if (denom > 1e-12) {
            double rel = diff / denom;
            if (rel > max_rel) max_rel = rel;
        }
        if (diff > tolerance) {
            passed = false;
        }
    }

    if (max_diff_out) *max_diff_out = max_diff;
    if (max_rel_diff_out) *max_rel_diff_out = max_rel;
    return passed;
}

const char* variant_to_string(kernel_variant_t variant) {
    switch (variant) {
        case VARIANT_NAIVE_IJK: return "naive_ijk";
        case VARIANT_IKJ: return "ikj";
        case VARIANT_BLOCKED_SERIAL: return "blocked_serial";
        case VARIANT_OMP_IKJ: return "omp_ikj";
        case VARIANT_OMP_BLOCKED: return "omp_blocked";
        case VARIANT_OMP_BLOCKED_COLLAPSE: return "omp_blocked_collapse";
        default: return "unknown";
    }
}

const char* schedule_to_string(omp_sched_type_t sched) {
    switch (sched) {
        case SCHEDULE_STATIC: return "static";
        case SCHEDULE_DYNAMIC: return "dynamic";
        default: return "unknown";
    }
}
