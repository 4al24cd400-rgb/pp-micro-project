#include "benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif

double get_wall_time(void) {
#if defined(_OPENMP)
    return omp_get_wtime();
#elif defined(_WIN32) || defined(_WIN64)
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static int compare_doubles(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static void execute_kernel(kernel_variant_t variant, const double* A, const double* B, double* C,
                           int n, int block_size, int num_threads, omp_sched_type_t sched) {
    switch (variant) {
        case VARIANT_NAIVE_IJK:
            matmul_naive_ijk(A, B, C, n);
            break;
        case VARIANT_IKJ:
            matmul_ikj(A, B, C, n);
            break;
        case VARIANT_BLOCKED_SERIAL:
            matmul_blocked_serial(A, B, C, n, block_size);
            break;
        case VARIANT_OMP_IKJ:
            matmul_omp_ikj(A, B, C, n, num_threads, sched);
            break;
        case VARIANT_OMP_BLOCKED:
            matmul_omp_blocked(A, B, C, n, block_size, num_threads, sched);
            break;
        case VARIANT_OMP_BLOCKED_COLLAPSE:
            matmul_omp_blocked_collapse(A, B, C, n, block_size, num_threads, sched);
            break;
        default:
            matmul_naive_ijk(A, B, C, n);
            break;
    }
}

bool run_benchmark(const bench_config_t* config, const double* A, const double* B, const double* ref_C, bench_result_t* result) {
    int n = config->matrix_size;
    double* test_C = allocate_matrix(n);
    if (!test_C) {
        fprintf(stderr, "Error: Failed to allocate test matrix C for benchmark.\n");
        return false;
    }

    /* 1. Correctness Validation */
    result->correctness_pass = true;
    result->max_abs_diff = 0.0;
    result->max_rel_diff = 0.0;

    if (config->verify && ref_C != NULL) {
        execute_kernel(config->variant, A, B, test_C, n, config->block_size, config->num_threads, config->schedule);
        result->correctness_pass = verify_matrices(ref_C, test_C, n, config->tolerance,
                                                   &result->max_abs_diff, &result->max_rel_diff);
        if (!result->correctness_pass) {
            fprintf(stderr, "Warning: Correctness check FAILED for variant %s (Max Abs Diff: %.2e, Max Rel Diff: %.2e)\n",
                    variant_to_string(config->variant), result->max_abs_diff, result->max_rel_diff);
        }
    }

    /* 2. Warm-up Iterations */
    for (int w = 0; w < config->warmup_runs; w++) {
        execute_kernel(config->variant, A, B, test_C, n, config->block_size, config->num_threads, config->schedule);
    }

    /* 3. Timed Repetitions */
    int reps = (config->repetitions > 0) ? config->repetitions : 5;
    double* times = (double*)malloc((size_t)reps * sizeof(double));
    if (!times) {
        free_matrix(test_C);
        return false;
    }

    double total_time = 0.0;
    for (int r = 0; r < reps; r++) {
        double t_start = get_wall_time();
        execute_kernel(config->variant, A, B, test_C, n, config->block_size, config->num_threads, config->schedule);
        double t_end = get_wall_time();

        double elapsed = t_end - t_start;
        times[r] = elapsed;
        total_time += elapsed;
    }

    /* 4. Statistical Analysis */
    qsort(times, (size_t)reps, sizeof(double), compare_doubles);

    result->min_time = times[0];
    result->max_time = times[reps - 1];
    result->mean_time = total_time / (double)reps;

    if (reps % 2 == 1) {
        result->median_time = times[reps / 2];
    } else {
        result->median_time = (times[reps / 2 - 1] + times[reps / 2]) * 0.5;
    }

    double var_sum = 0.0;
    for (int r = 0; r < reps; r++) {
        double d = times[r] - result->mean_time;
        var_sum += d * d;
    }
    result->stddev = sqrt(var_sum / (double)reps);
    result->repetitions_executed = reps;

    /* 5. GFLOPS calculation: 2 * N^3 FLOPs for N x N matrix multiplication */
    double total_flops = 2.0 * (double)n * (double)n * (double)n;
    if (result->median_time > 0.0) {
        result->gflops = (total_flops / (result->median_time * 1e9));
    } else {
        result->gflops = 0.0;
    }

    result->speedup = 1.0; /* Updated later by caller comparing to baseline */

    free(times);
    free_matrix(test_C);
    return true;
}

void write_csv_header(FILE* fp) {
    if (!fp) return;
    fprintf(fp, "timestamp,cpu,matrix_size,variant,loop_order,tile_size,threads,schedule,collapse,repetitions,min_time_s,mean_time_s,median_time_s,max_time_s,stddev_s,gflops,speedup,max_abs_diff,correctness\n");
    fflush(fp);
}

void append_csv_row(FILE* fp, const bench_config_t* config, const bench_result_t* result) {
    if (!fp) return;

    time_t now = time(NULL);
    char time_str[32];
    struct tm* tm_info = localtime(&now);
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", tm_info);

    const char* loop_order = (config->variant == VARIANT_NAIVE_IJK) ? "ijk" : "ikj";
    int is_collapse = (config->variant == VARIANT_OMP_BLOCKED_COLLAPSE) ? 1 : 0;
    const char* correctness_str = result->correctness_pass ? "PASS" : "FAIL";
    const char* cpu = (config->cpu_model && strlen(config->cpu_model) > 0) ? config->cpu_model : "Detected_CPU";

    fprintf(fp, "\"%s\",\"%s\",%d,\"%s\",\"%s\",%d,%d,\"%s\",%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.2e,\"%s\"\n",
            time_str,
            cpu,
            config->matrix_size,
            variant_to_string(config->variant),
            loop_order,
            config->block_size,
            config->num_threads,
            schedule_to_string(config->schedule),
            is_collapse,
            result->repetitions_executed,
            result->min_time,
            result->mean_time,
            result->median_time,
            result->max_time,
            result->stddev,
            result->gflops,
            result->speedup,
            result->max_abs_diff,
            correctness_str);
    fflush(fp);
}

/*
 * Automated Comprehensive Correctness Test Suite
 */
int run_correctness_test_suite(void) {
    printf("=================================================================\n");
    printf("        RUNNING MATRIX MULTIPLICATION UNIT TEST SUITE            \n");
    printf("=================================================================\n");

    const int test_sizes[] = {1, 2, 3, 7, 16, 65, 100, 128, 257};
    const int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    const int test_tiles[] = {8, 16, 32, 48, 64};
    const int num_tiles = sizeof(test_tiles) / sizeof(test_tiles[0]);

    int total_tests = 0;
    int passed_tests = 0;
    int failed_tests = 0;

    for (int s = 0; s < num_sizes; s++) {
        int n = test_sizes[s];
        double* A = allocate_matrix(n);
        double* B = allocate_matrix(n);
        double* C_ref = allocate_matrix(n);
        double* C_test = allocate_matrix(n);

        if (!A || !B || !C_ref || !C_test) {
            fprintf(stderr, "Allocation error in test suite for size %d\n", n);
            continue;
        }

        init_matrix_deterministic(A, n, 1.234);
        init_matrix_deterministic(B, n, 5.678);

        /* Compute reference with serial naive ijk */
        matmul_naive_ijk(A, B, C_ref, n);

        /* Test 1: IKJ loop order */
        total_tests++;
        matmul_ikj(A, B, C_test, n);
        double max_diff = 0.0, max_rel = 0.0;
        bool ok = verify_matrices(C_ref, C_test, n, DEFAULT_TOLERANCE, &max_diff, &max_rel);
        if (ok) {
            passed_tests++;
            printf("[PASS] Size %4dx%-4d | Variant: ikj                  | Max Diff: %.2e\n", n, n, max_diff);
        } else {
            failed_tests++;
            printf("[FAIL] Size %4dx%-4d | Variant: ikj                  | Max Diff: %.2e\n", n, n, max_diff);
        }

        /* Test 2: Blocked Serial with various tile sizes */
        for (int t = 0; t < num_tiles; t++) {
            int tile = test_tiles[t];
            total_tests++;
            matmul_blocked_serial(A, B, C_test, n, tile);
            ok = verify_matrices(C_ref, C_test, n, DEFAULT_TOLERANCE, &max_diff, &max_rel);
            if (ok) {
                passed_tests++;
                printf("[PASS] Size %4dx%-4d | Variant: blocked (tile=%-3d)     | Max Diff: %.2e\n", n, n, tile, max_diff);
            } else {
                failed_tests++;
                printf("[FAIL] Size %4dx%-4d | Variant: blocked (tile=%-3d)     | Max Diff: %.2e\n", n, n, tile, max_diff);
            }
        }

        /* Test 3: OpenMP IKJ (threads=4) */
        total_tests++;
        matmul_omp_ikj(A, B, C_test, n, 4, SCHEDULE_STATIC);
        ok = verify_matrices(C_ref, C_test, n, DEFAULT_TOLERANCE, &max_diff, &max_rel);
        if (ok) {
            passed_tests++;
            printf("[PASS] Size %4dx%-4d | Variant: omp_ikj (threads=4)   | Max Diff: %.2e\n", n, n, max_diff);
        } else {
            failed_tests++;
            printf("[FAIL] Size %4dx%-4d | Variant: omp_ikj (threads=4)   | Max Diff: %.2e\n", n, n, max_diff);
        }

        /* Test 4: OpenMP Blocked (threads=4, tile=32) */
        total_tests++;
        matmul_omp_blocked(A, B, C_test, n, 32, 4, SCHEDULE_STATIC);
        ok = verify_matrices(C_ref, C_test, n, DEFAULT_TOLERANCE, &max_diff, &max_rel);
        if (ok) {
            passed_tests++;
            printf("[PASS] Size %4dx%-4d | Variant: omp_blocked (t=4,B=32)| Max Diff: %.2e\n", n, n, max_diff);
        } else {
            failed_tests++;
            printf("[FAIL] Size %4dx%-4d | Variant: omp_blocked (t=4,B=32)| Max Diff: %.2e\n", n, n, max_diff);
        }

        /* Test 5: OpenMP Blocked Collapse(2) (threads=4, tile=32) */
        total_tests++;
        matmul_omp_blocked_collapse(A, B, C_test, n, 32, 4, SCHEDULE_STATIC);
        ok = verify_matrices(C_ref, C_test, n, DEFAULT_TOLERANCE, &max_diff, &max_rel);
        if (ok) {
            passed_tests++;
            printf("[PASS] Size %4dx%-4d | Variant: omp_collapse (t=4,B=32)| Max Diff: %.2e\n", n, n, max_diff);
        } else {
            failed_tests++;
            printf("[FAIL] Size %4dx%-4d | Variant: omp_collapse (t=4,B=32)| Max Diff: %.2e\n", n, n, max_diff);
        }

        free_matrix(A);
        free_matrix(B);
        free_matrix(C_ref);
        free_matrix(C_test);
    }

    printf("=================================================================\n");
    printf(" UNIT TEST SUMMARY: Total: %d, Passed: %d, Failed: %d\n", total_tests, passed_tests, failed_tests);
    printf(" RESULT: %s\n", (failed_tests == 0) ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    printf("=================================================================\n");

    return (failed_tests == 0) ? 0 : 1;
}
