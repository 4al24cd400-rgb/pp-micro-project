#include "matrix.h"
#include "benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

static void print_usage(const char* prog) {
    printf("===============================================================================\n");
    printf(" Dense Matrix Multiplication Benchmark with Cache Blocking & OpenMP Optimization\n");
    printf("===============================================================================\n");
    printf("Usage:\n");
    printf("  %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  --size, -n <N>          Matrix dimension N (default: 1024)\n");
    printf("  --tile, -b <B>          Cache blocking tile size B (default: 64)\n");
    printf("  --threads, -t <T>       Number of OpenMP threads (default: %d)\n", omp_get_max_threads());
    printf("  --variant, -v <VAR>     Kernel variant: naive_ijk, ikj, blocked_serial,\n");
    printf("                          omp_ikj, omp_blocked, omp_blocked_collapse, all (default: all)\n");
    printf("  --schedule <SCHED>      OpenMP schedule: static, dynamic (default: static)\n");
    printf("  --repetitions, -r <R>   Number of timed repetitions (default: 5)\n");
    printf("  --warmup, -w <W>        Number of untimed warmup runs (default: 1)\n");
    printf("  --tolerance <TOL>       Correctness validation tolerance (default: 1e-9)\n");
    printf("  --no-verify             Disable correctness verification against reference\n");
    printf("  --csv <path>            Path to output CSV file\n");
    printf("  --cpu <name>            CPU model name for logging metadata\n");
    printf("  --test                  Run comprehensive unit and correctness test suite\n");
    printf("  --help, -h              Display this help message and exit\n");
    printf("===============================================================================\n");
}

static kernel_variant_t parse_variant(const char* str, bool* is_all) {
    *is_all = false;
    if (strcmp(str, "all") == 0) {
        *is_all = true;
        return VARIANT_NAIVE_IJK;
    }
    if (strcmp(str, "naive") == 0 || strcmp(str, "naive_ijk") == 0 || strcmp(str, "ijk") == 0) return VARIANT_NAIVE_IJK;
    if (strcmp(str, "ikj") == 0) return VARIANT_IKJ;
    if (strcmp(str, "blocked") == 0 || strcmp(str, "blocked_serial") == 0) return VARIANT_BLOCKED_SERIAL;
    if (strcmp(str, "omp") == 0 || strcmp(str, "omp_ikj") == 0 || strcmp(str, "openmp") == 0) return VARIANT_OMP_IKJ;
    if (strcmp(str, "omp_blocked") == 0 || strcmp(str, "blocked_omp") == 0) return VARIANT_OMP_BLOCKED;
    if (strcmp(str, "omp_collapse") == 0 || strcmp(str, "omp_blocked_collapse") == 0 || strcmp(str, "collapse") == 0) return VARIANT_OMP_BLOCKED_COLLAPSE;
    return VARIANT_NAIVE_IJK;
}

static omp_sched_type_t parse_schedule(const char* str) {
    if (strcmp(str, "dynamic") == 0) return SCHEDULE_DYNAMIC;
    return SCHEDULE_STATIC;
}

int main(int argc, char* argv[]) {
    int n = 1024;
    int block_size = 64;
    int num_threads = omp_get_max_threads();
    kernel_variant_t variant = VARIANT_NAIVE_IJK;
    bool run_all_variants = true;
    omp_sched_type_t sched = SCHEDULE_STATIC;
    int repetitions = 5;
    int warmup_runs = 1;
    double tolerance = DEFAULT_TOLERANCE;
    bool verify = true;
    const char* csv_path = NULL;
    const char* cpu_model = "Intel Core Ultra 5 225H";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--test") == 0) {
            return run_correctness_test_suite();
        } else if ((strcmp(argv[i], "--size") == 0 || strcmp(argv[i], "-n") == 0) && i + 1 < argc) {
            n = atoi(argv[++i]);
            if (n <= 0) {
                fprintf(stderr, "Error: Invalid matrix size: %d. Must be > 0.\n", n);
                return 1;
            }
        } else if ((strcmp(argv[i], "--tile") == 0 || strcmp(argv[i], "-b") == 0) && i + 1 < argc) {
            block_size = atoi(argv[++i]);
            if (block_size <= 0) {
                fprintf(stderr, "Error: Invalid tile size: %d. Must be > 0.\n", block_size);
                return 1;
            }
        } else if ((strcmp(argv[i], "--threads") == 0 || strcmp(argv[i], "-t") == 0) && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
            if (num_threads <= 0) {
                fprintf(stderr, "Error: Invalid thread count: %d. Must be > 0.\n", num_threads);
                return 1;
            }
        } else if ((strcmp(argv[i], "--variant") == 0 || strcmp(argv[i], "-v") == 0) && i + 1 < argc) {
            variant = parse_variant(argv[++i], &run_all_variants);
        } else if (strcmp(argv[i], "--schedule") == 0 && i + 1 < argc) {
            sched = parse_schedule(argv[++i]);
        } else if ((strcmp(argv[i], "--repetitions") == 0 || strcmp(argv[i], "-r") == 0) && i + 1 < argc) {
            repetitions = atoi(argv[++i]);
            if (repetitions <= 0) repetitions = 5;
        } else if ((strcmp(argv[i], "--warmup") == 0 || strcmp(argv[i], "-w") == 0) && i + 1 < argc) {
            warmup_runs = atoi(argv[++i]);
            if (warmup_runs < 0) warmup_runs = 0;
        } else if (strcmp(argv[i], "--tolerance") == 0 && i + 1 < argc) {
            tolerance = atof(argv[++i]);
        } else if (strcmp(argv[i], "--no-verify") == 0) {
            verify = false;
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (strcmp(argv[i], "--cpu") == 0 && i + 1 < argc) {
            cpu_model = argv[++i];
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    printf("===============================================================================\n");
    printf(" DENSE MATRIX MULTIPLICATION BENCHMARK SUITE                                   \n");
    printf("===============================================================================\n");
    printf(" Matrix Size (N):    %d x %d (%.2f MB per matrix, Total: %.2f MB)\n",
           n, n, (double)(n * n * sizeof(double)) / (1024.0 * 1024.0),
           3.0 * (double)(n * n * sizeof(double)) / (1024.0 * 1024.0));
    printf(" Tile Size (B):      %d x %d\n", block_size, block_size);
    printf(" Threads:            %d\n", num_threads);
    printf(" Schedule:           %s\n", schedule_to_string(sched));
    printf(" Repetitions:        %d (Warmup: %d)\n", repetitions, warmup_runs);
    printf(" Validation:         %s (Tolerance: %.1e)\n", verify ? "Enabled" : "Disabled", tolerance);
    printf(" CSV Logging:        %s\n", csv_path ? csv_path : "None");
    printf("===============================================================================\n\n");

    /* Allocate matrices */
    double* A = allocate_matrix(n);
    double* B = allocate_matrix(n);
    double* C_ref = allocate_matrix(n);

    if (!A || !B || !C_ref) {
        fprintf(stderr, "Fatal Error: Memory allocation failed for N=%d.\n", n);
        free_matrix(A);
        free_matrix(B);
        free_matrix(C_ref);
        return 1;
    }

    init_matrix_deterministic(A, n, 1.234);
    init_matrix_deterministic(B, n, 5.678);

    /* Compute reference output */
    double t_ref_start = get_wall_time();
    matmul_ikj(A, B, C_ref, n);
    double t_ref_end = get_wall_time();
    double baseline_time = t_ref_end - t_ref_start;

    FILE* csv_fp = NULL;
    if (csv_path) {
        /* Open in append mode or create with header if new/empty */
        bool file_exists = false;
        FILE* check_fp = fopen(csv_path, "r");
        if (check_fp) {
            fseek(check_fp, 0, SEEK_END);
            if (ftell(check_fp) > 0) file_exists = true;
            fclose(check_fp);
        }
        csv_fp = fopen(csv_path, "a");
        if (csv_fp && !file_exists) {
            write_csv_header(csv_fp);
        }
    }

    kernel_variant_t variants_to_run[6];
    int count_variants = 0;

    if (run_all_variants) {
        variants_to_run[0] = VARIANT_NAIVE_IJK;
        variants_to_run[1] = VARIANT_IKJ;
        variants_to_run[2] = VARIANT_BLOCKED_SERIAL;
        variants_to_run[3] = VARIANT_OMP_IKJ;
        variants_to_run[4] = VARIANT_OMP_BLOCKED;
        variants_to_run[5] = VARIANT_OMP_BLOCKED_COLLAPSE;
        count_variants = 6;
    } else {
        variants_to_run[0] = variant;
        count_variants = 1;
    }

    printf("%-24s | %-8s | %-11s | %-11s | %-10s | %-8s | %-6s\n",
           "Variant", "TileSize", "Median Time", "Min Time", "GFLOPS", "Speedup", "Status");
    printf("-------------------------------------------------------------------------------------------------\n");

    for (int v = 0; v < count_variants; v++) {
        bench_config_t config;
        config.matrix_size = n;
        config.block_size = block_size;
        config.num_threads = num_threads;
        config.variant = variants_to_run[v];
        config.schedule = sched;
        config.repetitions = repetitions;
        config.warmup_runs = warmup_runs;
        config.tolerance = tolerance;
        config.verify = verify;
        config.csv_filepath = csv_path;
        config.cpu_model = cpu_model;

        bench_result_t result;
        bool ok = run_benchmark(&config, A, B, verify ? C_ref : NULL, &result);

        if (!ok) {
            fprintf(stderr, "Execution error for variant %s\n", variant_to_string(config.variant));
            continue;
        }

        /* Speedup relative to reference or single thread */
        if (baseline_time > 0.0 && result.median_time > 0.0) {
            result.speedup = baseline_time / result.median_time;
        } else {
            result.speedup = 1.0;
        }

        printf("%-24s | %-8d | %9.4f s | %9.4f s | %10.2f | %7.2fx | %s\n",
               variant_to_string(config.variant),
               config.block_size,
               result.median_time,
               result.min_time,
               result.gflops,
               result.speedup,
               result.correctness_pass ? "PASS" : "FAIL");

        if (csv_fp) {
            append_csv_row(csv_fp, &config, &result);
        }
    }

    printf("-------------------------------------------------------------------------------------------------\n\n");

    if (csv_fp) {
        fclose(csv_fp);
    }

    free_matrix(A);
    free_matrix(B);
    free_matrix(C_ref);

    return 0;
}
