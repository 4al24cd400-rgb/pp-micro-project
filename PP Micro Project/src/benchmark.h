#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "matrix.h"
#include <stdbool.h>
#include <stdio.h>

typedef struct {
    int matrix_size;
    int block_size;
    int num_threads;
    kernel_variant_t variant;
    omp_sched_type_t schedule;
    int repetitions;
    int warmup_runs;
    double tolerance;
    bool verify;
    const char* csv_filepath;
    const char* cpu_model;
} bench_config_t;

typedef struct {
    double min_time;
    double mean_time;
    double median_time;
    double max_time;
    double stddev;
    double gflops;
    double speedup;
    double max_abs_diff;
    double max_rel_diff;
    bool correctness_pass;
    int repetitions_executed;
} bench_result_t;

/* Monotonic high-resolution timer in seconds */
double get_wall_time(void);

/* Executes timed runs with warmups and statistical metric calculation */
bool run_benchmark(const bench_config_t* config, const double* A, const double* B, const double* ref_C, bench_result_t* result);

/* CSV output management */
void write_csv_header(FILE* fp);
void append_csv_row(FILE* fp, const bench_config_t* config, const bench_result_t* result);

/* Validation test suite */
int run_correctness_test_suite(void);

#endif /* BENCHMARK_H */
