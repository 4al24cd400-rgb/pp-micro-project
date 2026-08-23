# Comprehensive Academic Project Report

# Dense Matrix Multiplication with Cache Blocking — Tile Size Selection Tied to L1/L2 Cache Size, `collapse()` Decisions

---

## 1. Introduction

Dense matrix-matrix multiplication ($C = A \times B$) represents one of the foundational computational kernels in high-performance computing (HPC), numerical linear algebra, physics simulations, computer vision, and modern deep learning frameworks. In double-precision floating-point arithmetic ($\mathbb{R}^{N \times N}$), computing the matrix product of two $N \times N$ dense matrices requires $2N^3$ arithmetic floating-point operations ($N^3$ multiplications and $N^3$ additions) over $3N^2 \times 8\text{ bytes}$ of memory.

Mathematically, the theoretical arithmetic intensity of dense matrix multiplication scales linearly with problem dimension:

$$\text{Arithmetic Intensity} = \frac{\text{Floating Point Operations}}{\text{Memory Data Movement}} = \frac{2N^3\text{ FLOPs}}{3N^2 \times 8\text{ Bytes}} = \frac{N}{12}\text{ FLOPs/Byte}$$

For large matrix dimensions ($N \ge 1024$), this high arithmetic intensity suggests that matrix multiplication should ideally be compute-bound, operating near the peak floating-point throughput of modern processors. However, in practice, standard naive three-nested-loop implementations suffer from catastrophic performance degradation, operating at less than 2% of the processor's theoretical peak capability.

This severe performance gap is a direct manifestation of **the Memory Wall**—the architectural latency and bandwidth disparity between high-speed CPU computation units and high-capacity off-chip dynamic RAM (DRAM). Modern processors mitigate this latency through a multi-tiered hierarchical memory subsystem comprising per-core Level 1 Data (L1D) caches, Level 2 (L2) caches, shared Level 3 (L3) caches, and dynamic memory controllers.

This project presents an empirical and theoretical investigation into optimizing dense matrix multiplication on modern multicore architectures. We study:
1. **Loop Reordering ($ijk \to ikj$)**: Transforming non-contiguous memory access into unit-stride row streaming.
2. **Cache Blocking (Tiling)**: Constraining the active working dataset into high-speed L1/L2 hardware caches.
3. **Working-Set Capacity Modeling**: Quantifying theoretical upper bounds for tile sizes based on per-core L1D and L2 cache capacities.
4. **Multicore Parallelism with OpenMP**: Scaling computational throughput across multi-core architectures.
5. **Iteration Space Collapse (`collapse(2)`)**: Mitigating thread starvation and optimizing workload distribution across modern many-core CPUs.

---

## 2. Problem Statement

### 2.1 The Memory Bottleneck in Naive Matrix Multiplication
In the canonical naive formulation ($ijk$ order), computing an output element $C[i][j]$ requires computing the dot product of row $i$ of matrix $A$ and column $j$ of matrix $B$:

$$C[i][j] = \sum_{k=0}^{N-1} A[i][k] \times B[k][j]$$

In standard row-major memory layout (where matrix rows are laid out consecutively in linear memory addresses), accessing elements across a column of matrix $B$ ($B[k \cdot N + j]$ for $k = 0, 1, \dots, N-1$) involves a memory jump of $N \times 8\text{ bytes}$ between successive iterations of the inner loop $k$.

```
Matrix A (Row-Major: Contiguous Row)       Matrix B (Row-Major: Strided Column)
[ a00  a01  a02  a03 ]                     [ b00  b01  b02  b03 ]  <-- Stride of N elements
[ .    .    .    .   ]                     [ b10  b11  b12  b13 ]  <-- (Cache line wasted)
[ .    .    .    .   ]                     [ b20  b21  b22  b23 ]  <-- (Cache line wasted)
[ .    .    .    .   ]                     [ b30  b31  b32  b33 ]  <-- (Cache line wasted)
```

### 2.2 Microarchitectural Consequences
1. **Cache Line Underutilization**: A standard 64-byte hardware cache line holds 8 double-precision floats. When accessing column $j$ of $B$, the CPU brings a 64-byte cache line into L1D but utilizes only **1 double-precision float (8 bytes)** before jumping to the next row. The remaining 7 floats (87.5% of the fetched bandwidth) are evicted before being reused, causing severe cache line waste.
2. **Capacity & Conflict Misses**: When matrix dimension $N$ exceeds the cache capacity ($N \times 8\text{ bytes} > \text{Cache Size}$), cache lines fetched in earlier iterations are prematurely flushed to main memory.
3. **Inability to Vectorize**: Strided memory loads prevent modern SIMD execution units (AVX2, AVX-512, FMA) from executing vectorized multiply-accumulate instructions.
4. **Thread Starvation in Multicore Parallelism**: Naively parallelizing only the outermost loop ($ii$) creates only $\lceil N/B \rceil$ work items. When $N=1024$ and $B=128$, only 8 tasks exist, leaving cores completely idle on processors with $\ge 14$ logical cores.

---

## 3. Methodology

### 3.1 Algorithmic Transformations

#### Algorithm 1: Naive Triple-Loop ($ijk$)
The standard textbook implementation where the inner loop strides down columns of $B$.
```c
void matmul_naive_ijk(const double *A, const double *B, double *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j]; // Non-contiguous column access on B
            }
            C[i * N + j] = sum;
        }
    }
}
```

#### Algorithm 2: Loop-Reordered ($ikj$ Unit-Stride Streaming)
By interchanging loops $k$ and $j$, the scalar element $A[i][k]$ becomes invariant with respect to the innermost loop $j$ and is held inside a CPU register.
```c
void matmul_ikj(const double *A, const double *B, double *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double a_ik = A[i * N + k]; // Maintained in hardware register
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j]; // Contiguous row streaming
            }
        }
    }
}
```
* **Memory Traversal**: Both $B[k \cdot N + j]$ and $C[i \cdot N + j]$ traverse contiguous memory addresses with unit stride ($\Delta j = 1$).
* **Cache Efficiency**: Every 64-byte cache line fetch satisfies 8 consecutive inner-loop iterations.
* **Vectorization**: Compilers automatically generate packed SIMD instructions (FMA/AVX2).

#### Algorithm 3: Cache-Blocked Serial Matrix Multiplication
Partitions the computation into $B \times B$ square sub-matrices (tiles) to keep sub-matrices resident in cache.
```c
void matmul_blocked(const double *A, const double *B, double *C, int N, int tile_size) {
    int B_sz = tile_size;
    for (int ii = 0; ii < N; ii += B_sz) {
        int i_max = MIN(ii + B_sz, N);
        for (int kk = 0; kk < N; kk += B_sz) {
            int k_max = MIN(kk + B_sz, N);
            for (int jj = 0; jj < N; jj += B_sz) {
                int j_max = MIN(jj + B_sz, N);
                for (int i = ii; i < i_max; i++) {
                    for (int k = kk; k < k_max; k++) {
                        double a_ik = A[i * N + k];
                        for (int j = jj; j < j_max; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}
```

---

### 3.2 Theoretical Working-Set Capacity Model

For double-precision floating-point numbers ($8\text{ bytes/element}$) and a tile dimension of $B \times B$:
* Footprint of one tile: $W_{\text{tile}} = B^2 \times 8\text{ bytes}$
* **Three-Tile Active Working Set Model**:
  $$W_{3\text{-tile}} \approx A_{\text{tile}} + B_{\text{tile}} + C_{\text{tile}} = 3 \times B^2 \times 8 = 24 B^2\text{ bytes}$$

#### Mathematical Derivation of Theoretical Cache Bounds:
1. **L1D Cache Upper Bound ($W_{3\text{-tile}} \le \text{Size}_{\text{L1D}} = 48\text{ KB}$)**:
   $$24 B^2 \le 48 \times 1024 \implies B^2 \le 2048 \implies B \le \lfloor \sqrt{2048} \rfloor = 45$$
   Candidate tile sizes: $B \in \{8, 16, 32\}$.

2. **L2 Cache Upper Bound ($W_{3\text{-tile}} \le \text{Size}_{\text{L2}} = 2048\text{ KB}$)**:
   $$24 B^2 \le 2048 \times 1024 \implies B^2 \le 87381.33 \implies B \le \lfloor \sqrt{87381.33} \rfloor = 295$$
   Candidate tile sizes: $B \in \{48, 64, 96, 128, 192, 256\}$.

| Tile Size ($B$) | Single Tile Footprint | 3-Tile Working Set | % of L1D (48 KB) | % of L2 (2.0 MB) | Theoretical Classification |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **8 $\times$ 8** | 0.50 KB | 1.50 KB | 3.1% | 0.07% | L1D Resident (Fits easily in L1D) |
| **16 $\times$ 16** | 2.00 KB | 6.00 KB | 12.5% | 0.29% | L1D Resident (Fits in L1D) |
| **32 $\times$ 32** | 8.00 KB | 24.00 KB | 50.0% | 1.17% | L1D Resident (Optimal L1D Bound) |
| **48 $\times$ 48** | 18.00 KB | 54.00 KB | 112.5% | 2.64% | L2 Resident (Spills L1D, Fits in L2) |
| **64 $\times$ 64** | 32.00 KB | 96.00 KB | 200.0% | 4.69% | L2 Resident (Fits in L2) |
| **96 $\times$ 96** | 72.00 KB | 216.00 KB | 450.0% | 10.55% | L2 Resident (Fits in L2) |
| **128 $\times$ 128** | 128.00 KB | 384.00 KB | 800.0% | 18.75% | L2 Resident (Optimal L2 Bound) |
| **192 $\times$ 192** | 288.00 KB | 864.00 KB | 1800.0% | 42.19% | L2 Resident (Fits in L2) |
| **256 $\times$ 256** | 512.00 KB | 1536.00 KB | 3200.0% | 75.00% | L2 Resident (Upper L2 Limit) |
| **384 $\times$ 384** | 1152.00 KB | 3456.00 KB | 7200.0% | 168.75% | Exceeds L2 (Spills to L3/RAM) |
| **512 $\times$ 512** | 2048.00 KB | 6144.00 KB | 12800.0% | 300.00% | Exceeds L2 (Spills to L3/RAM) |

---

### 3.3 OpenMP Multithreading & `collapse(2)` Architecture

#### Strategy A: Non-Collapsed Outer-Loop Parallelism (`omp_blocked`)
```c
void matmul_omp_blocked(const double *A, const double *B, double *C, int N, int tile_size, int num_threads) {
    int B_sz = tile_size;
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int ii = 0; ii < N; ii += B_sz) {
        int i_max = MIN(ii + B_sz, N);
        for (int kk = 0; kk < N; kk += B_sz) {
            int k_max = MIN(kk + B_sz, N);
            for (int jj = 0; jj < N; jj += B_sz) {
                int j_max = MIN(jj + B_sz, N);
                for (int i = ii; i < i_max; i++) {
                    for (int k = kk; k < k_max; k++) {
                        double a_ik = A[i * N + k];
                        for (int j = jj; j < j_max; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}
```
* **Work Item Space**: Total iterations = $\lceil N / B \rceil$.
* **Starvation Issue**: At $N=1024, B=128$, only 8 tasks exist. On a 14-core processor, 6 cores remain 100% idle.

#### Strategy B: 2D Collapsed Iteration Space (`omp_blocked_collapse`)
```c
void matmul_omp_blocked_collapse(const double *A, const double *B, double *C, int N, int tile_size, int num_threads) {
    int B_sz = tile_size;
    #pragma omp parallel for collapse(2) num_threads(num_threads) schedule(static)
    for (int ii = 0; ii < N; ii += B_sz) {
        for (int jj = 0; jj < N; jj += B_sz) {
            int i_max = MIN(ii + B_sz, N);
            int j_max = MIN(jj + B_sz, N);
            for (int kk = 0; kk < N; kk += B_sz) {
                int k_max = MIN(kk + B_sz, N);
                for (int i = ii; i < i_max; i++) {
                    for (int k = kk; k < k_max; k++) {
                        double a_ik = A[i * N + k];
                        for (int j = jj; j < j_max; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}
```
* **Work Item Space**: Total iterations = $\lceil N / B \rceil^2$.
* **Load Balance**: At $N=1024, B=128$, the iteration space is flattened into $8 \times 8 = 64$ work items. All 14 threads receive balanced chunks ($\approx 4.5$ tiles/thread).
* **Data-Race Freedom**: Each $(ii, jj)$ pair writes exclusively to a disjoint sub-matrix block $C[ii \dots i_{\max}-1][jj \dots j_{\max}-1]$, guaranteeing thread safety without mutex or atomic overheads.

---

### 3.4 Rigorous Experimental Methodology & Benchmarking Protocol
1. **Memory Alignment**: All matrix buffers are dynamically allocated with 64-byte cache line alignment via `_aligned_malloc()` / `posix_memalign()`.
2. **Untimed Warmups**: Every tested configuration undergoes 1 warmup run to prime CPU frequency scaling and TLB tables.
3. **Statistical Filtering**: Each configuration is measured over 3–5 timed repetitions using `CLOCK_MONOTONIC` / `QueryPerformanceCounter` high-resolution timers, reporting the median runtime to eliminate OS noise.
4. **Boundary Robustness**: Matrices are clamped using dynamic min expressions $\min(ii + B, N)$ to ensure 100% correct execution for matrix dimensions that are not exact multiples of $B$.
5. **Numerical Verification**: Every computed output matrix is verified against a serial double-precision reference matrix:
   $$\max_{i,j} |C_{\text{test}}[i][j] - C_{\text{ref}}[i][j]| < 10^{-9}$$

---

## 4. Results & Analysis

### 4.1 Automated Unit Test Suite Validation
The complete C benchmark engine was evaluated against 81 test cases spanning matrix dimensions from $1 \times 1$ to $257 \times 257$ across all algorithmic variants:

```
=================================================================
        RUNNING MATRIX MULTIPLICATION UNIT TEST SUITE            
=================================================================
[PASS] Size    1x1    | Variant: ikj                  | Max Diff: 0.00e+00
[PASS] Size    1x1    | Variant: blocked (tile=8)     | Max Diff: 0.00e+00
[PASS] Size    1x1    | Variant: blocked (tile=16)    | Max Diff: 0.00e+00
[PASS] Size    1x1    | Variant: blocked (tile=32)    | Max Diff: 0.00e+00
[PASS] Size    1x1    | Variant: omp_ikj (threads=4)  | Max Diff: 0.00e+00
[PASS] Size    1x1    | Variant: omp_blocked (t=4,B=32)| Max Diff: 0.00e+00
[PASS] Size    1x1    | Variant: omp_collapse (t=4,B=32)| Max Diff: 0.00e+00
...
[PASS] Size  257x257  | Variant: ikj                  | Max Diff: 0.00e+00
[PASS] Size  257x257  | Variant: blocked (tile=64)    | Max Diff: 0.00e+00
[PASS] Size  257x257  | Variant: omp_collapse (t=4,B=32)| Max Diff: 0.00e+00
=================================================================
 UNIT TEST SUMMARY: Total: 81, Passed: 81, Failed: 0
 RESULT: ALL TESTS PASSED
=================================================================
```

---

### 4.2 Empirical Benchmark Performance ($N = 1024 \times 1024$)

The table below summarizes measured execution times, throughput in GFLOPS, speedup factors relative to the naive baseline, and parallel efficiency on the target Intel Core Ultra 5 225H processor (14 cores):

| Kernel Variant | Tile Size ($B$) | Threads ($T$) | Median Time (s) | Throughput (GFLOPS) | Speedup vs Naive | Speedup vs Serial IKJ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`naive_ijk`** | — | 1 | 3.7348 s | 0.575 GFLOPS | 1.00x | 0.09x |
| **`ikj` (Unit Stride)** | — | 1 | 0.3441 s | 6.241 GFLOPS | **10.85x** | 1.00x |
| **`blocked_serial`** | 16 | 1 | 0.4789 s | 4.484 GFLOPS | 7.80x | 0.72x |
| **`blocked_serial`** | 32 | 1 | 0.4190 s | 5.125 GFLOPS | 8.91x | 0.82x |
| **`blocked_serial`** | 64 | 1 | 0.3662 s | 5.864 GFLOPS | 10.20x | 0.94x |
| **`blocked_serial`** | **128** | 1 | **0.3392 s** | **6.331 GFLOPS** | **11.01x** | **1.01x** |
| **`omp_ikj`** | — | 4 | 0.1287 s | 16.685 GFLOPS | 29.02x | 2.67x |
| **`omp_ikj`** | — | 14 | 0.0519 s | 41.376 GFLOPS | 71.96x | 6.63x |
| **`omp_blocked`** | 64 | 4 | 0.1173 s | 18.307 GFLOPS | 31.84x | 2.93x |
| **`omp_blocked`** | 64 | 14 | 0.0460 s | 46.683 GFLOPS | 81.19x | 7.48x |
| **`omp_blocked_collapse`** | 64 | 4 | 0.1158 s | 18.544 GFLOPS | 32.25x | 2.97x |
| **`omp_blocked_collapse`** | **64** | **14** | **0.0410 s** | **52.376 GFLOPS** | **91.10x** | **8.39x** |

---

### 4.3 Key Observations & Visualizations

#### 1. Loop Reordering Impact ($ijk \to ikj$)
* Simply interchanging the inner loops from $ijk$ to $ikj$ reduced execution time from **3.735 seconds down to 0.344 seconds**, producing a massive **10.85x speedup** without any multithreading.
* This massive gain confirms that eliminating column strides and streaming contiguous 64-byte cache lines into AVX2 vector registers is the single most dominant single-core optimization.

#### 2. Tile Size Selection & Cache Hierarchy Bounds
* **L1D Bound ($B \le 32$)**: While $B=32$ ensures 100% L1D cache residency (24 KB footprint vs 48 KB L1D), the loop iteration overhead is higher and inner vector loops are short ($B=32$).
* **L2 Bound ($B = 64, 128$)**: $B=64$ (96 KB footprint) and $B=128$ (384 KB footprint) comfortably reside inside the 2.0 MB per-core L2 cache. This allows the compiler to generate longer unrolled SIMD loops while keeping memory accesses within L2 cache latency (sub-10 ns), yielding peak single-core and multi-core throughput.
* **Serial Peak**: Occurs at $B=128$ (6.33 GFLOPS).
* **Parallel Peak**: Occurs at $B=64$ with `collapse(2)` (52.38 GFLOPS).

#### 3. OpenMP Thread Scaling & `collapse(2)` Scalability
* For $T=14$ threads, `omp_blocked_collapse` achieves **52.38 GFLOPS** compared to **46.68 GFLOPS** for `omp_blocked` and **41.38 GFLOPS** for `omp_ikj`.
* Collapsing the two outer loops expands the iteration space from $\lceil 1024/64 \rceil = 16$ items to $16 \times 16 = 256$ items, ensuring balanced distribution across all 14 cores and eliminating thread starvation.

---

### 4.4 Generated Publication Charts & Visual Results

The empirical results and hardware interactions are visually depicted in the high-resolution charts below (stored in `figures/`):

#### Figure 1: Execution Time vs. Tile Size ($B = 16 \dots 128$)
![Execution Time vs Tile Size](figures/01_execution_time_tile_size.png)
*Shows execution time as a function of tile size $B$ across serial blocked, OpenMP blocked, and OpenMP blocked with `collapse(2)`. The curve illustrates the transition from loop overhead domination at small tiles ($B=16$) to cache-resident peak throughput ($B=64, 128$).*

---

#### Figure 2: Computational Throughput (GFLOPS) vs. Tile Size
![GFLOPS vs Tile Size](figures/02_gflops_tile_size.png)
*Demonstrates floating-point throughput across tile sizes. Peak parallel throughput of 52.38 GFLOPS is achieved at $B=64$ with `collapse(2)`.*

---

#### Figure 3: Execution Time Across Serial and OpenMP Variants
![Serial vs OpenMP](figures/03_serial_vs_openmp.png)
*Compares all 6 core algorithmic kernels, showing the 10.85x speedup from loop reordering ($ijk \to ikj$) and the further 8.39x scaling from multicore OpenMP parallelization.*

---

#### Figure 4: Parallel Speedup vs. Thread Count ($T = 1 \dots 14$)
![Thread Scaling](figures/04_thread_scaling.png)
*Illustrates strong scaling across physical processor cores against ideal linear speedup, demonstrating high parallel efficiency up to 14 threads.*

---

#### Figure 5: OpenMP `collapse(2)` vs. Non-Collapsed Outer Loop
![collapse(2) Comparison](figures/05_collapse_comparison.png)
*Visualizes the performance advantage of flattening the 2D tile iteration space, which avoids thread starvation when thread count approaches or exceeds $\lceil N/B \rceil$.*

---

#### Figure 6: Computational Throughput vs. Matrix Dimension ($N = 512, 1024$)
![Matrix Size Scaling](figures/06_matrix_size_scaling.png)
*Depicts algorithmic scaling across varying problem dimensions, highlighting sustained GFLOPS throughput as matrix dimensions scale.*

---

#### Figure 7: Empirical Throughput with Theoretical L1D and L2 Cache Boundaries
![Cache Tile Analysis](figures/07_cache_tile_analysis.png)
*Overlays actual measured throughput against the theoretical capacity boundaries for the per-core 48 KB L1D cache and 2.0 MB L2 cache.*

---

#### Figure 8: 2D Performance Heatmap (Tile Size $\times$ Thread Count)
![Tile-Thread Heatmap](figures/08_tile_thread_heatmap.png)
*A 2D intensity map visualizing the multi-parameter design space across thread counts ($T=1 \dots 14$) and tile sizes ($B=16 \dots 128$), highlighting the global optimal operating region.*

---

## 5. Conclusion

This academic micro-project investigated the empirical and theoretical principles of hardware cache locality, cache blocking, SIMD vectorization, and OpenMP multithreaded scalability on dense matrix multiplication.

### Summary of Key Findings:
1. **Locality Outweighs Raw Parallelism First**: Transforming loop traversal order from $ijk$ to $ikj$ produced a **10.85x speedup** purely by enabling continuous cache line streaming and SIMD auto-vectorization, proving that memory locality optimization must precede parallelization.
2. **Theoretical Cache Sizing vs Empirical Optimum**:
   - Theoretical 3-tile capacity analysis correctly identifies valid working-set ranges: $B \le 32$ for L1D (48 KB) and $B \le 256$ for L2 (2.0 MB).
   - Empirical measurements reveal that $B=64$ and $B=128$ provide the peak performance on modern x86-64 processors by balancing high L2 cache residency with maximized SIMD vector unrolling and minimal loop branch overhead.
3. **OpenMP `collapse(2)` Eliminates Thread Starvation**:
   - For high core counts ($T=14$), flattening the 2D tile iteration space with `#pragma omp parallel for collapse(2)` expanded available work items to $\lceil N/B \rceil^2$, preventing thread underutilization and achieving the overall peak throughput of **52.38 GFLOPS (91.10x total speedup over naive serial)**.
4. **Strict Reproducibility**: All 81 unit tests passed with double-precision accuracy ($\epsilon < 10^{-9}$), verifying that performance gains were achieved without sacrificing numerical correctness.

---
*Report generated from physical benchmark execution on Intel Core Ultra 5 225H with MinGW-W64 GCC 16.2.0.*
