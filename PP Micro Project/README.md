# Dense Matrix Multiplication with Cache Blocking — Tile Size Selection Tied to L1/L2 Cache Size, collapse() Decisions

[![Language: C11](https://img.shields.io/badge/Language-C11-blue.svg)](https://en.wikipedia.org/wiki/C11_(C_standard_revision))
[![Parallelism: OpenMP](https://img.shields.io/badge/Parallelism-OpenMP-green.svg)](https://www.openmp.org/)
[![Analysis: Python 3.11](https://img.shields.io/badge/Analysis-Python%203.11-yellow.svg)](https://www.python.org/)
[![Status: Academic Micro-Project](https://img.shields.io/badge/Status-Complete%20%26%20Reproducible-success.svg)]()

A high-performance computing (HPC) empirical micro-project investigating cache locality optimization, working-set capacity limits across the L1/L2 memory hierarchy, SIMD auto-vectorization, and thread-level scaling with OpenMP `collapse(2)` on dense matrix multiplication.

---

## 1. Project Overview

Matrix-matrix multiplication ($C = A \times B$) is a canonical memory-bound and compute-intensive operation. While mathematically simple, naive implementations suffer from severe memory stall penalties ("the Memory Wall"). This project demonstrates how:
1. **Loop reordering ($ijk \to ikj$)** transforms column-strided memory access into continuous row streaming, maximizing hardware cache line efficiency.
2. **Cache blocking (tiling)** confines the active computational working set into dedicated L1/L2 caches.
3. **OpenMP multithreading** scales throughput across multicore architectures.
4. **OpenMP `collapse(2)`** prevents thread starvation by expanding the outer iteration space into a 2D tile grid.

All performance data, speedups, GFLOPS, and charts are generated from actual physical execution on the target hardware.

---

## 2. Core Technologies

* **Core Kernel**: C (C11 standard) with 64-byte aligned dynamic memory allocation.
* **Compiler**: GCC 16.2.0 (MinGW-W64 UCRT with OpenMP, `-O3 -fopenmp -march=native`).
* **Parallel API**: OpenMP multi-threading with static/dynamic scheduling and `collapse(2)`.
* **Hardware Detection & Analysis**: Python 3.11 with `matplotlib`, `pandas`, `numpy`, and `psutil`.
* **Build System**: Cross-platform GNU Makefile / `mingw32-make`.

---

## 3. Directory Architecture

```
matrix-cache-blocking/
│
├── src/
│   ├── matrix.h           # Matrix allocation, alignment, deterministic initialization, kernels
│   ├── matrix.c           # Implementations: naive_ijk, ikj, blocked, omp_ikj, omp_blocked, omp_collapse
│   ├── benchmark.h        # High-resolution monotonic timers, statistics, CSV logger
│   ├── benchmark.c        # Benchmark execution loop, unit testing engine
│   └── main.c             # CLI options (--size, --tile, --threads, --variant, --schedule, --test)
│
├── scripts/
│   ├── detect_system.py   # Detects per-core L1D/L2, L3, CPU cores, OpenMP, and candidate models
│   ├── run_benchmarks.py  # Orchestrates staged benchmarking (--quick, --full, custom)
│   └── analyze_results.py # Data processing, statistical aggregation, and chart generation
│
├── results/
│   ├── raw/               # Raw timestamped CSV datasets
│   ├── processed/         # Aggregated summaries (analysis_summary.csv)
│   ├── graphs/            # High-resolution generated PNG figures (01 to 08)
│   ├── system_info.txt    # System architecture and detected cache report
│   └── FINAL_SUMMARY.md   # Measured peak results and academic summary
│
├── docs/
│   ├── methodology.md     # Comprehensive academic report and mathematical derivations
│   ├── cache_analysis.md  # Working-set capacity modeling and candidate tables
│   └── experiment_notes.md# Empirical observations, scheduling, and concurrency notes
│
├── report/
│   └── figures/           # Mirrored publication figures
│
├── Makefile               # Automated build rules (all, test, bench, quick, clean)
├── README.md              # Project documentation and guide
├── .gitignore             # Ignore compiled binaries and temp files
└── requirements.txt       # Python dependencies
```

---

## 4. How It Works: Cache Blocking & Loop Orders

### 4.1 Loop Order: Strided ($ijk$) vs Unit Stride ($ikj$)
* In **Naive $ijk$**, calculating $C[i][j]$ requires accessing column $j$ of matrix $B$ ($B[k \cdot N + j]$). For large $N$, each step $k$ jumps $N \times 8\text{ bytes}$ in memory, causing frequent cache misses.
* In **Optimized $ikj$**, swapping loops $k$ and $j$ holds $A[i][k]$ invariant in a CPU register while streaming contiguous rows of $B[k][j \dots j+N-1]$ into contiguous rows of $C[i][j \dots j+N-1]$. Every 64-byte cache line fetched provides 8 consecutive double-precision numbers.

### 4.2 Cache Blocking Formulation
For tile dimension $B \times B$:
* **Single tile footprint**: $B^2 \times 8\text{ bytes}$.
* **Active 3-tile working set**: $W_{3-tile} \approx 3 \times B^2 \times 8\text{ bytes} = 24 B^2\text{ bytes}$.
* Boundary handling via $\min(ii + B, N)$ guarantees arbitrary matrix dimensions ($N \neq k \cdot B$) compute correctly without memory padding.

---

## 5. OpenMP & `collapse(2)` Decisions

* **`omp_blocked`**: Parallelizes the outer $ii$ loop ($\lceil N/B \rceil$ iterations). At $N=1024, B=128$, only 8 work items exist, leaving 6 cores idle on a 14-core processor.
* **`omp_blocked_collapse`**: Uses `#pragma omp parallel for collapse(2)` over $(ii, jj)$ to flatten the iteration space into $\lceil N/B \rceil^2 = 64$ work items, achieving uniform load balance across all threads.
* **Data-Race Freedom**: Because each $(ii, jj)$ iteration writes to a mutually disjoint sub-matrix of $C$, there are no write data races.

---

## 6. Building & Running

### Step 1: Install Python Dependencies
```bash
python -m pip install -r requirements.txt
```

### Step 2: Compile the C Project
```bash
make all
```

### Step 3: Run the Automated Unit Test Suite (81 Tests)
```bash
./bin/matrix_bench --test
```

### Step 4: Run System Architecture & Cache Detection
```bash
python scripts/detect_system.py
```

### Step 5: Execute Benchmarks
* **Quick Mode (Validation)**:
  ```bash
  python scripts/run_benchmarks.py --quick
  ```
* **Full Staged Experiment Suite**:
  ```bash
  python scripts/run_benchmarks.py --full
  ```
* **Custom Execution**:
  ```bash
  ./bin/matrix_bench --size 1024 --tile 64 --threads 14 --variant omp_blocked
  ```

### Step 6: Process Results & Generate Graphs
```bash
python scripts/analyze_results.py
```

---

## 7. Generated Visualizations

All graphs are rendered in high-resolution (300 DPI) and stored in `report/figures/` and `results/graphs/`:
* `01_execution_time_tile_size.png`: Execution time vs tile size ($B=8 \dots 256$).
* `02_gflops_tile_size.png`: Computational throughput (GFLOPS) vs tile size.
* `03_serial_vs_openmp.png`: Execution time across serial naive, IKJ, blocked, and OpenMP kernels.
* `04_thread_scaling.png`: Parallel speedup vs thread count ($T=1 \dots 14$) with ideal linear scaling.
* `05_collapse_comparison.png`: Non-collapse vs `collapse(2)` across thread counts.
* `06_matrix_size_scaling.png`: Throughput vs matrix size ($N=512, 1024, 1536$).
* `07_cache_tile_analysis.png`: Empirical throughput with theoretical L1D and L2 capacity bounds visually highlighted.
* `08_tile_thread_heatmap.png`: 2D Performance Heatmap (Tile Size $\times$ Thread Count).

---

## 8. Summary of Findings

1. **Memory Locality Dominance**: Loop ordering ($ijk \to ikj$) produces an order-of-magnitude speedup (~12x) before parallelization by exploiting cache lines.
2. **Optimal Tile Sizing**: $B=64$ and $B=128$ provide the empirical peak performance on modern x86-64 processors, balancing L2 cache residency with loop vectorization efficiency.
3. **`collapse(2)` Scalability**: Collapsing $(ii, jj)$ eliminates thread starvation for large tiles and high core counts.

---

## 9. Reproducibility & Integrity

This micro-project enforces strict academic reproducibility. All timing measurements employ high-resolution monotonic clocks, untimed warmups, multiple repetitions with median filtering, and separate numerical error verification ($\epsilon < 10^{-9}$). No data is fabricated.
