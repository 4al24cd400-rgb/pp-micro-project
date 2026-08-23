# Academic Micro-Project Final Benchmark Summary

## Project: Dense Matrix Multiplication with Cache Blocking — Tile Size Selection Tied to L1/L2 Cache Size, collapse() Decisions

---

### 1. Hardware & Execution Environment
* **CPU Model**: Intel(R) Core(TM) Ultra 5 225H
* **Physical Cores**: 14
* **Logical Processors**: 14
* **L1 Data Cache (Per-Core)**: 48.0 KB
* **L2 Cache (Per-Core / Cluster)**: 2.0 MB (2048.0 KB)
* **L3 Cache (Shared LLC)**: 18.0 MB
* **Cache Line Size**: 64 bytes
* **Compiler**: GCC 16.2.0 (MinGW-W64 UCRT with OpenMP)
* **Dataset CSV**: `benchmark_20260823_233853.csv`

---

### 2. Measured Peak Performance & Best Configuration
* **Matrix Size Tested (N)**: 1024 x 1024
* **Optimal Tile Size (B)**: 64 x 64
* **Optimal Variant**: `omp_blocked_collapse`
* **Optimal Thread Count**: 14 Threads
* **Execution Time (Median)**: 0.041000 seconds
* **Peak Computational Throughput**: **52.38 GFLOPS**
* **Speedup vs Serial IKJ (T=1)**: **8.39x**
* **Correctness Status**: **PASS** (Numerical verification error < 1e-9)

---

### 3. Empirical Best Tile Sizes vs Theoretical Cache Models
* **Theoretical L1D Bound ($W_{3-tile} \le 48\text{ KB}$)**: Candidate tile sizes $B \in [8, 16, 32]$ ($B=32$ uses 24 KB, 50% L1D).
* **Theoretical L2 Bound ($W_{3-tile} \le 2\text{ MB}$)**: Candidate tile sizes $B \in [48, 64, 96, 128, 192, 256]$ ($B=64$ uses 96 KB; $B=128$ uses 384 KB).
* **Measured Best Serial Tile Size**: $B = 128$
* **Measured Best Parallel Tile Size**: $B = 64$

---

### 4. Key Findings & Research Questions Answered
1. **Loop-Order Locality**: Changing the loop order from $ijk$ to $ikj$ yields a dramatic throughput improvement because the inner loop steps across continuous contiguous memory addresses for both $B[k \cdot N + j]$ and $C[i \cdot N + j]$, fully exploiting 64-byte hardware cache lines (8 doubles per transaction).
2. **Cache-Aware Tile Selection**: Cache capacity analysis narrows candidate sizes to ranges that fit into L1D ($B \le 32$) and L2 ($B \le 256$). Hardware benchmarks reveal that while $B=32$ maximizes L1D residency, $B=64$ and $B=128$ achieve peak performance by balancing L1/L2 residency with loop overhead amortisation and compiler SIMD vectorization.
3. **OpenMP collapse(2) Impact**: Collapsing the two outer tile loops $(ii, jj)$ flattens the 2D tile grid into $\lceil N/B \rceil^2$ iterations. This provides superior thread work distribution and prevents thread starvation when the thread count is high relative to $\lceil N/B \rceil$.
4. **Distinction Between Theory & Measurement**: Theoretical calculations define memory capacity upper bounds, while empirical benchmarking captures hardware prefetcher dynamics, TLB hit rates, vector pipeline fill, and memory bus saturation.

---
*Generated automatically from actual physical hardware benchmark execution.*
