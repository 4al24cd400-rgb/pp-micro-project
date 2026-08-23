# Empirical Experiment Notes & High-Performance Computing Observations

## 1. Loop-Order Memory Access Dynamics ($ijk$ vs. $ikj$)

Empirical benchmark measurements consistently reveal a massive performance divergence between the classical $ijk$ loop order and the optimized $ikj$ loop order:

* **Naive $ijk$**: Traverses matrix $B$ by columns ($B[k \cdot N + j]$ with stride $N$). For $N=1024$, the memory gap between successive inner loop steps is $1024 \times 8 = 8,192\text{ bytes}$. Because this far exceeds the 64-byte hardware cache line, nearly every memory request triggers a cache miss on large matrices.
* **Optimized $ikj$**: Pins $A[i \cdot N + k]$ in a CPU vector register while streaming continuous rows of $B[k \cdot N + j]$ and accumulating into continuous rows of $C[i \cdot N + j]$ with unit stride ($\Delta j = 1$). Each 64-byte line fetched from DRAM / L2 supplies 8 consecutive doubles, yielding dramatic throughput gains (~10x to 15x over naive $ijk$ in serial execution).

---

## 2. Tile-Size Optimization Spectrum ($B=8 \dots 256$)

1. **Very Small Tiles ($B=8, 16$)**:
   - Small tiles suffer from excessive tile loop branch overhead ($\approx (N/B)^3$ outer loop iterations).
   - Inner loops are too short for compiler loop unrolling and SIMD pipeline saturation.
2. **Medium Tiles ($B=32, 64$)**:
   - $B=32$ maintains strict residency within the 48 KB L1D cache ($24\text{ KB}$ working set).
   - $B=64$ achieves an excellent sweet spot by providing enough inner work ($64$ iterations) for 256-bit AVX2 / FMA auto-vectorization while staying well within the 2.0 MB per-core L2 cache ($96\text{ KB}$ working set).
3. **Large Tiles ($B=128, 192, 256$)**:
   - $B=128$ ($384\text{ KB}$ working set) fully amortizes loop overhead and drives peak vector execution.
   - For very large matrices, $B > 256$ starts encountering L2 capacity pressure and TLB thrashing.

---

## 3. OpenMP Parallelization & `collapse(2)` Concurrency Analysis

### 3.1 Loop Scheduling Granularity
* **Standard Parallel Blocked (`#pragma omp parallel for`)**:
  - Parallelizes the outermost $ii$ loop.
  - Number of parallel work units = $\lceil N / B \rceil$.
  - At $N=1024, B=128$, only 8 outer tile strips exist. On a 14-core machine, 6 cores receive zero work, causing thread under-utilization.
* **Collapsed Parallel Blocked (`#pragma omp parallel for collapse(2)`)**:
  - Combines the $(ii, jj)$ tile loops into a 2D tile iteration grid.
  - Number of parallel work units = $\lceil N / B \rceil^2$.
  - At $N=1024, B=128$, work units increase from 8 to $8 \times 8 = 64$ tiles, distributing ~4.5 tiles per thread across all 14 cores and eliminating thread starvation.

### 3.2 Concurrency & False Sharing Verification
* **Data Race Freedom**: In `collapse(2)` over $(ii, jj)$, each thread is granted exclusive write access to a disjoint sub-matrix $C[ii..i_{max}, jj..j_{max}]$. Accumulation along $kk$ occurs sequentially within each thread's local execution. Direct write data races are physically impossible.
* **Cache Line Sharing**: In continuous row-major memory, tile boundary columns (e.g. column $j=B-1$ and $j=B$) may occupy the same 64-byte line (8 doubles). When adjacent tiles are updated concurrently by different cores, minor cache line invalidations can occur. Static block chunk scheduling groups contiguous tiles per thread, minimizing inter-core boundary transitions.

---

## 4. OpenMP Scheduling Policy Evaluation (Static vs. Dynamic)

* **Static Scheduling (`schedule(static)`)**:
  - Deterministically partitions iteration space into contiguous chunks at compile/startup time.
  - Overhead per iteration is virtually zero.
  - Because dense matrix multiplication has uniform $O(B^3)$ computational work per tile, static scheduling provides optimal load balancing with zero runtime scheduling jitter.
* **Dynamic Scheduling (`schedule(dynamic)`)**:
  - Uses an atomic runtime work queue where idle threads request new chunks dynamically.
  - Incurs mutex/atomic contention on the shared work queue, adding runtime overhead with no load-balancing benefit for uniform dense matrices.
