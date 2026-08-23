# Comprehensive Technical Report & Experimental Methodology

## Dense Matrix Multiplication with Cache Blocking — Tile Size Selection Tied to L1/L2 Cache Size, collapse() Decisions

---

## 1. Executive Summary & Research Formulation

Dense Matrix-Matrix Multiplication ($C = A \times B$) is a fundamental computational kernel in high-performance computing (HPC), numerical linear algebra, physical simulations, and machine learning workloads. For $N \times N$ dense matrices in double precision ($\mathbb{R}^{N \times N}$), the operation performs $2N^3$ floating-point operations over $3N^2 \times 8\text{ bytes}$ of data, yielding an algorithmic arithmetic intensity of:

$$\text{Arithmetic Intensity} = \frac{2N^3\text{ FLOPs}}{24N^2\text{ Bytes}} = \frac{N}{12}\text{ FLOPs/Byte}$$

Despite this theoretically high arithmetic intensity for large $N$, conventional naive implementations suffer severe memory-bandwidth bottlenecks ("the Memory Wall") due to poor spatial and temporal locality, low cache line utilization, and conflict misses in the CPU memory hierarchy.

### Primary Research Question
> **"How does cache-aware tile-size selection affect dense matrix multiplication performance, and how does OpenMP `collapse(2)` influence parallel scalability and cache locality across multicore processor architectures?"**

---

## 2. Theoretical Mathematical Models & Memory Hierarchy

### 2.1 Classical Matrix Multiplication ($C = A \times B$)
For matrices $A, B, C \in \mathbb{R}^{N \times N}$:

$$C_{i,j} = \sum_{k=0}^{N-1} A_{i,k} \cdot B_{k,j} \quad \text{for } 0 \le i, j < N$$

### 2.2 Memory Layout & Loop Order Formulations
All matrices are stored in 1D contiguous row-major format, aligned to 64-byte cache line boundaries:
* Element index: $\text{Offset}(i, j) = i \cdot N + j$

#### 1. Naive $ijk$ Loop Order
```c
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        double sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[i * N + k] * B[k * N + j]; // Strided access on B
        }
        C[i * N + j] = sum;
    }
}
```
* **Matrix $A$ Access**: Row-wise contiguous ($A[i \cdot N + k]$ steps by 1 element as $k$ increments).
* **Matrix $B$ Access**: Column-wise strided ($B[k \cdot N + j]$ steps by $N$ elements across successive $k$ iterations).
* **Cache Behavior**: When $N \times 8\text{ bytes} > \text{L1/L2 cache capacity}$, each inner iteration loads a distinct 64-byte cache line for $B$, utilizing only 1 of the 8 double-precision floats in each cache line before eviction, causing pervasive capacity and conflict misses.

#### 2. Loop-Order Optimized $ikj$ Loop Order
```c
for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
        double a_ik = A[i * N + k]; // Maintained in CPU register
        for (int j = 0; j < N; j++) {
            C[i * N + j] += a_ik * B[k * N + j]; // Unit stride streaming
        }
    }
}
```
* **Matrix $A$ Access**: $A[i \cdot N + k]$ is invariant in the innermost $j$ loop and is pinned inside a hardware register.
* **Matrix $B$ & $C$ Access**: Both $B[k \cdot N + j]$ and $C[i \cdot N + j]$ traverse contiguous memory addresses with unit stride ($\Delta j = 1$).
* **Cache Behavior**: Every 64-byte memory transaction brings 8 double-precision elements into L1D cache, yielding 87.5% spatial hit rate in the inner loop and enabling SIMD auto-vectorization (AVX2 / AVX-512).

---

## 3. Cache Blocking (Tiling) Architecture

Cache blocking partitions the 3D iteration space into $B \times B$ sub-matrices (tiles) such that the working set of active tiles remains resident in high-speed per-core cache memory throughout the inner computation.

```c
for (int ii = 0; ii < N; ii += B) {
    int i_max = MIN(ii + B, N);
    for (int kk = 0; kk < N; kk += B) {
        int k_max = MIN(kk + B, N);
        for (int jj = 0; jj < N; jj += B) {
            int j_max = MIN(jj + B, N);
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
```

### 3.1 Arbitrary Dimension Boundary Clamping
To ensure robust numerical computation without assuming $N \pmod B == 0$, boundary clamping functions $\min(ii + B, N)$ dynamically adjust edge tile dimensions, ensuring 100% numerical correctness for arbitrary matrix dimensions.

---

## 4. Cache-Aware Working-Set Model & Candidate Selection

### 4.1 Mathematical Formulation of Working Set
For double-precision floating-point ($8\text{ bytes/element}$) and square tile dimension $B \times B$:
* Footprint of a single tile: $W_{tile} = B^2 \times 8\text{ bytes}$
* **Three-Tile Active Working Set Model**:
  $$W_{3-tile} \approx A_{tile} + B_{tile} + C_{tile} = 3 \times B^2 \times 8\text{ bytes} = 24 B^2\text{ bytes}$$

### 4.2 Theoretical Working-Set Capacity vs Target Cache Levels

| Tile Size ($B$) | Single Tile Footprint | 3-Tile Working Set | % L1D (48 KB / core) | % L2 (2 MB / core) | Theoretical Category |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **8 $\times$ 8** | 0.50 KB | 1.50 KB | 3.1% | 0.07% | L1D Resident (Fits in L1D) |
| **16 $\times$ 16** | 2.00 KB | 6.00 KB | 12.5% | 0.29% | L1D Resident (Fits in L1D) |
| **32 $\times$ 32** | 8.00 KB | 24.00 KB | 50.0% | 1.17% | L1D Resident (Fits in L1D) |
| **48 $\times$ 48** | 18.00 KB | 54.00 KB | 112.5% | 2.64% | L2 Resident (Fits in L2) |
| **64 $\times$ 64** | 32.00 KB | 96.00 KB | 200.0% | 4.69% | L2 Resident (Fits in L2) |
| **96 $\times$ 96** | 72.00 KB | 216.00 KB | 450.0% | 10.55% | L2 Resident (Fits in L2) |
| **128 $\times$ 128** | 128.00 KB | 384.00 KB | 800.0% | 18.75% | L2 Resident (Fits in L2) |
| **192 $\times$ 192** | 288.00 KB | 864.00 KB | 1800.0% | 42.19% | L2 Resident (Fits in L2) |
| **256 $\times$ 256** | 512.00 KB | 1536.00 KB | 3200.0% | 75.00% | L2 Resident (Fits in L2) |

> [!NOTE]
> **Theoretical Candidate Formulation**:
> 1. Maximum L1D Bound ($W_{3-tile} \le 48\text{ KB}$): $B_{max, L1} = \lfloor \sqrt{48 \times 1024 / 24} \rfloor = 45 \implies B \in \{16, 32\}$.
> 2. Maximum L2 Bound ($W_{3-tile} \le 2048\text{ KB}$): $B_{max, L2} = \lfloor \sqrt{2048 \times 1024 / 24} \rfloor = 295 \implies B \in \{64, 96, 128, 192, 256\}$.

### 4.3 Why Theoretical Suitability $\neq$ Empirical Optimum
Theoretical cache capacity analysis defines the maximum upper bounds of working sets. However, the experimentally fastest tile size is determined by complex microarchitectural interactions:
1. **Cache Associativity & Conflict Misses**: Real caches (e.g. 12-way or 16-way set-associative) experience conflict misses if matrix row strides map to the same cache sets.
2. **Hardware Stream Prefetchers**: Modern CPUs detect sequential row streaming across $B$ and $C$ and prefetch lines into L1D/L2 before execution, reducing penalty for larger tiles.
3. **SIMD Vectorization & Loop Overhead**: Larger tile dimensions ($B=64, 128$) amortize tile loop branch overheads and allow compiler vectorization pipelines (FMA, AVX2) to run with maximum instruction throughput.
4. **Register Blocking**: The compiler schedules register allocation for unrolled micro-kernels within tiles.

---

## 5. OpenMP Multithreading & `collapse(2)` Analysis

### 5.1 Directive Formulations
1. **Outer Loop Parallelism (`omp_blocked`)**:
   ```c
   #pragma omp parallel for num_threads(T) schedule(static)
   for (int ii = 0; ii < N; ii += B) { ... }
   ```
   * Total loop iterations: $K_1 = \lceil N / B \rceil$.
   * When $N=1024, B=128$, total iterations $K_1 = 8$. For thread count $T=14$, only 8 threads receive work while 6 threads remain idle (thread starvation).

2. **2D Grid Collapsed Parallelism (`omp_blocked_collapse`)**:
   ```c
   #pragma omp parallel for collapse(2) num_threads(T) schedule(static)
   for (int ii = 0; ii < N; ii += B) {
       for (int jj = 0; jj < N; jj += B) {
           for (int kk = 0; kk < N; kk += B) { ... }
       }
   }
   ```
   * Total loop iterations: $K_2 = \lceil N / B \rceil^2$.
   * When $N=1024, B=128$, total iterations $K_2 = 8 \times 8 = 64$. All 14 threads receive balanced chunks ($64 / 14 \approx 4.5$ tiles/thread).

### 5.2 Concurrency, Data Races, and False Sharing Analysis
* **Data Races**: In `collapse(2)` over $(ii, jj)$, each thread is assigned a distinct output tile $C_{ii, jj}$. The accumulation over $kk$ runs serially within each thread. Because no two threads ever write to the same element of $C$, there are **ZERO data races**.
* **Cache-Line Sharing vs False Sharing**:
  - Distinct tiles own distinct matrix rows and columns.
  - However, in row-major layout, the end of one tile row and the start of an adjacent tile row within the same horizontal band can share a single 64-byte cache line (8 doubles).
  - When threads update adjacent tiles concurrently, cache coherence protocols (MESI/MOESI) may invalidate shared lines. Aligning tile row allocations and scheduling contiguous tile blocks mitigate this overhead.

---

## 6. Experimental Verification & Benchmarking Protocol

1. **Hardware Environment**:
   - Processor: Intel Core Ultra 5 225H (14 Cores / 14 Threads)
   - Cache Hierarchy: 48 KB L1D / core, 2 MB L2 / core, 18 MB shared L3
   - OS: Windows 11 (x86_64)
   - Compiler: GCC 16.2.0 (MinGW-W64 UCRT, `-O3 -fopenmp -march=native`)
2. **Correctness Validation**:
   - 81 automated unit tests across dimensions $1 \times 1$ to $257 \times 257$ and tile sizes $8$ to $64$.
   - Max absolute error threshold: $\epsilon < 10^{-9}$.
3. **Monotonic Timing Protocol**:
   - Monotonic high-resolution timer (`omp_get_wtime()`).
   - Untimed warmup iterations to initialize thread pools and pre-populate TLB/caches.
   - 5 timed repetitions per configuration.
   - Reported metrics: Minimum, Mean, Median, Maximum, Standard Deviation, GFLOPS, and Speedup.

---

## 7. Mathematical Performance Metrics

* **Execution Time ($T$)**: Median wall-clock duration of timed kernel execution in seconds.
* **Speedup ($S$)**:
  $$S = \frac{T_{\text{baseline}}}{T_{\text{variant}}}$$
* **Floating-Point Operations per Second (GFLOPS)**:
  $$\text{GFLOPS} = \frac{2 \times N^3}{T_{\text{median}} \times 10^9}$$

---

## 8. Limitations & Future Work

* **Hardware Heterogeneity**: Modern hybrid architectures (e.g. Intel Core Ultra with P-cores, E-cores, and LP E-cores) exhibit non-uniform core frequencies and cache sharing configurations.
* **NUMA & Memory Bandwidth**: For multi-socket systems, non-uniform memory access (NUMA) first-touch policies become dominant.
* **Future Work**: Register-level micro-kernel packing (BLIS-style multi-level cache packing) and GPU offloading (OpenMP target / CUDA).
