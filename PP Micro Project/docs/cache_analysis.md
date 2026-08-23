# Cache Hierarchy & Mathematical Working-Set Analysis

## 1. Physical Hardware Cache Hierarchy Detected

The experiments were executed on an Intel Core Ultra 5 225H processor featuring a hybrid heterogeneous architecture:

| Cache Level | Scope | Capacity (Detected) | Associativity / Line Size | Function & Latency |
| :---: | :---: | :---: | :---: | :---: |
| **L1 Data (L1D)** | Dedicated Per-Core | **48 KB** (P-Core) / **32 KB** (E-Core) | 12-way / 64-byte | Ultra-low latency (~4-5 cycles), private execution |
| **L1 Instruction (L1I)**| Dedicated Per-Core | **64 KB** | 8-way / 64-byte | Direct instruction pipeline fetch |
| **L1 Total Aggregate** | Across 14 Cores | **1,408 KB (1.4 MB)** | Multi-core sum | Cumulative L1 across all active CPU execution units |
| **L2 Unified** | Dedicated Per P-Core / Cluster | **2,048 KB (2.0 MB)** | 16-way / 64-byte | Mid-level cache (~14 cycles), non-inclusive |
| **L2 Total Aggregate** | Across All Clusters | **22,528 KB (22.0 MB)**| Multi-core sum | Cumulative L2 cache across all execution clusters |
| **L3 Shared (LLC)** | Shared Across Cores | **18,432 KB (18.0 MB)**| 12-way / 64-byte | Shared Last-Level Cache (~50-60 cycles) |
| **System Memory (DRAM)**| System Global | **16.0+ GB** | 64-byte bus width | Off-chip main memory (~100-200 cycles) |

> [!IMPORTANT]
> **Per-Core vs. Aggregate Distinction**:
> OpenMP work loops distribute tile blocks across separate CPU threads. Each thread executes independently on a single core (or execution unit) and is directly constrained by the **per-core L1D (48 KB)** and **per-core L2 (2.0 MB)** capacities rather than the aggregate sum. All primary working-set calculations are strictly evaluated against per-core capacities.

---

## 2. Working Set Formulations for Blocked Matrix Multiplication

For a double-precision floating-point kernel computing $C_{ii..i_{max}, jj..j_{max}} += A_{ii..i_{max}, kk..k_{max}} \times B_{kk..k_{max}, jj..j_{max}}$ with tile dimension $B$:

### 2.1 Three-Tile Working Set Formulation ($W_{3-tile}$)
$$\text{Memory per double element} = 8\text{ bytes}$$
$$\text{Footprint of one square tile } (B \times B) = B^2 \times 8\text{ bytes}$$
$$W_{3-tile} \approx \text{Footprint}(A_{tile}) + \text{Footprint}(B_{tile}) + \text{Footprint}(C_{tile}) = 3 \times B^2 \times 8\text{ bytes} = 24 B^2\text{ bytes}$$

### 2.2 Streaming vs Resident Tile Working Set ($W_{stream}$)
In our inner $ikj$ micro-kernel:
* $A[i \cdot N + k]$ is scalar in register ($8\text{ bytes}$).
* $C[i \cdot N + jj..j_{max}]$ is a single row segment of $C$ ($8 B\text{ bytes}$).
* $B[k \cdot N + jj..j_{max}]$ is a single row segment of $B$ ($8 B\text{ bytes}$).
* The full tile of $B$ ($B^2 \times 8\text{ bytes}$) is reused across all $i \in [ii, ii+B-1]$ iterations.
* Therefore, keeping $B_{tile}$ resident in L1/L2 cache while streaming rows of $A$ and $C$ requires:
  $$W_{stream} \approx B^2 \times 8 + 2 \times B \times 8\text{ bytes} = 8(B^2 + 2B)\text{ bytes}$$

---

## 3. Mathematical Candidate Evaluation Table

| Tile Size ($B$) | Single Tile ($8 B^2$) | 3-Tile Working Set ($24 B^2$) | % of L1D (48 KB) | % of L2 (2 MB) | Cache Fit Classification |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **8** | 0.50 KB | 1.50 KB | 3.1 % | 0.07 % | Fits L1D with extensive headroom |
| **16** | 2.00 KB | 6.00 KB | 12.5 % | 0.29 % | Fits L1D comfortably |
| **32** | 8.00 KB | 24.00 KB | 50.0 % | 1.17 % | **Optimal L1D Candidate** (50% headroom for line alignment & prefetcher) |
| **48** | 18.00 KB | 54.00 KB | 112.5 % | 2.64 % | Borderline L1D / Fits L2 |
| **64** | 32.00 KB | 96.00 KB | 200.0 % | 4.69 % | **Optimal L2 Candidate** (Balanced SIMD vectorization & L2 residency) |
| **96** | 72.00 KB | 216.00 KB | 450.0 % | 10.55 % | Fits L2 |
| **128** | 128.00 KB | 384.00 KB | 800.0 % | 18.75 % | **High-Throughput L2 Candidate** (Maximizes FMA pipeline utilization) |
| **192** | 288.00 KB | 864.00 KB | 1,800.0 % | 42.19 % | Fits L2 |
| **256** | 512.00 KB | 1,536.00 KB | 3,200.0 % | 75.00 % | Near L2 Capacity limit |
| **384** | 1,152.00 KB | 3,456.00 KB | 7,200.0 % | 168.75 % | Exceeds L2 / Spills to L3 LLC |
| **512** | 2,048.00 KB | 6,144.00 KB | 12,800.0 % | 300.00 % | Exceeds L2 / Heavy DRAM traffic |

---

## 4. Synthesis: Theoretical Bounds vs Empirical Reality

1. **Why $B=32$ is Theoretically Ideal for L1D**:
   At $B=32$, the active 3-tile working set is $24\text{ KB}$, occupying exactly 50% of the 48 KB L1D cache. The remaining 24 KB provides buffer space for hardware stream prefetching lines and avoids cache conflict thrashing across set-associative ways.
2. **Why $B=64$ and $B=128$ Win Empirically on Modern x86-64 CPUs**:
   - Modern AVX2 registers process 4 doubles (256 bits) per vector instruction with fused multiply-accumulate (FMA) latency of 4 cycles and throughput of 0.5 cycles.
   - Larger tiles ($B=64, 128$) allow the GCC loop vectorizer to completely unroll inner loops by 4x or 8x, filling execution ports with zero register stalls.
   - The 2.0 MB per-core L2 cache easily holds the 96 KB to 384 KB working sets of $B=64$ and $B=128$, achieving near-L1 access speeds due to non-blocking out-of-order L2 hits.
