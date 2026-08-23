#!/usr/bin/env python3
"""
run_benchmarks.py - Automated Benchmark Orchestration Engine
Executes staged experiments (Tile Size Exploration, Thread Scaling, collapse(2) Comparison,
and Scheduling) with rigorous correctness verification, monotonic timing, and structured CSV logging.
"""

import sys
import os
import subprocess
import argparse
import time
import datetime
import glob

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN_DIR = os.path.join(PROJECT_ROOT, "bin")
RAW_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "raw")
EXECUTABLE_NAME = "matrix_bench.exe" if os.name == "nt" else "matrix_bench"
EXECUTABLE_PATH = os.path.join(BIN_DIR, EXECUTABLE_NAME)

def ensure_compiled():
    """Ensures matrix_bench executable is built and up to date."""
    if not os.path.exists(EXECUTABLE_PATH):
        print(f"[*] Executable not found at {EXECUTABLE_PATH}. Building project...")
        # Check for make
        make_cmd = "make"
        if os.name == "nt":
            if os.path.exists("C:\\mingw64\\bin\\make.exe"):
                make_cmd = "C:\\mingw64\\bin\\make.exe"
            elif os.path.exists("C:\\mingw64\\bin\\mingw32-make.exe"):
                make_cmd = "C:\\mingw64\\bin\\mingw32-make.exe"
        
        env = os.environ.copy()
        if os.name == "nt" and os.path.exists("C:\\mingw64\\bin"):
            env["PATH"] = "C:\\mingw64\\bin;" + env.get("PATH", "")
        
        res = subprocess.run([make_cmd, "all"], cwd=PROJECT_ROOT, env=env)
        if res.returncode != 0 or not os.path.exists(EXECUTABLE_PATH):
            print(f"[!] Compilation failed. Cannot proceed with benchmarking.", file=sys.stderr)
            sys.exit(1)
        print("[+] Compilation successful.")

def run_single_config(size, tile, threads, variant, sched, reps, csv_path, verify=True):
    """Invokes matrix_bench with specific parameter configurations."""
    env = os.environ.copy()
    if os.name == "nt" and os.path.exists("C:\\mingw64\\bin"):
        env["PATH"] = "C:\\mingw64\\bin;" + env.get("PATH", "")

    args = [
        EXECUTABLE_PATH,
        "--size", str(size),
        "--tile", str(tile),
        "--threads", str(threads),
        "--variant", str(variant),
        "--schedule", str(sched),
        "--repetitions", str(reps),
        "--warmup", "1",
        "--csv", csv_path
    ]
    if not verify:
        args.append("--no-verify")

    res = subprocess.run(args, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"\n[!] Error executing benchmark: {' '.join(args)}", file=sys.stderr)
        print(res.stderr, file=sys.stderr)
        return False
    return True

def run_quick_suite(csv_path):
    """Quick mode: Fast validation of all kernels, key tile sizes, and thread counts."""
    print("=" * 80)
    print("              STARTING QUICK BENCHMARK SUITE (PIPELINE VERIFICATION)          ")
    print("=" * 80)

    # 1. Run unit test suite
    print("[*] Step 1: Running unit tests...")
    env = os.environ.copy()
    if os.name == "nt" and os.path.exists("C:\\mingw64\\bin"):
        env["PATH"] = "C:\\mingw64\\bin;" + env.get("PATH", "")
    test_res = subprocess.run([EXECUTABLE_PATH, "--test"], cwd=PROJECT_ROOT, env=env)
    if test_res.returncode != 0:
        print("[!] Unit tests failed! Aborting benchmark suite.", file=sys.stderr)
        sys.exit(1)
    print("[+] All unit tests passed!\n")

    # 2. Quick benchmark configurations
    configs = []
    # Baseline serial naive (only N=512 to save time)
    configs.append((512, 64, 1, "naive_ijk", "static", 3))
    configs.append((512, 64, 1, "ikj", "static", 3))
    configs.append((1024, 64, 1, "ikj", "static", 3))

    # Tile size variations (N=1024, threads=4)
    for tile in [16, 32, 64, 128]:
        configs.append((1024, tile, 1, "blocked_serial", "static", 3))
        configs.append((1024, tile, 4, "omp_blocked", "static", 3))
        configs.append((1024, tile, 4, "omp_blocked_collapse", "static", 3))

    # Thread scaling (N=1024, tile=64)
    for threads in [1, 2, 4, 8, 14]:
        configs.append((1024, 64, threads, "omp_ikj", "static", 3))
        configs.append((1024, 64, threads, "omp_blocked", "static", 3))
        configs.append((1024, 64, threads, "omp_blocked_collapse", "static", 3))

    total = len(configs)
    start_time = time.time()
    for idx, (size, tile, threads, variant, sched, reps) in enumerate(configs, 1):
        print(f"[{idx:2d}/{total:2d}] Running: N={size:<4} Tile={tile:<3} Threads={threads:<2} Var={variant:<20} Sched={sched} ...", end="", flush=True)
        t0 = time.time()
        ok = run_single_config(size, tile, threads, variant, sched, reps, csv_path)
        t_elapsed = time.time() - t0
        if ok:
            print(f" [DONE ({t_elapsed:.2f}s)]")
        else:
            print(f" [FAILED]")

    total_time = time.time() - start_time
    print(f"\n[+] Quick benchmark suite completed in {total_time:.2f} seconds.")
    print(f"[+] Structured results saved to: {csv_path}")

def run_full_suite(csv_path):
    """Full mode: Staged experimentation covering Phases A, B, C, and D."""
    print("=" * 80)
    print("             STARTING COMPREHENSIVE STAGED EXPERIMENT SUITE                   ")
    print("=" * 80)

    # 1. Run unit tests first
    print("[*] Validating correctness with unit test suite...")
    env = os.environ.copy()
    if os.name == "nt" and os.path.exists("C:\\mingw64\\bin"):
        env["PATH"] = "C:\\mingw64\\bin;" + env.get("PATH", "")
    test_res = subprocess.run([EXECUTABLE_PATH, "--test"], cwd=PROJECT_ROOT, env=env)
    if test_res.returncode != 0:
        print("[!] Unit tests failed! Aborting benchmark suite.", file=sys.stderr)
        sys.exit(1)
    print("[+] All unit tests passed!\n")

    configs = []

    # ---------------------------------------------------------
    # Baseline Reference Measurements
    # ---------------------------------------------------------
    for n in [512, 1024]:
        # Run naive_ijk for baseline reference comparison
        configs.append((n, 64, 1, "naive_ijk", "static", 3))
        configs.append((n, 64, 1, "ikj", "static", 5))
    configs.append((1536, 64, 1, "ikj", "static", 3))

    # ---------------------------------------------------------
    # Phase A — Tile-Size Exploration (N=1024)
    # ---------------------------------------------------------
    tile_sizes = [8, 16, 32, 48, 64, 96, 128, 192, 256]
    for tile in tile_sizes:
        # Serial blocked
        configs.append((1024, tile, 1, "blocked_serial", "static", 5))
        # Parallel blocked (threads=4 and threads=14)
        for threads in [4, 14]:
            configs.append((1024, tile, threads, "omp_blocked", "static", 5))
            configs.append((1024, tile, threads, "omp_blocked_collapse", "static", 5))

    # ---------------------------------------------------------
    # Phase B — Thread Scaling & Matrix Dimension Scaling
    # ---------------------------------------------------------
    matrix_sizes = [512, 1024, 1536]
    threads_list = [1, 2, 4, 8, 14]
    key_tiles = [32, 64, 128]

    for n in matrix_sizes:
        for t in threads_list:
            # Parallel IKJ
            configs.append((n, 64, t, "omp_ikj", "static", 5 if n <= 1024 else 3))
            # Blocked with key tiles
            for b in key_tiles:
                configs.append((n, b, t, "omp_blocked", "static", 5 if n <= 1024 else 3))

    # ---------------------------------------------------------
    # Phase C — collapse(2) In-Depth Comparison
    # ---------------------------------------------------------
    for n in [512, 1024, 1536]:
        for b in [32, 64, 128]:
            for t in [1, 2, 4, 8, 14]:
                # Add collapse(2) counter-parts for identical conditions
                configs.append((n, b, t, "omp_blocked_collapse", "static", 5 if n <= 1024 else 3))

    # ---------------------------------------------------------
    # Phase D — Scheduling Evaluation (Static vs Dynamic)
    # ---------------------------------------------------------
    for sched in ["static", "dynamic"]:
        for t in [4, 14]:
            for var in ["omp_blocked", "omp_blocked_collapse"]:
                configs.append((1024, 64, t, var, sched, 5))

    # Remove duplicates preserving order
    unique_configs = []
    seen = set()
    for c in configs:
        if c not in seen:
            seen.add(c)
            unique_configs.append(c)

    total = len(unique_configs)
    print(f"[*] Total benchmark configurations to execute: {total}")
    start_time = time.time()

    for idx, (size, tile, threads, variant, sched, reps) in enumerate(unique_configs, 1):
        print(f"[{idx:3d}/{total:3d}] N={size:<4} Tile={tile:<3} Threads={threads:<2} Var={variant:<20} Sched={sched:<7} ...", end="", flush=True)
        t0 = time.time()
        ok = run_single_config(size, tile, threads, variant, sched, reps, csv_path, verify=(idx <= 10 or size <= 512))
        t_elapsed = time.time() - t0
        if ok:
            print(f" [DONE ({t_elapsed:.2f}s)]")
        else:
            print(f" [FAILED]")

    total_time = time.time() - start_time
    print(f"\n[+] Full benchmark suite completed in {total_time:.2f} seconds ({total_time/60.0:.2f} minutes).")
    print(f"[+] Structured results saved to: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Matrix Multiplication Benchmark Runner")
    parser.add_argument("--quick", action="store_true", help="Run fast verification benchmark suite")
    parser.add_argument("--full", action="store_true", help="Run full staged experiment suite")
    parser.add_argument("--sizes", nargs="+", type=int, help="Custom matrix sizes (e.g. --sizes 512 1024)")
    parser.add_argument("--tiles", nargs="+", type=int, help="Custom tile sizes (e.g. --tiles 32 64 128)")
    parser.add_argument("--threads", nargs="+", type=int, help="Custom thread counts (e.g. --threads 1 2 4 8 14)")
    parser.add_argument("--variants", nargs="+", type=str, help="Custom variants to test")
    parser.add_argument("--repetitions", type=int, default=5, help="Number of repetitions per config")
    parser.add_argument("--output", type=str, help="Custom CSV output path")
    args = parser.parse_args()

    ensure_compiled()
    os.makedirs(RAW_RESULTS_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        csv_path = args.output
    else:
        csv_path = os.path.join(RAW_RESULTS_DIR, f"benchmark_{timestamp}.csv")

    latest_csv = os.path.join(RAW_RESULTS_DIR, "latest_benchmark.csv")

    if args.full:
        run_full_suite(csv_path)
    elif args.quick or (not args.sizes and not args.tiles):
        run_quick_suite(csv_path)
    else:
        # Custom runs
        sizes = args.sizes or [1024]
        tiles = args.tiles or [64]
        threads_list = args.threads or [4]
        variants = args.variants or ["blocked_serial", "omp_blocked", "omp_blocked_collapse"]
        reps = args.repetitions

        total = len(sizes) * len(tiles) * len(threads_list) * len(variants)
        print(f"[*] Executing {total} custom configurations...")
        idx = 1
        for s in sizes:
            for b in tiles:
                for t in threads_list:
                    for v in variants:
                        print(f"[{idx}/{total}] N={s} B={b} T={t} V={v} ...", end="", flush=True)
                        ok = run_single_config(s, b, t, v, "static", reps, csv_path)
                        print(" [DONE]" if ok else " [FAILED]")
                        idx += 1

    # Update latest_benchmark.csv copy
    try:
        import shutil
        shutil.copyfile(csv_path, latest_csv)
    except Exception:
        pass

    return 0

if __name__ == "__main__":
    sys.exit(main())
