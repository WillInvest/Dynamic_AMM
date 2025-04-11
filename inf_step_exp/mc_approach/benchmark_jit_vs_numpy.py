import numpy as np
from numba import jit, prange, config
import time
from tqdm import tqdm
import multiprocessing
import os

# Print CPU information
def print_cpu_info():
    physical_cores = multiprocessing.cpu_count()
    try:
        # On macOS and Linux
        logical_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        # Fallback for other systems
        logical_cores = physical_cores
    
    print(f"\nCPU Information:")
    print(f"Physical cores: {physical_cores}")
    print(f"Logical cores: {logical_cores}")
    print(f"Numba thread count: {config.NUMBA_DEFAULT_NUM_THREADS}")

# JIT version with loops
@jit(nopython=True)
def distribute_case_jit(x_dis, y_dis, prices, gammas, L):
    x_new = x_dis.copy()
    y_new = y_dis.copy()
    fees_inc = np.zeros_like(x_dis)
    fees_out = np.zeros_like(x_dis)
    
    for i in range(prices.shape[0]):
        for j in range(prices.shape[1]):
            for k in range(len(gammas)):
                price = prices[i, j]
                gamma = gammas[k]
                price_ratio = y_dis[i, j, k] / x_dis[i, j, k]
                
                if price > price_ratio / (1-gamma):
                    x_new[i,j,k] = L / np.sqrt((1-gamma)*price)
                    y_new[i,j,k] = L*np.sqrt((1-gamma)*price)
                    fees_inc[i,j,k] = (gamma/(1-gamma)) * (L*np.sqrt((1-gamma)*price)-y_dis[i,j,k])
                    fees_out[i,j,k] = gamma * price * (x_dis[i,j,k] - L/np.sqrt((1-gamma)*price))
    
    return x_new, y_new, fees_inc, fees_out

# Parallel JIT version
@jit(nopython=True, parallel=True)
def distribute_case_parallel(x_dis, y_dis, prices, gammas, L):
    x_new = x_dis.copy()
    y_new = y_dis.copy()
    fees_inc = np.zeros_like(x_dis)
    fees_out = np.zeros_like(x_dis)
    
    # Use prange for parallel execution of the outer loop
    for i in prange(prices.shape[0]):
        for j in range(prices.shape[1]):
            for k in range(len(gammas)):
                price = prices[i, j]
                gamma = gammas[k]
                price_ratio = y_dis[i, j, k] / x_dis[i, j, k]
                
                if price > price_ratio / (1-gamma):
                    x_new[i,j,k] = L / np.sqrt((1-gamma)*price)
                    y_new[i,j,k] = L*np.sqrt((1-gamma)*price)
                    fees_inc[i,j,k] = (gamma/(1-gamma)) * (L*np.sqrt((1-gamma)*price)-y_dis[i,j,k])
                    fees_out[i,j,k] = gamma * price * (x_dis[i,j,k] - L/np.sqrt((1-gamma)*price))
    
    return x_new, y_new, fees_inc, fees_out

# NumPy vectorized version
def distribute_case_numpy(x_dis, y_dis, prices, gammas, L):
    # Expand dimensions for broadcasting
    prices_expanded = np.tile(prices[:, :, np.newaxis], (1, 1, len(gammas)))
    gammas_expanded = np.tile(gammas[np.newaxis, np.newaxis, :], (prices.shape[0], prices.shape[1], 1))
    
    # Calculate masks
    price_ratio = y_dis / x_dis
    upper_mask = prices_expanded > price_ratio / (1-gammas_expanded)
    
    # Initialize output arrays
    x_new = x_dis.copy()
    y_new = y_dis.copy()
    fees_inc = np.zeros_like(x_dis)
    fees_out = np.zeros_like(x_dis)
    
    # Update upper case
    x_new[upper_mask] = L / np.sqrt((1-gammas_expanded[upper_mask])*prices_expanded[upper_mask])
    y_new[upper_mask] = L*np.sqrt((1-gammas_expanded[upper_mask])*prices_expanded[upper_mask])
    fees_inc[upper_mask] = (gammas_expanded[upper_mask]/(1-gammas_expanded[upper_mask])) * \
        (L*np.sqrt((1-gammas_expanded[upper_mask])*prices_expanded[upper_mask])-y_dis[upper_mask])
    fees_out[upper_mask] = gammas_expanded[upper_mask] * prices_expanded[upper_mask] * \
        (x_dis[upper_mask] - L/np.sqrt((1-gammas_expanded[upper_mask])*prices_expanded[upper_mask]))
    
    return x_new, y_new, fees_inc, fees_out

def run_benchmark(num_seeds=1000000, num_sigmas=5, num_gammas=20, num_runs=5):
    # Print CPU information first
    print_cpu_info()
    
    print(f"\nRunning benchmark with:")
    print(f"- {num_seeds} seeds")
    print(f"- {num_sigmas} sigma values")
    print(f"- {num_gammas} gamma values")
    print(f"- {num_runs} runs each")
    print("\nInitializing data...")
    
    # Initialize test data
    x_dis = np.ones((num_seeds, num_sigmas, num_gammas)) * 1000
    y_dis = np.ones((num_seeds, num_sigmas, num_gammas)) * 1000
    prices = np.random.lognormal(0, 0.1, (num_seeds, num_sigmas))
    gammas = np.linspace(0.0005, 0.01, num_gammas)
    L = np.sqrt(1000 * 1000)
    
    # Warm up JIT
    print("Warming up JIT...")
    _ = distribute_case_jit(x_dis[:2], y_dis[:2], prices[:2], gammas[:2], L)
    _ = distribute_case_parallel(x_dis[:2], y_dis[:2], prices[:2], gammas[:2], L)
    
    # Benchmark JIT version
    print("\nBenchmarking JIT version...")
    jit_times = []
    for i in tqdm(range(num_runs)):
        start = time.time()
        x_new_jit, y_new_jit, fees_inc_jit, fees_out_jit = distribute_case_jit(x_dis, y_dis, prices, gammas, L)
        jit_times.append(time.time() - start)
    
    # Benchmark Parallel JIT version
    print("\nBenchmarking Parallel JIT version...")
    parallel_times = []
    for i in tqdm(range(num_runs)):
        start = time.time()
        x_new_par, y_new_par, fees_inc_par, fees_out_par = distribute_case_parallel(x_dis, y_dis, prices, gammas, L)
        parallel_times.append(time.time() - start)
    
    # Benchmark NumPy version
    print("\nBenchmarking NumPy version...")
    numpy_times = []
    for i in tqdm(range(num_runs)):
        start = time.time()
        x_new_np, y_new_np, fees_inc_np, fees_out_np = distribute_case_numpy(x_dis, y_dis, prices, gammas, L)
        numpy_times.append(time.time() - start)
    
    # Print results
    print("\nResults:")
    print(f"JIT average time: {np.mean(jit_times):.4f} seconds")
    print(f"Parallel JIT average time: {np.mean(parallel_times):.4f} seconds")
    print(f"NumPy average time: {np.mean(numpy_times):.4f} seconds")
    print(f"\nSpeedup ratios:")
    print(f"JIT vs NumPy: {np.mean(numpy_times)/np.mean(jit_times):.2f}x")
    print(f"Parallel JIT vs NumPy: {np.mean(numpy_times)/np.mean(parallel_times):.2f}x")
    print(f"Parallel JIT vs Serial JIT: {np.mean(jit_times)/np.mean(parallel_times):.2f}x")
    
    # Verify results match
    print("\nVerifying results match...")
    x_new_jit, y_new_jit, fees_inc_jit, fees_out_jit = distribute_case_jit(x_dis, y_dis, prices, gammas, L)
    x_new_par, y_new_par, fees_inc_par, fees_out_par = distribute_case_parallel(x_dis, y_dis, prices, gammas, L)
    x_new_np, y_new_np, fees_inc_np, fees_out_np = distribute_case_numpy(x_dis, y_dis, prices, gammas, L)
    
    print("Checking JIT vs NumPy:")
    x_match = np.allclose(x_new_jit, x_new_np, rtol=1e-10)
    y_match = np.allclose(y_new_jit, y_new_np, rtol=1e-10)
    fees_inc_match = np.allclose(fees_inc_jit, fees_inc_np, rtol=1e-10)
    fees_out_match = np.allclose(fees_out_jit, fees_out_np, rtol=1e-10)
    print(f"Results match: {all([x_match, y_match, fees_inc_match, fees_out_match])}")
    
    print("\nChecking Parallel vs NumPy:")
    x_match = np.allclose(x_new_par, x_new_np, rtol=1e-10)
    y_match = np.allclose(y_new_par, y_new_np, rtol=1e-10)
    fees_inc_match = np.allclose(fees_inc_par, fees_inc_np, rtol=1e-10)
    fees_out_match = np.allclose(fees_out_par, fees_out_np, rtol=1e-10)
    print(f"Results match: {all([x_match, y_match, fees_inc_match, fees_out_match])}")

if __name__ == "__main__":
    run_benchmark() 