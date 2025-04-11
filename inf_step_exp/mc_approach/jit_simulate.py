import numpy as np
from tqdm import tqdm
import polars as pl
import time
import gc
from tqdm import tqdm
import os
from jit_update import _update_prices, _update_distribute_case, _update_rebalance_case

class AMMSimulator:
    def __init__(self, x=1000, y=1000, s0=1, drift=0, 
                 dt=1/(365*24), steps=100, seed=None,
                 num_paths=int(1e6), gamma_values=None, sigma_values=None):
        """
        Initialize the AMM Simulator that runs multiple simulations with different seeds
        """
        self.x = x
        self.y = y
        self.s0 = s0
        self.drift = drift
        self.dt = dt
        self.steps = steps
        self.gamma_values = gamma_values
        self.sigma_values = sigma_values
        self.num_paths = num_paths
        self.L = np.sqrt(self.x * self.y)
        self.x_init = x
        self.y_init = y
        self.s_init = s0
        self.epsilon = 1e-6
        
        # Use timestamp as seed to generate random seeds, which are used to generate price paths
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.seed = hash(timestamp) % (2**32) if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        self.z_generator = self.z_yield()
        self.reset()
        
    def reset(self):
        ns = len(self.sigma_values) # number of sigma values
        ng = len(self.gamma_values) # number of gamma values
        self.prices = np.zeros((self.num_paths, ns), dtype=np.float64)
        self.x_dis = np.zeros((self.num_paths, ns, ng), dtype=np.float64)
        self.y_dis = np.zeros((self.num_paths, ns, ng), dtype=np.float64)
        self.dis_inc_fees = np.zeros((self.num_paths, ns, ng), dtype=np.float64)
        self.dis_out_fees = np.zeros((self.num_paths, ns, ng), dtype=np.float64)
        self.x_rinc = np.zeros((self.num_paths, ns, ng), dtype=np.float64)
        self.y_rinc = np.zeros((self.num_paths, ns, ng), dtype=np.float64)
        self.x_rout = np.zeros((self.num_paths, ns, ng), dtype=np.float64)
        self.y_rout = np.zeros((self.num_paths, ns, ng), dtype=np.float64)
        self.prices[:, :] = self.s0
        self.x_dis[:, :, :] = self.x_init
        self.y_dis[:, :, :] = self.y_init
        self.x_rinc[:, :, :] = self.x_init
        self.y_rinc[:, :, :] = self.y_init
        self.x_rout[:, :, :] = self.x_init
        self.y_rout[:, :, :] = self.y_init
        
    def z_yield(self):
        """
        Generator function that directly yields standard normal random numbers for each seed.
        
        Yields:
            numpy.ndarray: Array of standard normal random numbers, one for each seed.
        """
        while True:
            # Generate one standard normal number for each seed
            yield self.rng.normal(0, 1, size=self.num_paths)
            
    def update_prices(self):
        """
        Generate price series for single step and all seeds and sigma values
        """
        z = next(self.z_generator) # (num_paths, )
        
        self.prices = _update_prices(self.prices, z, self.sigma_values, self.dt)
        
        return self.prices
    
    def update_distribute_case(self):
        """
        Update the distribute case
        """
        # assert L is equal to self.L for all entries
        L = np.sqrt(self.x_dis * self.y_dis)
        assert np.all(np.abs(L - self.L) < self.epsilon), "Constant product is violated for the distribute case"
        
        # Update using JIT-compiled function
        x_new, y_new, fees_inc, fees_out = _update_distribute_case(
            self.x_dis, self.y_dis, self.prices, 
            self.gamma_values, self.L, self.epsilon
        )
        
        # Update state
        self.x_dis = x_new
        self.y_dis = y_new
        self.dis_inc_fees += fees_inc
        self.dis_out_fees += fees_out
        
        # Verify updates
        L_new = np.sqrt(self.x_dis * self.y_dis)
        assert np.all(np.abs(L_new - self.L) < self.epsilon), "Constant product is violated after distribute case update"
                
    def update_rebalance_case(self):
        """
        Update the rebalance case
        """
        L_rinc = np.sqrt(self.x_rinc * self.y_rinc)
        L_rout = np.sqrt(self.x_rout * self.y_rout)
        
        # Update using JIT-compiled function
        x_rinc_new, y_rinc_new, x_rout_new, y_rout_new = _update_rebalance_case(
            self.x_rinc, self.y_rinc, self.x_rout, self.y_rout,
            self.prices, self.gamma_values, L_rinc, L_rout
        )
        
        assert np.all(x_rinc_new*y_rinc_new >= self.x_rinc * self.y_rinc), "Constant product is violated for the rebalance case"
        assert np.all(x_rout_new*y_rout_new >= self.x_rout * self.y_rout), "Constant product is violated for the rebalance case"
        
        # Update state
        self.x_rinc = x_rinc_new
        self.y_rinc = y_rinc_new
        self.x_rout = x_rout_new
        self.y_rout = y_rout_new
        
        
    def simulate(self, output_dir=None):
        """
        Simulate the AMM for all seeds and sigma values
        """
        print(f"Simulating {self.num_paths} paths with {len(self.sigma_values)} sigma values and {len(self.gamma_values)} gamma values for {self.steps} steps")
        print(f"Range of sigma values: {max(self.sigma_values)} to {min(self.sigma_values)}")
        print(f"Range of gamma values: {max(self.gamma_values)} to {min(self.gamma_values)}")
        print(f"Simulation Starts...")
        
        current_time = time.time()
        for _ in tqdm(range(self.steps), desc="Simulating Progress"):
            self.update_prices()
            self.update_distribute_case()
            self.update_rebalance_case()
            
        print(f"Simulation Finished...")
        print(f"Time taken: {time.time() - current_time} seconds")
        
        if output_dir is not None:
            self.save_results(output_dir)
                    
        
    def save_results(self, output_dir=None):
        """
        Save the simulation results efficiently using Polars and Parquet format
        """
        print(f"Saving results...")
        total_length = self.num_paths * len(self.sigma_values) * len(self.gamma_values)
        progress_bar = tqdm(total=total_length, desc="Saving Progress")
        batch_size = 10000
        current_batch = []
        chunk_idx = 0
        
        for path_idx in range(self.num_paths):
            for sigma_idx, sigma in enumerate(self.sigma_values):
                for gamma_idx, gamma in enumerate(self.gamma_values):
                    record = {
                        'path_idx': path_idx,
                        'sigma': sigma,
                        'gamma': gamma,
                        'price': self.prices[path_idx, sigma_idx],
                        'x_dis': self.x_dis[path_idx, sigma_idx, gamma_idx],
                        'y_dis': self.y_dis[path_idx, sigma_idx, gamma_idx],
                        'x_rinc': self.x_rinc[path_idx, sigma_idx, gamma_idx],
                        'y_rinc': self.y_rinc[path_idx, sigma_idx, gamma_idx],
                        'x_rout': self.x_rout[path_idx, sigma_idx, gamma_idx],
                        'y_rout': self.y_rout[path_idx, sigma_idx, gamma_idx],
                        'dis_inc_fees': self.dis_inc_fees[path_idx, sigma_idx, gamma_idx],
                        'dis_out_fees': self.dis_out_fees[path_idx, sigma_idx, gamma_idx]
                    }
                    current_batch.append(record)
                    progress_bar.update(1)
                    
                    # When batch is full, write to parquet
                    if len(current_batch) >= batch_size:
                        # Convert batch to DataFrame and write
                        df = pl.DataFrame(current_batch)
                        parquet_path = f"{output_dir}/simulation_results_{chunk_idx}.parquet"
                        df.write_parquet(parquet_path, compression='zstd')
                        
                        # Clear batch and memory
                        current_batch = []
                        del df
                        gc.collect()
                        chunk_idx += 1
        
        # Write any remaining records
        if current_batch:
            df = pl.DataFrame(current_batch)
            parquet_path = f"{output_dir}/simulation_results_{chunk_idx}.parquet"
            df.write_parquet(parquet_path, compression='zstd')
            del df
            gc.collect()
        
        progress_bar.close()
        print(f"Results saved to {output_dir}")
        
        self.combine_chunks(output_dir, delete_chunks=True)
        print(f"Combined results saved to {output_dir}")
            
    def combine_chunks(self,output_dir, delete_chunks=True):
        """
        Combine all chunk files into a single DataFrame.
        """
        # clear memory
        gc.collect()
        self.cleanup()
        
        # Find all chunk files
        chunk_files = sorted([f for f in os.listdir(output_dir) 
                            if f.startswith('simulation_results_') and f.endswith('.parquet')])
        
        if not chunk_files:
            raise ValueError(f"No chunk files found in {output_dir}")
        
        # Read and combine all chunks
        print(f"Combining {len(chunk_files)} chunks...")
        dfs = []
        for chunk_file in tqdm(chunk_files, desc="Reading chunks"):
            df = pl.read_parquet(os.path.join(output_dir, chunk_file))
            dfs.append(df)
        
        # Concatenate all DataFrames
        combined_df = pl.concat(dfs)
        
        # Sort the final DataFrame
        combined_df = combined_df.sort(['path_idx', 'sigma', 'gamma'])
        
        # Write the combined file
        combined_file = os.path.join(output_dir, 'simulation_results_combined.parquet')
        combined_df.write_parquet(combined_file, compression='zstd')
        print(f"Combined data written to {combined_file}")
        
        # Delete individual chunk files if requested
        if delete_chunks:
            print("Deleting individual chunk files...")
            for chunk_file in tqdm(chunk_files, desc="Deleting chunks"):
                os.remove(os.path.join(output_dir, chunk_file))
            print(f"Deleted {len(chunk_files)} chunk files")
        
        return combined_df
    
        
    def cleanup(self):
        """Clean up resources and memory"""
        # Clear large arrays
        if hasattr(self, 'prices'):
            del self.prices
        if hasattr(self, 'x_dis'):
            del self.x_dis
        if hasattr(self, 'y_dis'):
            del self.y_dis
        if hasattr(self, 'dis_inc_fees'):
            del self.dis_inc_fees
        if hasattr(self, 'dis_out_fees'):
            del self.dis_out_fees
        if hasattr(self, 'x_rinc'):
            del self.x_rinc
        if hasattr(self, 'y_rinc'):
            del self.y_rinc
        if hasattr(self, 'x_rout'):
            del self.x_rout
        if hasattr(self, 'y_rout'):
            del self.y_rout
            
        # Force garbage collection
        gc.collect()
        
    def __del__(self):
        """
        Destructor - called when the object is about to be destroyed
        """
        try:
            self.cleanup()
        except Exception as e:
            print(f"Error during destructor cleanup: {e}")
                    

if __name__ == "__main__":
    # Generate timestamp for unique filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f'/Users/haofu/Desktop/AMM/Dynamic_AMM/inf_step_exp/mc_approach/crazy_simulation_results/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Use smaller values for profiling
    num_paths = 1000000
    gamma_values = np.round(np.arange(0.0005, 0.0105, 0.0005), 4)
    sigma_values = [0.002, 0.004, 0.006, 0.008, 0.01]
    steps = 100  # Changed from 5 to 10000 for full simulation
    
    simulator = AMMSimulator(num_paths=num_paths, 
                         gamma_values=gamma_values, 
                         sigma_values=sigma_values, 
                         steps=steps, seed=0)
    simulator.simulate(output_dir)
            
