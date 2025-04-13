import numpy as np
from tqdm import tqdm
import polars as pl
import time
import gc
from tqdm import tqdm
import os
from jit_update import _simulate

class AMMSimulator:
    def __init__(self, x=1000, y=1000, s0=1, drift=0, 
                 dt=1/(365*24), steps=100, seed=None, epoch=None,
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
        self.sigma = sigma_values[0]
        self.gamma = gamma_values[0]
        self.num_paths = num_paths
        self.L = np.sqrt(self.x * self.y)
        self.x_init = x
        self.y_init = y
        self.s_init = s0
        self.epsilon = 1e-6
        self.epoch = epoch
        
        # Use timestamp as seed to generate random seeds, which are used to generate price paths
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        seed = hash(timestamp) % (2**32) if seed is None else seed
        np.random.seed(seed)
        self.seeds = np.random.randint(0, 2**32, size=self.num_paths)
    
    def jit_simulate(self, output_dir):
        """
        Simulate the AMM for all seeds, sigma values and gamma values 
        """
        
        (final_x_dis, final_y_dis, final_xinc_reb, final_yinc_reb, 
        final_xout_reb, final_yout_reb, fees_inc_dis, fees_out_dis, 
        final_prices) = _simulate(
           self.x_init, self.y_init, self.s0, self.steps, self.num_paths, self.sigma_values, self.gamma_values, self.dt, self.seeds
       )
        
        chunk_size = 100000
        current_batch = []
        chunk_idx = 0
        os.makedirs(output_dir, exist_ok=True)
        for i in range(self.num_paths):
            for j, sigma in enumerate(self.sigma_values):
                for k, gamma in enumerate(self.gamma_values):
                    current_batch.append({
                        'sigma': sigma,
                        'gamma': gamma,
                        'final_price': final_prices[i, j],
                        'final_x': final_x_dis[i, j, k],
                        'final_y': final_y_dis[i, j, k],
                        'final_xinc': final_xinc_reb[i, j, k],
                        'final_yinc': final_yinc_reb[i, j, k],
                        'final_xout': final_xout_reb[i, j, k],
                        'final_yout': final_yout_reb[i, j, k],
                        'fees_inc': fees_inc_dis[i, j, k],
                        'fees_out': fees_out_dis[i, j, k]
                    })
                    
                    if len(current_batch) >= chunk_size:
                        df = pl.DataFrame(current_batch)
                        parquet_path = f"{output_dir}/simulation_results_{chunk_idx}.parquet"
                        df.write_parquet(parquet_path, compression='zstd')
                        current_batch = []
                        chunk_idx += 1
        if current_batch:
            df = pl.DataFrame(current_batch)
            parquet_path = f"{output_dir}/simulation_results_epoch{self.epoch}_chunk{chunk_idx}.parquet"
            df.write_parquet(parquet_path, compression='zstd')
        combined_df = self.combine_chunks(output_dir)
        return combined_df
    
                                    
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
        dfs = []
        for chunk_file in chunk_files:
            df = pl.read_parquet(os.path.join(output_dir, chunk_file))
            dfs.append(df)
        
        # Concatenate all DataFrames
        combined_df = pl.concat(dfs)
        
        # Sort the final DataFrame
        combined_df = combined_df.sort(['sigma', 'gamma'])
        
        # Write the combined file
        combined_file = os.path.join(output_dir, f'combined_simulation_results_epoch{self.epoch}.parquet')
        combined_df.write_parquet(combined_file, compression='zstd')
        
        # Delete individual chunk files if requested
        if delete_chunks:
            for chunk_file in chunk_files:
                os.remove(os.path.join(output_dir, chunk_file))
        
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
    output_dir = f'/home/shiftpub/Dynamic_AMM/inf_step_exp/mc_approach/jit_simulation_results/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    num_epochs = 100
    for epoch in tqdm(range(num_epochs)):
        # Use smaller values for profiling
        num_paths = 100
        gamma_values = np.round(np.arange(0.0005, 0.0105, 0.0005), 4)
        sigma_values = np.round(np.arange(0.002, 0.022, 0.002), 3)
        steps = 1000000
        
        simulator = AMMSimulator(num_paths=num_paths, 
                            gamma_values=gamma_values, 
                            sigma_values=sigma_values, 
                            steps=steps, epoch=epoch)
        results_df = simulator.jit_simulate(output_dir)
        
                
