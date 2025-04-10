import numpy as np
from tqdm import tqdm
import polars as pl
import multiprocessing as mp
import os
import time
import gc
from tqdm import tqdm
import psutil
import pickle

class AMMSimulator:
    def __init__(self, x=1000, y=1000, s0=1, drift=0, 
                 dt=1/(365*24), steps=100,
                 num_seeds=None, gamma_values=None, sigma_values=None):
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
        self.num_seeds = num_seeds
        self.L = np.sqrt(self.x * self.y)
        self.x_init = x
        self.y_init = y
        self.s_init = s0
        self.epsilon = 1e-6
        
        # Use timestamp as seed to generate random seeds, which are used to generate price paths
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        seed_hash = hash(timestamp) % (2**32) 
        random_seeds = np.random.RandomState(seed_hash).randint(0, 2**32, size=self.num_seeds)
        self.rngs = [np.random.default_rng(seed) for seed in random_seeds]
        self.z_generator = self.z_yield()
        self.reset()
        
    def reset(self):
        ns = len(self.sigma_values) # number of sigma values
        ng = len(self.gamma_values) # number of gamma values
        self.prices = np.zeros((self.num_seeds, ns), dtype=np.float64)
        self.x_dis = np.zeros((self.num_seeds, ns, ng), dtype=np.float64)
        self.y_dis = np.zeros((self.num_seeds, ns, ng), dtype=np.float64)
        self.dis_inc_fees = np.zeros((self.num_seeds, ns, ng), dtype=np.float64)
        self.dis_out_fees = np.zeros((self.num_seeds, ns, ng), dtype=np.float64)
        self.x_rinc = np.zeros((self.num_seeds, ns, ng), dtype=np.float64)
        self.y_rinc = np.zeros((self.num_seeds, ns, ng), dtype=np.float64)
        self.x_rout = np.zeros((self.num_seeds, ns, ng), dtype=np.float64)
        self.y_rout = np.zeros((self.num_seeds, ns, ng), dtype=np.float64)
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
            yield np.array([rng.normal(0, 1) for rng in self.rngs])
            
    def update_prices(self):
        """
        Generate price series for single step and all seeds and sigma values
        """
        z = next(self.z_generator) # (num_seeds, )
        
        for i, sigma in enumerate(self.sigma_values):
            drift = -0.5 * sigma**2 * self.dt
            diffusion = sigma * np.sqrt(self.dt)
            self.prices[:, i] = self.prices[:, i] * np.exp(drift + diffusion * z)
            
        return self.prices
    
    def update_distribute_case(self):
        """
        Update the distribute case
        """
        # assert L is equal to self.L for all entries
        L = np.sqrt(self.x_dis * self.y_dis)
        assert np.all(np.abs(L - self.L) < self.epsilon), "Constant product is violated for the distribute case"
        
        prices = np.tile(self.prices[:, :, np.newaxis], (1, 1, len(self.gamma_values)))
        gammas = np.tile(self.gamma_values[np.newaxis, np.newaxis, :], (self.num_seeds, len(self.sigma_values), 1))
        upper_threshold = (self.y_dis / self.x_dis) / (1-gammas)
        lower_threshold = (self.y_dis / self.x_dis) * (1-gammas)
        upper_mask = prices > upper_threshold
        lower_mask = prices < lower_threshold
        
        # update x_dis, y_dis, and fees for upper case
        self.dis_inc_fees[upper_mask] += (gammas[upper_mask]/(1-gammas[upper_mask])) * \
            (L[upper_mask]*np.sqrt((1-gammas[upper_mask])*prices[upper_mask])-self.y_dis[upper_mask]) # incoming fees collected from delta_y
        self.dis_out_fees[upper_mask] += (gammas[upper_mask]) * prices[upper_mask] * \
            (self.x_dis[upper_mask] - L[upper_mask]/np.sqrt((1-gammas[upper_mask])*prices[upper_mask])) # outgoing fees collected from delta_x
        self.x_dis[upper_mask] = L[upper_mask] / np.sqrt((1-gammas[upper_mask])*prices[upper_mask])
        self.y_dis[upper_mask] = L[upper_mask]*np.sqrt((1-gammas[upper_mask])*prices[upper_mask])
        assert np.all(self.dis_out_fees[upper_mask] >= 0), f"Outgoing fees, {self.dis_out_fees[upper_mask]}, are negative for the Distribute upper case"
        assert np.all(self.dis_inc_fees[upper_mask] >= 0), f"Incoming fees, {self.dis_inc_fees[upper_mask]}, are negative for the Distribute upper case"
        assert np.all(self.dis_out_fees[upper_mask] - self.dis_inc_fees[upper_mask] > -self.epsilon), f"Outgoing fees, {self.dis_out_fees[upper_mask]}, are significantly less than incoming fees, {self.dis_inc_fees[upper_mask]}, for the Distribute upper case"
        
        # update x_dis, y_dis, and fees for lower case
        self.dis_inc_fees[lower_mask] += (gammas[lower_mask]/(1-gammas[lower_mask])) * prices[lower_mask] * \
            (L[lower_mask]*np.sqrt((1-gammas[lower_mask])/prices[lower_mask]) - self.x_dis[lower_mask]) # incoming fees collected from delta_x
        self.dis_out_fees[lower_mask] += (gammas[lower_mask]) * \
            (self.y_dis[lower_mask] - L[lower_mask]*np.sqrt(prices[lower_mask]/(1-gammas[lower_mask]))) # outgoing fees collected from delta_y
        self.x_dis[lower_mask] = L[lower_mask]*np.sqrt((1-gammas[lower_mask])/prices[lower_mask])
        self.y_dis[lower_mask] = L[lower_mask]*np.sqrt(prices[lower_mask]/(1-gammas[lower_mask]))
        assert np.all(self.dis_out_fees[lower_mask] >= 0), f"Outgoing fees, {self.dis_out_fees[lower_mask]}, are negative for the Distribute lower case"
        assert np.all(self.dis_inc_fees[lower_mask] >= 0), f"Incoming fees, {self.dis_inc_fees[lower_mask]}, are negative for the Distribute lower case"
        assert np.all(self.dis_out_fees[lower_mask] - self.dis_inc_fees[lower_mask] > -self.epsilon), f"Outgoing fees, {self.dis_out_fees[lower_mask]}, are significantly less than incoming fees, {self.dis_inc_fees[lower_mask]}, for the Distribute lower case"
                
    def update_rebalance_case(self):
        """
        Update the rebalance case
        """
        L_rinc = np.sqrt(self.x_rinc * self.y_rinc)
        L_rout = np.sqrt(self.x_rout * self.y_rout)
        assert np.all(L_rinc >= self.L), "Constant product is violated for the rebalance case"
        assert np.all(L_rout >= self.L), "Constant product is violated for the rebalance case"
        
        prices = np.tile(self.prices[:, :, np.newaxis], (1, 1, len(self.gamma_values)))
        gammas = np.tile(self.gamma_values[np.newaxis, np.newaxis, :], (self.num_seeds, len(self.sigma_values), 1))
        upper_threshold = (self.y_rinc / self.x_rinc) / (1-gammas)
        lower_threshold = (self.y_rinc / self.x_rinc) * (1-gammas)
        upper_mask = prices > upper_threshold
        lower_mask = prices < lower_threshold
        
        # update x_rinc, y_rinc, and fees for incoming upper case
        rinc_upper_a = 1-gammas[upper_mask]
        rinc_upper_b = (2-gammas[upper_mask])*self.y_rinc[upper_mask]
        rinc_upper_c = self.y_rinc[upper_mask]**2 - L_rinc[upper_mask]**2 * prices[upper_mask] * (1-gammas[upper_mask])
        rinc_upper_delta_y = (-rinc_upper_b + np.sqrt(rinc_upper_b**2 - 4*rinc_upper_a*rinc_upper_c)) / (2*rinc_upper_a)
        self.y_rinc[upper_mask] = self.y_rinc[upper_mask] + rinc_upper_delta_y
        self.x_rinc[upper_mask] = self.y_rinc[upper_mask] / (prices[upper_mask] * (1-gammas[upper_mask]))
        assert np.all(np.sqrt(self.x_rinc[upper_mask] * self.y_rinc[upper_mask]) - L_rinc[upper_mask] > -self.epsilon), "Constant product is violated for the Rebalance upper case"
        
        # update x_rout, y_rout, and fees for outgoing upper case
        rout_upper_a = 1-gammas[upper_mask]
        rout_upper_b = -(2-gammas[upper_mask])*self.x_rout[upper_mask]
        rout_upper_c = self.x_rout[upper_mask]**2 - L_rout[upper_mask]**2 / ((1-gammas[upper_mask])*prices[upper_mask])
        rout_upper_delta_x = (-rout_upper_b - np.sqrt(rout_upper_b**2 - 4*rout_upper_a*rout_upper_c)) / (2*rout_upper_a)
        self.x_rout[upper_mask] = self.x_rout[upper_mask] - (1-gammas[upper_mask]) * rout_upper_delta_x
        self.y_rout[upper_mask] = self.x_rout[upper_mask] * (1-gammas[upper_mask]) * prices[upper_mask]
        assert np.all(np.sqrt(self.x_rout[upper_mask] * self.y_rout[upper_mask]) - L_rout[upper_mask] > -self.epsilon), "Constant product is violated for the Rebalance outgoing upper case"
        
        # update x_rinc, y_rinc, and fees for incoming lower case
        rinc_lower_a = 1-gammas[lower_mask]
        rinc_lower_b = (2-gammas[lower_mask])*self.x_rinc[lower_mask]
        rinc_lower_c = self.x_rinc[lower_mask]**2 - L_rinc[lower_mask]**2 * (1-gammas[lower_mask]) / prices[lower_mask]
        rinc_lower_delta_x = (-rinc_lower_b + np.sqrt(rinc_lower_b**2 - 4*rinc_lower_a*rinc_lower_c)) / (2*rinc_lower_a)
        self.x_rinc[lower_mask] = self.x_rinc[lower_mask] + rinc_lower_delta_x
        self.y_rinc[lower_mask] = self.x_rinc[lower_mask] * prices[lower_mask] / (1-gammas[lower_mask])
        assert np.all(np.sqrt(self.x_rinc[lower_mask] * self.y_rinc[lower_mask]) - L_rinc[lower_mask] > -self.epsilon), "Constant product is violated for the Rebalance incoming lower case"
                
        # update x_rout, y_rout, and fees for outgoing lower case
        rout_lower_a = 1-gammas[lower_mask]
        rout_lower_b = -(2-gammas[lower_mask])*self.y_rout[lower_mask]
        rout_lower_c = self.y_rout[lower_mask]**2 - L_rout[lower_mask]**2 * prices[lower_mask] / (1-gammas[lower_mask])
        rout_lower_delta_y = (-rout_lower_b - np.sqrt(rout_lower_b**2 - 4*rout_lower_a*rout_lower_c)) / (2*rout_lower_a)
        self.y_rout[lower_mask] = self.y_rout[lower_mask] - (1-gammas[lower_mask]) * rout_lower_delta_y
        self.x_rout[lower_mask] = self.y_rout[lower_mask] * (1-gammas[lower_mask]) / prices[lower_mask]
        assert np.all(np.sqrt(self.x_rout[lower_mask] * self.y_rout[lower_mask]) - L_rout[lower_mask] > -self.epsilon), "Constant product is violated for the Rebalance outgoing lower case"
        
        
    def simulate(self, output_dir):
        """
        Simulate the AMM for all seeds and sigma values
        """
        print(f"Simulating {self.num_seeds} seeds with {len(self.sigma_values)} sigma values and {len(self.gamma_values)} gamma values for {self.steps} steps")
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
        print(f"Saving results...")
        
        simulation_results = {}
        total_length = self.num_seeds * len(self.sigma_values) * len(self.gamma_values)
        progress_bar = tqdm(total=total_length, desc="Saving Progress")
        chunk_idx = 0
        for seed_idx in range(self.num_seeds):
            for sigma_idx, sigma in enumerate(self.sigma_values):
                for gamma_idx, gamma in enumerate(self.gamma_values):
                    simulation_results[seed_idx, sigma, gamma] = \
                    {
                        'prices': self.prices[seed_idx, sigma_idx],
                        'x_dis': self.x_dis[seed_idx, sigma_idx, gamma_idx],
                        'y_dis': self.y_dis[seed_idx, sigma_idx, gamma_idx],
                        'x_rinc': self.x_rinc[seed_idx, sigma_idx, gamma_idx],
                        'y_rinc': self.y_rinc[seed_idx, sigma_idx, gamma_idx],
                        'x_rout': self.x_rout[seed_idx, sigma_idx, gamma_idx],
                        'y_rout': self.y_rout[seed_idx, sigma_idx, gamma_idx],
                        'dis_inc_fees': self.dis_inc_fees[seed_idx, sigma_idx, gamma_idx],
                        'dis_out_fees': self.dis_out_fees[seed_idx, sigma_idx, gamma_idx]
                    }
                    progress_bar.update(1)
            if seed_idx % 10000 == 0:
                self.save_results(simulation_results, chunk_idx, output_dir)
                simulation_results = {}
                chunk_idx += 1
        progress_bar.close()

        self.save_results(simulation_results, chunk_idx, output_dir)
                    
        
    def save_results(self, simulation_results, chunk_idx, output_dir=None):
        """
        Save the simulation results efficiently using Polars and Parquet format
        """
        if not simulation_results:
            return None
            
        # Convert dictionary to a list of records for Polars DataFrame
        records = []
        for (seed_idx, sigma, gamma), data in simulation_results.items():
            record = {
                'seed_idx': seed_idx,
                'sigma': sigma,
                'gamma': gamma,
                'price': data['prices'],
                'x_dis': data['x_dis'],
                'y_dis': data['y_dis'],
                'x_rinc': data['x_rinc'],
                'y_rinc': data['y_rinc'],
                'x_rout': data['x_rout'],
                'y_rout': data['y_rout'],
                'dis_inc_fees': data['dis_inc_fees'],
                'dis_out_fees': data['dis_out_fees']
            }
            records.append(record)
        
        # Convert to Polars DataFrame
        results_df = pl.DataFrame(records)
        
        # Save to parquet file with compression
        parquet_path = f"{output_dir}/simulation_results_{chunk_idx}.parquet"
        results_df.write_parquet(parquet_path, compression='zstd')
        
        print(f"Results saved to {parquet_path}")
        
        # Clear memory
        del results_df
        gc.collect()
        
    
    @staticmethod
    def combine_chunks(output_dir, delete_chunks=True):
        """
        Combine all chunk files into a single DataFrame.
        """
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
        combined_df = combined_df.sort(['seed_idx', 'sigma', 'gamma'])
        
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
                    

if __name__ == "__main__":
    
    # Generate timestamp for unique filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f'/home/shiftpub/Dynamic_AMM/inf_step_exp/mc_approach/crazy_simulation_results/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    num_seeds = 1000000
    gamma_values = np.round(np.arange(0.0005, 0.0105, 0.0005), 4)
    sigma_values = [0.002, 0.004, 0.006, 0.008, 0.01]
    steps = 10000
    simulator = AMMSimulator(num_seeds=num_seeds, gamma_values=gamma_values, sigma_values=sigma_values, steps=steps)
    simulator.simulate(output_dir)
    
    # Combine all chunks into a single file
    print("Combining all chunks into a single file...")
    gc.collect()
    AMMSimulator.combine_chunks(output_dir, delete_chunks=True)
    print("Simulation and data processing complete!")