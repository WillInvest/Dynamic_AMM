import numpy as np
from tqdm import tqdm
import polars as pl
import multiprocessing as mp
import os
import time
from copy import deepcopy
import gc  # Add garbage collector import

class AMMSimulator:
    def __init__(self, x=1000, y=1000, gamma=0.002, s0=1, drift=0, sigma=0.2, 
                 dt=1/(365), steps=10000, num_paths=1):
        """
        Initialize the AMM Simulator that runs multiple simulations with different seeds
        """
        self.x = x
        self.y = y
        self.gamma = gamma
        self.s0 = s0
        self.drift = drift
        self.sigma = sigma
        self.dt = dt
        self.steps = steps
        self.num_paths = num_paths
        self.L = np.sqrt(self.x * self.y)
        
        self.x_init = x
        self.y_init = y
        self.s_init = s0
        
    def reset(self):
        self.x = self.x_init
        self.y = self.y_init
        self.s0 = self.s_init
        
    def generate_prices(self, seed):
        """
        Generate price series for all seeds using fully vectorized GBM
        """
        # Number of time steps
        self.reset()
        n = self.steps
        t = np.arange(1, n+1) * self.dt
        
        # Initialize random number generator
        np.random.seed(seed)
        
        # Generate random normal variables
        Z = np.random.normal(0, 1, size=(self.num_paths, n))
        
        # Initialize price array for all seeds
        prices = np.zeros((self.num_paths, n + 1))
        prices[:, 0] = self.s0
        
        # Vectorized computation
        drift_term = np.broadcast_to((self.drift - 0.5 * self.sigma**2) * t, (self.num_paths, n))
        diffusion_term = self.sigma * np.sqrt(self.dt) * np.cumsum(Z, axis=1)
        prices[:, 1:] = self.s0 * np.exp(drift_term + diffusion_term)
        
        return prices
    
    def simulate(self, seed):
        """
        Simulate AMM behavior for all four scenarios
        """
        # Generate price series
        seed = int(time.time())
        prices = self.generate_prices(seed)
        results = {}

        # Create an array of gamma values
        gamma_values = [0.002]
        num_gammas = len(gamma_values)
        
        # Create gamma matrix
        gamma_matrix = np.tile(gamma_values, (self.num_paths, 1))
        
        # Initialize arrays for all scenarios
        # Distribute scenario (shared pool)
        dis_x = np.tile(self.x, (self.num_paths, num_gammas))
        dis_y = np.tile(self.y, (self.num_paths, num_gammas))
        dis_inc_fees = np.zeros((self.num_paths, num_gammas), dtype=np.float64)
        dis_out_fees = np.zeros((self.num_paths, num_gammas), dtype=np.float64)
        
        # Reinvest scenario (separate pools)
        re_inc_x = np.tile(self.x, (self.num_paths, num_gammas))
        re_inc_y = np.tile(self.y, (self.num_paths, num_gammas))
        
        re_out_x = np.tile(self.x, (self.num_paths, num_gammas))
        re_out_y = np.tile(self.y, (self.num_paths, num_gammas))
        
        terminal_prices = np.tile(prices[:, -1][:, np.newaxis], (1, num_gammas))

        # Iterate through time steps
        for t in range(self.steps):
            # Get current price with broadcasting
            S1 = np.tile(prices[:, t+1][:, np.newaxis], (1, num_gammas))
            
            # Calculate thresholds for each scenario (3 sets)
            # 1. Distribute thresholds
            dis_upper = dis_y / ((1 - gamma_matrix) * dis_x)
            dis_lower = (1 - gamma_matrix) * dis_y / dis_x
            
            # 2. Reinvest Incoming thresholds
            re_inc_upper = re_inc_y / ((1 - gamma_matrix) * re_inc_x)
            re_inc_lower = (1 - gamma_matrix) * re_inc_y / re_inc_x
            
            # 3. Reinvest Outgoing thresholds
            re_out_upper = re_out_y / ((1 - gamma_matrix) * re_out_x)
            re_out_lower = (1 - gamma_matrix) * re_out_y / re_out_x
            
            # Create masks for each scenario
            dis_mask_upper = S1 > dis_upper
            dis_mask_lower = S1 < dis_lower
            dis_mask_no_trade = ~(dis_mask_upper | dis_mask_lower)
            
            re_inc_mask_upper = S1 > re_inc_upper
            re_inc_mask_lower = S1 < re_inc_lower
            re_inc_mask_no_trade = ~(re_inc_mask_upper | re_inc_mask_lower)
            
            re_out_mask_upper = S1 > re_out_upper
            re_out_mask_lower = S1 < re_out_lower
            re_out_mask_no_trade = ~(re_out_mask_upper | re_out_mask_lower)
            
            # Create temporary arrays for new values
            dis_x_new = np.zeros_like(dis_x, dtype=np.float64)
            dis_y_new = np.zeros_like(dis_y, dtype=np.float64)
            dis_inc_fees_step = np.zeros_like(dis_x, dtype=np.float64)
            dis_out_fees_step = np.zeros_like(dis_x, dtype=np.float64)
            
            re_inc_x_new = np.zeros_like(re_inc_x, dtype=np.float64)
            re_inc_y_new = np.zeros_like(re_inc_y, dtype=np.float64)
            
            re_out_x_new = np.zeros_like(re_out_x, dtype=np.float64)
            re_out_y_new = np.zeros_like(re_out_y, dtype=np.float64)
            
            # ==== DISTRIBUTE (shared for both inc/out) ====
            # Upper threshold crossed (price increased)
            dis_x_new[dis_mask_upper] = self.L / np.sqrt((1 - gamma_matrix)[dis_mask_upper] * S1[dis_mask_upper])
            dis_y_new[dis_mask_upper] = self.L * np.sqrt((1 - gamma_matrix)[dis_mask_upper] * S1[dis_mask_upper])
            
            # Lower threshold crossed (price decreased)
            dis_x_new[dis_mask_lower] = self.L * np.sqrt((1 - gamma_matrix)[dis_mask_lower] / S1[dis_mask_lower])
            dis_y_new[dis_mask_lower] = self.L * np.sqrt(S1[dis_mask_lower] / (1 - gamma_matrix[dis_mask_lower]))
            
            # No trade
            dis_x_new[dis_mask_no_trade] = dis_x[dis_mask_no_trade]
            dis_y_new[dis_mask_no_trade] = dis_y[dis_mask_no_trade]
            
            # Calculate fees for distribute scenarios
            # Incoming fees
            dis_inc_fees_step[dis_mask_upper] = gamma_matrix[dis_mask_upper] / (1 - gamma_matrix[dis_mask_upper]) * (dis_y_new[dis_mask_upper] - dis_y[dis_mask_upper])
            dis_inc_fees_step[dis_mask_lower] = gamma_matrix[dis_mask_lower] / (1 - gamma_matrix[dis_mask_lower]) * S1[dis_mask_lower] * (dis_x_new[dis_mask_lower] - dis_x[dis_mask_lower])
            
            # Outgoing fees
            dis_out_fees_step[dis_mask_upper] = gamma_matrix[dis_mask_upper] * S1[dis_mask_upper] * (dis_x[dis_mask_upper] - dis_x_new[dis_mask_upper])
            dis_out_fees_step[dis_mask_lower] = gamma_matrix[dis_mask_lower] * (dis_y[dis_mask_lower] - dis_y_new[dis_mask_lower])
            
            # ==== REINVEST INCOMING ====
            # Upper threshold crossed
            delta_upper = gamma_matrix[re_inc_mask_upper]**2 * re_inc_y[re_inc_mask_upper]**2 + 4 * self.L**2 * (1-gamma_matrix[re_inc_mask_upper])**2 * S1[re_inc_mask_upper]
            a_upper = (1-gamma_matrix[re_inc_mask_upper])
            b_upper = (2-gamma_matrix[re_inc_mask_upper]) * re_inc_y[re_inc_mask_upper]
            delta_y = (-b_upper + np.sqrt(delta_upper)) / (2 * a_upper)
            re_inc_y_new[re_inc_mask_upper] = re_inc_y[re_inc_mask_upper] + delta_y
            re_inc_x_new[re_inc_mask_upper] = re_inc_y_new[re_inc_mask_upper] / ((1-gamma_matrix[re_inc_mask_upper]) * S1[re_inc_mask_upper])
            
            # Lower threshold crossed
            delta_lower = gamma_matrix[re_inc_mask_lower]**2 * re_inc_x[re_inc_mask_lower]**2 + 4 * self.L**2 * (1-gamma_matrix[re_inc_mask_lower])**2 / S1[re_inc_mask_lower]
            a_lower = (1-gamma_matrix[re_inc_mask_lower])
            b_lower = (2-gamma_matrix[re_inc_mask_lower]) * re_inc_x[re_inc_mask_lower]
            delta_x = (-b_lower + np.sqrt(delta_lower)) / (2 * a_lower)
            re_inc_x_new[re_inc_mask_lower] = re_inc_x[re_inc_mask_lower] + delta_x
            re_inc_y_new[re_inc_mask_lower] = re_inc_x_new[re_inc_mask_lower] * S1[re_inc_mask_lower] / (1-gamma_matrix[re_inc_mask_lower])
            
            # No trade
            re_inc_x_new[re_inc_mask_no_trade] = re_inc_x[re_inc_mask_no_trade]
            re_inc_y_new[re_inc_mask_no_trade] = re_inc_y[re_inc_mask_no_trade]
            
            # ==== REINVEST OUTGOING ====
            # Upper threshold crossed 
            delta_upper = gamma_matrix[re_out_mask_upper]**2 * re_out_x[re_out_mask_upper]**2 + 4 * self.L**2 / S1[re_out_mask_upper]
            a_upper = (1-gamma_matrix[re_out_mask_upper])
            b_upper = -(2-gamma_matrix[re_out_mask_upper]) * re_out_x[re_out_mask_upper]
            delta_x = (-b_upper - np.sqrt(delta_upper)) / (2 * a_upper)
            re_out_x_new[re_out_mask_upper] = re_out_x[re_out_mask_upper] - (1-gamma_matrix[re_out_mask_upper]) * delta_x
            re_out_y_new[re_out_mask_upper] = re_out_x_new[re_out_mask_upper] * (1-gamma_matrix[re_out_mask_upper]) * S1[re_out_mask_upper]

            # Lower threshold crossed
            delta_lower = gamma_matrix[re_out_mask_lower]**2 * re_out_y[re_out_mask_lower]**2 + 4 * self.L**2 * S1[re_out_mask_lower]
            a_lower = (1-gamma_matrix[re_out_mask_lower])
            b_lower = -(2-gamma_matrix[re_out_mask_lower]) * re_out_y[re_out_mask_lower]
            delta_y = (-b_lower - np.sqrt(delta_lower)) / (2 * a_lower)
            re_out_y_new[re_out_mask_lower] = re_out_y[re_out_mask_lower] - (1-gamma_matrix[re_out_mask_lower]) * delta_y
            re_out_x_new[re_out_mask_lower] = re_out_y_new[re_out_mask_lower] * (1-gamma_matrix[re_out_mask_lower]) / S1[re_out_mask_lower]

            # No trade
            re_out_x_new[re_out_mask_no_trade] = re_out_x[re_out_mask_no_trade]
            re_out_y_new[re_out_mask_no_trade] = re_out_y[re_out_mask_no_trade]
            
            # Update state for next iteration
            dis_x = dis_x_new
            dis_y = dis_y_new
            dis_inc_fees += dis_inc_fees_step
            dis_out_fees += dis_out_fees_step
            
            re_inc_x = re_inc_x_new
            re_inc_y = re_inc_y_new
            
            re_out_x = re_out_x_new
            re_out_y = re_out_y_new
        
        # Calculate final pool values for all scenarios
        dis_inc_values = terminal_prices * dis_x + dis_y + dis_inc_fees
        dis_out_values = terminal_prices * dis_x + dis_y + dis_out_fees
        re_inc_values = terminal_prices * re_inc_x + re_inc_y
        re_out_values = terminal_prices * re_out_x + re_out_y
        
        # Calculate differences for each path
        diff_dis_out_in = dis_out_values - dis_inc_values
        diff_re_out_in = re_out_values - re_inc_values
        diff_dis_out_re_out = dis_out_values - re_out_values
        diff_dis_in_re_in = dis_inc_values - re_inc_values

        # Calculate mean values across paths for each gamma
        for i, gamma in enumerate(gamma_values):
            # Store results for all four scenarios plus differences
            results[gamma] = (
                np.mean(dis_inc_values[:, i]),  # distribute incoming pool value
                np.mean(dis_inc_fees[:, i]),    # distribute incoming fee value
                np.mean(dis_out_values[:, i]),  # distribute outgoing pool value 
                np.mean(dis_out_fees[:, i]),    # distribute outgoing fee value
                np.mean(re_inc_values[:, i]),   # reinvest incoming pool value
                np.mean(re_out_values[:, i]),   # reinvest outgoing pool value
                np.mean(diff_dis_out_in[:, i]),  # mean diff: dis_out - dis_in
                np.mean(diff_re_out_in[:, i]),   # mean diff: re_out - re_in
                np.mean(diff_dis_out_re_out[:, i]), # mean diff: dis_out - re_out
                np.mean(diff_dis_in_re_in[:, i])    # mean diff: dis_in - re_in
            )
        
        return results
    
    def parallel_simulate(self, output_dir, num_seeds, start_seed=0):
        """
        Run simulations in parallel across multiple seeds
        
        Parameters:
        -----------
        num_seeds : int
            Number of seeds to simulate
        start_seed : int
            Starting seed value
            
        Returns:
        --------
        DataFrame
            DataFrame containing all simulation results
        """
        
        # Create a pool with number of workers equal to CPU count
        # n_workers = mp.cpu_count()
        n_workers = 20
        
        # Create the seed range
        seeds = range(start_seed, start_seed + num_seeds)
        
        # Initialize arrays to collect results
        all_data = []
        
        # Run simulations in parallel
        with mp.Pool(n_workers) as pool:
            # Map the simulate function to each seed
            results = list(tqdm(
                pool.imap(self.simulate, seeds),
                total=num_seeds,
                desc=f"Processing seeds for sigma={self.sigma}"
            ))
        
        print("All simulations complete. Combining results...")
        
        # Extract and combine results
        for seed_result in results:
            for gamma, (dis_inc_val, dis_inc_fee, dis_out_val, dis_out_fee, 
                        re_inc_val, re_out_val, 
                        diff_dis_out_in, diff_re_out_in, 
                        diff_dis_out_re_out, diff_dis_in_re_in) in seed_result.items():
                all_data.append({
                    'sigma': self.sigma,
                    'gamma': gamma,
                    'dis_inc_value': dis_inc_val,
                    'dis_inc_fee': dis_inc_fee,
                    'dis_out_value': dis_out_val,
                    'dis_out_fee': dis_out_fee,
                    're_inc_value': re_inc_val,
                    're_out_value': re_out_val,
                    'diff_dis_out_in': diff_dis_out_in,
                    'diff_re_out_in': diff_re_out_in,
                    'diff_dis_out_re_out': diff_dis_out_re_out,
                    'diff_dis_in_re_in': diff_dis_in_re_in
                })
        
        # Convert to DataFrame
        # sort by sigma and gamma
        results_df = pl.DataFrame(all_data)
        results_df = results_df.sort(['sigma', 'gamma'])
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        results_df.write_csv(f"{output_dir}/results_{self.sigma}.csv")
            
        return results_df

    def simulate_all_paths(self, seed):
        """
        Simulate AMM behavior for all four scenarios and store all path results with full time evolution
        
        Returns:
        --------
        dict
            Dictionary containing all path results for each gamma value with full time evolution
        """
        # Generate price series use seed generated by time
        prices = self.generate_prices(seed)
        results = {}

        # Create an array of gamma values
        gamma_values = [0.002]
        num_gammas = len(gamma_values)
        
        # Create gamma matrix
        gamma_matrix = np.tile(gamma_values, (self.num_paths, 1))
        
        # Initialize arrays for all scenarios with time dimension
        # Distribute scenario (shared pool)
        dis_x = np.tile(self.x, (self.num_paths, num_gammas, self.steps + 1)).astype(np.float64)
        dis_y = np.tile(self.y, (self.num_paths, num_gammas, self.steps + 1)).astype(np.float64)
        dis_inc_fees = np.zeros((self.num_paths, num_gammas, self.steps + 1), dtype=np.float64)
        dis_out_fees = np.zeros((self.num_paths, num_gammas, self.steps + 1), dtype=np.float64)
        
        # Reinvest scenario (separate pools)
        re_inc_x = np.tile(self.x, (self.num_paths, num_gammas, self.steps + 1)).astype(np.float64)
        re_inc_y = np.tile(self.y, (self.num_paths, num_gammas, self.steps + 1)).astype(np.float64)
        
        re_out_x = np.tile(self.x, (self.num_paths, num_gammas, self.steps + 1)).astype(np.float64)
        re_out_y = np.tile(self.y, (self.num_paths, num_gammas, self.steps + 1)).astype(np.float64)

        # Iterate through time steps
        for t in tqdm(range(self.steps)):
            # Get current price with broadcasting
            S1 = np.tile(prices[:, t+1][:, np.newaxis], (1, num_gammas))
            
            # Calculate thresholds for each scenario (3 sets)
            # 1. Distribute thresholds
            dis_upper = dis_y[:, :, t] / ((1 - gamma_matrix) * dis_x[:, :, t])
            dis_lower = (1 - gamma_matrix) * dis_y[:, :, t] / dis_x[:, :, t]
            
            # 2. Reinvest Incoming thresholds
            re_inc_upper = re_inc_y[:, :, t] / ((1 - gamma_matrix) * re_inc_x[:, :, t])
            re_inc_lower = (1 - gamma_matrix) * re_inc_y[:, :, t] / re_inc_x[:, :, t]
            
            # 3. Reinvest Outgoing thresholds
            re_out_upper = re_out_y[:, :, t] / ((1 - gamma_matrix) * re_out_x[:, :, t])
            re_out_lower = (1 - gamma_matrix) * re_out_y[:, :, t] / re_out_x[:, :, t]
            
            # Create masks for each scenario
            dis_mask_upper = S1 > dis_upper
            dis_mask_lower = S1 < dis_lower
            dis_mask_no_trade = ~(dis_mask_upper | dis_mask_lower)
            
            re_inc_mask_upper = S1 > re_inc_upper
            re_inc_mask_lower = S1 < re_inc_lower
            re_inc_mask_no_trade = ~(re_inc_mask_upper | re_inc_mask_lower)
            
            re_out_mask_upper = S1 > re_out_upper
            re_out_mask_lower = S1 < re_out_lower
            re_out_mask_no_trade = ~(re_out_mask_upper | re_out_mask_lower)
            
            # Create temporary arrays for new values
            dis_x_new = np.zeros_like(dis_x[:, :, t], dtype=np.float64)
            dis_y_new = np.zeros_like(dis_y[:, :, t], dtype=np.float64)
            dis_inc_fees_step = np.zeros_like(dis_x[:, :, t], dtype=np.float64)
            dis_out_fees_step = np.zeros_like(dis_x[:, :, t], dtype=np.float64)
            
            re_inc_x_new = np.zeros_like(re_inc_x[:, :, t], dtype=np.float64)
            re_inc_y_new = np.zeros_like(re_inc_y[:, :, t], dtype=np.float64)
            
            re_out_x_new = np.zeros_like(re_out_x[:, :, t], dtype=np.float64)
            re_out_y_new = np.zeros_like(re_out_y[:, :, t], dtype=np.float64)
            
            # ==== DISTRIBUTE (shared for both inc/out) ====
            # Upper threshold crossed (price increased)
            dis_x_new[dis_mask_upper] = self.L / np.sqrt((1 - gamma_matrix)[dis_mask_upper] * S1[dis_mask_upper])
            dis_y_new[dis_mask_upper] = self.L * np.sqrt((1 - gamma_matrix)[dis_mask_upper] * S1[dis_mask_upper])
            
            # Lower threshold crossed (price decreased)
            dis_x_new[dis_mask_lower] = self.L * np.sqrt((1 - gamma_matrix)[dis_mask_lower] / S1[dis_mask_lower])
            dis_y_new[dis_mask_lower] = self.L * np.sqrt(S1[dis_mask_lower] / (1 - gamma_matrix[dis_mask_lower]))
            
            # No trade
            dis_x_new[dis_mask_no_trade] = dis_x[:, :, t][dis_mask_no_trade]
            dis_y_new[dis_mask_no_trade] = dis_y[:, :, t][dis_mask_no_trade]
            
            # Calculate fees for distribute scenarios
            # Incoming fees
            dis_inc_fees_step[dis_mask_upper] = gamma_matrix[dis_mask_upper] / (1 - gamma_matrix[dis_mask_upper]) * (dis_y_new[dis_mask_upper] - dis_y[:, :, t][dis_mask_upper])
            dis_inc_fees_step[dis_mask_lower] = gamma_matrix[dis_mask_lower] / (1 - gamma_matrix[dis_mask_lower]) * S1[dis_mask_lower] * (dis_x_new[dis_mask_lower] - dis_x[:, :, t][dis_mask_lower])
            
            # Outgoing fees
            dis_out_fees_step[dis_mask_upper] = gamma_matrix[dis_mask_upper] * S1[dis_mask_upper] * (dis_x[:, :, t][dis_mask_upper] - dis_x_new[dis_mask_upper])
            dis_out_fees_step[dis_mask_lower] = gamma_matrix[dis_mask_lower] * (dis_y[:, :, t][dis_mask_lower] - dis_y_new[dis_mask_lower])
            
            # ==== REINVEST INCOMING ====
            # Upper threshold crossed
            delta_upper = gamma_matrix[re_inc_mask_upper]**2 * re_inc_y[:, :, t][re_inc_mask_upper]**2 +\
                4 * (re_inc_y[:, :, t][re_inc_mask_upper] * re_inc_x[:, :, t][re_inc_mask_upper]) * (1-gamma_matrix[re_inc_mask_upper])**2 * S1[re_inc_mask_upper]
            a_upper = (1-gamma_matrix[re_inc_mask_upper])
            b_upper = (2-gamma_matrix[re_inc_mask_upper]) * re_inc_y[:, :, t][re_inc_mask_upper]
            delta_y = (-b_upper + np.sqrt(delta_upper)) / (2 * a_upper)
            re_inc_y_new[re_inc_mask_upper] = re_inc_y[:, :, t][re_inc_mask_upper] + delta_y
            re_inc_x_new[re_inc_mask_upper] = re_inc_y_new[re_inc_mask_upper] / ((1-gamma_matrix[re_inc_mask_upper]) * S1[re_inc_mask_upper])
            
            # Lower threshold crossed
            delta_lower = gamma_matrix[re_inc_mask_lower]**2 * re_inc_x[:, :, t][re_inc_mask_lower]**2 +\
                4 * (re_inc_x[:, :, t][re_inc_mask_lower] * re_inc_y[:, :, t][re_inc_mask_lower]) * (1-gamma_matrix[re_inc_mask_lower])**2 / S1[re_inc_mask_lower]
            a_lower = (1-gamma_matrix[re_inc_mask_lower])
            b_lower = (2-gamma_matrix[re_inc_mask_lower]) * re_inc_x[:, :, t][re_inc_mask_lower]
            delta_x = (-b_lower + np.sqrt(delta_lower)) / (2 * a_lower)
            re_inc_x_new[re_inc_mask_lower] = re_inc_x[:, :, t][re_inc_mask_lower] + delta_x
            re_inc_y_new[re_inc_mask_lower] = re_inc_x_new[re_inc_mask_lower] * S1[re_inc_mask_lower] / (1-gamma_matrix[re_inc_mask_lower])
            
            # No trade
            re_inc_x_new[re_inc_mask_no_trade] = re_inc_x[:, :, t][re_inc_mask_no_trade]
            re_inc_y_new[re_inc_mask_no_trade] = re_inc_y[:, :, t][re_inc_mask_no_trade]
            
            # ==== REINVEST OUTGOING ====
            # Upper threshold crossed 
            delta_upper = gamma_matrix[re_out_mask_upper]**2 * re_out_x[:, :, t][re_out_mask_upper]**2 +\
                4 * (re_out_x[:, :, t][re_out_mask_upper] * re_out_y[:, :, t][re_out_mask_upper]) * (1-gamma_matrix[re_out_mask_upper])**2 / S1[re_out_mask_upper]
            a_upper = (1-gamma_matrix[re_out_mask_upper])
            b_upper = -(2-gamma_matrix[re_out_mask_upper]) * re_out_x[:, :, t][re_out_mask_upper]
            delta_x = (-b_upper - np.sqrt(delta_upper)) / (2 * a_upper)
            re_out_x_new[re_out_mask_upper] = re_out_x[:, :, t][re_out_mask_upper] - (1-gamma_matrix[re_out_mask_upper]) * delta_x
            re_out_y_new[re_out_mask_upper] = re_out_x_new[re_out_mask_upper] * (1-gamma_matrix[re_out_mask_upper]) * S1[re_out_mask_upper]

            # Lower threshold crossed
            delta_lower = gamma_matrix[re_out_mask_lower]**2 * re_out_y[:, :, t][re_out_mask_lower]**2 +\
                4 * (re_out_y[:, :, t][re_out_mask_lower] * re_out_x[:, :, t][re_out_mask_lower]) * (1-gamma_matrix[re_out_mask_lower])**2 * S1[re_out_mask_lower]
            a_lower = (1-gamma_matrix[re_out_mask_lower])
            b_lower = -(2-gamma_matrix[re_out_mask_lower]) * re_out_y[:, :, t][re_out_mask_lower]
            delta_y = (-b_lower - np.sqrt(delta_lower)) / (2 * a_lower)
            re_out_y_new[re_out_mask_lower] = re_out_y[:, :, t][re_out_mask_lower] - (1-gamma_matrix[re_out_mask_lower]) * delta_y
            re_out_x_new[re_out_mask_lower] = re_out_y_new[re_out_mask_lower] * (1-gamma_matrix[re_out_mask_lower]) / S1[re_out_mask_lower]

            # No trade
            re_out_x_new[re_out_mask_no_trade] = re_out_x[:, :, t][re_out_mask_no_trade]
            re_out_y_new[re_out_mask_no_trade] = re_out_y[:, :, t][re_out_mask_no_trade]
            
            # Update state for next iteration
            dis_x[:, :, t+1] = dis_x_new
            dis_y[:, :, t+1] = dis_y_new
            dis_inc_fees[:, :, t+1] = dis_inc_fees[:, :, t] + dis_inc_fees_step
            dis_out_fees[:, :, t+1] = dis_out_fees[:, :, t] + dis_out_fees_step
            
            re_inc_x[:, :, t+1] = re_inc_x_new
            re_inc_y[:, :, t+1] = re_inc_y_new
            
            re_out_x[:, :, t+1] = re_out_x_new
            re_out_y[:, :, t+1] = re_out_y_new

        # Store all path results for each gamma with full time evolution
        for i, gamma in enumerate(gamma_values):
            results[gamma] = {
                'prices': prices,  # shape: (num_paths, time_steps)
                'dis_x': dis_x[:, i, :],  # shape: (num_paths, time_steps)
                'dis_y': dis_y[:, i, :],
                'dis_inc_fees': dis_inc_fees[:, i, :],
                'dis_out_fees': dis_out_fees[:, i, :],
                're_inc_x': re_inc_x[:, i, :],
                're_inc_y': re_inc_y[:, i, :],
                're_out_x': re_out_x[:, i, :],
                're_out_y': re_out_y[:, i, :]
            }
        
        return results

    def parallel_simulate_all_paths(self, output_dir, num_seeds, start_seed=0):
        """
        Run simulations in parallel across multiple seeds and store all path results with full time evolution,
        writing out results in chunks to manage memory usage.
        """
        n_workers = 20
        chunk_size = 1000000  # Number of rows per chunk
        current_chunk = 0
        
        # Create the seed range
        seeds = range(start_seed, start_seed + num_seeds)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize arrays to collect results
        all_data = []
        total_rows = 0
        
        # Calculate total number of rows for progress bar
        gamma_values = np.round(np.arange(0.005, 0.010, 0.005), 3)
        total_expected_rows = num_seeds * self.num_paths * len(gamma_values) * (self.steps + 1)
        pbar = tqdm(total=total_expected_rows, desc=f"Processing rows for sigma={self.sigma}")
        
        # Run simulations in parallel
        with mp.Pool(n_workers) as pool:
            results = list(tqdm(
                pool.imap(self.simulate_all_paths, seeds),
                total=num_seeds,
                desc=f"Processing seeds for sigma={self.sigma}"
            ))
        
        # Process results and write in chunks
        for seed_idx, seed_result in enumerate(results):
            for gamma, path_results in seed_result.items():
                for path_idx in range(self.num_paths):
                    for time_idx in range(self.steps + 1):
                        all_data.append({
                            'seed_idx': seed_idx,
                            'path_idx': path_idx,
                            'time_idx': time_idx,
                            'gamma': gamma,
                            'price': path_results['prices'][path_idx, time_idx],  
                            'dis_x': path_results['dis_x'][path_idx, time_idx],
                            'dis_y': path_results['dis_y'][path_idx, time_idx],
                            'dis_inc_fees': path_results['dis_inc_fees'][path_idx, time_idx],
                            'dis_out_fees': path_results['dis_out_fees'][path_idx, time_idx],
                            're_inc_x': path_results['re_inc_x'][path_idx, time_idx],
                            're_inc_y': path_results['re_inc_y'][path_idx, time_idx],
                            're_out_x': path_results['re_out_x'][path_idx, time_idx],
                            're_out_y': path_results['re_out_y'][path_idx, time_idx]
                        })
                        total_rows += 1
                        pbar.update(1)
                        
                        # Write chunk if we've reached the chunk size
                        if total_rows >= chunk_size:
                            # Convert to DataFrame and sort
                            results_df = pl.DataFrame(all_data)
                            results_df = results_df.sort(['seed_idx', 'path_idx', 'time_idx', 'gamma'])
                            
                            # Write chunk to parquet file
                            chunk_file = f"{output_dir}/all_paths_results_{self.sigma}_chunk_{current_chunk}.parquet"
                            results_df.write_parquet(chunk_file, compression='zstd')
                            
                            # Clear memory
                            del results_df
                            gc.collect()  # Force garbage collection
                            
                            # Reset for next chunk
                            all_data = []
                            total_rows = 0
                            current_chunk += 1
        
        # Write any remaining data
        if all_data:
            results_df = pl.DataFrame(all_data)
            results_df = results_df.sort(['seed_idx', 'path_idx', 'time_idx', 'gamma'])
            chunk_file = f"{output_dir}/all_paths_results_{self.sigma}_chunk_{current_chunk}.parquet"
            results_df.write_parquet(chunk_file, compression='zstd')
            
            # Clear memory after final write
            del results_df
            gc.collect()
        
        pbar.close()
        print(f"All results written in {current_chunk + 1} chunks for sigma={self.sigma}")

    @staticmethod
    def combine_chunks_for_sigma(sigma, output_dir, delete_chunks=True):
        """
        Combine all chunk files for a specific sigma value into a single DataFrame.
        """
        # Find all chunk files for the given sigma
        chunk_files = sorted([f for f in os.listdir(output_dir) 
                            if f.startswith(f'all_paths_results_{sigma}_chunk_') and f.endswith('.parquet')])
        
        if not chunk_files:
            raise ValueError(f"No chunk files found for sigma={sigma}")
        
        # Read and combine all chunks
        print(f"Combining {len(chunk_files)} chunks for sigma={sigma}...")
        dfs = []
        for chunk_file in tqdm(chunk_files, desc="Reading chunks"):
            df = pl.read_parquet(os.path.join(output_dir, chunk_file))
            dfs.append(df)
        
        # Concatenate all DataFrames
        combined_df = pl.concat(dfs)
        
        # Sort the final DataFrame
        combined_df = combined_df.sort(['gamma'])
        
        # Write the combined file
        combined_file = os.path.join(output_dir, f'all_paths_results_{sigma}_combined.parquet')
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
    import pandas as pd
    import os
    output_dir = '/home/shiftpub/Dynamic_AMM/inf_step_exp/mc_approach/mc_results_temp'
    os.makedirs(output_dir, exist_ok=True)
    sigma = [0.006]
    for s in sigma:
        simulator = AMMSimulator(
            x=1000,          # Initial X reserve
            y=1000,          # Initial Y reserve
            gamma=None,      # Fee parameter (not used in vectorized version)
            s0=1.0,          # Initial price
            drift=0.0,       # No drift
            sigma=s,
            dt=1/(365*24),
            steps=1000000,
            num_paths=1
        )
        simulator.parallel_simulate_all_paths(output_dir, num_seeds=1, start_seed=0)
        simulator.combine_chunks_for_sigma(s, output_dir, delete_chunks=True)
