import numpy as np
from tqdm import tqdm
import polars as pl
import os
import time
import gc
from tqdm import tqdm

class AMMSimulator:
    def __init__(self, x=1000, y=1000, s0=1, 
                 dt=1/(365*24), steps=100,
                 num_seeds=None, gamma_values=None,
                 mu_values=None,sigma_values=None):
        """
        Initialize the AMM Simulator that runs multiple simulations with different seeds
        """
        self.x = x
        self.y = y
        self.s0 = s0
        self.dt = dt
        self.steps = steps
        self.gamma_values = gamma_values
        self.mu_values = mu_values
        self.sigma_values = sigma_values
        self.num_seeds = num_seeds
        self.L = np.sqrt(self.x * self.y)
        self.x_init = x
        self.y_init = y
        self.s_init = s0
        self.epsilon = 1e-4
        self.test_mode = False
        
        # Use timestamp as seed to generate random seeds, which are used to generate price paths
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.seed = hash(timestamp) % (2**32)
        self.rng = np.random.default_rng(self.seed)
        self.z_generator = self.z_yield()
        self.reset()
        
    def reset(self):
        ns = len(self.sigma_values) # number of sigma values
        ng = len(self.gamma_values) # number of gamma values
        nm = len(self.mu_values) # number of mu values
        self.prices = np.zeros((self.num_seeds, nm, ns), dtype=np.float64)
        self.x_dis = np.zeros((self.num_seeds, nm, ns, ng), dtype=np.float64)
        self.y_dis = np.zeros((self.num_seeds, nm, ns, ng), dtype=np.float64)
        self.dis_inc_fees = np.zeros((self.num_seeds, nm, ns, ng), dtype=np.float64)
        self.dis_out_fees = np.zeros((self.num_seeds, nm, ns, ng), dtype=np.float64)
        self.x_rinc = np.zeros((self.num_seeds, nm, ns, ng), dtype=np.float64)
        self.y_rinc = np.zeros((self.num_seeds, nm, ns, ng), dtype=np.float64)
        self.x_rout = np.zeros((self.num_seeds, nm, ns, ng), dtype=np.float64)
        self.y_rout = np.zeros((self.num_seeds, nm, ns, ng), dtype=np.float64)
        self.prices[:, :, :] = self.s0
        self.x_dis[:, :, :, :] = self.x_init
        self.y_dis[:, :, :, :] = self.y_init
        self.x_rinc[:, :, :, :] = self.x_init
        self.y_rinc[:, :, :, :] = self.y_init
        self.x_rout[:, :, :, :] = self.x_init
        self.y_rout[:, :, :, :] = self.y_init
        
    def z_yield(self):
        """
        Generator function that yields standard normal random numbers with shape (num_seeds, nm, ns)
        
        Yields:
            numpy.ndarray: Array of standard normal random numbers with shape (num_seeds, nm, ns)
        """
        while True:
            yield self.rng.normal(0, 1, size=(self.num_seeds, len(self.mu_values), len(self.sigma_values)))
            
    def update_prices(self):
        """
        Generate price series for single step and all seeds and sigma values
        """
        z = next(self.z_generator)  # shape: (n, m, s)
        # Calculate drift and diffusion for all mu and sigma values at once
        drift = (self.mu_values[:, np.newaxis] - 0.5 * self.sigma_values[np.newaxis, :]**2) * self.dt  # (m, 1) - (1, s) = (m, s)
        diffusion = self.sigma_values[np.newaxis, :] * np.sqrt(self.dt)  # (1, s)
        
        # Reshape drift and diffusion to match z's shape
        drift = drift[np.newaxis, :, :]  # (1, m, s)
        diffusion = diffusion[np.newaxis, :, :]  # (1, 1, s)
        # Now all shapes align correctly for broadcasting
        self.prices = self.prices * np.exp(drift + diffusion * z) # (1, m, s) + (1, 1, s) * (n, m, s) = (1, m, s) + (n, m, s) = (n, m, s)
        return self.prices
    
    def update_distribute_case(self):
        """
        Update the distribute case
        """
        # assert L is equal to self.L for all entries
        L = np.sqrt(self.x_dis * self.y_dis)
        assert np.all(np.abs(L - self.L) < self.epsilon), "Constant product is violated for the distribute case"
        prices = np.tile(self.prices[:, :, :, np.newaxis], (1, 1, 1, len(self.gamma_values)))
        # print(f"prices shape: {prices.shape}")
        gammas = np.tile(self.gamma_values[np.newaxis, np.newaxis, np.newaxis, :], (self.num_seeds, len(self.mu_values), len(self.sigma_values), 1))
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
     
        # update x_dis, y_dis, and fees for lower case
        self.dis_inc_fees[lower_mask] += (gammas[lower_mask]/(1-gammas[lower_mask])) * prices[lower_mask] * \
            (L[lower_mask]*np.sqrt((1-gammas[lower_mask])/prices[lower_mask]) - self.x_dis[lower_mask]) # incoming fees collected from delta_x
        self.dis_out_fees[lower_mask] += (gammas[lower_mask]) * \
            (self.y_dis[lower_mask] - L[lower_mask]*np.sqrt(prices[lower_mask]/(1-gammas[lower_mask]))) # outgoing fees collected from delta_y
        self.x_dis[lower_mask] = L[lower_mask]*np.sqrt((1-gammas[lower_mask])/prices[lower_mask])
        self.y_dis[lower_mask] = L[lower_mask]*np.sqrt(prices[lower_mask]/(1-gammas[lower_mask]))
     
        if self.test_mode:
            if not np.all(self.dis_out_fees >= 0):
                invalid_indices = np.where(self.dis_out_fees < 0)
                print(f"Outgoing fees are negative for the Distribute upper case at indices: {invalid_indices}")
                print(f"Invalid outgoing fees values: {self.dis_out_fees[invalid_indices]}")
                raise AssertionError("Outgoing fees are negative for the Distribute upper case")

            if not np.all(self.dis_inc_fees >= 0):
                invalid_indices = np.where(self.dis_inc_fees < 0)
                print(f"Incoming fees are negative for the Distribute upper case at indices: {invalid_indices}")
                print(f"Invalid incoming fees values: {self.dis_inc_fees[invalid_indices]}")
                raise AssertionError("Incoming fees are negative for the Distribute upper case")

            if not np.all(self.dis_out_fees - self.dis_inc_fees >= -self.epsilon):
                invalid_indices = np.where(self.dis_out_fees - self.dis_inc_fees < -self.epsilon)
                print(f"Outgoing fees are significantly less than incoming fees for the Distribute upper case at indices: {invalid_indices}")
                print(f"Outgoing fees: {self.dis_out_fees[invalid_indices]}")
                print(f"Incoming fees: {self.dis_inc_fees[invalid_indices]}")
                raise AssertionError("Outgoing fees are significantly less than incoming fees for the Distribute upper case")
            
    def update_rebalance_case(self):
        """
        Update the rebalance case
        """
        L_rinc = np.sqrt(self.x_rinc * self.y_rinc)
        L_rout = np.sqrt(self.x_rout * self.y_rout)
        assert np.all(L_rinc - self.L >= -self.epsilon), "Constant product is violated for the rebalance case"
        assert np.all(L_rout - self.L >= -self.epsilon), "Constant product is violated for the rebalance case"
        
        prices = np.tile(self.prices[:, :, :, np.newaxis], (1, 1, 1, len(self.gamma_values)))
        gammas = np.tile(self.gamma_values[np.newaxis, np.newaxis, np.newaxis, :], (self.num_seeds, len(self.mu_values), len(self.sigma_values), 1))
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

        # update x_rout, y_rout, and fees for outgoing upper case
        self.x_rout[upper_mask] = np.sqrt(L_rout[upper_mask]**2 / (1-gammas[upper_mask]) / prices[upper_mask])
        self.y_rout[upper_mask] = np.sqrt(L_rout[upper_mask]**2 * (1-gammas[upper_mask]) * prices[upper_mask])
        
        # update x_rinc, y_rinc, and fees for incoming lower case
        rinc_lower_a = 1-gammas[lower_mask]
        rinc_lower_b = (2-gammas[lower_mask])*self.x_rinc[lower_mask]
        rinc_lower_c = self.x_rinc[lower_mask]**2 - L_rinc[lower_mask]**2 * (1-gammas[lower_mask]) / prices[lower_mask]
        rinc_lower_delta_x = (-rinc_lower_b + np.sqrt(rinc_lower_b**2 - 4*rinc_lower_a*rinc_lower_c)) / (2*rinc_lower_a)
        self.x_rinc[lower_mask] = self.x_rinc[lower_mask] + rinc_lower_delta_x
        self.y_rinc[lower_mask] = self.x_rinc[lower_mask] * prices[lower_mask] / (1-gammas[lower_mask])
                
        # update x_rout, y_rout, and fees for outgoing lower case
        self.x_rout[lower_mask] = np.sqrt(L_rout[lower_mask]**2 * (1-gammas[lower_mask]) / prices[lower_mask])
        self.y_rout[lower_mask] = np.sqrt(L_rout[lower_mask]**2 * prices[lower_mask] / (1-gammas[lower_mask]))
        
        if self.test_mode:
            if not np.all(np.sqrt(self.x_rinc * self.y_rinc) - L_rinc >= -self.epsilon):
                invalid_indices = np.where(np.sqrt(self.x_rinc * self.y_rinc) - L_rinc < -self.epsilon)
                print(f"Constant product is violated for the Rebalance incoming lower case at indices: {invalid_indices}")
                print(f"Invalid constant product values: {np.sqrt(self.x_rinc[invalid_indices] * self.y_rinc[invalid_indices]) - L_rinc[invalid_indices]}")
                raise AssertionError("Constant product is violated for the Rebalance incoming lower case")
            
            if not np.all(np.sqrt(self.x_rout * self.y_rout) - L_rout >= -self.epsilon):
                invalid_indices = np.where(np.sqrt(self.x_rout * self.y_rout) - L_rout < -self.epsilon)
                print(f"Constant product is violated for the Rebalance upper case at indices: {invalid_indices}")
                print(f"Invalid constant product values: {np.sqrt(self.x_rout[invalid_indices] * self.y_rout[invalid_indices]) - L_rout[invalid_indices]}")
                raise AssertionError("Constant product is violated for the Rebalance upper case")
        
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
        
        prices = self.prices[:, :, :, np.newaxis]  # shape: (n, m, s, 1)
    
        # Calculate pool values
        pv_dis = prices * self.x_dis + self.y_dis  # shape: (n, m, s, g)
        pv_rinc = prices * self.x_rinc + self.y_rinc  # shape: (n, m, s, g)
        pv_rout = prices * self.x_rout + self.y_rout  # shape: (n, m, s, g)
        
        # Create arrays for indices
        mu_indices = np.tile(np.repeat(np.arange(len(self.mu_values)), len(self.sigma_values) * len(self.gamma_values)), self.num_seeds)
        sigma_indices = np.tile(np.repeat(np.arange(len(self.sigma_values)), len(self.gamma_values)), self.num_seeds * len(self.mu_values))
        gamma_indices = np.tile(np.arange(len(self.gamma_values)), self.num_seeds * len(self.mu_values) * len(self.sigma_values))
    
        # Reshape the arrays to match the indices
        pv_dis = pv_dis.reshape(-1)
        pv_rinc = pv_rinc.reshape(-1)
        pv_rout = pv_rout.reshape(-1)
        dis_inc_fees = self.dis_inc_fees.reshape(-1)
        dis_out_fees = self.dis_out_fees.reshape(-1)
    
        # Create DataFrame directly from arrays
        results_df = pl.DataFrame({
            'mu': self.mu_values[mu_indices],
            'sigma': self.sigma_values[sigma_indices],
            'gamma': self.gamma_values[gamma_indices],
            'pv_dis': pv_dis,
            'pv_rinc': pv_rinc,
            'pv_rout': pv_rout,
            'dis_inc_fees': dis_inc_fees,
            'dis_out_fees': dis_out_fees
        })
        
        # Save to parquet file with compression
        parquet_path = f"{output_dir}/simulation_results_steps_{self.steps}.parquet"
        results_df.write_parquet(parquet_path, compression='zstd')
    
        print(f"Results saved to {parquet_path}")
    
        # Clear memory
        del results_df
        gc.collect()  
        
    def simulate_full_history(self, output_dir):
        """
        Simulate the AMM for all seeds and sigma values, recording full history
        """
        print(f"Simulating {self.num_seeds} seeds with {len(self.sigma_values)} sigma values and {len(self.gamma_values)} gamma values for {self.steps} steps")
        print(f"Range of sigma values: {max(self.sigma_values)} to {min(self.sigma_values)}")
        print(f"Range of gamma values: {max(self.gamma_values)} to {min(self.gamma_values)}")
        print(f"Simulation Starts...")
    
        # Initialize arrays to store history
        # Shape: (steps+1, num_seeds, num_mu, num_sigma, num_gamma)
        history_prices = np.zeros((self.steps + 1, self.num_seeds, len(self.mu_values), len(self.sigma_values), len(self.gamma_values)))
        history_x_dis = np.zeros((self.steps + 1, self.num_seeds, len(self.mu_values), len(self.sigma_values), len(self.gamma_values)))
        history_y_dis = np.zeros((self.steps + 1, self.num_seeds, len(self.mu_values), len(self.sigma_values), len(self.gamma_values)))
        history_dis_inc_fees = np.zeros((self.steps + 1, self.num_seeds, len(self.mu_values), len(self.sigma_values), len(self.gamma_values)))
        history_dis_out_fees = np.zeros((self.steps + 1, self.num_seeds, len(self.mu_values), len(self.sigma_values), len(self.gamma_values)))
        history_x_rinc = np.zeros((self.steps + 1, self.num_seeds, len(self.mu_values), len(self.sigma_values), len(self.gamma_values)))
        history_y_rinc = np.zeros((self.steps + 1, self.num_seeds, len(self.mu_values), len(self.sigma_values), len(self.gamma_values)))
        history_x_rout = np.zeros((self.steps + 1, self.num_seeds, len(self.mu_values), len(self.sigma_values), len(self.gamma_values)))
        history_y_rout = np.zeros((self.steps + 1, self.num_seeds, len(self.mu_values), len(self.sigma_values), len(self.gamma_values)))
        history_vDI = np.zeros((self.steps + 1, self.num_seeds, len(self.mu_values), len(self.sigma_values), len(self.gamma_values)))
        history_vDO = np.zeros((self.steps + 1, self.num_seeds, len(self.mu_values), len(self.sigma_values), len(self.gamma_values)))
        history_vRI = np.zeros((self.steps + 1, self.num_seeds, len(self.mu_values), len(self.sigma_values), len(self.gamma_values)))
        history_vRO = np.zeros((self.steps + 1, self.num_seeds, len(self.mu_values), len(self.sigma_values), len(self.gamma_values)))
        history_aDI = np.zeros((self.steps + 1, self.num_seeds, len(self.mu_values), len(self.sigma_values), len(self.gamma_values)))
        history_aDO = np.zeros((self.steps + 1, self.num_seeds, len(self.mu_values), len(self.sigma_values), len(self.gamma_values)))
        history_aRI = np.zeros((self.steps + 1, self.num_seeds, len(self.mu_values), len(self.sigma_values), len(self.gamma_values)))
        history_aRO = np.zeros((self.steps + 1, self.num_seeds, len(self.mu_values), len(self.sigma_values), len(self.gamma_values)))
        gammas = np.tile(self.gamma_values[np.newaxis, np.newaxis, np.newaxis, :], (self.num_seeds, len(self.mu_values), len(self.sigma_values), 1))
        # Store initial values
        initial_values = self.x * self.s0 + self.y
        initial_price = np.tile(self.prices[:, :, :, np.newaxis], (1, 1, 1, len(self.gamma_values)))
        history_prices[0] = initial_price
        history_x_dis[0] = self.x_dis
        history_y_dis[0] = self.y_dis
        history_x_rinc[0] = self.x_rinc
        history_y_rinc[0] = self.y_rinc
        history_x_rout[0] = self.x_rout
        history_y_rout[0] = self.y_rout
        history_vDI[0] = self.x_dis * initial_price + self.y_dis + self.dis_inc_fees - initial_values
        history_vDO[0] = self.x_dis * initial_price + self.y_dis + self.dis_out_fees - initial_values
        history_vRI[0] = self.x_rinc * initial_price + self.y_rinc - initial_values
        history_vRO[0] = self.x_rout * initial_price + self.y_rout - initial_values
        current_time = time.time()
        for step in tqdm(range(self.steps), desc="Simulating Progress"):
            self.update_prices()
            self.update_distribute_case()
            self.update_rebalance_case()
        
            # Store current values
            history_prices[step + 1] = np.tile(self.prices[:, :, :, np.newaxis], (1, 1, 1, len(self.gamma_values)))
            history_x_dis[step + 1] = self.x_dis
            history_y_dis[step + 1] = self.y_dis
            history_dis_inc_fees[step + 1] = self.dis_inc_fees
            history_dis_out_fees[step + 1] = self.dis_out_fees
            history_x_rinc[step + 1] = self.x_rinc
            history_y_rinc[step + 1] = self.y_rinc
            history_x_rout[step + 1] = self.x_rout
            history_y_rout[step + 1] = self.y_rout
            history_vDI[step + 1] = self.x_dis * history_prices[step + 1] + self.y_dis + self.dis_inc_fees - \
                (history_x_dis[0] * history_prices[step + 1] + history_y_dis[0])
            history_vDO[step + 1] = self.x_dis * history_prices[step + 1] + self.y_dis + self.dis_out_fees - \
                (history_x_dis[0] * history_prices[step + 1] + history_y_dis[0])
            history_vRI[step + 1] = self.x_rinc * history_prices[step + 1] + self.y_rinc - \
                (history_x_rinc[0] * history_prices[step + 1] + history_y_rinc[0])
            history_vRO[step + 1] = self.x_rout * history_prices[step + 1] + self.y_rout - \
                (history_x_rout[0] * history_prices[step + 1] + history_y_rout[0])
            history_aDI[step + 1] = np.maximum((history_x_dis[step] - history_x_dis[step + 1]) * history_prices[step + 1],
                                               (history_y_dis[step] - history_y_dis[step + 1])) - \
                                    np.maximum((history_x_dis[step + 1] - history_x_dis[step]) * history_prices[step + 1] / (1-gammas),
                                               (history_y_dis[step + 1] - history_y_dis[step]) / (1-gammas))
            history_aDO[step + 1] = np.maximum((history_x_dis[step] - history_x_dis[step + 1]) * history_prices[step + 1] * (1-gammas),
                                               (history_y_dis[step] - history_y_dis[step + 1]) * (1-gammas)) - \
                                    np.maximum((history_x_dis[step + 1] - history_x_dis[step]) * history_prices[step + 1],
                                               (history_y_dis[step + 1] - history_y_dis[step]))
            history_aRI[step + 1] = np.maximum((history_x_rinc[step] - history_x_rinc[step + 1]) * history_prices[step + 1],
                                               (history_y_rinc[step] - history_y_rinc[step + 1])) - \
                                    np.maximum((history_x_rinc[step + 1] - history_x_rinc[step]) * history_prices[step + 1],
                                               (history_y_rinc[step + 1] - history_y_rinc[step]))
            history_aRO[step + 1] = np.maximum((history_x_rout[step] - history_x_rout[step + 1]) * history_prices[step + 1] * (1-gammas),
                                               (history_y_rout[step] - history_y_rout[step + 1]) * (1-gammas)) - \
                                    np.maximum((history_x_rout[step + 1] - history_x_rout[step]) * history_prices[step + 1],
                                               (history_y_rout[step + 1] - history_y_rout[step]))
            
        print(f"Simulation Finished...")
        print(f"Time taken: {time.time() - current_time} seconds")
        print(f"Saving results...")
    
        # Create arrays for indices
        steps_indices = np.repeat(np.arange(self.steps + 1), self.num_seeds * len(self.mu_values) * len(self.sigma_values) * len(self.gamma_values))
        mu_indices = np.tile(np.repeat(np.arange(len(self.mu_values)), len(self.sigma_values) * len(self.gamma_values)), (self.steps + 1) * self.num_seeds)
        sigma_indices = np.tile(np.repeat(np.arange(len(self.sigma_values)), len(self.gamma_values)), (self.steps + 1) * self.num_seeds * len(self.mu_values))
        gamma_indices = np.tile(np.arange(len(self.gamma_values)), (self.steps + 1) * self.num_seeds * len(self.mu_values) * len(self.sigma_values))
    
        # Reshape the arrays to match the indices
        history_prices = history_prices.reshape(-1)
        history_x_dis = history_x_dis.reshape(-1)
        history_y_dis = history_y_dis.reshape(-1)
        history_dis_inc_fees = history_dis_inc_fees.reshape(-1)
        history_dis_out_fees = history_dis_out_fees.reshape(-1)
        history_x_rinc = history_x_rinc.reshape(-1)
        history_y_rinc = history_y_rinc.reshape(-1)
        history_x_rout = history_x_rout.reshape(-1)
        history_y_rout = history_y_rout.reshape(-1)
        history_vDI = history_vDI.reshape(-1)
        history_vDO = history_vDO.reshape(-1)
        history_vRI = history_vRI.reshape(-1)
        history_vRO = history_vRO.reshape(-1)
        history_aDI = history_aDI.reshape(-1)
        history_aDO = history_aDO.reshape(-1)
        history_aRI = history_aRI.reshape(-1)
        history_aRO = history_aRO.reshape(-1)
        # Create DataFrame directly from arrays
        results_df = pl.DataFrame({
            't': steps_indices,
            'mu': self.mu_values[mu_indices],
            'sigma': self.sigma_values[sigma_indices],
            'gamma': self.gamma_values[gamma_indices],
            'p': history_prices,
            'xD': history_x_dis,
            'yD': history_y_dis,
            'fDI': history_dis_inc_fees,
            'fDO': history_dis_out_fees,
            'xRI': history_x_rinc,
            'yRI': history_y_rinc,
            'xRO': history_x_rout,
            'yRO': history_y_rout,
            'vDI': history_vDI,
            'vDO': history_vDO,
            'vRI': history_vRI,
            'vRO': history_vRO,
            'aDI': history_aDI,
            'aDO': history_aDO,
            'aRI': history_aRI,
            'aRO': history_aRO
        })
    
        # Save to parquet file with compression
        parquet_path = f"{output_dir}/simulation_full_history_steps_{self.steps}.parquet"
        results_df.write_parquet(parquet_path, compression='zstd')
    
        print(f"Results saved to {parquet_path}")
    
        # Clear memory
        del results_df
        gc.collect()

if __name__ == "__main__":
    
    # Generate timestamp for unique filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f'/home/shiftpub/Dynamic_AMM/inf_step_exp/mc_approach/crazy_simulation_results/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    x = 100; y = 100
    num_seeds = 1
    gamma_values = np.array([0.003])
    sigma_values = np.array([0.1])
    mu_values = np.array([-1])
    steps_values = np.array([100])

    # steps_values = np.array([5, 10, 20, 50, 100, 500, 1000, 5000, 10000])
    dt = 1/(365*24)
    for steps in steps_values:
        simulator = AMMSimulator(x=x, y=y, num_seeds=num_seeds, gamma_values=gamma_values, sigma_values=sigma_values, mu_values=mu_values, steps=steps, dt=dt)
        simulator.simulate_full_history(output_dir)


    
