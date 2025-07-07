import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm

class AMM:
    def __init__(self, 
                 L=100, 
                 gamma=0.003, 
                 sigma=0.5, 
                 delta_t=1, 
                 mu=0.0, 
                 fee_model='distribute', 
                 device='cuda'):
        """
        Initialize AMM parameters
        
        Args:
            L: Constant product parameter (default 100)
            gamma: Fee rate (default 0.003)
            sigma: Volatility parameter (default 0.5)
            delta_t: Time step (default 1)
            mu: Drift parameter (default 0.0)
            fee_model: Fee model type ('distribute' or 'reinvest')
            fee_source: Fee source type ('incoming' or 'outgoing')
        """
        self.device = device
        self.L = L
        self.gamma = gamma
        self.sigma = sigma
        self.delta_t = delta_t
        self.mu = mu
        self.fee_model = fee_model
        self.discount_factor = np.exp(-self.mu * self.delta_t)
        
    def reset(self, sigma=None, gamma=None):
        if sigma is not None:
            self.sigma = sigma
        if gamma is not None:
            self.gamma = gamma
    
    def generate_training_data(self, k: float, num_samples: int) -> torch.Tensor:
        """
        Generate training data with price p fixed at price ratio y/x
    
        Args:
            num_samples: Number of samples to generate
            k: relative position of the price p
                - k in [-1, 1]
                - k = -1: price is at the lower bound
                - k = 1: price is at the upper bound
                - k = 0: price is at the middle
        Returns:
            Tensor of shape (num_samples, 3) containing (p, x, y)
        """
        # Generate data on CPU efficiently
        min_x = self.L * 0.5
        max_x = self.L * 2
        x_values = np.linspace(min_x, max_x, num_samples)
        y_values = self.L**2 / x_values
        p_values = (y_values / x_values) / (1 - self.gamma)**(k)
        
        # Stack into a numpy array
        states_np = np.stack([p_values, x_values, y_values], axis=1)
        
        # Convert to tensor in single operation and move to device
        states = torch.tensor(states_np, dtype=torch.float64, device=self.device)
        
        return states
    
    def calculate_fee(self, p, x, y):
        alpha = self.L * np.sqrt((1-self.gamma) * p) * np.exp(self.mu*self.delta_t/2 - self.sigma**2 * self.delta_t/8)
        beta = self.L * ((2-self.gamma)/np.sqrt(1-self.gamma)) * np.sqrt(p) * np.exp((0.5*self.mu-0.125*self.sigma**2)*self.delta_t)
        d1 = (np.log((1-self.gamma)*y/(p*x)) - self.mu*self.delta_t) / (self.sigma * np.sqrt(self.delta_t))
        d2 = (np.log(y/((1-self.gamma)*p*x)) - self.mu*self.delta_t) / (self.sigma * np.sqrt(self.delta_t))
        incoming_term1 = alpha * (norm.cdf(d1) + norm.cdf(-d2))
        incoming_term2 = np.exp(self.mu*self.delta_t) * p*x * norm.cdf(d1 - 0.5*self.sigma*np.sqrt(self.delta_t))
        incoming_term3 = y * norm.cdf(-d2 - 0.5*self.sigma*np.sqrt(self.delta_t))
        outgoing_term1 = alpha * (norm.cdf(d1) + norm.cdf(-d2))
        outgoing_term2 = np.exp(self.mu*self.delta_t) * p*x * norm.cdf(-d2 + 0.5*self.sigma*np.sqrt(self.delta_t))
        outgoing_term3 = y * norm.cdf(d1 + 0.5*self.sigma*np.sqrt(self.delta_t))
        pool_term1 = beta * (norm.cdf(d1) + norm.cdf(-d2))
        pool_term2 = 2 * self.L * np.sqrt(p) * (norm.cdf(d2 + 0.5*self.sigma*np.sqrt(self.delta_t)) - norm.cdf(d1 + 0.5*self.sigma*np.sqrt(self.delta_t)))
        
        incoming_fee = (self.gamma/(1-self.gamma)) * (incoming_term1 - incoming_term2 - incoming_term3)
        outgoing_fee = self.gamma * (-outgoing_term1 + outgoing_term2 + outgoing_term3)
        pool_value = pool_term1 + pool_term2
        
        return incoming_fee, outgoing_fee, pool_value

    def calculate_fee_grid(self, x=1000, y=1000, p=1):
        """
        Calculate fees and pool value for a grid of sigma and gamma values using fully vectorized operations
        
        Args:
            x: Amount of asset X (default 1000)
            y: Amount of asset Y (default 1000)
            p: Price ratio (default 1)
            
        Returns:
            Dictionary containing:
            - sigma_grid: Array of sigma values
            - gamma_grid: Array of gamma values
            - incoming_fee_grid: 2D array of incoming fees
            - outgoing_fee_grid: 2D array of outgoing fees
            - pool_value_grid: 2D array of pool values
        """
        # Create grids for sigma and gamma
        sigma_values = np.round(np.arange(0.2, 2.1, 0.2), 2)
        gamma_values = np.round(np.arange(0.0005, 0.0505, 0.0005), 4)
        sigma_grid, gamma_grid = np.meshgrid(sigma_values, gamma_values)
        
        # Calculate L
        L = np.sqrt(x * y)
        
        # Pre-calculate common terms
        sqrt_delta_t = np.sqrt(self.delta_t)
        mu_delta_t = self.mu * self.delta_t
        
        # Calculate gamma-dependent terms (all with same shape as input grid)
        one_minus_gamma = 1 - gamma_grid
        sqrt_one_minus_gamma = np.sqrt(one_minus_gamma)
        gamma_ratio = gamma_grid / one_minus_gamma
        
        # Calculate sigma-dependent terms
        sigma_sq = sigma_grid**2
        
        # Calculate alpha and beta grids
        alpha_grid_incoming = L * np.sqrt(one_minus_gamma * p) * np.exp(mu_delta_t/2 - sigma_sq * self.delta_t/8)
        alpha_grid_outgoing = L * np.sqrt(p/one_minus_gamma) * np.exp(mu_delta_t/2 - sigma_sq * self.delta_t/8)
        
        beta_grid = L * ((2-gamma_grid)/sqrt_one_minus_gamma) * np.sqrt(p) * np.exp((0.5*self.mu-0.125*sigma_sq)*self.delta_t)
        
        # Calculate d1 and d2 grids
        d1_grid = (np.log(one_minus_gamma*y/(p*x)) - mu_delta_t) / (sigma_grid * sqrt_delta_t)
        d2_grid = (np.log(y/(one_minus_gamma*p*x)) - mu_delta_t) / (sigma_grid * sqrt_delta_t)
        d1_plus_grid = d1_grid + 0.5*sigma_grid*sqrt_delta_t
        d2_plus_grid = d2_grid + 0.5*sigma_grid*sqrt_delta_t
        d1_minus_grid = d1_grid - 0.5*sigma_grid*sqrt_delta_t
        d2_minus_grid = d2_grid - 0.5*sigma_grid*sqrt_delta_t
        
        # Calculate all terms using the grids (all with same shape)
        incoming_term1_grid = alpha_grid_incoming * (norm.cdf(d1_grid) + norm.cdf(-d2_grid))
        incoming_term2_grid = np.exp(mu_delta_t) * p*x * norm.cdf(d1_minus_grid)
        incoming_term3_grid = y * norm.cdf(-d2_plus_grid)
        
        outgoing_term1_grid = alpha_grid_outgoing * (norm.cdf(d1_grid) + norm.cdf(-d2_grid))
        outgoing_term2_grid = np.exp(mu_delta_t) * p*x * norm.cdf(-d2_minus_grid)
        outgoing_term3_grid = y * norm.cdf(d1_plus_grid)
        
        pool_term1_grid = beta_grid * (norm.cdf(d1_grid) + norm.cdf(-d2_grid))
        pool_term2_grid = 2 * L * np.sqrt(p) * (norm.cdf(d2_plus_grid) - norm.cdf(d1_plus_grid))
        
        # Calculate final results using the grids
        incoming_fee_grid = gamma_ratio * (incoming_term1_grid - incoming_term2_grid - incoming_term3_grid)
        outgoing_fee_grid = gamma_grid * (-outgoing_term1_grid + outgoing_term2_grid + outgoing_term3_grid)
        pool_value_grid = pool_term1_grid + pool_term2_grid
        
        results = {
                      'sigma_grid': sigma_grid,
                      'gamma_grid': gamma_grid,
                      'incoming_fee_grid': incoming_fee_grid,
                      'outgoing_fee_grid': outgoing_fee_grid,
                      'pool_value_grid': pool_value_grid
                  }
        
        
        
        return results
    
    
if __name__ == "__main__":
    amm = AMM(delta_t=1/(365*24))
    results = amm.calculate_fee_grid()
    
    # Flatten the grids and create a DataFrame
    df = pd.DataFrame({
        'sigma': results['sigma_grid'].flatten(),
        'gamma': results['gamma_grid'].flatten(),
        'incoming_fee': results['incoming_fee_grid'].flatten(),
        'outgoing_fee': results['outgoing_fee_grid'].flatten(),
        'pool_value': results['pool_value_grid'].flatten()
    })
    
    # sort by sigma and gamma
    df = df.sort_values(by=['sigma', 'gamma'])
    
    # Display the first few rows
    print("\nFirst few rows of the results:")
    print(df.head())
    
    # Display some basic statistics
    print("\nBasic statistics:")
    print(df.describe())
    
    # Optional: Save to CSV
    df.to_csv('fee_grid_results.csv', index=False)
