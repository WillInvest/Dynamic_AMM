import numpy as np
import os
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
                 fee_source='incoming',
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
        self.fee_source = fee_source
        self.discount_factor = np.exp(-self.mu * self.delta_t)
    
    def generate_training_data(self, num_samples: int) -> torch.Tensor:
        """
        Generate training data with price p fixed at price ratio y/x
    
        Args:
            num_samples: Number of samples to generate
        
        Returns:
            Tensor of shape (num_samples, 3) containing (p, x, y)
        """
        # Generate data on CPU efficiently
        min_x = self.L * 1
        max_x = self.L * 1
        x_values = np.linspace(min_x, max_x, num_samples)
        y_values = self.L**2 / x_values
        p_values = y_values / x_values
        
        # Stack into a numpy array
        states_np = np.stack([p_values, x_values, y_values], axis=1)
        
        # Convert to tensor in single operation and move to device
        states = torch.tensor(states_np, dtype=torch.float64, device=self.device)
        
        return states
    
    # def calculate_ingoing_fee(self, p, x, y):
    #     """
    #     Calculate expected ingoing fee for distribute model
    #     Works with either numpy arrays or single values
    #     """
    #     alpha = self.L * np.sqrt((1-self.gamma) * p) * np.exp(self.mu*self.delta_t/2 - self.sigma**2 * self.delta_t/8)
        
    #     d1 = (np.log((1-self.gamma)*y/(p*x)) - self.mu*self.delta_t) / (self.sigma * np.sqrt(self.delta_t))
    #     d2 = (np.log(y/((1-self.gamma)*p*x)) - self.mu*self.delta_t) / (self.sigma * np.sqrt(self.delta_t))
        
    #     term1 = alpha * (norm.cdf(d1) + norm.cdf(-d2))
    #     term2 = np.exp(self.mu*self.delta_t) * p*x * norm.cdf(d1 - 0.5*self.sigma*np.sqrt(self.delta_t))
    #     term3 = y * norm.cdf(-d2 - 0.5*self.sigma*np.sqrt(self.delta_t))
        
    #     return (self.gamma/(1-self.gamma)) * (term1 - term2 - term3)
    
    # def calculate_outgoing_fee(self, p, x, y):
    #     """
    #     Calculate expected outgoing fee
    #     Works with either numpy arrays or single values
    #     """
    #     alpha = self.L * np.sqrt((1-self.gamma) * p) * np.exp(self.mu*self.delta_t/2 - self.sigma**2 * self.delta_t/8)
        
    #     d1 = (np.log((1-self.gamma)*y/(p*x)) - self.mu*self.delta_t) / (self.sigma * np.sqrt(self.delta_t))
    #     d2 = (np.log(y/((1-self.gamma)*p*x)) - self.mu*self.delta_t) / (self.sigma * np.sqrt(self.delta_t))
        
    #     term1 = alpha * (norm.cdf(d1) + norm.cdf(-d2))
    #     term2 = np.exp(self.mu*self.delta_t) * p*x * norm.cdf(-d2 + 0.5*self.sigma*np.sqrt(self.delta_t))
    #     term3 = y * norm.cdf(d1 + 0.5*self.sigma*np.sqrt(self.delta_t))
        
    #     return self.gamma * (-term1 + term2 + term3)
    
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


    # def calculate_fee_ingoing(self, p, x, y):
    #     """
    #     Calculate expected ingoing fee for distribute model
    #     Works with either numpy arrays or single values
    #     """
    #     alpha = self.L * np.sqrt((1-self.gamma) * p) * np.exp(-self.sigma**2 * self.delta_t / 8)
        
    #     d1 = np.log((1-self.gamma)*y/(p*x)) / (self.sigma * np.sqrt(self.delta_t))
    #     d2 = np.log(y/((1-self.gamma)*p*x)) / (self.sigma * np.sqrt(self.delta_t))
        
    #     term1 = alpha * (norm.cdf(d1) + norm.cdf(-d2))
    #     term2 = p*x * norm.cdf(d1 - 0.5*self.sigma*np.sqrt(self.delta_t))
    #     term3 = y * norm.cdf(-d2 - 0.5*self.sigma*np.sqrt(self.delta_t))
        
    #     return (self.gamma/(1-self.gamma)) * (term1 - term2 - term3)

    # def calculate_fee_outgoing(self, p, x, y):
    #     """
    #     Calculate expected outgoing fee
    #     Works with either numpy arrays or single values
    #     """
    #     alpha = self.L * np.sqrt((1-self.gamma) * p) * np.exp(-self.sigma**2 * self.delta_t / 8)
        
    #     d1 = np.log((1-self.gamma)*y/(p*x)) / (self.sigma * np.sqrt(self.delta_t))
    #     d2 = np.log(y/((1-self.gamma)*p*x)) / (self.sigma * np.sqrt(self.delta_t))
        
    #     term1 = alpha * (norm.cdf(d1) + norm.cdf(-d2))
    #     term2 = p*x * norm.cdf(-d2 + 0.5*self.sigma*np.sqrt(self.delta_t))
    #     term3 = y * norm.cdf(d1 + 0.5*self.sigma*np.sqrt(self.delta_t))
        
    #     return self.gamma * (-term1 + term2 + term3)
