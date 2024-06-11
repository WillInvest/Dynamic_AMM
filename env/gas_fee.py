import numpy as np
import matplotlib.pyplot as plt


class GasFeeSimulator:
    def __init__(self, start_price=1e-4, mu=0, sigma=0.2, epsilon=0.01, dt=0.01, deterministic=False, steps=500):
        self.initial_price = start_price
        self.price = start_price
        self.mu = mu
        self.sigma = sigma
        self.epsilon = epsilon
        self.sigma = sigma
        self.dt = dt
        self.deterministic = deterministic  # Flag to control stochastic/deterministic behavior
        self.shock_index = 0  # Index to track the current shock
        self.steps = steps
        self.path = self.get_zigzag(steps=self.steps, high=1e-3, low=1e-5)
        self.min_gwei = 1
        self.max_gwei = 1000
        self.spread = self.max_gwei - self.min_gwei
        
    def get_zigzag(self, steps, high, low):
        
        # Define the basic zigzag pattern: rise to 1.5, drop to 0.5, return to 1
        rise_steps = steps // 3
        drop_steps = steps // 3
        return_steps = steps - (rise_steps + drop_steps)
        
        # Create the rise sequence from 1 to 1.5
        rise_sequence = np.linspace(1, high, rise_steps)
        
        # Create the drop sequence from 1.5 to 0.5
        drop_sequence = np.linspace(high, low, drop_steps)
        
        # Create the return sequence from 0.5 to 1
        return_sequence = np.linspace(low, 1, return_steps)
        
        # Concatenate the sequences to form the full zigzag pattern
        return np.concatenate((rise_sequence, drop_sequence, return_sequence))
    
    def next(self):
        if self.deterministic:
            
            # Use a predetermined shock from the list
            self.price = self.initial_price * self.path[self.shock_index%self.steps] 
            self.shock_index += 1
        else:
            # Stochastic update, random shock
            shock = np.random.normal()
            # Update the current price using the GBM formula
            self.price *= np.exp(
                (self.mu - 0.5 * self.sigma ** 2) * self.dt + self.sigma * np.sqrt(self.dt) * shock)
        

    def reset(self):
        self.price = self.initial_price
        self.shock_index = 0  # Reset shock index
