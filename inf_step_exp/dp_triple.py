import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm
from tqdm import tqdm

class ValueFunctionNN(nn.Module):
    def __init__(self, hidden_dim=64, normalize=True):
        super(ValueFunctionNN, self).__init__()
        self.normalize = normalize
        
        # Statistics for normalization
        self.register_buffer('p_mean', torch.tensor(1.0))
        self.register_buffer('p_std', torch.tensor(0.5))
        self.register_buffer('x_mean', torch.tensor(10000.0))
        self.register_buffer('x_std', torch.tensor(2000.0))
        self.register_buffer('y_mean', torch.tensor(10000.0))
        self.register_buffer('y_std', torch.tensor(2000.0))
        
        # Neural network layers
        self.network = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def normalize_input(self, state):
        if not self.normalize:
            return state
            
        normalized = torch.zeros_like(state)
        normalized[:, 0] = (state[:, 0] - self.p_mean) / self.p_std
        normalized[:, 1] = (state[:, 1] - self.x_mean) / self.x_std
        normalized[:, 2] = (state[:, 2] - self.y_mean) / self.y_std
        return normalized
    
    def forward(self, state):
        # Normalize the input state
        normalized_state = self.normalize_input(state)
        # Process through the network
        return self.network(normalized_state)

class AMM:
    def __init__(self, 
                 L=10000, 
                 gamma=0.003, 
                 sigma=0.5, 
                 delta_t=1, 
                 mu=0.0, 
                 fee_model='distribute', 
                 fee_source='incoming'):
        """
        Initialize AMM parameters
        
        Args:
            L: Constant product parameter (default 10000)
            gamma: Fee rate (default 0.003)
            sigma: Volatility parameter (default 0.5)
            delta_t: Time step (default 1)
            mu: Drift parameter (default 0.0)
            fee_model: Fee model type ('distribute' or 'reinvest')
            fee_source: Fee source type ('incoming' or 'outgoing')
        """
        self.L = L
        self.gamma = gamma
        self.sigma = sigma
        self.delta_t = delta_t
        self.mu = mu
        self.fee_model = fee_model
        self.fee_source = fee_source
        self.discount_factor = np.exp(-self.mu * self.delta_t)
        
        # Initialize neural network
        self.value_network = ValueFunctionNN()
        self.target_network = ValueFunctionNN()
        self.target_network.load_state_dict(self.value_network.state_dict())  # Initialize with same weights
    
    def update_target_network(self, tau=0.01):
        """Soft update target network: θ_target = τ*θ_current + (1-τ)*θ_target"""
        for target_param, current_param in zip(self.target_network.parameters(), self.value_network.parameters()):
            target_param.data.copy_(tau * current_param.data + (1.0 - tau) * target_param.data)
    
    def compute_normalization_stats(self, states):
        """
        Compute normalization statistics from a batch of states
        """
        p = states[:, 0]
        x = states[:, 1]
        y = states[:, 2]
    
        stats = {
            'p_mean': p.mean().item(),
            'p_std': max(p.std().item(), 1e-5),  # Prevent division by zero
            'x_mean': x.mean().item(),
            'x_std': max(x.std().item(), 1e-5),
            'y_mean': y.mean().item(),
            'y_std': max(y.std().item(), 1e-5)
        }
    
        # Update buffers in both networks
        for key, value in stats.items():
            getattr(self.value_network, key).fill_(value)
            getattr(self.target_network, key).fill_(value)
    
        print(f"Normalization statistics updated: {stats}")
        return stats
    
    def generate_training_data(self, num_samples):
        """
        Generate training data for the value function approximation
        
        Args:
            num_samples: Number of state samples to generate
        
        Returns:
            Tensor of shape (num_samples, 3) containing (p, x, y)
        """    
        # Generate x values uniformly between 50 and 100
        x_values = torch.FloatTensor(num_samples).uniform_(50, 150)
        
        # Calculate corresponding y values using constant product L = x*y
        y_values = self.L**2 / x_values
        
        # Calculate price bounds for each x,y pair
        price_ratio = y_values / x_values
        p_min = price_ratio * (1 - self.gamma)
        p_max = price_ratio / (1 - self.gamma)
        
        # Generate random prices within the bounds for each pair
        p_values = torch.FloatTensor(num_samples).uniform_(0, 1) * (p_max - p_min) + p_min
        
        # Stack into a single array
        states = torch.stack([p_values, x_values, y_values], dim=1)
        
        return states

    def calculate_fee_ingoing(self, p, x, y):
        """
        Calculate expected ingoing fee for distribute model
        """
        alpha = self.L * np.sqrt((1-self.gamma) * p) * np.exp(-self.sigma**2 * self.delta_t / 8)
        
        d1 = np.log((1-self.gamma)*y/(p*x)) / (self.sigma * np.sqrt(self.delta_t))
        d2 = np.log(y/((1-self.gamma)*p*x)) / (self.sigma * np.sqrt(self.delta_t))
        
        term1 = alpha * (norm.cdf(d1) + norm.cdf(-d2))
        term2 = p*x * norm.cdf(d1 - 0.5*self.sigma**2*np.sqrt(self.delta_t))
        term3 = y * norm.cdf(-d2 - 0.5*self.sigma**2*np.sqrt(self.delta_t))
        
        return (self.gamma/(1-self.gamma)) * (term1 - term2 - term3)

    def calculate_fee_outgoing(self, p, x, y):
        """
        Calculate expected outgoing fee
        """
        alpha = self.L * np.sqrt((1-self.gamma) * p) * np.exp(-self.sigma**2 * self.delta_t / 8)
        
        d1 = np.log((1-self.gamma)*y/(p*x)) / (self.sigma * np.sqrt(self.delta_t))
        d2 = np.log(y/((1-self.gamma)*p*x)) / (self.sigma * np.sqrt(self.delta_t))
        
        term1 = alpha * (norm.cdf(d1) + norm.cdf(-d2))
        term2 = p*x * norm.cdf(-d2 + 0.5*self.sigma**2*np.sqrt(self.delta_t))
        term3 = y * norm.cdf(d1 + 0.5*self.sigma**2*np.sqrt(self.delta_t))
        
        return self.gamma * (-term1 + term2 + term3)

    def calculate_grid_targets(self):
        """
        Calculate target values using triple integration over x, p, and p'
    
        This function generates a grid of states by integrating over x and p dimensions,
        then calculates targets by integrating over future prices p', properly accounting
        for probability distributions and integration weights in all dimensions.
    
        Returns:
            states: Generated states of shape (n_states, 3) containing (p, x, y)
            targets: Target values for the generated states
        """
        # Number of integration points for each dimension
        n_x_points = 100  # Points for x dimension
        n_p_points = 50   # Points for each p dimension (given an x)
        n_p_prime_points = 100  # Points for future price integration
    
        # Total number of states we'll generate
        n_states = n_x_points * n_p_points
    
        # Initialize arrays for states and targets
        all_states = []
        all_targets = []
    
        # Set up x-dimension integration
        x_min, x_max = 9500.0, 10500.0
        x_points, x_weights = np.polynomial.legendre.leggauss(n_x_points)
    
        # Transform x points from [-1, 1] to [x_min, x_max]
        x_points = 0.5 * (x_max - x_min) * x_points + 0.5 * (x_max + x_min)
        x_transformed_weights = x_weights * 0.5 * (x_max - x_min)
    
        # Define PDF for x (uniform distribution)
        x_pdf_values = np.ones_like(x_points) / (x_max - x_min)
    
        # For each x point
        for x_idx, x in tqdm(enumerate(x_points), desc="X points", total=len(x_points)):
            # Calculate corresponding y based on constant product formula
            y = self.L**2 / x
        
            # Calculate valid price range for this (x, y) pair
            price_ratio = y / x
            p_min = price_ratio * (1 - self.gamma)
            p_max = price_ratio / (1 - self.gamma)
        
            # Set up p-dimension integration
            p_points, p_weights = np.polynomial.legendre.leggauss(n_p_points)
        
            # Transform p points from [-1, 1] to [p_min, p_max]
            p_points = 0.5 * (p_max - p_min) * p_points + 0.5 * (p_max + p_min)
            p_transformed_weights = p_weights * 0.5 * (p_max - p_min)
        
            # Define PDF for p (uniform distribution within valid range)
            p_pdf_values = np.ones_like(p_points) / (p_max - p_min)
        
            # For each p point
            for p_idx, p in enumerate(p_points):
                # Create state (p, x, y)
                state = torch.tensor([p, x, y], dtype=torch.float32).to(self.device)
                all_states.append(state)
            
                # Calculate immediate reward for this state
                immediate_reward = p * x + y
            
                # Now calculate expected future value by integrating over p'
                # Set up p'-dimension integration
                log_p = np.log(p)
                log_p_mean = log_p + (self.mu - 0.5 * self.sigma**2) * self.delta_t
                log_p_std = self.sigma * np.sqrt(self.delta_t)
            
                # Set integration bounds for log-p'
                log_p_min = log_p_mean - 4 * log_p_std
                log_p_max = log_p_mean + 4 * log_p_std
            
                # Get quadrature points and weights for p' integration
                p_prime_points, p_prime_weights = np.polynomial.legendre.leggauss(n_p_prime_points)
            
                # Transform to log-price space
                log_p_prime_points = 0.5 * (log_p_max - log_p_min) * p_prime_points + 0.5 * (log_p_max + log_p_min)
                p_prime_transformed_weights = p_prime_weights * 0.5 * (log_p_max - log_p_min)
            
                # Convert to actual prices
                p_prime_points = torch.tensor(np.exp(log_p_prime_points), dtype=torch.float32).to(self.device)
            
                # Calculate new states based on AMM mechanics
                p_upper = price_ratio / (1 - self.gamma)
                p_lower = price_ratio * (1 - self.gamma)
            
                new_x = torch.zeros_like(p_prime_points).to(self.device)
                new_y = torch.zeros_like(p_prime_points).to(self.device)
            
                # Apply AMM rules using masks
                above_mask = p_prime_points > p_upper
                below_mask = p_prime_points < p_lower
                within_mask = ~(above_mask | below_mask)
            
                if self.fee_model == 'distribute':
                    # Calculate updated x,y 
                    new_x[above_mask] = self.L / torch.sqrt((1 - self.gamma) * p_prime_points[above_mask])
                    new_y[above_mask] = self.L * torch.sqrt((1 - self.gamma) * p_prime_points[above_mask])
                    new_y[below_mask] = self.L * torch.sqrt(p_prime_points[below_mask] / (1 - self.gamma))
                    new_x[below_mask] = self.L * torch.sqrt((1 - self.gamma) / p_prime_points[below_mask])
            
                new_x[within_mask] = x
                new_y[within_mask] = y
            
                # Verify constant product condition
                constant_product = torch.full_like(new_x, self.L**2)
                if not torch.allclose(new_x * new_y, constant_product, rtol=1e-4):
                    max_deviation = torch.max(torch.abs(new_x * new_y - self.L**2))
                    print(f"Warning: Constant product condition violated: max deviation = {max_deviation}")
            
                # Stack future states
                next_states = torch.stack([p_prime_points, new_x, new_y], dim=1)
            
                # Get value estimates from target network
                with torch.no_grad():
                    values = self.target_network(next_states).squeeze()
            
                # Calculate PDF values for log-normal distribution (p' given p)
                log_terms = (torch.log(p_prime_points/p) - (self.mu - 0.5 * self.sigma**2) * self.delta_t)**2
                denominator = 2 * self.sigma**2 * self.delta_t
                pi_term = torch.tensor(np.sqrt(2 * np.pi * self.delta_t), dtype=torch.float32).to(self.device)
                p_prime_pdf_values = torch.exp(-log_terms / denominator) / (p_prime_points * self.sigma * pi_term)  
                # Calculate expected value through integration over p'
                integrand = values * p_prime_pdf_values
                p_prime_integral = torch.sum(integrand * torch.tensor(p_prime_transformed_weights, dtype=torch.float32))
            
                # Add fees if using distribute model
                if self.fee_model == 'distribute':
                    if self.fee_source == 'incoming':
                        fee = self.calculate_fee_ingoing(p, x, y)
                    elif self.fee_source == 'outgoing':
                        fee = self.calculate_fee_outgoing(p, x, y)
                    p_prime_integral += fee
            
                # Calculate target using Bellman equation (max of immediate reward and discounted future value)
                target_value = max(immediate_reward, self.discount_factor * p_prime_integral)
            
                # Weight the target by the probability of this state (x, p)
                # This accounts for the PDF values and integration weights of x and p dimensions
                all_targets.append(target_value)
    
        # Convert lists to tensors
        states_tensor = torch.stack(all_states)
        targets_tensor = torch.tensor(all_targets, dtype=torch.float32).unsqueeze(1)
    
        return states_tensor, targets_tensor

    def train_value_function(self, num_epochs=100, batch_size=64, learning_rate=0.001):
        """
        Train the value function using grid integration for state and target generation
        """
        # Get states and targets using grid integration
        print("Generating grid states and calculating targets...")
        states, targets = self.calculate_grid_targets()
    
        # Compute normalization statistics
        self.compute_normalization_stats(states)
    
        # Create dataset and dataloader
        dataset = TensorDataset(states, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
        # Define optimizer
        optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95)
        # Loss history
        losses = []
    
        # Training loop
        print(f"Training value function on {len(states)} grid states...")
        for epoch in range(num_epochs):
            epoch_losses = []
        
            for batch_states, batch_targets in dataloader:
                # Forward pass
                predicted_values = self.value_network(batch_states)
            
                # Calculate loss
                loss = nn.MSELoss()(predicted_values, batch_targets)
            
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)
                optimizer.step()
            
                epoch_losses.append(loss.item())
        
            # Record average loss
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            scheduler.step()
            # Update target network
            if (epoch + 1) % 100 == 0:
                self.update_target_network(tau=0.001)
                
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # save the value network
        torch.save(self.value_network.state_dict(), f'value_network_triple_{self.fee_model}_{self.fee_source}.pth')
    
        return losses

def main():
    # Initialize AMM with different fee models
    distribute_amm = AMM(
        L=10000,
        gamma=0.003,
        sigma=0.5,
        delta_t=1,
        mu=0.0,
        fee_model='distribute',
        fee_source='incoming'
    )
    
    # Train both models
    print("\nTraining value function for distribute fee model...")
    distribute_losses = distribute_amm.train_value_function(
        num_epochs=2000,
        batch_size=64,
        learning_rate=0.0003
    )
    # save the losses to a csv
    np.savetxt('distribute_losses.csv', distribute_losses, delimiter=',')
    
    # Plot training losses
    plt.figure(figsize=(10, 6))
    plt.plot(distribute_losses, label='Distribute Fee Model')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('amm_rl_training_loss.png')
    plt.show()

if __name__ == "__main__":
    main()