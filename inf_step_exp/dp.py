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
        
        # Initialize neural network with L parameter
        self.value_network = ValueFunctionNN()
        self.target_network = ValueFunctionNN()
        self.target_network.load_state_dict(self.value_network.state_dict())  # Initialize with same weights
    
        self.value_network.to(device)
        self.target_network.to(device)
        
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

        print(f"p_mean: {stats['p_mean']}, p_std: {stats['p_std']}, x_mean: {stats['x_mean']}, x_std: {stats['x_std']}, y_mean: {stats['y_mean']}, y_std: {stats['y_std']}")
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
        max_x = 10500
        min_x = 9500
        x_values = torch.FloatTensor(num_samples).uniform_(min_x, max_x)
        
        # Calculate corresponding y values using constant product L = x*y
        y_values = self.L**2 / x_values
        
        # Calculate price bounds for each x,y pair
        price_ratio = y_values / x_values
        p_min = price_ratio * (1 - self.gamma)
        p_max = price_ratio / (1 - self.gamma)
        
        # Generate random prices within the bounds for each pair
        p_values = torch.FloatTensor(num_samples).uniform_(0, 1) * (p_max - p_min) + p_min
        
        # Stack into a single array
        states = torch.stack([p_values, x_values, y_values], dim=1).to(self.device)
        
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

    def calculate_batch_targets(self, states):
        """
        Calculate target values for training (max of immediate reward and expected future value)
        
        Args:
            states: Current states (p, x, y) of shape (batch_size, 3)
        
        Returns:
            Target values for training
        """
        batch_size = states.shape[0]
        expected_values = torch.zeros(batch_size, 1).to(self.device)
        
        # Calculate immediate rewards (p*x + y)
        p, x, y = states[:, 0], states[:, 1], states[:, 2]
        immediate_reward = p * x + y
        
        # Pre-compute quadrature points and weights
        num_points = 100
        points, weights = np.polynomial.legendre.leggauss(num_points)
        
        # Calculate expected value for each state
        for b in range(batch_size):
            p = states[b, 0].item()
            x = states[b, 1].item()
            y = states[b, 2].item()
            
            # Setup distribution parameters
            log_p = np.log(p)
            log_p_mean = log_p + (self.mu - 0.5 * self.sigma**2) * self.delta_t
            log_p_std = self.sigma * np.sqrt(self.delta_t)
            
            # Transform integration points
            log_p_min = log_p_mean - 3 * log_p_std
            log_p_max = log_p_mean + 3 * log_p_std
            log_p_points = 0.5 * (log_p_max - log_p_min) * points + 0.5 * (log_p_max + log_p_min)
            transformed_weights = weights * 0.5 * (log_p_max - log_p_min)
            
            # Create price points tensor
            p_points = torch.tensor(np.exp(log_p_points), dtype=torch.float32).to(self.device)
            
            # Calculate new states based on AMM mechanics
            price_ratio = y / x
            p_upper = price_ratio / (1 - self.gamma)
            p_lower = price_ratio * (1 - self.gamma)
            
            new_x = torch.zeros_like(p_points).to(self.device)
            new_y = torch.zeros_like(p_points).to(self.device)
            # Apply AMM rules using masks
            above_mask = p_points > p_upper
            below_mask = p_points < p_lower
            within_mask = ~(above_mask | below_mask)
            
            if self.fee_model == 'distribute':
                # calculate updated x,y 
                new_x[above_mask] = self.L / torch.sqrt((1 - self.gamma) * p_points[above_mask])
                new_y[above_mask] = self.L * torch.sqrt((1 - self.gamma) * p_points[above_mask])
                new_y[below_mask] = self.L * torch.sqrt(p_points[below_mask] / (1 - self.gamma))
                new_x[below_mask] = self.L * torch.sqrt((1 - self.gamma) / p_points[below_mask])
            
            new_x[within_mask] = x
            new_y[within_mask] = y
            
            # make sure new_x * new_y = L^2
            constant_product = torch.full_like(new_x, self.L**2).to(self.device)
            assert torch.allclose(new_x * new_y, constant_product, rtol=1e-4), \
                f"Constant product condition violated: max deviation = {torch.max(torch.abs(new_x * new_y - self.L**2))}"
            
            next_states = torch.stack([p_points, new_x, new_y], dim=1).to(self.device)
            
            # Process all points at once
            with torch.no_grad():
                values = self.target_network(next_states).squeeze()
            
            # Calculate PDF values
            log_terms = (torch.log(p_points/p) - (self.mu - 0.5 * self.sigma**2) * self.delta_t)**2
            denominator = 2 * self.sigma**2 * self.delta_t
            pi_term = torch.tensor(np.sqrt(2 * np.pi * self.delta_t), dtype=torch.float32).to(self.device)
            pdf_values = torch.exp(-log_terms / denominator) / (p_points * self.sigma * pi_term)    
                    
            # Calculate integrand and expected value
            integrand = values * pdf_values
            expected_values[b] = torch.sum(integrand * torch.tensor(transformed_weights, dtype=torch.float32).to(self.device))
            
            # Add fees if using distribute model
            if self.fee_model == 'distribute':
                if self.fee_source == 'incoming':
                    fee = self.calculate_fee_ingoing(p, x, y)
                elif self.fee_source == 'outgoing':
                    fee = self.calculate_fee_outgoing(p, x, y)
                expected_values[b] += fee
        
        # Calculate target as maximum of immediate reward and expected future value
        target = torch.maximum(immediate_reward.unsqueeze(1), self.discount_factor * expected_values)
        return target

    def train_value_function(self, num_epochs=100, batch_size=128, learning_rate=0.001):
        """
        Train the value function approximation network
        """
        # Define optimizer
        optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        # Loss history
        losses = []
        states = self.generate_training_data(num_samples=5000)
        self.compute_normalization_stats(states)
        # Print header
        header = "| {:^5} | {:^20} | {:^12} | {:^10} |".format("Epoch", "Loss", "LR", "Tau")
        separator = "|" + "-" * 7 + "|" + "-" * 22 + "|" + "-" * 14 + "|" + "-" * 12 + "|"
        tqdm.write("\nTraining Progress:")
        tqdm.write(separator)
        tqdm.write(header)
        tqdm.write(separator)
        
        # Training loop
        pbar = tqdm(range(num_epochs), desc="Training value function", disable=False)
        for epoch in pbar:
            epoch_losses = []
            # self.compute_normalization_stats(states)
            # Create data loader
            dataset = TensorDataset(states)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            for batch_states, in dataloader:
                # Calculate target values
                target = self.calculate_batch_targets(batch_states)
                
                # Forward pass
                predicted_value = self.value_network(batch_states)
                
                # Calculate loss
                loss = nn.MSELoss()(predicted_value, target)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            # Record average loss
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            # Step the scheduler
            # scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Update target network
            tau = 0.0001
            self.update_target_network(tau=tau)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 1 == 0:
                progress = "| {:>5d} | {:>12.6f} | {:>12.6f} | {:>10.6f} |".format(
                    epoch + 1, avg_loss, current_lr, tau
                )
                tqdm.write(progress)
        
        # Print final separator
        tqdm.write(separator)
        
        # save the value network
        torch.save(self.value_network.state_dict(), f'mc_value_network_{self.fee_model}_{self.fee_source}.pth')
        
        return losses

def main():
    # Check if CUDA (GPU) is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize AMM with different fee models
    for fee_source in ['incoming', 'outgoing']:
    
        distribute_amm = AMM(
            L=10000,
            gamma=0.003,
            sigma=0.5,
            delta_t=1,
            mu=0.0,
            fee_model='distribute',
            fee_source=fee_source,
            device=device
        )
    
        # Train both models
        print(f"\nTraining value function for distribute {fee_source} fee model...")
        distribute_losses = distribute_amm.train_value_function(
            num_epochs=500,
            batch_size=256,
            learning_rate=0.03
        )

if __name__ == "__main__":
    main()