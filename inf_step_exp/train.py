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

# Fix the deprecated warning while maintaining precision
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    # Set default device instead of tensor type
    torch.set_default_device('cuda')

class ValueFunctionNN(nn.Module):
    def __init__(self, L, hidden_dim=64, normalize=True):
        super(ValueFunctionNN, self).__init__()
        self.normalize = normalize
        self.L = L
        
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
    
    def normalize_input(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: torch.Tensor of shape (batch_size, 3) containing (p, x, y)
        Returns:
            torch.Tensor of normalized state values
        """
        if not self.normalize:
            return state
            
        normalized: torch.Tensor = torch.zeros_like(state, dtype=torch.float64)  # shape: (batch_size, 3)
        normalized[:, 0] = state[:, 0]  # Keep price as is
        normalized[:, 1] = state[:, 1] / self.L
        normalized[:, 2] = state[:, 2] / self.L
        return normalized
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: torch.Tensor of shape (batch_size, 3) containing (p, x, y)
        Returns:
            torch.Tensor of shape (batch_size, 1) containing predicted values
        """
        # Normalize the input state
        normalized_state: torch.Tensor = self.normalize_input(state)
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
        
        # Pre-compute constants
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.pi_term = np.sqrt(2 * np.pi * self.delta_t)
        self.alpha_factor = np.exp(-self.sigma**2 * self.delta_t / 8)
        
        # Initialize neural network with L parameter
        self.value_network = ValueFunctionNN(L=L)
        self.target_network = ValueFunctionNN(L=L)
        self.target_network.load_state_dict(self.value_network.state_dict())  # Initialize with same weights
    
        self.value_network.to(device)
        self.target_network.to(device)
        
    def update_target_network(self, tau=0.01):
        """Soft update target network: θ_target = τ*θ_current + (1-τ)*θ_target"""
        for target_param, current_param in zip(self.target_network.parameters(), self.value_network.parameters()):
            target_param.data.copy_(tau * current_param.data + (1.0 - tau) * target_param.data)
    
    def generate_training_data(self, num_samples: int) -> torch.Tensor:
        """
        Generate training data with price p fixed at price ratio y/x
    
        Args:
            num_samples: Number of samples to generate
        
        Returns:
            Tensor of shape (num_samples, 3) containing (p, x, y)
        """
        # Generate data on CPU efficiently
        min_x = 9500
        max_x = 10500
        x_values = np.linspace(min_x, max_x, num_samples)
        y_values = self.L**2 / x_values
        p_values = y_values / x_values
        
        # Stack into a numpy array
        states_np = np.stack([p_values, x_values, y_values], axis=1)
        
        # Convert to tensor in single operation and move to device
        states = torch.tensor(states_np, dtype=torch.float64, device=self.device)
        
        return states

    def calculate_fee_ingoing(self, p, x, y):
        """
        Calculate expected ingoing fee for distribute model
        Works with either numpy arrays or single values
        """
        alpha = self.L * np.sqrt((1-self.gamma) * p) * self.alpha_factor
        
        d1 = np.log((1-self.gamma)*y/(p*x)) / (self.sigma * self.sqrt_delta_t)
        d2 = np.log(y/((1-self.gamma)*p*x)) / (self.sigma * self.sqrt_delta_t)
        
        term1 = alpha * (norm.cdf(d1) + norm.cdf(-d2))
        term2 = p*x * norm.cdf(d1 - 0.5*self.sigma*self.sqrt_delta_t)
        term3 = y * norm.cdf(-d2 - 0.5*self.sigma*self.sqrt_delta_t)
        
        return (self.gamma/(1-self.gamma)) * (term1 - term2 - term3)

    def calculate_fee_outgoing(self, p, x, y):
        """
        Calculate expected outgoing fee
        Works with either numpy arrays or single values
        """
        alpha = self.L * np.sqrt((1-self.gamma) * p) * self.alpha_factor
        
        d1 = np.log((1-self.gamma)*y/(p*x)) / (self.sigma * self.sqrt_delta_t)
        d2 = np.log(y/((1-self.gamma)*p*x)) / (self.sigma * self.sqrt_delta_t)
        
        term1 = alpha * (norm.cdf(d1) + norm.cdf(-d2))
        term2 = p*x * norm.cdf(-d2 + 0.5*self.sigma*self.sqrt_delta_t)
        term3 = y * norm.cdf(d1 + 0.5*self.sigma*self.sqrt_delta_t)
        
        return self.gamma * (-term1 + term2 + term3)

    def calculate_batch_targets(self, states: torch.Tensor) -> torch.Tensor:
        """
        Calculate target values for training using hybrid CPU/GPU approach
        
        Args:
            states: torch.Tensor of shape (batch_size, 3) containing (p, x, y)
        Returns:
            torch.Tensor of shape (batch_size, 1) containing target values
        """
        # Move states to CPU for extensive numpy calculations
        states_np = states.cpu().numpy()
        batch_size = states_np.shape[0]
        
        # Calculate immediate rewards (p*x + y)
        p = states_np[:, 0]
        x = states_np[:, 1]
        y = states_np[:, 2]
        immediate_reward = p * x + y
        
        # Pre-compute quadrature points and weights once
        num_points = 100
        points, weights = np.polynomial.legendre.leggauss(num_points)
        
        # Pre-allocate arrays
        expected_values = np.zeros(batch_size)
        
        # Calculate expected value for each state
        for b in range(batch_size):
            # Setup distribution parameters
            log_p = np.log(p[b])
            log_p_mean = log_p + (self.mu - 0.5 * self.sigma**2) * self.delta_t
            log_p_std = self.sigma * self.sqrt_delta_t
            
            # Transform integration points
            log_p_min = log_p_mean - 3 * log_p_std
            log_p_max = log_p_mean + 3 * log_p_std
            log_p_points = 0.5 * (log_p_max - log_p_min) * points + 0.5 * (log_p_max + log_p_min)
            transformed_weights = weights * 0.5 * (log_p_max - log_p_min)
            p_points = np.exp(log_p_points)
            
            # Calculate new states based on AMM mechanics
            price_ratio = y[b] / x[b]
            p_upper = price_ratio / (1 - self.gamma)
            p_lower = price_ratio * (1 - self.gamma)
            
            new_x = np.zeros_like(p_points)
            new_y = np.zeros_like(p_points)
            
            # Apply AMM rules using masks
            above_mask = p_points > p_upper
            below_mask = p_points < p_lower
            within_mask = ~(above_mask | below_mask)
            
            if self.fee_model == 'distribute':
                # calculate updated x,y 
                new_x[above_mask] = self.L / np.sqrt((1 - self.gamma) * p_points[above_mask])
                new_y[above_mask] = self.L * np.sqrt((1 - self.gamma) * p_points[above_mask])
                new_y[below_mask] = self.L * np.sqrt(p_points[below_mask] / (1 - self.gamma))
                new_x[below_mask] = self.L * np.sqrt((1 - self.gamma) / p_points[below_mask])
            
            new_x[within_mask] = x[b]
            new_y[within_mask] = y[b]
            
            # Check constant product
            new_product = new_x * new_y
            if not np.allclose(new_product, self.L**2, rtol=1e-6, atol=1e-8):
                max_deviation = np.max(np.abs(new_product - self.L**2))
                print(f"Warning: Constant product violated, max deviation: {max_deviation}")
            
            # Create next states tensor for GPU inference
            next_states_np = np.column_stack((p_points, new_x, new_y))
            next_states = torch.tensor(next_states_np, dtype=torch.float64, device=self.device)
            
            # GPU inference
            with torch.no_grad():
                values = self.target_network(next_states).cpu().numpy().flatten()
            
            # Calculate PDF and integrate on CPU
            log_terms = (np.log(p_points/p[b]) - (self.mu - 0.5 * self.sigma**2) * self.delta_t)**2
            denominator = 2 * self.sigma**2 * self.delta_t
            pdf_values = np.exp(-log_terms / denominator) / (p_points * self.sigma * self.pi_term)
            
            integrand = values * pdf_values
            expected_values[b] = np.sum(integrand * transformed_weights)
            
            # Add fees if using distribute model
            if self.fee_model == 'distribute':
                if self.fee_source == 'incoming':
                    fee = self.calculate_fee_ingoing(p[b], x[b], y[b])
                elif self.fee_source == 'outgoing':
                    fee = self.calculate_fee_outgoing(p[b], x[b], y[b])
                expected_values[b] += fee
        
        # Calculate target
        targets_np = np.maximum(immediate_reward, self.discount_factor * expected_values)
        
        # Convert back to tensor for training
        targets = torch.tensor(targets_np, dtype=torch.float64, device=self.device).unsqueeze(1)
        
        return targets

    def train_value_function(self, num_epochs: int = 100, batch_size: int = 128, 
                            learning_rate: float = 0.001, verbose: bool = False, 
                            progress_bar: bool = False) -> list[float]:
        """
        Train the value function using a large pre-generated dataset
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Size of training batches
            learning_rate: Learning rate for optimizer
            verbose: Whether to print progress details
            progress_bar: Whether to show progress bar
        Returns:
            list[float]: History of training losses
        """
        # Generate training data - make sure it's a PyTorch tensor
        training_states = self.generate_training_data(num_samples=10000)
    
        print(f"Training states are a {type(training_states)} with device {training_states.device}")

    
        # Create dataset from tensor
        dataset = TensorDataset(training_states)
        generator = torch.Generator(device=self.device)
        # Create dataloader without generator (fix for the error)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    
        # Define optimizer, scheduler and criterion
        optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        criterion = nn.MSELoss()
    
        # Loss history
        losses = []
        best_loss = float('inf')
    
        # Print header with wider columns
        if verbose:
            header = "| {:^5} | {:^20} | {:^20} | {:^10} | {:^12} |".format(
                "Epoch", "Loss", "Min Loss", "LR", "Tau"
            )
            separator = "-" * len(header)
            print("\nTraining Progress:")
            print(separator)
            print(header)
            print(separator)
    
        # Training loop
        desc = f"{self.fee_model}-{self.fee_source}-{self.gamma*10000}bp-{self.sigma}s"
        pbar = tqdm(range(num_epochs), desc=desc, disable=not progress_bar)
    
        for epoch in pbar:
            epoch_losses = []
        
            for batch_states, in dataloader:
                # Make sure batch is on the correct device
                batch_states = batch_states.to(self.device)
            
                # Forward pass
                predicted_value = self.value_network(batch_states)
                targets = self.calculate_batch_targets(batch_states)
            
                # Compute loss
                loss = criterion(predicted_value, targets)
            
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm(self.value_network.parameters(), max_norm=1.0)
                optimizer.step()
            
                epoch_losses.append(loss.item())
            
            # Record average loss
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
            
            # Step the scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Update target network
            tau = 0.0001
            self.update_target_network(tau=tau)
            
            # Update progress bar
            if progress_bar:
                if avg_loss > 1e6:
                    formatted_loss = f"{avg_loss:.2e}"
                else:
                    formatted_loss = f"{avg_loss:.2f}"
                    
                if best_loss > 1e6:
                    formatted_best = f"{best_loss:.2e}"
                else:
                    formatted_best = f"{best_loss:.2f}"
                
                pbar.set_postfix({'loss': formatted_loss, 'best': formatted_best})
            
            # Print progress with scientific notation
            if verbose and (epoch + 1) % 1 == 0:
                progress = "| {:>5d} | {:>20.6e} | {:>20.6e} | {:>10.6f} | {:>12.6f} |".format(
                    epoch + 1, avg_loss, best_loss, current_lr, tau
                )
                print(progress)
            
            # Save model if we hit a new best loss
            if is_best:
                root_dir = '/home/shiftpub/Dynamic_AMM/models'
                os.makedirs(root_dir, exist_ok=True)
                torch.save(
                    self.value_network.state_dict(), 
                    f'{root_dir}/optimal_mc_value_network_{self.fee_model}_{self.fee_source}_gamma_{self.gamma}_sigma_{self.sigma}.pth'
                )
        
        if verbose:
            print(separator)
            print(f"\nTraining completed! Best loss: {best_loss:.6e}")
            print(f"Model saved as: optimal_mc_value_network_{self.fee_model}_{self.fee_source}_gamma_{self.gamma}_sigma_{self.sigma}.pth")
        
        return losses

def main():
    # Check if CUDA (GPU) is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize AMM with different fee models
    for fee_source in ['incoming', 'outgoing']:
        for gamma in [0.0005, 0.005, 0.05, 0.5]:
            for sigma in [0.1, 0.5, 1, 2]:
                distribute_amm = AMM(
                    L=10000,
                    gamma=gamma,
                    sigma=sigma,
                    delta_t=1,
                    mu=0.0,
                    fee_model='distribute',
                    fee_source=fee_source,
                    device=device
                )
            
                # Train both models
                distribute_losses = distribute_amm.train_value_function(
                    num_epochs=1000,
                    batch_size=256,
                    learning_rate=0.003,
                    verbose=True,
                    progress_bar=False
                )

if __name__ == "__main__":
    main()