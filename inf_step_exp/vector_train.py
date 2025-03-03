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
        
        # nn.init.normal_(self.network[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.network[-1].bias, self.L)  # Start near the typical reward
    
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
        normalized[:, 1] = state[:, 1] 
        normalized[:, 2] = state[:, 2]
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
        min_x = self.L * 0.95
        max_x = self.L * 1.05
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
        alpha = self.L * np.sqrt((1-self.gamma) * p) * np.exp(-self.sigma**2 * self.delta_t / 8)
        
        d1 = np.log((1-self.gamma)*y/(p*x)) / (self.sigma * np.sqrt(self.delta_t))
        d2 = np.log(y/((1-self.gamma)*p*x)) / (self.sigma * np.sqrt(self.delta_t))
        
        term1 = alpha * (norm.cdf(d1) + norm.cdf(-d2))
        term2 = p*x * norm.cdf(d1 - 0.5*self.sigma*np.sqrt(self.delta_t))
        term3 = y * norm.cdf(-d2 - 0.5*self.sigma*np.sqrt(self.delta_t))
        
        return (self.gamma/(1-self.gamma)) * (term1 - term2 - term3)

    def calculate_fee_outgoing(self, p, x, y):
        """
        Calculate expected outgoing fee
        Works with either numpy arrays or single values
        """
        alpha = self.L * np.sqrt((1-self.gamma) * p) * np.exp(-self.sigma**2 * self.delta_t / 8)
        
        d1 = np.log((1-self.gamma)*y/(p*x)) / (self.sigma * np.sqrt(self.delta_t))
        d2 = np.log(y/((1-self.gamma)*p*x)) / (self.sigma * np.sqrt(self.delta_t))
        
        term1 = alpha * (norm.cdf(d1) + norm.cdf(-d2))
        term2 = p*x * norm.cdf(-d2 + 0.5*self.sigma*np.sqrt(self.delta_t))
        term3 = y * norm.cdf(d1 + 0.5*self.sigma*np.sqrt(self.delta_t))
        
        return self.gamma * (-term1 + term2 + term3)

    def calculate_batch_targets(self, states: torch.Tensor) -> torch.Tensor:
        """
        Calculate target values for training using vectorized approach
    
        Args:
            states: torch.Tensor of shape (batch_size, 3) containing (p, x, y)
        Returns:
            torch.Tensor of shape (batch_size, 1) containing target values
        """
        # Move states to CPU for extensive numpy calculations
        states_np = states.cpu().numpy()
        batch_size = states_np.shape[0]
    
        # Extract state components
        p = states_np[:, 0]
        x = states_np[:, 1]
        y = states_np[:, 2]
        immediate_reward = (p * x + y) # / self.L
    
        # Pre-compute quadrature points and weights once
        num_points = 500
        points, weights = np.polynomial.legendre.leggauss(num_points)
    
        # Setup distribution parameters
        log_p = np.log(p)
        log_p_mean = log_p[:, np.newaxis] + (self.mu - 0.5 * self.sigma**2) * self.delta_t
        log_p_std = self.sigma * np.sqrt(self.delta_t)
    
        # Transform integration points for all batch items at once
        log_p_min = log_p_mean - 3 * log_p_std
        log_p_max = log_p_mean + 3 * log_p_std
        # Reshape for broadcasting: (batch_size, num_points)
        log_p_points = 0.5 * (log_p_max - log_p_min) * points[np.newaxis, :] + 0.5 * (log_p_max + log_p_min)
        transformed_weights = weights[np.newaxis, :] * 0.5 * (log_p_max - log_p_min)
    
        # Exponential to get price points (batch_size, num_points)
        p_points = np.exp(log_p_points)
    
        # Calculate new states based on AMM mechanics
        price_ratio = y / x  # (batch_size,)
        p_upper = price_ratio / (1 - self.gamma)  # (batch_size,)
        p_lower = price_ratio * (1 - self.gamma)  # (batch_size,)
    
        # Expand dimensions for broadcasting
        p_upper = p_upper[:, np.newaxis]  # (batch_size, 1)
        p_lower = p_lower[:, np.newaxis]  # (batch_size, 1)
    
        # Initialize arrays for new x, y values with broadcast dimensions
        new_x = np.zeros((batch_size, num_points))
        new_y = np.zeros((batch_size, num_points))
    
        # Create masks for the conditions (batch_size, num_points)
        above_mask = p_points > p_upper
        below_mask = p_points < p_lower
        within_mask = ~(above_mask | below_mask)
    
        if self.fee_model == 'distribute':
            # Vectorized calculations for above mask
            new_x[above_mask] = self.L / np.sqrt((1 - self.gamma) * p_points[above_mask])
            new_y[above_mask] = self.L * np.sqrt((1 - self.gamma) * p_points[above_mask])
        
            # Vectorized calculations for below mask
            new_x[below_mask] = self.L * np.sqrt((1 - self.gamma) / p_points[below_mask])
            new_y[below_mask] = self.L * np.sqrt(p_points[below_mask] / (1 - self.gamma))
    
        # Use broadcasting for within mask
        x_expanded = x[:, np.newaxis]  # (batch_size, 1)
        y_expanded = y[:, np.newaxis]  # (batch_size, 1)
    
        # Fill in 'within' values
        new_x[within_mask] = x_expanded.repeat(num_points, axis=1)[within_mask]
        new_y[within_mask] = y_expanded.repeat(num_points, axis=1)[within_mask]
    
        # Check constant product (optional, can be removed for performance)
        new_product = new_x * new_y
        if not np.allclose(new_product, self.L**2, rtol=1e-6, atol=1e-8):
            max_deviation = np.max(np.abs(new_product - self.L**2))
            print(f"Warning: Constant product violated, max deviation: {max_deviation}")
    
        # Reshape for processing with target network
        # Flatten batch dimension and points dimension, then add feature dimension
        next_states_flat = np.zeros((batch_size * num_points, 3))
        next_states_flat[:, 0] = p_points.flatten()
        next_states_flat[:, 1] = new_x.flatten()
        next_states_flat[:, 2] = new_y.flatten()
    
        # Convert to tensor for GPU inference
        next_states = torch.tensor(next_states_flat, dtype=torch.float64, device=self.device)
    
        # GPU inference
        with torch.no_grad():
            flat_values = self.target_network(next_states).cpu().numpy()
            pred_values = self.value_network(next_states).cpu().numpy()
    
        # Reshape back to (batch_size, num_points)
        values = flat_values.reshape(batch_size, num_points)
    
        # Calculate PDF vectorized
        log_terms = (log_p_points - log_p[:, np.newaxis] - (self.mu - 0.5 * self.sigma**2) * self.delta_t)**2
        denominator = 2 * self.sigma**2 * self.delta_t
        pdf_values = np.exp(-log_terms / denominator) / (p_points * self.sigma * np.sqrt(2 * np.pi * self.delta_t))
    
        # Vectorized integration
        integrand = values * pdf_values
        expected_values = np.sum(integrand * transformed_weights, axis=1)
    
        # Add fees if using distribute modelx
        if self.fee_model == 'distribute':
            if self.fee_source == 'incoming':
                fees = self.calculate_fee_ingoing(p, x, y)
            elif self.fee_source == 'outgoing':
                fees = self.calculate_fee_outgoing(p, x, y)
            expected_values += fees
    
        # Calculate target
        targets_np = np.maximum(immediate_reward, self.discount_factor * expected_values)
    
        # Convert back to tensor for training
        targets = torch.tensor(targets_np * self.L, dtype=torch.float64, device=self.device).unsqueeze(1)
    
        return targets

    def train_value_function(self, tau: float = 0.01, num_epochs: int = 100, batch_size: int = 128, 
                             learning_rate: float = 0.001, verbose: bool = False, 
                             progress_bar: bool = False, patience: int = 10) -> list[float]:
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
        training_states = self.generate_training_data(num_samples=1)

        print(f"Training states are a {type(training_states)} with device {training_states.device}")

        # Create dataset from tensor
        dataset = TensorDataset(training_states)
        generator = torch.Generator(device=self.device)

        # Define optimizer, scheduler and criterion
        optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        criterion = nn.MSELoss()

        # Loss history
        losses = []
        best_loss = float('inf')
    
        # Early stopping counter
        epochs_without_improvement = 0
        best_model_state = None

        # Print header with wider columns
        if verbose:
            header = "| {:^5} | {:^20} | {:^20} | {:^10} | {:^12} | {:^8} |".format(
                "Epoch", "Loss", "Min Loss", "LR", "Tau", "No Impr."
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
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

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
                # clip_grad_norm(self.value_network.parameters(), max_norm=1.0)
                optimizer.step()
        
                epoch_losses.append(loss.item())
        
            # Record average loss
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
        
            # Check if this is the best loss so far
            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_without_improvement = 0
                # Save the best model state
                best_model_state = {k: v.cpu().clone() for k, v in self.value_network.state_dict().items()}
                # Save model
                root_dir = '/home/shiftpub/Dynamic_AMM/models'
                os.makedirs(root_dir, exist_ok=True)
                torch.save(
                    self.value_network.state_dict(), 
                    f'{root_dir}/{self.fee_model}_{self.fee_source}_gamma_{self.gamma}_sigma_{self.sigma}.pth'
                )
            else:
                epochs_without_improvement += 1
                # if epochs_without_improvement >= patience:
                #     if verbose:
                #         print(f"\nEarly stopping triggered after {epoch + 1} epochs without improvement")
                #     break
        
            # Step the scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        
            # Update target network
            tau = scheduler.get_last_lr()[0] * 10
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
            
                pbar.set_postfix({
                    'loss': formatted_loss, 
                    'best': formatted_best, 
                    'no_impr': epochs_without_improvement
                })
        
            # Print progress with scientific notation
            if verbose and (epoch + 1) % 1 == 0:
                progress = "| {:>5d} | {:>20.6e} | {:>20.6e} | {:>10.6f} | {:>12.6f} | {:>8d} |".format(
                    epoch + 1, avg_loss, best_loss, current_lr, tau, epochs_without_improvement
                )
                print(progress)
    
        # After training or early stopping, load the best model
        if best_model_state is not None:
            self.value_network.load_state_dict(best_model_state)
    
        if verbose:
            print(separator)
            if epochs_without_improvement >= patience:
                print(f"\nTraining stopped early after {epoch + 1} epochs due to no improvement for {patience} epochs")
            else:
                print(f"\nTraining completed for all {num_epochs} epochs")
            print(f"Best loss: {best_loss:.6e}")
            print(f"Model saved as: optimal_mc_value_network_{self.fee_model}_{self.fee_source}_gamma_{self.gamma}_sigma_{self.sigma}.pth")
    
        return losses

def main():
    # Check if CUDA (GPU) is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize AMM with different fee models
    for fee_source in ['incoming', 'outgoing']:
        for gamma in [0.0005]:
            for sigma in [1]:
                distribute_amm = AMM(
                    L=1000,
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
                    batch_size=1,
                    learning_rate=0.0003,
                    tau=0.001,
                    verbose=True,
                    progress_bar=False
                )

if __name__ == "__main__":
    main()