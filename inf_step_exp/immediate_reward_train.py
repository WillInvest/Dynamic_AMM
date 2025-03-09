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
        # nn.init.constant_(self.network[-1].bias)  # Start near the typical reward
    
    # def normalize_input(self, state: torch.Tensor) -> torch.Tensor:
    #     """
    #     Args:
    #         state: torch.Tensor of shape (batch_size, 3) containing (p, x, y)
    #     Returns:
    #         torch.Tensor of normalized state values
    #     """
    #     if not self.normalize:
    #         return state
            
    #     normalized: torch.Tensor = torch.zeros_like(state, dtype=torch.float64)  # shape: (batch_size, 3)
    #     normalized[:, 0] = state[:, 0]  # Keep price as is
    #     normalized[:, 1] = state[:, 1] / self.L
    #     normalized[:, 2] = state[:, 2] / self.L
    #     return normalized
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: torch.Tensor of shape (batch_size, 3) containing (p, x, y)
        Returns:
            torch.Tensor of shape (batch_size, 1) containing predicted values
        """
        # Normalize the input state
        # normalized_state: torch.Tensor = self.normalize_input(state)
        # Process through the network
        return self.network(state)

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
        self.value_network.to(device)
    
    def generate_training_data(self, num_samples: int) -> torch.Tensor:
        """
        Generate training data with price p fixed at price ratio y/x
    
        Args:
            num_samples: Number of samples to generate
        
        Returns:
            Tensor of shape (num_samples, 3) containing (p, x, y)
        """

        min_x = self.L * 0.2
        max_x = self.L * 5
        x_values = np.linspace(min_x, max_x, num_samples)
        y_values = self.L**2 / x_values
        p_values = y_values / x_values
        
        # Stack into a numpy array
        states_np = np.stack([p_values, x_values, y_values], axis=1)
        
        # Convert to tensor in single operation and move to device
        states = torch.tensor(states_np, dtype=torch.float64, device=self.device)
        
        return states

    def calculate_batch_targets(self, states: torch.Tensor) -> torch.Tensor:
        """
        Calculate target values for training using vectorized approach with trapezoidal integration.
        Uses the immediate reward of the next state rather than the current state.
    
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
    
        # We won't use immediate reward of current state anymore
        # Instead, we'll calculate it for each next state

        # Setup integration parameters
        # Setup integration parameters
        # num_points = 500  # Number of points for integration
        
        # # Generate random samples from log-normal distribution
        # log_p = np.log(p)
        # drift = (self.mu - 0.5 * self.sigma**2) * self.delta_t
        # log_p_std = self.sigma * np.sqrt(self.delta_t)
    
        # # Generate samples for all batch items at once
        # # Shape: (batch_size, num_samples)
        # log_p_points = np.random.normal(
        #     loc=log_p.reshape(-1, 1) + drift, 
        #     scale=log_p_std, 
        #     size=(batch_size, num_points)
        # )

        # # Calculate corresponding prices
        # p_points = np.exp(log_p_points)

        # # Calculate new states based on AMM mechanics
        # price_ratio = y / x  # (batch_size,)
        # p_upper = price_ratio / (1 - self.gamma)  # (batch_size,)
        # p_lower = price_ratio * (1 - self.gamma)  # (batch_size,)

        # # Expand dimensions for broadcasting
        # p_upper = p_upper.reshape(-1, 1)  # (batch_size, 1)
        # p_lower = p_lower.reshape(-1, 1)  # (batch_size, 1)

        # # Initialize arrays for new x, y values
        # new_x = np.zeros((batch_size, num_points))
        # new_y = np.zeros((batch_size, num_points))

        # # Create masks for the conditions
        # above_mask = p_points > p_upper
        # below_mask = p_points < p_lower
        # within_mask = ~(above_mask | below_mask)

        # if self.fee_model == 'distribute':
        #     # Vectorized calculations for each price region
        #     new_x[above_mask] = self.L / np.sqrt((1 - self.gamma) * p_points[above_mask])
        #     new_y[above_mask] = self.L * np.sqrt((1 - self.gamma) * p_points[above_mask])

        #     new_x[below_mask] = self.L * np.sqrt((1 - self.gamma) / p_points[below_mask])
        #     new_y[below_mask] = self.L * np.sqrt(p_points[below_mask] / (1 - self.gamma))

        # # Use broadcasting for within mask
        # x_expanded = x.reshape(-1, 1)  # (batch_size, 1)
        # y_expanded = y.reshape(-1, 1)  # (batch_size, 1)

        # # Fill in 'within' values
        # new_x[within_mask] = np.repeat(x_expanded, num_points, axis=1)[within_mask]
        # new_y[within_mask] = np.repeat(y_expanded, num_points, axis=1)[within_mask]

        # # Calculate immediate reward for each NEXT state
        # # Instead of using the current state's immediate reward
        # next_immediate_reward = p_points * new_x + new_y
        # # Multiply by weights, sum, and scale by dx/2
        # expected_values = np.mean(next_immediate_reward, axis=1)

        # Apply discount factor if needed
        # if self.discount_factor < 1.0:
        #     expected_values *= self.discount_factor
        immediate_reward = p * x + y
        targets = torch.tensor(immediate_reward, dtype=torch.float64, device=self.device).unsqueeze(1)
    
        return targets

    def train_value_function(self, num_epochs: int = 100, batch_size: int = 128, 
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
        training_states = self.generate_training_data(num_samples=100000)
        
        
        # Create dataset from tensor
        dataset = TensorDataset(training_states)
        generator = torch.Generator(device=self.device)

        # Define optimizer, scheduler and criterion
        optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
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
                root_dir = '/home/shiftpub/Dynamic_AMM/pretrained_models'
                os.makedirs(root_dir, exist_ok=True)
                torch.save(
                    self.value_network.state_dict(), 
                    f'{root_dir}/pretrained_immediate_{self.fee_model}_{self.fee_source}.pth'
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs without improvement")
                    if verbose:
                        print(f"\nEarly stopping triggered after {epoch + 1} epochs without improvement")
                    break
        
            # Step the scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        
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
            if verbose and (epoch + 1) % 10 == 0:
                progress = "| {:>5d} | {:>20.6e} | {:>20.6e} | {:>10.6f} | {:>8d} |".format(
                    epoch + 1, avg_loss, best_loss, current_lr, epochs_without_improvement
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
        distribute_amm = AMM(
            L=1000,
            fee_model='distribute',
            fee_source=fee_source,
            device=device
        )
    
        # Train both models
        distribute_losses = distribute_amm.train_value_function(
            num_epochs=1000,
            batch_size=1280,
            learning_rate=0.0003,
            verbose=True,
            progress_bar=False
        )

if __name__ == "__main__":
    main()