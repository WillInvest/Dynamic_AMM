import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm

# Set default tensor type to float64 (double precision)
torch.set_default_tensor_type(torch.DoubleTensor)
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)

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
        min_x = 9500
        max_x = 10500
        # Generate x values using Gauss-Legendre points for better coverage
        # x_points, _ = np.polynomial.legendre.leggauss(num_samples)
        # Transform x points from [-1, 1] to [min_x, max_x]
        # x_values = 0.5 * (max_x - min_x) * x_points + 0.5 * (max_x + min_x)
        x_values = torch.linspace(min_x, max_x, num_samples)
    
        # Calculate corresponding y values using constant product formula
        y_values = self.L**2 / x_values
    
        # Set price exactly at the price ratio (p = y/x)
        p_values = y_values / x_values
    
        # Stack into a single array and move to device
        states = torch.stack([p_values, x_values, y_values], dim=1).to(self.device)
    
        return states

    def calculate_fee_ingoing(self, p: float, x: float, y: float) -> float:
        """
        Calculate expected ingoing fee for distribute model
        
        Args:
            p: Current price
            x: Token X amount
            y: Token Y amount
        Returns:
            float: Expected ingoing fee
        """
        alpha = self.L * np.sqrt((1-self.gamma) * p) * np.exp(-self.sigma**2 * self.delta_t / 8)
        
        d1 = np.log((1-self.gamma)*y/(p*x)) / (self.sigma * np.sqrt(self.delta_t))
        d2 = np.log(y/((1-self.gamma)*p*x)) / (self.sigma * np.sqrt(self.delta_t))
        
        term1 = alpha * (norm.cdf(d1) + norm.cdf(-d2))
        term2 = p*x * norm.cdf(d1 - 0.5*self.sigma*np.sqrt(self.delta_t))
        term3 = y * norm.cdf(-d2 - 0.5*self.sigma*np.sqrt(self.delta_t))
        
        return (self.gamma/(1-self.gamma)) * (term1 - term2 - term3)

    def calculate_fee_outgoing(self, p: float, x: float, y: float) -> float:
        """
        Calculate expected outgoing fee
        
        Args:
            p: Current price
            x: Token X amount
            y: Token Y amount
        Returns:
            float: Expected outgoing fee
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
        Calculate target values for training
        
        Args:
            states: torch.Tensor of shape (batch_size, 3) containing (p, x, y)
        Returns:
            torch.Tensor of shape (batch_size, 1) containing target values
        """
        batch_size: int = states.shape[0]
        expected_values: torch.Tensor = torch.zeros(batch_size, 1).to(self.device)
        
        # Calculate immediate rewards (p*x + y)
        p: torch.Tensor = states[:, 0]  # shape: (batch_size,)
        x: torch.Tensor = states[:, 1]  # shape: (batch_size,)
        y: torch.Tensor = states[:, 2]  # shape: (batch_size,)
        immediate_reward: torch.Tensor = p * x + y  # shape: (batch_size,)
        
        # Pre-compute quadrature points and weights
        num_points = 50
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
            p_points = torch.tensor(np.exp(log_p_points), dtype=torch.float64).to(self.device)
            
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
            new_product = new_x * new_y
            max_deviation = torch.max(torch.abs(new_product - constant_product))
            max_deviation_index = torch.argmax(torch.abs(new_product - constant_product))
            rel_deviation = max_deviation / constant_product[max_deviation_index]
            # Use a more reasonable tolerance for floating point comparisons
            assert torch.allclose(new_product, constant_product, rtol=1e-6, atol=1e-8), \
                f"Constant product condition violated: max deviation = {max_deviation}, relative deviation = {rel_deviation}"
            
            next_states: torch.Tensor = torch.stack([p_points, new_x, new_y], dim=1).to(self.device)  # shape: (num_points, 3)
            
            # Move to GPU only for network inference
            with torch.no_grad():
                values: torch.Tensor = self.target_network(next_states).squeeze()  # shape: (num_points,)
            
            # calculate pdf
            log_terms: torch.Tensor = (torch.log(p_points/p) - (self.mu - 0.5 * self.sigma**2) * self.delta_t)**2
            denominator = 2 * self.sigma**2 * self.delta_t
            pi_term: torch.Tensor = torch.tensor(np.sqrt(2 * np.pi * self.delta_t), dtype=torch.float64).to(self.device)
            pdf_values: torch.Tensor = torch.exp(-log_terms / denominator) / (p_points * self.sigma * pi_term)    
            integrand = values * pdf_values
            expected_values[b] = torch.sum(integrand * torch.tensor(transformed_weights, dtype=torch.float64).to(self.device))
            
            # Add fees if using distribute model
            if self.fee_model == 'distribute':
                if self.fee_source == 'incoming':
                    fee = self.calculate_fee_ingoing(p, x, y)
                elif self.fee_source == 'outgoing':
                    fee = self.calculate_fee_outgoing(p, x, y)
                expected_values[b] += fee
        
        # Calculate target and move back to GPU for training
        target = torch.maximum(immediate_reward.unsqueeze(1), self.discount_factor * expected_values)
        return target.to(self.device)

    def train_value_function(self, num_epochs: int = 100, batch_size: int = 128, learning_rate: float = 0.001) -> list[float]:
        """
        Train the value function using a large pre-generated dataset
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Size of training batches
            learning_rate: Learning rate for optimizer
        Returns:
            list[float]: History of training losses
        """
        # Generate training data 
        training_states = self.generate_training_data(num_samples=500)         
        dataset = TensorDataset(training_states)
        
        # Create a generator for the dataloader that matches the device
        generator = torch.Generator(device=self.device)
        
        # Define optimizer, scheduler and criterion
        optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        criterion = nn.MSELoss()
        
        # Loss history
        losses = []
        best_loss = float('inf')
        
        # Print header
        header = "| {:^5} | {:^12} | {:^12} | {:^10} | {:^12} |".format(
            "Epoch", "Loss", "Min Loss", "LR", "Tau"
        )
        separator = "-" * len(header)
        print("\nTraining Progress:")
        print(separator)
        print(header)
        print(separator)
        
        # Training loop
        pbar = tqdm(range(num_epochs), desc="Training value function", disable=True)
        for epoch in pbar:
            epoch_losses = []
            # Add generator to DataLoader
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
            
            for batch_states, in dataloader:
                batch_states = batch_states.to(self.device)
                targets = self.calculate_batch_targets(batch_states)
                predicted_value: torch.Tensor = self.value_network(batch_states)
                loss = criterion(predicted_value, targets)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm(self.value_network.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_losses.append(loss.item())
            
            # Record average loss
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            best_loss = min(best_loss, avg_loss)
            
            # Step the scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Update target network
            tau = 0.0001
            self.update_target_network(tau=tau)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                progress = "| {:>5d} | {:>12.6f} | {:>12.6f} | {:>10.6f} | {:>12.6f} |".format(
                    epoch + 1, avg_loss, best_loss, current_lr, tau
                )
                print(progress)
                
                # Save model if we hit a new best loss
                if avg_loss == best_loss:
                    torch.save(self.value_network.state_dict(), 
                             f'optimal_mc_value_network_{self.fee_model}_{self.fee_source}.pth')
        
        print(separator)
        print(f"\nTraining completed! Best loss: {best_loss:.6f}")
        print(f"Model saved as: optimal_mc_value_network_{self.fee_model}_{self.fee_source}.pth")
        
        return losses

def main():
   
    # Check if CUDA (GPU) is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize AMM with different fee models
    for fee_source in ['incoming', 'outgoing']:
    
        distribute_amm = AMM(
            L=10000,
            gamma=0.1,
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
            num_epochs=1000,
            batch_size=64,
            learning_rate=0.003
        )

if __name__ == "__main__":
    main()