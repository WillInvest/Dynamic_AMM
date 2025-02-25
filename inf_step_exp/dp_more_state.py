import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm
from tqdm import tqdm

class ValueFunctionNN(nn.Module):
    """Neural network to approximate the value function V(p, x, y, γ)"""
    
    def __init__(self, hidden_dim=64):
        super(ValueFunctionNN, self).__init__()
        # Input dimension is 3: price (p), token x amount, token y amount
        self.network = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """
        Forward pass through the network
        
        Args:
            state: Tensor of shape (batch_size, 4) containing (p, x, y, γ)
        
        Returns:
            Value function estimate V(p, x, y, γ)
        """
        return self.network(state)

class AMM:
    def __init__(self, 
                 L=10000, 
                 delta_t=1, 
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
        self.delta_t = delta_t
        self.fee_model = fee_model
        self.fee_source = fee_source
        
        # Initialize neural network
        self.value_network = ValueFunctionNN()
        self.target_network = ValueFunctionNN()
        self.target_network.load_state_dict(self.value_network.state_dict())  # Initialize with same weights
    
    def update_target_network(self, tau=0.01):
        """Soft update target network: θ_target = τ*θ_current + (1-τ)*θ_target"""
        for target_param, current_param in zip(self.target_network.parameters(), self.value_network.parameters()):
            target_param.data.copy_(tau * current_param.data + (1.0 - tau) * target_param.data)
    
    def generate_training_data(self, num_samples):
        """
        Generate training data for the value function approximation
        
        Args:
            num_samples: Number of state samples to generate
        
        Returns:
            Tensor of shape (num_samples, 3) containing (p, x, y)
        """   
        # generate num_samples for each combination of gamma, sigma, and drift
        gammas = [0.001, 0.003, 0.005]
        sigmas = [0.1, 0.4]
        
        # Generate x values uniformly between 50 and 100
        x_values = torch.FloatTensor(num_samples).uniform_(50, 150)
        
        # Calculate corresponding y values using constant product L = x*y
        y_values = self.L**2 / x_values
        
        # add each combination of sigma and drift to the states
        states = []
        for gamma in gammas:
            for sigma in sigmas:
                # Calculate price bounds for each x,y pair
                price_ratio = y_values / x_values
                p_min = price_ratio * (1 - gamma)
                p_max = price_ratio / (1 - gamma)
        
                # Create tensors for gamma and sigma
                gamma_tensor = torch.tensor(gamma, dtype=torch.float32)
                sigma_tensor = torch.tensor(sigma, dtype=torch.float32)
                gamma_expanded = torch.full_like(x_values, gamma_tensor)
                sigma_expanded = torch.full_like(x_values, sigma_tensor)
                
                # Generate random p_values between p_min and p_max
                p_values = torch.rand(num_samples) * (p_max - p_min) + p_min
                
                # Stack the state components
                state = torch.stack([p_values, x_values, y_values, gamma_expanded, sigma_expanded], dim=1)
                states.append(state)
        
        return torch.cat(states, dim=0)

    def calculate_fee_ingoing(self, p, x, y, gamma, sigma):
        """
        Calculate expected ingoing fee for distribute model
        """
        alpha = self.L * np.sqrt((1-gamma) * p) * np.exp(-sigma**2 * self.delta_t / 8)
        
        d1 = np.log((1-gamma)*y/(p*x)) / (sigma * np.sqrt(self.delta_t))
        d2 = np.log(y/((1-gamma)*p*x)) / (sigma * np.sqrt(self.delta_t))
        
        term1 = alpha * (norm.cdf(d1) + norm.cdf(-d2))
        term2 = p*x * norm.cdf(d1 - 0.5*sigma**2*np.sqrt(self.delta_t))
        term3 = y * norm.cdf(-d2 - 0.5*sigma**2*np.sqrt(self.delta_t))
        
        return (gamma/(1-gamma)) * (term1 - term2 - term3)

    def calculate_fee_outgoing(self, p, x, y, gamma, sigma):
        """
        Calculate expected outgoing fee
        """
        alpha = self.L * np.sqrt((1-gamma) * p) * np.exp(-sigma**2 * self.delta_t / 8)
        
        d1 = np.log((1-gamma)*y/(p*x)) / (sigma * np.sqrt(self.delta_t))
        d2 = np.log(y/((1-gamma)*p*x)) / (sigma * np.sqrt(self.delta_t))
        
        term1 = alpha * (norm.cdf(d1) + norm.cdf(-d2))
        term2 = p*x * norm.cdf(-d2 + 0.5*sigma**2*np.sqrt(self.delta_t))
        term3 = y * norm.cdf(d1 + 0.5*sigma**2*np.sqrt(self.delta_t))
        
        return gamma * (-term1 + term2 + term3)

    def calculate_batch_targets(self, states):
        """
        Calculate target values for training (max of immediate reward and expected future value)
        
        Args:
            states: Current states (p, x, y) of shape (batch_size, 3)
        
        Returns:
            Target values for training
        """
        batch_size = states.shape[0]
        expected_values = torch.zeros(batch_size, 1)
        
        # Calculate immediate rewards (p*x + y)
        p, x, y = states[:, 0], states[:, 1], states[:, 2]
        immediate_reward = p * x + y
        
        # Pre-compute quadrature points and weights
        num_points = 50
        points, weights = np.polynomial.legendre.leggauss(num_points)
        
        # Calculate expected value for each state
        for b in range(batch_size):
            p = states[b, 0].item()
            x = states[b, 1].item()
            y = states[b, 2].item()
            gamma = states[b, 3].item()
            sigma = states[b, 4].item()
            
            # Setup distribution parameters
            log_p = np.log(p)
            log_p_mean = log_p + (- 0.5 * sigma**2) * self.delta_t
            log_p_std = sigma * np.sqrt(self.delta_t)
            
            # Transform integration points
            log_p_min = log_p_mean - 4 * log_p_std
            log_p_max = log_p_mean + 4 * log_p_std
            log_p_points = 0.5 * (log_p_max - log_p_min) * points + 0.5 * (log_p_max + log_p_min)
            transformed_weights = weights * 0.5 * (log_p_max - log_p_min)
            
            # Create price points tensor
            p_points = torch.tensor(np.exp(log_p_points), dtype=torch.float32)
            
            # Calculate new states based on AMM mechanics
            price_ratio = y / x
            p_upper = price_ratio / (1 - gamma)
            p_lower = price_ratio * (1 - gamma)
            
            new_x = torch.zeros_like(p_points)
            new_y = torch.zeros_like(p_points)
            # Apply AMM rules using masks
            above_mask = p_points > p_upper
            below_mask = p_points < p_lower
            within_mask = ~(above_mask | below_mask)
            
            if self.fee_model == 'distribute':
                # calculate updated x,y 
                new_x[above_mask] = self.L / torch.sqrt((1 - gamma) * p_points[above_mask])
                new_y[above_mask] = self.L * torch.sqrt((1 - gamma) * p_points[above_mask])
                new_y[below_mask] = self.L * torch.sqrt(p_points[below_mask] / (1 - gamma))
                new_x[below_mask] = self.L * torch.sqrt((1 - gamma) / p_points[below_mask])
            
            new_x[within_mask] = x
            new_y[within_mask] = y
            
            # make sure new_x * new_y = L^2
            constant_product = torch.full_like(new_x, self.L**2)
            gamma_tensor = torch.full_like(new_x, gamma)
            sigma_tensor = torch.full_like(new_x, sigma)
            assert torch.allclose(new_x * new_y, constant_product, rtol=1e-4), \
                f"Constant product condition violated: max deviation = {torch.max(torch.abs(new_x * new_y - self.L**2))}"
            
            next_states = torch.stack([p_points, new_x, new_y, gamma_tensor, sigma_tensor], dim=1)
            
            # Process all points at once
            with torch.no_grad():
                values = self.target_network(next_states).squeeze()
            
            # Calculate PDF values
            log_terms = (torch.log(p_points/p) - (- 0.5 * sigma**2) * self.delta_t)**2
            denominator = 2 * sigma**2 * self.delta_t
            pdf_values = torch.exp(-log_terms / denominator) / (p_points * sigma * np.sqrt(2 * np.pi * self.delta_t))
            
            # Calculate integrand and expected value
            integrand = values * pdf_values
            expected_values[b] = torch.sum(integrand * torch.tensor(transformed_weights, dtype=torch.float32))
            
            # Add fees if using distribute model
            if self.fee_model == 'distribute':
                if self.fee_source == 'incoming':
                    fee = self.calculate_fee_ingoing(p, x, y, gamma, sigma)
                elif self.fee_source == 'outgoing':
                    fee = self.calculate_fee_outgoing(p, x, y, gamma, sigma)
                expected_values[b] += fee
        
        # Calculate target as maximum of immediate reward and expected future value
        target = torch.maximum(immediate_reward.unsqueeze(1), np.exp(0*self.delta_t)*expected_values)
        return target

    def train_value_function(self, num_epochs=100, batch_size=128, learning_rate=0.001):
        """
        Train the value function approximation network
        """
        # Generate training data
        print("Generating training data...")
        states = self.generate_training_data(num_samples=100)
        
        # Create data loader
        dataset = TensorDataset(states)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Define optimizer
        optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        
        # Loss history
        losses = []
        
        # Print header
        header = "| {:^5} | {:^12} | {:^12} | {:^10} |".format("Epoch", "Loss", "LR", "Tau")
        separator = "|" + "-" * 7 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 12 + "|"
        tqdm.write("\nTraining Progress:")
        tqdm.write(separator)
        tqdm.write(header)
        tqdm.write(separator)
        
        # Training loop
        pbar = tqdm(range(num_epochs), desc="Training value function")
        for epoch in pbar:
            epoch_losses = []
            
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
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            # Record average loss
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            # Step the scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Update target network
            tau = 0.0005 if avg_loss < 10 else 0.005
            self.update_target_network(tau=tau)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 1 == 0:
                progress = "| {:>5d} | {:>12.6f} | {:>12.6f} | {:>10.6f} |".format(
                    epoch + 1, avg_loss, current_lr, tau
                )
                tqdm.write(progress)
        
        # Print final separator
        tqdm.write(separator)
        return losses

def main():
    # Initialize AMM with different fee models
    distribute_amm = AMM(
        L=100,
        delta_t=1,
        fee_model='distribute',
        fee_source='incoming'
    )
    
    # Train both models
    print("\nTraining value function for distribute fee model...")
    distribute_losses = distribute_amm.train_value_function(
        num_epochs=500,
        batch_size=128,
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