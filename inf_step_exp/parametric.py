import torch    
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.integrate import quad
from CPMM import AMM

class ParametricValueModel:
    def __init__(self, mu, sigma, gamma, delta_t=1):
        """
        Initialize the parametric value model
        
        Parameters:
        -----------
        mu : float
            Drift parameter
        sigma : float
            Volatility parameter
        gamma : float
            AMM gamma parameter
        delta_t : float
            Time step
        """
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.delta_t = delta_t
        self.discount_factor = np.exp(-mu * delta_t)
        
    def generate_raw_data(self, L, num_samples=100):
        """
        Generate initial states
        
        Parameters:
        -----------
        num_samples : int
            Number of initial states to sample
            
        Returns:
        --------
        states_df : DataFrame
            DataFrame with initial states
        """
        # Create an AMM instance
        amm = AMM(
            L=L,
            gamma=self.gamma,
            sigma=self.sigma,
            delta_t=self.delta_t,
            mu=self.mu,
            fee_model='distribute',
            fee_source='outgoing',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Generate initial states
        states = amm.generate_training_data(num_samples=num_samples)
        states_np = states.cpu().numpy()
        # Calculate expected fees using original state values
        original_p = states_np[:, 0]
        original_x = states_np[:, 1]
        original_y = states_np[:, 2]
        expected_incoming_fee = amm.calculate_fee_ingoing(original_p, original_x, original_y)
        expected_outgoing_fee = amm.calculate_fee_outgoing(original_p, original_x, original_y)

        # Create DataFrame with initial states
        states_df = pd.DataFrame({
            'state_idx': np.arange(len(states_np)),
            'p': states_np[:, 0],
            'x': states_np[:, 1],
            'y': states_np[:, 2],
            'immediate_reward': states_np[:, 0] * states_np[:, 1] + states_np[:, 2],
            'sqrt_p': np.sqrt(states_np[:, 0]),
            'expected_incoming_fee': expected_incoming_fee,
            'expected_outgoing_fee': expected_outgoing_fee
        })
        
        # Add price bounds for AMM mechanics
        states_df['price_ratio'] = states_df['y'] / states_df['x']
        states_df['p_upper'] = states_df['price_ratio'] / (1 - self.gamma)
        states_df['p_lower'] = states_df['price_ratio'] * (1 - self.gamma)
        
        # Store AMM for later use
        self.amm = amm
        
        return states_df
    
    def generate_trapezoidal_points(self, num_points=500):
        """
        Generate evenly-spaced points and weights for trapezoidal integration
        
        Parameters:
        -----------
        num_points : int
            Number of integration points
            
        Returns:
        --------
        points : numpy array
            Integration points (log price changes)
        weights : numpy array
            Integration weights
        """
        # Calculate drift and diffusion for log price
        drift = (self.mu - 0.5 * self.sigma**2) * self.delta_t
        diffusion = self.sigma * np.sqrt(self.delta_t)
        
        # Define integration range (±5 standard deviations around drift)
        std_range = 5
        lower_bound = drift - std_range * diffusion
        upper_bound = drift + std_range * diffusion
        
        # Generate evenly spaced points
        points = np.linspace(lower_bound, upper_bound, num_points)
        
        # Calculate weights for trapezoidal rule
        # For trapezoid rule, all interior points have equal weight, and the endpoints have half weight
        dx = (upper_bound - lower_bound) / (num_points - 1)
        weights = np.ones(num_points) * dx
        weights[0] = weights[-1] = dx / 2
        
        # Calculate the probability density function (PDF) for each point
        # since we're doing a weighted integral against the lognormal distribution
        pdf = np.exp(-(points - drift)**2 / (2 * diffusion**2)) / (diffusion * np.sqrt(2 * np.pi))
        
        # Multiply weights by PDF values
        weights = weights * pdf
        
        return points, weights
    
    def calculate_future_states(self, initial_state, integration_points):
        """
        Calculate future states based on initial state and integration points
        
        Parameters:
        -----------
        initial_state : Series
            Initial state (p, x, y, etc.)
        integration_points : numpy array
            Log price changes for future scenarios
            
        Returns:
        --------
        future_states : dict
            Dictionary with future states (prices, x, y)
        """
        initial_p = initial_state['p']
        initial_x = initial_state['x']
        initial_y = initial_state['y']
        p_upper = initial_state['p_upper']
        p_lower = initial_state['p_lower']
        
        # Calculate new prices using log-normal model
        new_prices = initial_p * np.exp(integration_points)
        
        # Initialize arrays for new x, y values
        new_x = np.zeros_like(new_prices)
        new_y = np.zeros_like(new_prices)
        
        # Create masks for the conditions
        above_mask = new_prices > p_upper
        below_mask = new_prices < p_lower
        within_mask = ~(above_mask | below_mask)
        
        # Calculate new x, y values based on conditions
        new_x[above_mask] = self.amm.L / np.sqrt((1 - self.gamma) * new_prices[above_mask])
        new_y[above_mask] = self.amm.L * np.sqrt((1 - self.gamma) * new_prices[above_mask])
        
        new_x[below_mask] = self.amm.L * np.sqrt((1 - self.gamma) / new_prices[below_mask])
        new_y[below_mask] = self.amm.L * np.sqrt(new_prices[below_mask] / (1 - self.gamma))
        
        new_x[within_mask] = initial_x
        new_y[within_mask] = initial_y
        
        future_states = {
            'new_prices': new_prices,
            'new_x': new_x,
            'new_y': new_y,
            'above_mask': above_mask,
            'below_mask': below_mask,
            'within_mask': within_mask
        }
        
        return future_states
    
    def calculate_continuation_values(self, future_states, fees, C):
        """
        Calculate continuation values for future states
        
        Parameters:
        -----------
        future_states : dict
            Dictionary with future states
        fees : dict
            Dictionary with expected fees
        C : float
            Coefficient for V(p) = C*sqrt(p)
            
        Returns:
        --------
        continuation_values : dict
            Dictionary with continuation values
        """
        new_prices = future_states['new_prices']
        
        # Calculate future values based on new prices
        future_values = C * np.sqrt(new_prices)
        
        # Calculate continuation values (the future value plus fees)
        # Uses the expected fees calculated based on original p, x, y
        continuation_in = fees['expected_incoming_fee'] + future_values
        continuation_out = fees['expected_outgoing_fee'] + future_values
        
        continuation_values = {
            'continuation_in': continuation_in,
            'continuation_out': continuation_out
        }
        
        return continuation_values
    
    def integrate_continuation_values(self, continuation_values, weights):
        """
        Integrate continuation values using quadrature weights
        
        Parameters:
        -----------
        continuation_values : dict
            Dictionary with continuation values
        weights : numpy array
            Integration weights
            
        Returns:
        --------
        expected_values : dict
            Dictionary with expected future values
        """
        continuation_in = continuation_values['continuation_in']
        continuation_out = continuation_values['continuation_out']
        
        # Calculate weighted sum
        expected_in = np.sum(continuation_in * weights)
        expected_out = np.sum(continuation_out * weights)
        
        # Apply discount factor
        expected_in *= self.discount_factor
        expected_out *= self.discount_factor
        
        expected_values = {
            'expected_in': expected_in,
            'expected_out': expected_out
        }
        
        return expected_values
    
    def generate_parametric_data(self, Cin, Cout, num_samples=100, num_integration_points=500):
        """
        Generate data for parametric model using trapezoidal integration
        
        Parameters:
        -----------
        Cin : float
            Coefficient for V(p) = Cin*sqrt(p)
        Cout : float
            Coefficient for V(p) = Cout*sqrt(p)
        num_samples : int
            Number of initial states to sample
        num_integration_points : int
            Number of integration points
            
        Returns:
        --------
        final_df : DataFrame
            DataFrame with initial states and expected future values
        """
        # Step 1: Generate raw data
        raw_data = self.generate_raw_data(num_samples)
        
        # Step 2: Generate trapezoidal points and weights
        points, weights = self.generate_trapezoidal_points(num_integration_points)
        
        # Step 3-7: Process each state to get expected future values
        expected_values = []
        
        for _, row in raw_data.iterrows():
            # Step 3: Calculate future states
            future_states = self.calculate_future_states(row, points)
            
            # Step 5: Calculate expected new value using trapezoidal integration
            new_prices = future_states['new_prices']
            future_values_in = Cin * np.sqrt(new_prices)
            future_values_out = Cout * np.sqrt(new_prices)
            expected_new_value_in = np.sum(future_values_in * weights)
            expected_new_value_out = np.sum(future_values_out * weights)
            
            # Calculate expected values
            expected_in = row['expected_incoming_fee'] + expected_new_value_in
            expected_out = row['expected_outgoing_fee'] + expected_new_value_out
            
            # Step 7: Calculate metrics with shortcut names
            current_pool = row['p'] * row['x'] + row['y']
            current_value_in = Cin * np.sqrt(row['p'])
            current_value_out = Cout * np.sqrt(row['p'])

            # Calculate discounted values
            discounted_in_value = self.discount_factor * expected_in
            discounted_out_value = self.discount_factor * expected_out
            
            # Calculate squared errors
            # se_in = np.sqrt((current_value_in - discounted_in_value)**2)
            # se_out = np.sqrt((current_value_out - discounted_out_value)**2)
            se_in = np.sqrt((current_value_in - max(current_pool, discounted_in_value))**2)
            se_out = np.sqrt((current_value_out - max(current_pool, discounted_out_value))**2)
            
            expected_values.append({
                'state_idx': row['state_idx'],
                'PV0': current_pool,
                'V0_in': current_value_in,
                'V0_out': current_value_out,
                'Ein': row['expected_incoming_fee'],
                'Eout': row['expected_outgoing_fee'],
                'EV_in': expected_new_value_in,
                'EV_out': expected_new_value_out,
                'Vd_in': discounted_in_value,
                'Vd_out': discounted_out_value,
                'SE_In': se_in,
                'SE_Out': se_out
            })
        
        # Create DataFrame with expected values
        expected_df = pd.DataFrame(expected_values)
        
        # Merge with raw data
        final_df = pd.merge(raw_data, expected_df, on='state_idx')
        
        return final_df
    
    def find_optimal_c_in_out(self, num_samples=100, num_integration_points=500, initial_c=1.0):
        """
        Find the optimal Cin and Cout values that separately minimize SE_In and SE_Out
        
        Parameters:
        -----------
        num_samples : int
            Number of initial states to sample
        num_integration_points : int
            Number of integration points
        initial_c : float
            Initial guess for both Cin and Cout
            
        Returns:
        --------
        optimal_cin : float
            Optimal value for Cin that minimizes SE_In
        optimal_cout : float
            Optimal value for Cout that minimizes SE_Out
        optimal_data_in : DataFrame
            DataFrame with optimal results using Cin
        optimal_data_out : DataFrame
            DataFrame with optimal results using Cout
        """
        from scipy.optimize import minimize
        
        # Define the objective function for minimizing SE_In
        def objective_in(c):
            # Generate data with the given C value
            data = self.generate_parametric_data(
                Cin=c[0],
                Cout=1,
                num_samples=num_samples,
                num_integration_points=num_integration_points
            )
            
            # Return average SE_In
            avg_se_in = data['SE_In'].mean()
            print(f"Trying Cin = {c[0]:.6f}, Avg SE_In = {avg_se_in:.6f}")
            return avg_se_in
        
        # Define the objective function for minimizing SE_Out
        def objective_out(c):
            # Generate data with the given C value
            data = self.generate_parametric_data(
                Cin=1,
                Cout=c[0],
                num_samples=num_samples,
                num_integration_points=num_integration_points
            )
            
            # Return average SE_Out
            avg_se_out = data['SE_Out'].mean()
            print(f"Trying Cout = {c[0]:.6f}, Avg SE_Out = {avg_se_out:.6f}")
            return avg_se_out
        
        # Run the optimization for Cin (minimizing SE_In)
        print("Starting optimization to find optimal Cin (minimizing SE_In)...")
        result_in = minimize(
            objective_in,
            x0=[initial_c],
            method='Nelder-Mead',
            options={'xtol': 1e-6, 'disp': True}
        )
        
        # Run the optimization for Cout (minimizing SE_Out)
        print("\nStarting optimization to find optimal Cout (minimizing SE_Out)...")
        result_out = minimize(
            objective_out,
            x0=[initial_c],
            method='Nelder-Mead',
            options={'xtol': 1e-6, 'disp': True}
        )
        
        # Generate final data with optimal Cin and Cout
        optimal_cin = result_in.x[0]
        optimal_cout = result_out.x[0]
        
        optimal_data = self.generate_parametric_data(
            Cin=optimal_cin,
            Cout=optimal_cout,
            num_samples=num_samples,
            num_integration_points=num_integration_points
        )
        
        print(f"\nOptimization complete!")
        print(f"Optimal Cin = {optimal_cin:.6f} (minimizes SE_In)")
        print(f"Optimal Cout = {optimal_cout:.6f} (minimizes SE_Out)")
        print(f"Final average SE_In with Cin = {optimal_data['SE_In'].mean():.6f}")
        print(f"Final average SE_Out with Cin = {optimal_data['SE_Out'].mean():.6f}")

        
        return optimal_cin, optimal_cout, optimal_data
        

def main():
    # Initialize the parametric value model
    mu = 0.1
    sigma = 0.5
    gamma = 0.03
    delta_t = 1
    
    # Create the parametric value model
    model = ParametricValueModel(mu, sigma, gamma, delta_t)
    
    # Find separate optimal Cin and Cout to minimize SE_In and SE_Out
    # using trapezoidal integration
    print("Using trapezoidal integration to find optimal C values...")
    optimal_cin, optimal_cout, optimal_data = model.find_optimal_c_in_out(
        num_samples=100,
        num_integration_points=500,
        initial_c=1.0
    )
    
    # Select columns to display for dual C results
    display_columns = [
        'state_idx', 'p', 'x', 'y', 'PV0', 
        'V0_in', 'V0_out',
        'Ein', 'Eout', 
        'EV_in', 'EV_out',
        'Vd_in', 'Vd_out',
        'SE_In', 'SE_Out'
    ]
    
    # Print the dual C results
    print("\nFinal results with optimal Cin and Cout:")
    print(optimal_data[display_columns].head().to_markdown())
    
    # Save results to CSV
    optimal_data.to_csv('optimal_dual_c_results.csv', index=False)
    print("Results saved to 'optimal_dual_c_results.csv'")

if __name__ == "__main__":
    main()