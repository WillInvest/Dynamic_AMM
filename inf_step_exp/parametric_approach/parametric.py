import torch    
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.integrate import quad
from CPMM import AMM

class ParametricValueModel:
    def __init__(self, L, mu, sigma, gamma, delta_t=0.0001):
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
        self.L = L
        self.amm = AMM(
            L=L,
            gamma=gamma,
            sigma=sigma,
            delta_t=delta_t,
            mu=mu
        )
    def generate_raw_data(self, k=0, num_samples=1):
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
        
        # Generate initial states
        states = self.amm.generate_training_data(k=k,num_samples=num_samples)
        states_np = states.cpu().numpy()
        # Calculate expected fees using original state values
        original_p = states_np[:, 0]
        original_x = states_np[:, 1]
        original_y = states_np[:, 2]
        expected_incoming_fee, expected_outgoing_fee, pool_value = self.amm.calculate_fee(original_p, original_x, original_y)

        # Create DataFrame with initial states
        states_df = pd.DataFrame({
            'state_idx': np.arange(len(states_np)),
            'p': states_np[:, 0],
            'x': states_np[:, 1],
            'y': states_np[:, 2],
            'current_pool': states_np[:, 0] * states_np[:, 1] + states_np[:, 2],
            'fin': expected_incoming_fee,
            'fout': expected_outgoing_fee
        })
        
        return states_df
    
    def generate_trapezoidal_points(self, initial_p, num_points=500):
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
        x0 = np.log(initial_p)
        drift = (self.mu - 0.5 * self.sigma**2) * self.delta_t
        diffusion = self.sigma * np.sqrt(self.delta_t)
        
        # Define integration range (±5 standard deviations around drift)
        std_range = 3
        lower_bound = x0 + drift - std_range * diffusion
        upper_bound = x0 + drift + std_range * diffusion
        
        # Generate evenly spaced points
        points = np.linspace(lower_bound, upper_bound, int(num_points))
        
        # Calculate weights for trapezoidal rule
        # For trapezoid rule, all interior points have equal weight, and the endpoints have half weight
        dx = (upper_bound - lower_bound) / (num_points - 1)
        raw_weights = np.ones(num_points) * dx
        raw_weights[0] = raw_weights[-1] = dx / 2
        
        # Calculate the probability density function (PDF) for each point
        # since we're doing a weighted integral against the lognormal distribution
        pdf = np.exp(-(points - (x0+drift))**2 / (2 * diffusion**2)) / (diffusion * np.sqrt(2 * np.pi))
        
        # Multiply weights by PDF values
        weights = raw_weights * pdf
        
        return np.exp(points), raw_weights, pdf, weights
    
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
        p_upper = initial_p / (1 - self.gamma)
        p_lower = initial_p * (1 - self.gamma)
        
        # Calculate new prices using log-normal model
        new_prices = integration_points
        
        # Initialize arrays for new x, y values
        new_x = np.zeros_like(new_prices)
        new_y = np.zeros_like(new_prices)
        
        # Create masks for the conditions
        above_mask = new_prices > p_upper
        below_mask = new_prices < p_lower
        within_mask = ~(above_mask | below_mask)
        
        # Calculate new x, y values based on conditions
        new_x[above_mask] = self.L / np.sqrt((1 - self.gamma) * new_prices[above_mask])
        new_y[above_mask] = self.L * np.sqrt((1 - self.gamma) * new_prices[above_mask])
        
        new_x[below_mask] = self.L * np.sqrt((1 - self.gamma) / new_prices[below_mask])
        new_y[below_mask] = self.L * np.sqrt(new_prices[below_mask] / (1 - self.gamma))
        
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
        continuation_in = fees['fin'] + future_values
        continuation_out = fees['fout'] + future_values
        
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
    
    def generate_parametric_data(self, Cin, Cout, k=0, num_samples=100, num_integration_points=500):
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
        raw_data = self.generate_raw_data(k=k, num_samples=num_samples)
        expected_values = []
        se = []
        for _, row in raw_data.iterrows():
            # Step 3: Calculate future states
            points, raw_weights, pdf, weights = self.generate_trapezoidal_points(row['p'], num_integration_points)
            future_states = self.calculate_future_states(row, points)
            future_states['weights'] = weights
            future_states['pdf'] = pdf
            # Step 5: Calculate expected new value using trapezoidal integration
            new_prices = future_states['new_prices']
            future_values_in = Cin * np.sqrt(new_prices)
            future_values_out = Cout * np.sqrt(new_prices)
            expected_new_value_in = np.sum(future_values_in * weights)
            expected_new_value_out = np.sum(future_values_out * weights)
            
            # Calculate expected values
            expected_in = row['fin'] + expected_new_value_in
            expected_out = row['fout'] + expected_new_value_out
            # Step 7: Calculate metrics with shortcut names
            current_pool = row['p'] * row['x'] + row['y']
            current_value_in = Cin * np.sqrt(row['p'])
            current_value_out = Cout * np.sqrt(row['p'])

            # Calculate discounted values
            discounted_in_value = self.discount_factor * expected_in
            discounted_out_value = self.discount_factor * expected_out
            
            # Calculate squared errors
            se_max_in = np.sqrt((current_value_in - max(current_pool, discounted_in_value))**2)
            se_max_out = np.sqrt((current_value_out - max(current_pool, discounted_out_value))**2)
            se_cont_in = np.sqrt((current_value_in - discounted_in_value)**2)
            se_cont_out = np.sqrt((current_value_out - discounted_out_value)**2)
            
            expected_values.append({
                'state_idx': row['state_idx'],
                'V0_in': current_value_in,
                'V0_out': current_value_out,
                'EDV_in': discounted_in_value,
                'EDV_out': discounted_out_value
            })
            se.append({
                'SE_max_in': se_max_in,
                'SE_max_out': se_max_out,
                'SE_cont_in': se_cont_in,
                'SE_cont_out': se_cont_out
            })
            
        # Create DataFrame with expected values
        expected_df = pd.DataFrame(expected_values)
        se = pd.DataFrame(se, columns=['SE_max_in', 'SE_max_out', 'SE_cont_in', 'SE_cont_out'])
        # Merge with raw data
        final_df = pd.merge(raw_data, expected_df, on='state_idx')
        
        return final_df, se
    
    def find_optimal_c_in_out(self, k=0, num_samples=100, num_integration_points=500, initial_c=1.0, maximum=True):
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
            data, se = self.generate_parametric_data(
                Cin=c[0],
                Cout=1,
                k=k,
                num_samples=num_samples,
                num_integration_points=num_integration_points
            )
            
            # Return average SE_In
            if maximum:
                avg_se_in = se['SE_max_in'].mean()
            else:
                avg_se_in = se['SE_cont_in'].mean()
            return avg_se_in
        
        # Define the objective function for minimizing SE_Out
        def objective_out(c):
            # Generate data with the given C value
            data, se = self.generate_parametric_data(
                Cin=1,
                Cout=c[0],
                k=k,
                num_samples=num_samples,
                num_integration_points=num_integration_points
            )
            
            # Return average SE_Out
            if maximum:
                avg_se_out = se['SE_max_out'].mean()
            else:
                avg_se_out = se['SE_cont_out'].mean()
            return avg_se_out
        
        # Run the optimization for Cin (minimizing SE_In)
        result_in = minimize(
            objective_in,
            x0=[initial_c],
            method='Nelder-Mead'
        )
        
        # Run the optimization for Cout (minimizing SE_Out)
        result_out = minimize(
            objective_out,
            x0=[initial_c],
            method='Nelder-Mead'
        )
        
        # Generate final data with optimal Cin and Cout
        optimal_cin = result_in.x[0]
        optimal_cout = result_out.x[0]
        
        optimal_data, se = self.generate_parametric_data(
            Cin=optimal_cin,
            Cout=optimal_cout,
            k=k,
            num_samples=num_samples,
            num_integration_points=num_integration_points
        )
        
        return optimal_cin, optimal_cout, optimal_data, se
        

def parallel_run():
    from tqdm import tqdm

    results = []
    L_list = [100, 1000, 10000]
    k_list = [-1, -0.5, 0, 0.5, 1]
    mu_list = [0.0, 0.2, 0.4]
    sigma_list = [0.1, 0.2, 0.3]
    gamma_list = [0.0005, 0.003, 0.01]
    delta_t_list = [1, 2, 3]
    num_sample_list = [10, 100, 1000]
    num_integration_points_list = [10, 100, 1000]

    # Calculate total number of iterations
    total_iterations = len(L_list) * len(mu_list) * len(sigma_list) * len(gamma_list) * len(delta_t_list) * len(num_sample_list) * len(num_integration_points_list)

    # Create progress bar
    pbar = tqdm(total=total_iterations, desc="Processing parameter combinations")

    for L in L_list:
        for k in k_list:
            for mu in mu_list:
                for sigma in sigma_list:
                    for gamma in gamma_list:
                        for delta_t in delta_t_list:
                            for num_sample in num_sample_list:
                                for num_integration_points in num_integration_points_list:
                                    model = ParametricValueModel(L=L,mu=mu, sigma=sigma, gamma=gamma, delta_t=delta_t)
                                    optimal_cin, optimal_cout, optimal_data, se = model.find_optimal_c_in_out(num_samples=num_sample, num_integration_points=num_integration_points, initial_c=1.0, maximum=False)
                                    results.append({
                                        'L': L,
                                        'mu': mu,
                                        'sigma': sigma,
                                        'gamma': gamma,
                                        'delta_t': delta_t,
                                        'num_sample': num_sample,
                                        'num_integration_points': num_integration_points,
                                        'optimal_cin': optimal_cin,
                                        'optimal_cout': optimal_cout
                                        })
                                    # Update progress bar
                                    pbar.update(1)
                                    # Update description with current parameters
                                    pbar.set_description(f"L={L}, k={k}, mu={mu}, sigma={sigma}, gamma={gamma}, num_sample={num_sample}, num_integration_points={num_integration_points}")

    # Close progress bar
    pbar.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv('results.csv', index=False)

if __name__ == "__main__":
    parallel_run()