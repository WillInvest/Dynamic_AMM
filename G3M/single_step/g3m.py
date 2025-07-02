#%%
import numpy as np
from scipy.stats import norm
import pandas as pd

def calculate_G3M_single_step_fees(sigma: float, w: float, gamma: float,
                                   pv0: float = 1.0, L: float = 1.0, 
                                   delta_t: float = 1.0) -> float:
    """
    Calculate the total expected fees from both tokens X and Y in the G3M model.
    
    Args:
        sigma: Volatility parameter
        w: Weight parameter
        gamma: Fee parameter
        pv0: Initial pool value
        L: Total liquidity
        delta_t: Time step
        
    Returns:
        float: Total expected fees from both tokens
    """
    # Common terms
    sqrt_delta_t = np.sqrt(delta_t)
    sigma_sqrt_dt = sigma * sqrt_delta_t
    w_ratio = w / (1 - w)  # w/(1-w) for Phi arguments
    w_ratio_inv = (1 - w) / w  # (1-w)/w for first terms
    gamma_term = gamma / (1 - gamma)
    x_t = (L/((1-w)*pv0)**(1-w))**(1/w)
    y_t = (1-w) * pv0
    S_t = w_ratio * y_t / x_t
    
    # Calculate fee from token X
    x_fee = gamma_term * (
        # First term
        L * ((w_ratio_inv / (1 - gamma)) ** (w-1)) * (S_t ** w) * 
        np.exp(0.5 * sigma**2 * delta_t * w * (w - 1)) *
        norm.cdf(
            (1 / sigma_sqrt_dt) * (
                np.log(w_ratio * (1 - gamma) * y_t / (S_t * x_t)) +
                (0.5 - w) * sigma**2 * delta_t
            )
        ) -
        # Second term
        S_t * x_t *
        norm.cdf(
            (1 / sigma_sqrt_dt) * (
                np.log(w_ratio * y_t * (1 - gamma) / (S_t * x_t )) -
                0.5 * sigma**2 * delta_t
            )
        )
    )
    
    # Calculate fee from token Y
    y_fee = gamma_term * (
        # First term
        L * ((w_ratio_inv * (1 - gamma) * S_t) ** w) *
        np.exp(0.5 * sigma**2 * delta_t * w * (w - 1)) *
        norm.cdf(
            (-1 / sigma_sqrt_dt) * (
                np.log(w_ratio * y_t / (S_t * x_t * (1 - gamma))) +
                (0.5 - w) * sigma**2 * delta_t
            )
        ) -
        # Second term
        y_t *
        norm.cdf(
            (-1 / sigma_sqrt_dt) * (
                np.log(w_ratio * y_t / (S_t * x_t * (1 - gamma))) +
                0.5 * sigma**2 * delta_t
            )
        )
    )
    
    print(f"x_fee: {x_fee:.8f}, y_fee: {y_fee:.8f}")
    
    return x_fee + y_fee


def calculate_G3M_expected_pool_value(sigma: float, w: float, gamma: float,
                                     pv0: float = 1.0, L: float = 1.0, 
                                     delta_t: float = 1.0) -> float:
    """
    Calculate the expected pool value E[PV_{t+1}] for the G3M model.
    
    Args:
        sigma: Volatility parameter
        w: Weight parameter
        gamma: Fee parameter
        x_t: Current amount of token X
        y_t: Current amount of token Y
        delta_t: Time step
        
    Returns:
        float: Expected pool value at t+1
    """
    # Common terms
    sigma_sqrt_dt = sigma * np.sqrt(delta_t)
    x_t = (L/((1-w)*pv0)**(1-w))**(1/w)
    y_t = (1-w) * pv0
    S_t = (w / (1 - w)) * (y_t / x_t)
    
    # Calculate π₁ and π₂
    pi_1 = (1 - w) / (w * (1 - gamma))
    pi_2 = (1 - w) * (1 - gamma) / w

    # Calculate ν_t
    nu_t = L * (S_t ** w) * np.exp(0.5 * w * (w - 1) * sigma**2 * delta_t)
    
    # Calculate λ₁ and λ₂
    w_ratio = w / (1 - w)
    lambda_1 = w_ratio * y_t / ((1 - gamma) * S_t * x_t)
    lambda_2 = w_ratio * (1 - gamma) * y_t / (S_t * x_t)
    
    # Calculate D terms
    D_1_plus = (1 / sigma_sqrt_dt) * (np.log(lambda_1) + 0.5 * sigma**2 * delta_t)
    D_1_minus = (1 / sigma_sqrt_dt) * (np.log(lambda_1) - 0.5 * sigma**2 * delta_t)
    D_2_plus = (1 / sigma_sqrt_dt) * (np.log(lambda_2) + 0.5 * sigma**2 * delta_t)
    D_2_minus = (1 / sigma_sqrt_dt) * (np.log(lambda_2) - 0.5 * sigma**2 * delta_t)
    
    # First part of the formula
    first_term = nu_t * (
        (pi_2**w + pi_2**(w-1)) * norm.cdf(-D_1_plus + w * sigma * np.sqrt(delta_t)) +
        (pi_1**w + pi_1**(w-1)) * norm.cdf(D_2_plus - w * sigma * np.sqrt(delta_t))
    )
    
    # Second part of the formula
    second_term = y_t * (norm.cdf(D_1_plus) - norm.cdf(D_2_plus)) + \
        x_t * S_t * (norm.cdf(D_1_minus) - norm.cdf(D_2_minus))
    return first_term + second_term


def calculate_G3M_expected_fee(sigma: float, w: float, gamma: float,
                              pv0: float = 1.0, L: float = 1.0, 
                              delta_t: float = 1.0) -> float:
    """
    Calculate the expected fee E[F_{t+1}] for the G3M model.
    
    Args:
        sigma: Volatility parameter
        w: Weight parameter
        gamma: Fee parameter
        x_t: Current amount of token X
        y_t: Current amount of token Y
        delta_t: Time step
        
    Returns:
        float: Expected fee at t+1
    """
    # Common terms
    sigma_sqrt_dt = sigma * np.sqrt(delta_t)
    gamma_term = gamma / (1 - gamma)
    x_t = (L/((1-w)*pv0)**(1-w))**(1/w)
    y_t = (1-w) * pv0
    S_t = (w / (1 - w)) * (y_t / x_t)
    
    # Calculate π₁ and π₂
    pi_1 = (1 - w) / (w * (1 - gamma))
    pi_2 = (1 - w) * (1 - gamma) / w

    # Calculate ν_t
    nu_t = L * (S_t ** w) * np.exp(0.5 * w * (w - 1) * sigma**2 * delta_t)
    
    # Calculate λ₁ and λ₂
    w_ratio = w / (1 - w)
    lambda_1 = w_ratio * y_t / ((1 - gamma) * S_t * x_t)
    lambda_2 = w_ratio * (1 - gamma) * y_t / (S_t * x_t)
    
    # Calculate D terms
    D_1_plus = (1 / sigma_sqrt_dt) * (np.log(lambda_1) + 0.5 * sigma**2 * delta_t)
    D_1_minus = (1 / sigma_sqrt_dt) * (np.log(lambda_1) - 0.5 * sigma**2 * delta_t)
    D_2_plus = (1 / sigma_sqrt_dt) * (np.log(lambda_2) + 0.5 * sigma**2 * delta_t)
    D_2_minus = (1 / sigma_sqrt_dt) * (np.log(lambda_2) - 0.5 * sigma**2 * delta_t)
    
    incoming_fee_x = nu_t * pi_1 ** (w-1) * norm.cdf(D_2_plus - w * sigma * np.sqrt(delta_t)) - S_t * x_t * norm.cdf(D_2_minus)
    incoming_fee_y = nu_t * pi_2 ** w * norm.cdf(-D_1_plus + w * sigma * np.sqrt(delta_t)) - y_t * norm.cdf(-D_1_plus)
    incoming_fee = gamma_term * (incoming_fee_x + incoming_fee_y)
    
    outgoing_fee_x = S_t * x_t * norm.cdf(-D_1_minus) - nu_t * pi_2 ** (w-1) * norm.cdf(-D_1_plus + w * sigma * np.sqrt(delta_t)) 
    outgoing_fee_y = y_t * norm.cdf(D_2_plus) - nu_t * pi_1 ** w * norm.cdf(D_2_plus - w * sigma * np.sqrt(delta_t))
    outgoing_fee = gamma * (outgoing_fee_x + outgoing_fee_y)
    return incoming_fee, outgoing_fee
    

def find_optimal_gamma_with_optimizer():
    """
    Find the optimal fee rates (gamma) that maximize incoming fees and outgoing fees separately
    for each combination of sigma and w using scipy's optimizer.
    """
    from scipy.optimize import minimize_scalar
    
    # Define objective functions (negative because we want to maximize)
    def objective_incoming(gamma, sigma, w):
        incoming_fee, _ = calculate_G3M_expected_fee(sigma, w, gamma)
        return -incoming_fee
    
    def objective_outgoing(gamma, sigma, w):
        _, outgoing_fee = calculate_G3M_expected_fee(sigma, w, gamma)
        return -outgoing_fee
    
    # Parameters to test
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    w_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = []
    
    # Optimize for each combination
    for sigma in sigmas:
        for w in w_values:
            # Optimize for incoming fees
            result_incoming = minimize_scalar(
                objective_incoming,
                args=(sigma, w),
                bounds=(0.001, 0.9),
                method='bounded'
            )
            
            # Optimize for outgoing fees
            result_outgoing = minimize_scalar(
                objective_outgoing,
                args=(sigma, w),
                bounds=(0.001, 0.9),
                method='bounded'
            )
            
            if result_incoming.success and result_outgoing.success:
                optimal_gamma_incoming = result_incoming.x
                optimal_gamma_outgoing = result_outgoing.x
                
                # Calculate fees at both optimal points
                incoming_fee_at_incoming, _ = calculate_G3M_expected_fee(
                    sigma, w, optimal_gamma_incoming
                )
                _, outgoing_fee_at_outgoing = calculate_G3M_expected_fee(
                    sigma, w, optimal_gamma_outgoing
                )
                
                results.append({
                    'sigma': sigma,
                    'w': w,
                    'optimal_gamma_incoming': optimal_gamma_incoming,
                    'optimal_gamma_outgoing': optimal_gamma_outgoing,
                    'incoming_fee_at_incoming': incoming_fee_at_incoming,
                    'outgoing_fee_at_outgoing': outgoing_fee_at_outgoing
                })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df


