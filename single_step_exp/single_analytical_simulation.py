import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.stats import norm
from tqdm import tqdm
import pandas as pd

def integrate_incoming_fee(L, x, gamma, p0, mu, sigma):
    y = L**2 / x
    price_ratio = y/x
    spread_lower_bound = price_ratio * (1-gamma)
    spread_upper_bound = price_ratio / (1-gamma)
    
    def gbm_pdf(p1, p0, mu, sigma, t=1):
        m = np.log(p0) + (mu - 0.5 * sigma**2) * t
        s = sigma * np.sqrt(t)
        return 1 / (p1 * s * np.sqrt(2 * np.pi)) * \
               np.exp(-(np.log(p1) - m)**2 / (2 * s**2))
        
    def lower_integrand(p1):
        delta_x = 1/(1-gamma) * (L * np.sqrt((1-gamma) / p1) - x)
        return gamma * p1 * delta_x * gbm_pdf(p1, p0, mu, sigma)
    
    def upper_integrand(p1):
        delta_y = 1/(1-gamma) * (L * np.sqrt((1-gamma) * p1) - y)
        return gamma * delta_y * gbm_pdf(p1, p0, mu, sigma)
    
    epsilon = 1e-10
    infinity = 100 * p0
    
    integral_lower, error_lower = integrate.quad(lower_integrand, epsilon, spread_lower_bound)
    integral_upper, error_upper = integrate.quad(upper_integrand, spread_upper_bound, infinity)
    
    return integral_lower + integral_upper

def analytical_incoming_fee(L, p0, x, gamma, sigma, mu, delta_t=1):
    y = L**2 / x
    
    d1 = (np.log((1-gamma) * y / (p0 * x)) - mu * delta_t) / (sigma * np.sqrt(delta_t))
    d2 = (np.log(y / ((1-gamma) * x * p0)) - mu * delta_t) / (sigma * np.sqrt(delta_t))
    
    alpha = L*np.sqrt((1-gamma)*p0) * np.exp(mu/2 - sigma**2/8) * delta_t
    
    return gamma/(1-gamma) * (
        alpha * (norm.cdf(d1) + norm.cdf(-d2)) -\
        np.exp(mu*delta_t)*p0*x*norm.cdf(d1 - sigma*np.sqrt(delta_t)/2) -\
        y*norm.cdf(-d2 - sigma*np.sqrt(delta_t)/2)
        )

def integrate_outgoing_fee(L, x, gamma, p0, mu, sigma):
    y = L**2 / x
    price_ratio = y/x
    spread_lower_bound = price_ratio * (1-gamma)
    spread_upper_bound = price_ratio / (1-gamma)
    
    def gbm_pdf(p1, p0, mu, sigma, t=1):
        m = np.log(p0) + (mu - 0.5 * sigma**2) * t
        s = sigma * np.sqrt(t)
        return 1 / (p1 * s * np.sqrt(2 * np.pi)) * \
               np.exp(-(np.log(p1) - m)**2 / (2 * s**2))
        
    def lower_integrand(p1):
        delta_y = y - L * np.sqrt(p1/(1-gamma))
        return gamma * delta_y * gbm_pdf(p1, p0, mu, sigma)
    
    def upper_integrand(p1):
        delta_x = x - L * np.sqrt(1/((1-gamma) * p1))
        return gamma * p1 * delta_x * gbm_pdf(p1, p0, mu, sigma)
    
    epsilon = 1e-10
    infinity = 100 * p0
    
    integral_lower, error_lower = integrate.quad(lower_integrand, epsilon, spread_lower_bound)
    integral_upper, error_upper = integrate.quad(upper_integrand, spread_upper_bound, infinity)
    
    return integral_lower + integral_upper

def analytical_outgoing_fee(L, p0, x, gamma, sigma, mu, delta_t=1):
    y = L**2 / x
    
    d1 = (np.log((1-gamma) * y / (p0 * x)) - mu * delta_t) / (sigma * np.sqrt(delta_t))
    d2 = (np.log(y / ((1-gamma) * x * p0)) - mu * delta_t) / (sigma * np.sqrt(delta_t))
    
    alpha = L*np.sqrt(p0/(1-gamma)) * np.exp(mu/2 - sigma**2/8) * delta_t
    
    return gamma * (
        - alpha * (norm.cdf(d1) + norm.cdf(-d2)) +\
        np.exp(mu*delta_t)*p0*x*norm.cdf(-d2 + sigma*np.sqrt(delta_t)/2) +\
        y*norm.cdf(d1 + sigma*np.sqrt(delta_t)/2)
        )

def integrate_pool_value(L, x, gamma, p0, mu, sigma):
    y = L**2 / x
    price_ratio = y/x
    spread_lower_bound = price_ratio * (1-gamma)
    spread_upper_bound = price_ratio / (1-gamma)
    
    def gbm_pdf(p1, p0, mu, sigma, t=1):
        m = np.log(p0) + (mu - 0.5 * sigma**2) * t
        s = sigma * np.sqrt(t)
        return 1 / (p1 * s * np.sqrt(2 * np.pi)) * \
               np.exp(-(np.log(p1) - m)**2 / (2 * s**2))
        
    def lower_integrand(p1):
        y1 = L * np.sqrt(p1/(1-gamma))
        x1 = L * np.sqrt((1-gamma)/p1)
        return (p1*x1 + y1) * gbm_pdf(p1, p0, mu, sigma)
    
    def mid_integrand(p1):
        return (p0*x + y) * gbm_pdf(p1, p0, mu, sigma)
        
    
    def upper_integrand(p1):
        y1 = L * np.sqrt(p1/(1-gamma))
        x1 = L * np.sqrt((1-gamma)/p1)
        return (p1*x1 + y1) * gbm_pdf(p1, p0, mu, sigma)
    
    epsilon = 1e-10
    infinity = 100 * p0
    
    integral_lower, error_lower = integrate.quad(lower_integrand, epsilon, spread_lower_bound)
    integral_mid, error_mid = integrate.quad(mid_integrand, spread_lower_bound, spread_upper_bound)
    integral_upper, error_upper = integrate.quad(upper_integrand, spread_upper_bound, infinity)
    
    return integral_lower + integral_mid + integral_upper

def analytical_pool_value(L, p0, x, gamma, sigma, mu, delta_t=1):
    y = L**2 / x
    
    d1 = (np.log((1-gamma) * y / (p0 * x)) - mu * delta_t) / (sigma * np.sqrt(delta_t))
    d2 = (np.log(y / ((1-gamma) * x * p0)) - mu * delta_t) / (sigma * np.sqrt(delta_t))
    
    beta = L * (2-gamma) * np.sqrt(p0/(1-gamma)) * np.exp(mu/2 - sigma**2/8) * delta_t
        
    return beta * (norm.cdf(d1) + norm.cdf(-d2)) + (p0*x + y) * (norm.cdf(d2+sigma*np.sqrt(delta_t)/2) - norm.cdf(d1+sigma*np.sqrt(delta_t)/2))


def collect_results():
    # Calculate total iterations for overall progress tracking
    gammas = np.round(np.arange(0.0001, 0.01001, 0.0001), 4)
    x_values = np.round(np.arange(10, 210, 10), 2)
    relative_p_values = np.round(np.arange(0, 1.01, 0.1), 2)
    mu_values = np.round(np.arange(0, 0.45, 0.05), 2)
    sigma_values = np.round(np.arange(0.1, 0.45, 0.05), 2)
    
    total_iterations = len(gammas) * len(x_values) * len(relative_p_values) * len(mu_values) * len(sigma_values)
    print(f"Total iterations to process: {total_iterations}")
    
    results = []
    L = 100
    
    # Create overall progress bar
    overall_pbar = tqdm(total=total_iterations, desc="Overall progress", position=0)
    current_iteration = 0
    
    for gamma in gammas:
        for x in x_values:
            y = L**2 / x
            price_ratio = y/x
            upper_bound = price_ratio / (1-gamma)
            lower_bound = price_ratio * (1-gamma)
            
            # Inner loop progress bar
            for relative_p in relative_p_values:
                p0 = lower_bound + relative_p * (upper_bound - lower_bound)
                for mu in mu_values:
                    for sigma in sigma_values:
                        analytical_incoming = analytical_incoming_fee(L=L, x=x, gamma=gamma, p0=p0, mu=mu, sigma=sigma)
                        integrate_incoming = integrate_incoming_fee(L=L, x=x, gamma=gamma, p0=p0, mu=mu, sigma=sigma)
                        analytical_outgoing = analytical_outgoing_fee(L=L, x=x, gamma=gamma, p0=p0, mu=mu, sigma=sigma)
                        integrate_outgoing = integrate_outgoing_fee(L=L, x=x, gamma=gamma, p0=p0, mu=mu, sigma=sigma)
                        analytical_pool = analytical_pool_value(L=L, x=x, gamma=gamma, p0=p0, mu=mu, sigma=sigma)
                        integrate_pool = integrate_pool_value(L=L, x=x, gamma=gamma, p0=p0, mu=mu, sigma=sigma)
                            
                        results.append({
                            'L': L,
                            'x': x,
                            'y': y,
                            'gamma': gamma,
                            'relative_p': relative_p,
                            'p0': p0,
                            'drift': mu,
                            'sigma': sigma,
                            'analytical_incoming_fee': analytical_incoming,
                            'integrate_incoming_fee': integrate_incoming,
                            'analytical_outgoing_fee': analytical_outgoing,
                            'integrate_outgoing_fee': integrate_outgoing,
                            'analytical_pool_value': analytical_pool,
                            'integrate_pool_value': integrate_pool
                        })
                        
                        # Update overall progress
                        current_iteration += 1
                        overall_pbar.update(1)
    # Close progress bar
    overall_pbar.close()
                            
    results_df = pd.DataFrame(results)
    results_df.to_csv("comprehensive_comparison_results.csv")
    print(f"Completed all {current_iteration} iterations")
    print(results_df.head().to_markdown())
    
collect_results()