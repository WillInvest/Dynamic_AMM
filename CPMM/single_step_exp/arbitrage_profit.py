import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.stats import norm
from tqdm import tqdm
import pandas as pd
import scipy

def integrate_incoming_fee(L, x, gamma, p0, mu, sigma, delta_t=1/(365*24)):
    y = L**2 / x
    price_ratio = y/x
    spread_lower_bound = price_ratio * (1-gamma)
    spread_upper_bound = price_ratio / (1-gamma)
    
    eps = 1e-6
    dist = scipy.stats.lognorm(sigma * np.sqrt(delta_t), scale=np.exp(mu - 0.5 * sigma**2 * delta_t ))
    lb = dist.ppf(eps)
    ub = dist.ppf(1-eps)
    
    def gbm_pdf(p1, p0, mu, sigma, t=delta_t):
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
    
    
    integral_lower, error_lower = integrate.quad(lower_integrand, lb, spread_lower_bound)
    integral_upper, error_upper = integrate.quad(upper_integrand, spread_upper_bound, ub)
    
    return integral_lower + integral_upper

def integrate_outgoing_fee(L, x, gamma, p0, mu, sigma, delta_t=1/(365*24)):
    y = L**2 / x
    price_ratio = y/x
    spread_lower_bound = price_ratio * (1-gamma)
    spread_upper_bound = price_ratio / (1-gamma)
    
    eps = 1e-6
    dist = scipy.stats.lognorm(sigma * np.sqrt(delta_t), scale=np.exp(mu - 0.5 * sigma**2 * delta_t))
    lb = dist.ppf(eps)
    ub = dist.ppf(1-eps)
    
    def gbm_pdf(p1, p0, mu, sigma, t=delta_t):
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
    
    
    integral_lower, error_lower = integrate.quad(lower_integrand, lb, spread_lower_bound, points=10000)
    integral_upper, error_upper = integrate.quad(upper_integrand, spread_upper_bound, ub, points=10000)
    
    return integral_lower + integral_upper

def incoming_arbitrage_profit(L, x, gamma, p0, mu, sigma, delta_t=1/(365*24)):
    y = L**2 / x
    price_ratio = y/x
    spread_lower_bound = price_ratio * (1-gamma)
    spread_upper_bound = price_ratio / (1-gamma)
    
    eps = 1e-6
    dist = scipy.stats.lognorm(sigma * np.sqrt(delta_t), scale=np.exp(mu - 0.5 * sigma**2 * delta_t))
    lb = dist.ppf(eps)
    ub = dist.ppf(1-eps)
    
    def gbm_pdf(p1, p0, mu, sigma, t=delta_t):
        m = np.log(p0) + (mu - 0.5 * sigma**2) * t
        s = sigma * np.sqrt(t)
        return 1 / (p1 * s * np.sqrt(2 * np.pi)) * \
               np.exp(-(np.log(p1) - m)**2 / (2 * s**2))
        
    def lower_integrand(p1):
        delta_y = y - L * np.sqrt(p1/(1-gamma))
        delta_x = 1/(1-gamma) * (L * np.sqrt((1-gamma) / p1) - x)
        profit = (delta_y - delta_x * p1)
        assert profit > 0, f"Profit is negative: {profit}"
        return profit * gbm_pdf(p1, p0, mu, sigma)
    
    def upper_integrand(p1):
        delta_x = x - L * np.sqrt(1/((1-gamma) * p1))
        delta_y = 1/(1-gamma) * (L * np.sqrt((1-gamma) * p1) - y)
        profit = (delta_x * p1 - delta_y)
        assert profit > 0, f"Profit is negative: {profit}"
        return profit * gbm_pdf(p1, p0, mu, sigma)
    
    
    integral_lower, error_lower = integrate.quad(lower_integrand, lb*spread_lower_bound, spread_lower_bound, epsabs=1e-10, epsrel=1e-10)
    integral_upper, error_upper = integrate.quad(upper_integrand, spread_upper_bound, ub*spread_upper_bound, epsabs=1e-10, epsrel=1e-10)
    
    return integral_lower + integral_upper


def outgoing_arbitrage_profit(L, x, gamma, p0, mu, sigma, delta_t=1/(365*24)):
    y = L**2 / x
    price_ratio = y/x
    spread_lower_bound = price_ratio * (1-gamma)
    spread_upper_bound = price_ratio / (1-gamma)
    
    eps = 1e-6
    dist = scipy.stats.lognorm(sigma * np.sqrt(delta_t), scale=np.exp(mu - 0.5 * sigma**2 * delta_t))
    lb = dist.ppf(eps)
    ub = dist.ppf(1-eps)
    
    def gbm_pdf(p1, p0, mu, sigma, t=delta_t):
        m = np.log(p0) + (mu - 0.5 * sigma**2) * t
        s = sigma * np.sqrt(t)
        return 1 / (p1 * s * np.sqrt(2 * np.pi)) * \
               np.exp(-(np.log(p1) - m)**2 / (2 * s**2))
        
    def lower_integrand(p1):
        delta_x = L * np.sqrt((1-gamma)/p1) - x
        delta_y = (1-gamma) * (y - L * np.sqrt(p1/(1-gamma)))
        profit = (delta_y - delta_x * p1)
        assert profit > 0, f"Profit is negative: {profit}"
        return profit * gbm_pdf(p1, p0, mu, sigma)
    
    def upper_integrand(p1):
        delta_x = (1-gamma) * (x - L / np.sqrt((1-gamma) * p1))
        delta_y = L * np.sqrt(p1*(1-gamma)) - y
        profit = (delta_x * p1 - delta_y)
        assert profit > 0, f"Profit is negative: {profit}"
        return profit * gbm_pdf(p1, p0, mu, sigma)
    
    integral_lower, error_lower = integrate.quad(lower_integrand, lb*spread_lower_bound, spread_lower_bound, epsabs=1e-10, epsrel=1e-10)
    integral_upper, error_upper = integrate.quad(upper_integrand, spread_upper_bound, ub*spread_upper_bound, epsabs=1e-10, epsrel=1e-10)
    
    return integral_lower + integral_upper


def collect_results():
    # Calculate total iterations for overall progress tracking
    # gammas = np.round(np.arange(0.0005, 0.0205, 0.0005), 4)
    
    opt_df = pd.read_csv('/Users/haofu/Desktop/AMM/Dynamic_AMM/fee_optimization_results_one_year.csv')
    
    gammas = [0.3]
    x_values = [1e6]
    relative_p_values = [1]
    mu_values = [0]
    sigma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    total_iterations = len(gammas) * len(x_values) * len(relative_p_values) * len(mu_values) * len(sigma_values)
    print(f"Total iterations to process: {total_iterations}")
    
    results = []
    L = 1e6
    
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
                p0 = 1
                for mu in mu_values:
                    for sigma in sigma_values:
                        opt_gamma_in = opt_df[opt_df['sigma'] == sigma]['opt_gamma_in'].values[0]
                        opt_gamma_out = opt_df[opt_df['sigma'] == sigma]['opt_gamma_out'].values[0]
                                                    
                        delta_t = 1
                        fix_incoming_fee = integrate_incoming_fee(L=L, x=x, gamma=gamma, p0=p0, mu=mu, sigma=sigma, delta_t=delta_t)
                        fix_outgoing_fee = integrate_outgoing_fee(L=L, x=x, gamma=gamma, p0=p0, mu=mu, sigma=sigma, delta_t=delta_t)
                        fix_arbitrage_incoming = incoming_arbitrage_profit(L=L, x=x, gamma=gamma, p0=p0, mu=mu, sigma=sigma, delta_t=delta_t)
                        fix_arbitrage_outgoing = outgoing_arbitrage_profit(L=L, x=x, gamma=gamma, p0=p0, mu=mu, sigma=sigma, delta_t=delta_t)
                        opt_incoming_fee = integrate_incoming_fee(L=L, x=x, gamma=opt_gamma_in, p0=p0, mu=mu, sigma=sigma, delta_t=delta_t)
                        opt_outgoing_fee = integrate_outgoing_fee(L=L, x=x, gamma=opt_gamma_out, p0=p0, mu=mu, sigma=sigma, delta_t=delta_t)
                        opt_arbitrage_incoming = incoming_arbitrage_profit(L=L, x=x, gamma=opt_gamma_in, p0=p0, mu=mu, sigma=sigma, delta_t=delta_t)
                        opt_arbitrage_outgoing = outgoing_arbitrage_profit(L=L, x=x, gamma=opt_gamma_out, p0=p0, mu=mu, sigma=sigma, delta_t=delta_t)
                        
                        results.append({
                            'L': L,
                            'x': x,
                            'y': y,
                            'gamma': gamma,
                            'relative_p': relative_p,
                            'p0': p0,
                            'drift': mu,
                            'sigma': sigma,
                            'incoming_fee': fix_incoming_fee,
                            'outgoing_fee': fix_outgoing_fee,
                            'arbitrage_incoming': fix_arbitrage_incoming,
                            'arbitrage_outgoing': fix_arbitrage_outgoing,
                            'opt_incoming_fee': opt_incoming_fee,
                            'opt_outgoing_fee': opt_outgoing_fee,
                            'opt_arbitrage_incoming': opt_arbitrage_incoming,
                            'opt_arbitrage_outgoing': opt_arbitrage_outgoing
                        })
                        
                        # Update overall progress
                        current_iteration += 1
                        overall_pbar.update(1)
    # Close progress bar
    overall_pbar.close()
                            
    results_df = pd.DataFrame(results)
    # sort by sigma and gamma
    results_df = results_df.sort_values(by=['sigma', 'gamma'])
    results_df.to_csv("single_step_arbitrage_profit_results_one_year.csv")
    print(f"Completed all {current_iteration} iterations")
    # print(results_df.head().to_markdown())
    
collect_results()