import numpy as np
from numba import jit
from numba import prange
import os

os.environ['OMP_MAX_ACTIVE_LEVELS'] = '2'  # Set maximum active levels for nested parallelism

parallel = True

@jit(nopython=True)
def _update_prices(prices, z, sigma_values, dt):
    """JIT-compiled price update function"""
    for i, sigma in enumerate(sigma_values):
        drift = -0.5 * sigma**2 * dt
        diffusion = sigma * np.sqrt(dt)
        prices[:, i] = prices[:, i] * np.exp(drift + diffusion * z)
    return prices

@jit(nopython=True, parallel=parallel)
def _update_distribute_case(x_dis, y_dis, prices, gammas, L, epsilon):
    """JIT-compiled distribute case calculations"""
    x_new = x_dis.copy()
    y_new = y_dis.copy()
    fees_inc = np.zeros_like(x_dis)
    fees_out = np.zeros_like(x_dis)
    
    for i in prange(prices.shape[0]):  # num_paths
        for j in range(prices.shape[1]):  # num_sigmas
            for k in range(len(gammas)):  # num_gammas
                price = prices[i, j]
                gamma = gammas[k]
                price_ratio = y_dis[i, j, k] / x_dis[i, j, k]
                
                if price > price_ratio / (1-gamma):  # upper case
                    x_new[i,j,k] = L / np.sqrt((1-gamma)*price)
                    y_new[i,j,k] = L*np.sqrt((1-gamma)*price)
                    fees_inc[i,j,k] = (gamma/(1-gamma)) * (L*np.sqrt((1-gamma)*price)-y_dis[i,j,k])
                    fees_out[i,j,k] = gamma * price * (x_dis[i,j,k] - L/np.sqrt((1-gamma)*price))
                elif price < price_ratio * (1-gamma):  # lower case
                    x_new[i,j,k] = L*np.sqrt((1-gamma)/price)
                    y_new[i,j,k] = L*np.sqrt(price/(1-gamma))
                    fees_inc[i,j,k] = (gamma/(1-gamma)) * price * (L*np.sqrt((1-gamma)/price) - x_dis[i,j,k])
                    fees_out[i,j,k] = gamma * (y_dis[i,j,k] - L*np.sqrt(price/(1-gamma)))
    
    return x_new, y_new, fees_inc, fees_out

@jit(nopython=True, parallel=parallel)
def _update_rebalance_case(x_rinc, y_rinc, x_rout, y_rout, prices, gammas, L_rinc, L_rout):
    """JIT-compiled rebalance case calculations"""
    x_rinc_new = x_rinc.copy()
    y_rinc_new = y_rinc.copy()
    x_rout_new = x_rout.copy()
    y_rout_new = y_rout.copy()
    
    for i in prange(prices.shape[0]):
        for j in range(prices.shape[1]):
            for k in range(len(gammas)):
                price = prices[i, j]
                gamma = gammas[k]
                
                # Incoming case
                price_ratio_inc = y_rinc[i,j,k] / x_rinc[i,j,k]
                if price > price_ratio_inc / (1-gamma):  # upper case
                    a = 1.0 - gamma
                    b = (2.0 - gamma) * y_rinc[i,j,k]
                    c = y_rinc[i,j,k]**2 - L_rinc[i,j,k]**2 * price * (1.0 - gamma)
                    discriminant = b**2 - 4.0 * a * c
                    sqrt_discriminant = np.sqrt(discriminant)
                    delta_y = (-b + sqrt_discriminant) / (2.0 * a)
                    y_rinc_new[i,j,k] = y_rinc[i,j,k] + delta_y
                    x_rinc_new[i,j,k] = y_rinc_new[i,j,k] / (price * (1.0 - gamma))
                elif price < price_ratio_inc * (1-gamma):  # lower case
                    a = 1.0 - gamma
                    b = (2.0 - gamma) * x_rinc[i,j,k]
                    c = x_rinc[i,j,k]**2 - L_rinc[i,j,k]**2 * (1.0 - gamma) / price
                    discriminant = b**2 - 4.0 * a * c
                    sqrt_discriminant = np.sqrt(discriminant)
                    delta_x = (-b + sqrt_discriminant) / (2.0 * a)
                    x_rinc_new[i,j,k] = x_rinc[i,j,k] + delta_x
                    y_rinc_new[i,j,k] = x_rinc_new[i,j,k] * price / (1.0 - gamma)
                
                # Outgoing case
                price_ratio_out = y_rout[i,j,k] / x_rout[i,j,k]
                if price > price_ratio_out / (1-gamma):  # upper case
                    a = 1.0 - gamma
                    b = -(2.0 - gamma) * x_rout[i,j,k]
                    c = x_rout[i,j,k]**2 - L_rout[i,j,k]**2 / ((1.0 - gamma) * price)
                    discriminant = b**2 - 4.0 * a * c
                    sqrt_discriminant = np.sqrt(discriminant)
                    delta_x = (-b - sqrt_discriminant) / (2.0 * a)
                    x_rout_new[i,j,k] = x_rout[i,j,k] - (1.0 - gamma) * delta_x
                    y_rout_new[i,j,k] = x_rout_new[i,j,k] * (1.0 - gamma) * price
                elif price < price_ratio_out * (1-gamma):  # lower case
                    a = 1.0 - gamma
                    b = -(2.0 - gamma) * y_rout[i,j,k]
                    c = y_rout[i,j,k]**2 - L_rout[i,j,k]**2 * price / (1.0 - gamma)
                    discriminant = b**2 - 4.0 * a * c
                    sqrt_discriminant = np.sqrt(discriminant)
                    delta_y = (-b - sqrt_discriminant) / (2.0 * a)
                    y_rout_new[i,j,k] = y_rout[i,j,k] - (1.0 - gamma) * delta_y
                    x_rout_new[i,j,k] = y_rout_new[i,j,k] * (1.0 - gamma) / price
    
    return x_rinc_new, y_rinc_new, x_rout_new, y_rout_new