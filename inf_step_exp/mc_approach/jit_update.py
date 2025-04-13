import numpy as np
from numba import jit
from numba import prange
import os
import time
from numba import config

# Set the number of threads for parallel processing
# You can adjust this value based on your system's capabilities
num_threads = 20  # Change this to your desired number of threads
config.THREADING_LAYER = 'threadsafe'
os.environ['NUMBA_NUM_THREADS'] = str(num_threads)

os.environ['OMP_MAX_ACTIVE_LEVELS'] = '2'  # Set maximum active levels for nested parallelism

parallel = True

@jit(nopython=True, parallel=parallel)
def _simulate(x_init, y_init, s0, steps, num_paths, sigma_values, gamma_values, dt, seeds):
    """JIT-compiled simulate function for both distribute and rebalance cases"""
    num_sigmas = len(sigma_values)
    num_gammas = len(gamma_values)
    final_x_dis = np.zeros((num_paths, num_sigmas, num_gammas), dtype=np.float64)
    final_y_dis = np.zeros((num_paths, num_sigmas, num_gammas), dtype=np.float64)
    final_xinc_reb = np.zeros((num_paths, num_sigmas, num_gammas), dtype=np.float64)
    final_yinc_reb = np.zeros((num_paths, num_sigmas, num_gammas), dtype=np.float64)
    final_xout_reb = np.zeros((num_paths, num_sigmas, num_gammas), dtype=np.float64)
    final_yout_reb = np.zeros((num_paths, num_sigmas, num_gammas), dtype=np.float64)
    fees_inc_dis = np.zeros((num_paths, num_sigmas, num_gammas), dtype=np.float64)
    fees_out_dis = np.zeros((num_paths, num_sigmas, num_gammas), dtype=np.float64)
    final_prices = np.zeros((num_paths, num_sigmas), dtype=np.float64)
    
    for i in prange(num_paths): # num_paths
        for j in prange(num_sigmas): # num_sigmas
            seed = seeds[i]
            np.random.seed(seed)
            t = np.arange(1, steps+1) * dt
            z = np.random.normal(0, 1, size=(steps,))
            # create price paths
            drift = -0.5 * sigma_values[j]**2 * t
            diffusion = sigma_values[j] * np.sqrt(dt) * np.cumsum(z)
            price = s0 * np.exp(drift + diffusion)
            
            for k in prange(num_gammas): # num_gammas
                # Initialize with scalar values
                xdis = float(x_init)
                ydis = float(y_init)
                xinc = float(x_init)
                yinc = float(y_init)
                xout = float(x_init)
                yout = float(y_init)
                
                for t in range(steps): # num_steps
                    L_dis = np.sqrt(xdis * ydis)
                    L_inc = np.sqrt(xinc * yinc)
                    L_out = np.sqrt(xout * yout)
                    gamma = gamma_values[k]
                    current_price = price[t]
                    
                    # Distribute case
                    if current_price > (ydis / xdis) / (1-gamma): # upper case
                        fees_inc_dis[i,j,k] += (gamma/(1-gamma)) * (L_dis*np.sqrt((1-gamma)*current_price)-ydis)
                        fees_out_dis[i,j,k] += gamma * current_price * (xdis-L_dis/np.sqrt((1-gamma)*current_price))
                        xdis = L_dis / np.sqrt((1-gamma)*current_price)
                        ydis = L_dis*np.sqrt((1-gamma)*current_price)                        
                    elif current_price < (ydis / xdis) * (1-gamma): # lower case
                        fees_inc_dis[i,j,k] += (gamma/(1-gamma)) * current_price * (L_dis*np.sqrt((1-gamma)/current_price) - xdis)
                        fees_out_dis[i,j,k] += gamma * (ydis-L_dis*np.sqrt(current_price/(1-gamma)))
                        xdis = L_dis*np.sqrt((1-gamma)/current_price)
                        ydis = L_dis*np.sqrt(current_price/(1-gamma))

                    # Rebalance Incoming case
                    if current_price > (yinc / xinc) / (1-gamma):  # upper case
                        a = 1.0 - gamma
                        b = (2.0 - gamma) * yinc
                        c = yinc**2 - L_inc**2 * current_price * (1.0 - gamma)
                        delta_y = (-b + np.sqrt(b**2 - 4.0 * a * c)) / (2.0 * a)
                        yinc += delta_y
                        xinc = yinc / (current_price * (1.0 - gamma))
                    elif current_price < (yinc / xinc) * (1-gamma):  # lower case
                        a = 1.0 - gamma
                        b = (2.0 - gamma) * xinc
                        c = xinc**2 - L_inc**2 * (1.0 - gamma) / current_price
                        delta_x = (-b + np.sqrt(b**2 - 4.0 * a * c)) / (2.0 * a)
                        xinc += delta_x
                        yinc = xinc * current_price / (1.0 - gamma)
                        
                    # Rebalance Outgoing case
                    if current_price > (yout / xout) / (1-gamma):  # upper case
                        a = 1.0 - gamma
                        b = -(2.0 - gamma) * xout
                        c = xout**2 - L_out**2 / ((1.0 - gamma) * current_price)
                        delta_x = (-b - np.sqrt(b**2 - 4.0 * a * c)) / (2.0 * a)
                        xout -= (1.0 - gamma) * delta_x
                        yout = xout * (1.0 - gamma) * current_price
                    elif current_price < (yout / xout) * (1-gamma):  # lower case
                        a = 1.0 - gamma
                        b = -(2.0 - gamma) * yout
                        c = yout**2 - L_out**2 * current_price / (1.0 - gamma)
                        delta_y = (-b - np.sqrt(b**2 - 4.0 * a * c)) / (2.0 * a)
                        yout -= (1.0 - gamma) * delta_y
                        xout = yout * (1.0 - gamma) / current_price
                    
                final_x_dis[i,j,k] = xdis
                final_y_dis[i,j,k] = ydis
                final_xinc_reb[i,j,k] = xinc
                final_yinc_reb[i,j,k] = yinc
                final_xout_reb[i,j,k] = xout
                final_yout_reb[i,j,k] = yout
                final_prices[i,j] = current_price
            
    results = (final_x_dis, final_y_dis,
               final_xinc_reb, final_yinc_reb,
               final_xout_reb, final_yout_reb,
               fees_inc_dis, fees_out_dis,
               final_prices)
            
    return results



                    
                    
                    