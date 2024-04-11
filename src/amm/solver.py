# importing src directory
import sys
sys.path.append('..')
# library imports
import numpy as np

def find_root_bisection(func, *, 
                        tolerance=1e-20, 
                        max_iterations=10000, 
                        left_bound=-np.inf, 
                        right_bound=np.inf):
    # Find the initial range where the root lies
    a = -1
    b = 1

    while True:
        a = max(2 * a, left_bound) if np.isinf(left_bound) else left_bound
        b = min(2 * b, right_bound) if np.isinf(right_bound) else right_bound
        if func(a) * func(b) <= 0: break
    init_a, init_b = a, b
    # Perform bisection method
    iterations = 0
    while (b - a) / 2 > tolerance and iterations < max_iterations:
        c = (a + b) / 2
        if func(c) == 0:
            break
        if func(c) * func(a) < 0:
            b = c
        else:
            a = c
        iterations += 1
    if iterations >= max_iterations:
        print("MAX Iteraion reached.")
    root = (a + b) / 2
    return root, {'final_interval': (a, b),
                  'init': (init_a, init_b),
                  "final_func_values": (func(a), func((a + b) / 2), func(b)),
                  'init_func_values': (func(init_a), func(init_b))}
