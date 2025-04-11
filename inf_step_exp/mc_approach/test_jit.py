import numpy as np
from numba import jit
import time

# Outside class implementation - pure function
@jit(nopython=True)
def compute_array(x, y):
    result = 0.0
    for i in range(len(x)):
        for j in range(len(y)):
            result += np.sqrt(x[i] * y[j])
    return result

class FastClass:
    def __init__(self):
        self.data = np.array([1.0, 2.0, 3.0])
    
    def compute(self, x, y):
        """Wrapper method that calls the JIT-compiled function"""
        return compute_array(x, y)

class SlowClass:
    def __init__(self):
        self.data = np.array([1.0, 2.0, 3.0])
    
    def compute(self, x, y):
        """Same computation but without JIT"""
        result = 0.0
        for i in range(len(x)):
            for j in range(len(y)):
                result += np.sqrt(x[i] * y[j])
        return result

def test_performance():
    # Test data
    x = np.random.random(1000)
    y = np.random.random(1000)
    
    # Warm up JIT compilation (first call is slower due to compilation)
    _ = compute_array(x[:2], y[:2])
    
    # Test JIT function through class wrapper
    fast_obj = FastClass()
    start = time.time()
    result1 = fast_obj.compute(x, y)
    fast_time = time.time() - start
    print(f"Fast (JIT) time: {fast_time:.4f} seconds")
    
    # Test non-JIT method
    slow_obj = SlowClass()
    start = time.time()
    result2 = slow_obj.compute(x, y)
    slow_time = time.time() - start
    print(f"Slow (no JIT) time: {slow_time:.4f} seconds")
    
    print(f"Results match: {abs(result1 - result2) < 1e-10}")
    print(f"Speedup: {slow_time/fast_time:.1f}x")

if __name__ == "__main__":
    test_performance() 