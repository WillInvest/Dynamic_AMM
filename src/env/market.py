import numpy as np

class GBMPriceSimulator:
    def __init__(self, start_price=1, mu=0.1, sigma=0.5, epsilon=0.0005, dt=0.01):
        self.initial_price = start_price  # Store the initial price
        self.initial_mu = mu  # Store the initial drift coefficient
        self.initial_sigma = sigma  # Store the initial volatility coefficient
        self.initial_epsilon = epsilon  # Store the initial proportional difference for bid/ask prices
        self.initial_dt = dt  # Store the initial time step

        self.current_price = start_price  # Current midpoint price
        self.mu = mu  # Drift coefficient
        self.sigma = sigma  # Volatility coefficient
        self.epsilon = epsilon  # Proportional difference for bid/ask prices
        self.dt = dt  # Time step

    def get_bid_price(self):
        """Calculate and return the current bid price."""
        return self.current_price / (1 + self.epsilon)

    def get_ask_price(self):
        """Calculate and return the current ask price."""
        return self.current_price * (1 + self.epsilon)

    def next(self):
        """Advance the price using the GBM formula to the next time step."""
        # Generate a random standard normal value for the shock
        shock = np.random.normal()
        # Update the current price based on the GBM formula
        self.current_price *= np.exp((self.mu - 0.5 * self.sigma**2) * self.dt + self.sigma * np.sqrt(self.dt) * shock)
        
    def reset(self):
        """Reset the simulator to the initial state."""
        self.current_price = self.initial_price
        self.mu = self.initial_mu
        self.sigma = self.initial_sigma
        self.epsilon = self.initial_epsilon
        self.dt = self.initial_dt

    
if __name__ == '__main__':
    # Example of using the class
    simulator = GBMPriceSimulator()
    print("Initial midpoint price:", simulator.current_price)
    print("Initial bid price:", simulator.get_bid_price())
    print("Initial ask price:", simulator.get_ask_price())

    # Simulate a few steps
    for _ in range(5):
        simulator.next()
        print("\nNext step prices:")
        print("Ask price:", simulator.get_ask_price())
        print("Midpoint price:", simulator.current_price)
        print("Bid price:", simulator.get_bid_price())
