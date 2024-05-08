import numpy as np

class GBMPriceSimulator:
    def __init__(self, start_price=1.0, mu=0.1, sigma=0.5, epsilon=0.05, dt=0.01):
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
