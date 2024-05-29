import numpy as np
import matplotlib.pyplot as plt


class GBMPriceSimulator:
    def __init__(self, start_price=100, mu=0.1, sigma=0.5, epsilon=0.01, dt=0.01, deterministic=False, random=False):
        self.initial_price = start_price
        self.initial_mu = mu
        self.initial_sigma = sigma
        self.initial_epsilon = epsilon
        self.initial_dt = dt
        self.AP = start_price
        self.BP = start_price
        self.current_price = start_price
        self.mu = mu
        self.sigma = sigma
        self.epsilon = epsilon
        self.dt = dt
        self.deterministic = deterministic  # Flag to control stochastic/deterministic behavior
        self.shock_index = 0  # Index to track the current shock
        self.pathA = self.get_zigzag(1.5, 0.5, 1.2, 0.8, random=random)
        self.pathB = self.get_zigzag(1.2, 0.8, 1.1, 0.9, random=random)
        
    def generate_gbm_series(self, S0, mu, sigma, T, dt, n_series):
        """
        Generate multiple GBM series.

        Parameters:
        S0 (float): Initial stock price.
        mu (float): Drift coefficient.
        sigma (float): Volatility coefficient.
        T (int): Total time in years.
        dt (float): Time step in years.
        n_series (int): Number of GBM series to generate.

        Returns:
        np.array: A numpy array containing GBM paths.
        """
        n_steps = int(T / dt)  # Number of time steps
        timesteps = np.linspace(0, T, n_steps)
        S = np.zeros((n_steps, n_series))
        S[0] = S0

        for t in range(1, n_steps):
            Z = np.random.normal(0, 1, n_series)
            S[t] = S[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

        return S
        
    def get_zigzag(self, high1, low1, high2, low2, random=False):
        # total steps
        steps = 500

        
        # if random:
        #     # Parameters
        #     S0 = 100         # Initial price
        #     mu = 0.05        # Drift coefficient
        #     sigma = 0.2      # Volatility
        #     T = 1            # Total time in years
        #     dt = 0.002        # Time step in years

        #     # Generate the GBM series
        #     gbm_series = self.generate_gbm_series(S0, mu, sigma, T, dt, 1)
            
        #     return gbm_series
        
        # else:
        # Define the basic zigzag pattern: rise to 1.5, drop to 0.5, return to 1
        rise1_steps = steps // 5
        drop1_steps = steps // 5
        rise2_steps = steps // 5
        drop2_steps = steps // 5
        return_steps = steps - (rise1_steps + rise2_steps + drop1_steps + drop2_steps)
    
        # Create the rise sequence from 1 to 1.5
        rise1_sequence = np.linspace(1, high1, rise1_steps)
    
        # Create the drop sequence from 1.5 to 0.5
        drop1_sequence = np.linspace(high1, low1, drop1_steps)
    
        # Create the return sequence from 0.5 to 1
        rise2_sequence = np.linspace(low1, high2, rise2_steps)
        
        # Create the return sequence from 0.5 to 1
        drop2_sequence = np.linspace(high2, low2, drop2_steps)
        
        # Create the return sequence from 0.5 to 1
        return_sequence = np.linspace(low2, 1, return_steps)
        
        # Concatenate the sequences to form the full zigzag pattern
        return np.concatenate((rise1_sequence, drop1_sequence,
                                rise2_sequence, drop2_sequence,
                                return_sequence))

    def get_bid_price(self, token):
        if token == "A":
            return self.AP / (1 + 2 * self.epsilon)
        elif token == "B":
            return self.BP / (1 + self.epsilon)
        else:
            print("Invalid input")
            

    def get_ask_price(self, token):
        if token == "A":
            return self.AP * (1 + 2 * self.epsilon)
        elif token == "B":
            return self.BP * (1 + self.epsilon)
        else:
            print("Invalid input")

    def next(self):
        if self.deterministic:
            
            # Use a predetermined shock from the list
            self.AP = self.initial_price * self.pathA[self.shock_index%500] 
            self.BP = self.initial_price * self.pathB[self.shock_index%500]
            self.shock_index += 1
        else:
            # Stochastic update, random shock
            shock1 = np.random.normal()
            shock2 = np.random.normal()
            # Update the current price using the GBM formula
            self.AP *= np.exp(
                (self.mu - 0.5 * (self.sigma + 0.1) ** 2) * self.dt + self.sigma * np.sqrt(self.dt) * shock1)
            self.BP *= np.exp(
                (self.mu - 0.5 * (self.sigma - 0.1) ** 2) * self.dt + self.sigma * np.sqrt(self.dt) * shock2)

        

    def reset(self):
        self.AP = self.initial_price
        self.BP = self.initial_price
        self.mu = self.initial_mu
        self.sigma = self.initial_sigma
        self.epsilon = self.initial_epsilon
        self.dt = self.initial_dt
        self.shock_index = 0  # Reset shock index


if __name__ == '__main__':
    # Set the seed for reproducibility
    np.random.seed(123)

    
    market_simulator = GBMPriceSimulator(start_price=100, deterministic=True, random=True)

    askA = []
    askB = []
    bidA = []
    bidB = []
    ask_ratio = []
    bid_ratio = []
    for _ in range(50):
        market_simulator.next()
        askA.append(market_simulator.get_ask_price('A'))
        askB.append(market_simulator.get_ask_price('B'))
        bidA.append(market_simulator.get_bid_price('A'))
        bidB.append(market_simulator.get_bid_price('B'))
        ask_ratio.append(market_simulator.get_ask_price('A') / market_simulator.get_bid_price('B'))
        bid_ratio.append(market_simulator.get_bid_price('A') / market_simulator.get_ask_price('B'))



    # Plotting the prices
    plt.figure(figsize=(10, 10))
    # First subplot: Bid and Ask Prices for Tokens A and B
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
    plt.plot(bidA, label='Market Bid for A')
    plt.plot(askA, label='Market Ask for A')
    plt.plot(bidB, label='Market Bid for B')
    plt.plot(askB, label='Market Ask for B')
    plt.title('Bid and Ask Prices for Tokens A and B')
    plt.xlabel('Step')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # Second subplot: Bid and Ask Ratios
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
    plt.plot(bid_ratio, label='Bid Ratio')
    plt.plot(ask_ratio, label='Ask Ratio')
    plt.title('Bid and Ask Ratios')
    plt.xlabel('Step')
    plt.ylabel('Ratio')
    plt.legend()
    plt.grid(True)

    # Display the plots
    plt.tight_layout()
    plt.show()