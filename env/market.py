import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class MarketSimulator:
    def __init__(self,
                 start_price=100,
                 mu=0.1,
                 sigma=0.2,
                 epsilon=0.01,
                 dt=0.01,
                 deterministic=False,
                 steps=500,
                 seed=0):
        
        # self.max_seed = 30
        self.seed = seed 
        self.rng = np.random.default_rng(self.seed)
        self.initial_price = start_price
        self.initial_mu = mu
        self.initial_sigma = sigma
        self.initial_epsilon = epsilon
        self.initial_dt = dt
        self.AP = start_price
        self.BP = start_price
        self.current_price = start_price
        self.mu = mu
        self.sigma = sigma if sigma is not None else self.get_random_sigma()
        self.epsilon = epsilon
        self.sigmaA = self.sigma
        self.sigmaB = self.sigmaA/2
        self.dt = dt
        self.deterministic = deterministic  # Flag to control stochastic/deterministic behavior
        self.index = 0  # Index to track the current shock
        self.steps = steps
        self.pathA = self.get_zigzag(steps=self.steps, high=1.5, low=0.5)
        self.pathB = self.get_zigzag(steps=self.steps, high=1.2, low=0.8)

    def get_random_sigma(self):
        return self.rng.uniform(0.01, 1.0)
    
    def get_zigzag(self, steps, high, low):
        
        # Define the basic zigzag pattern: rise to 1.5, drop to 0.5, return to 1
        rise_steps = steps // 3
        drop_steps = steps // 3
        return_steps = steps - (rise_steps + drop_steps)
        
        # Create the rise sequence from 1 to 1.5
        rise_sequence = np.linspace(1, high, rise_steps)
        
        # Create the drop sequence from 1.5 to 0.5
        drop_sequence = np.linspace(high, low, drop_steps)
        
        # Create the return sequence from 0.5 to 1
        return_sequence = np.linspace(low, 1, return_steps)
        
        # Concatenate the sequences to form the full zigzag pattern
        return np.concatenate((rise_sequence, drop_sequence, return_sequence))

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
        
        # self.sigma = self.get_random_sigma()
        # self.sigmaA = self.sigma
        # self.sigmaB = self.sigmaA/2
        
        if self.deterministic:
            
            # Use a predetermined shock from the list
            self.AP = self.initial_price * self.pathA[self.index%self.steps] 
            self.BP = self.initial_price * self.pathB[self.index%self.steps]
            self.index += 1
        else:
            # Stochastic update, random shock
            shock1 = self.rng.normal()
            shock2 = self.rng.normal()

            # Update the current price using the GBM formula
            self.AP *= np.exp(
                (self.mu - 0.5 * self.sigmaA ** 2) * self.dt + self.sigmaA * np.sqrt(self.dt) * shock1)
            self.BP *= np.exp(
                (self.mu - 0.5 * self.sigmaB ** 2) * self.dt + self.sigmaB * np.sqrt(self.dt) * shock2)
        

    def reset(self):
        self.AP = self.initial_price
        self.BP = self.initial_price
        self.mu = self.initial_mu
        self.sigma = self.initial_sigma
        self.epsilon = self.initial_epsilon
        self.dt = self.initial_dt
        self.index = 0  # Reset shock index
        # self.seed += 1
        # self.rng = np.random.default_rng(self.seed)

if __name__ == '__main__':
    # Set the seed for reproducibility
    # np.random.seed(123)

    market_simulator = MarketSimulator(start_price=1, deterministic=True)
    ask_price = []
    bid_price = []
    for i in range(500):
        ask_a = market_simulator.get_ask_price('A')
        ask_b = market_simulator.get_ask_price('B')
        bid_a = market_simulator.get_bid_price('A')
        bid_b = market_simulator.get_bid_price('B')
        ask_price.append(ask_a / bid_b)
        bid_price.append(bid_a / ask_b)
        market_simulator.next()
        
    plt.figure(figsize=(20, 10))
    plt.plot(np.arange(500), ask_price, label='Market_ask_price')
    plt.plot(np.arange(500), bid_price, label='Market_bid_price')
    plt.title("market price simulation from GBM with 500 steps")
    plt.xlabel("steps")
    plt.ylabel("price ratio: (USD/A) / (USD/B)")
    plt.grid(True)
    plt.legend()
    plt.savefig('market_simulation_500_steps.png')
    
        