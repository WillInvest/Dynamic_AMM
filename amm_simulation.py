from amm import SimpleFeeAMM
from utils import parse_input, set_market_trade
from fee import TriangleFee, PercentFee
from threading import Thread
import numpy as np

# Function for brownian motion
def brownian_motion(initial_price, drift, volatility, time_steps):
    dt = 1  # time step
    prices = [initial_price]

    for _ in range(time_steps):
        # Generate a random movement
        random_movement = np.random.normal(0, np.sqrt(dt)) * volatility

        # Update price using Brownian motion formula
        new_price = prices[-1] + drift * dt + random_movement
        prices.append(new_price)

    return prices

# Function to execute in the setup thread
def set_trade_after_execution(amm_instance, MP, inv1, inv2):
    # Define the function to run after trade execution
    set_market_trade(amm_instance, MP, inv1, inv2)

def main():
    fee1 = PercentFee(0.01)
    fee2 = TriangleFee({1: 0.3, 100: 0.05, 500: 0.005, 1000: 0.0005, 10000: 0.00005})
    amm = SimpleFeeAMM(fee_structure=fee2)
    
    # Initialize Brownian motion parameters
    initial_price = 10
    drift = 0.1
    volatility = 0.2
    time_steps = 100
    market_prices = brownian_motion(initial_price, drift, volatility, time_steps)

    print("Initial AMM:")
    print(amm)

    for i in range(len(market_prices)):
        s2string = input("Input string (i.e. A B 1): ")
        if s2string == 'r':
            amm = SimpleFeeAMM(fee_structure=fee2)
            print("Reset amm")
            print(amm)
            continue  # reset

        order = parse_input(s2string)
        s1, s2, s2_in = order
        print(order)

        success, trade_info = amm.trade_swap(s1, s2, s2_in)
        if success:
            # Start a thread to execute set_trade_after_execution function
            print(market_prices[i])
            t = Thread(target=set_trade_after_execution, args=(amm, market_prices[i], 'B', 'A'))
            t.start()

        print("Updated portfolio:")
        print(amm)

if __name__ == "__main__":
    main()
