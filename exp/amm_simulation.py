from amm import SimpleFeeAMM
from utils import parse_input, set_market_trade
from fee import TriangleFee, PercentFee
from threading import Thread
import numpy as np
import time
import matplotlib.pyplot as plt

import wandb


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
        
        

def run_gbm(initial_price, drift, volatility, time_steps):
    dt = 0.5  # time step
    prices = [initial_price]

    for _ in range(time_steps):
        # Generate a random movement
        random_movement = np.random.normal(0, np.sqrt(dt)) * volatility

        # Update price using Brownian motion formula
        new_price = prices[-1] + prices[-1]*(drift * dt + random_movement)
        yield new_price


    
def main2():
    config_dict = {"fee_structure": "",
              "fee_scheme": "simple",
              "initial_price": 10,
              "drift": 0.1,
              "volatility": 0.1,
              "time_steps": 200,
              "num_agents": 1}
    # 
    use_wandb = False
    

    
    # config = wandb.config
    if config_dict['fee_structure'] == 'percent':
        fee = PercentFee(0.01)
    elif config_dict['fee_structure'] == 'triangle':
        fee = TriangleFee({1: 0.3, 100: 0.05, 500: 0.005, 1000: 0.0005, 10000: 0.00005})   
    else: 
        fee = None  #When it's None it will automatically choose the noFee() function 
    amm = SimpleFeeAMM(fee_structure=fee)
    
    # Initialize Brownian motion parameters
    initial_price = config_dict['initial_price']
    drift = config_dict['drift']
    volatility = config_dict['volatility']
    time_steps = config_dict['time_steps']
    # market_prices = brownian_motion(initial_price, drift, volatility, time_steps)

    market_prices = [float(initial_price)]
    
    if use_wandb:
        wandb.init(config=config_dict, 
            project="amm_simulation")
    
    
    recall = False
    open_AMMPrices = []
    open_MarketPrices = []
    
    def market_price_thread(initial_price, drift, volatility, time_steps, market_prices):

        for price in run_gbm(initial_price, drift, volatility, time_steps):
            market_prices.append(price)
            time.sleep(0.5)
            
    def arbitrage_thread(amm, market_prices):
        while not recall:
            curr_MP = market_prices[-1]
            set_market_trade(amm, curr_MP, "B", "A")
            time.sleep(2)
            
    def observer_thread(amm, market_prices):
        while not recall:
            curr_MP = market_prices[-1]
            # wandb.log({"AMM Price": amm.asset_ratio("B", "A"), "Market Price": curr_MP})
            open_AMMPrices.append(amm.asset_ratio("B", "A"))
            open_MarketPrices.append(curr_MP)
            time.sleep(1)
    
    threads = []
    for _ in range(config_dict['num_agents']):
        threads.append(Thread(target=arbitrage_thread, args=(amm, market_prices)))
        
    threads.append(Thread(target=market_price_thread, args=(initial_price, drift, volatility, time_steps, market_prices)))
    threads.append(Thread(target=observer_thread, args=(amm, market_prices)))

    for t in threads:
        t.start()

    print("Initial AMM:")
    print(amm)
    
    threads[-2].join()
    recall = True
    
    for t in threads:
        t.join(timeout = 0.1)
    

    
    if use_wandb:
        wandb.finish()
    
    Amm_price = open_AMMPrices
    Market_price = open_MarketPrices

    # Example price arrays (replace these with your actual price data)
    price_array_1 = Amm_price
    price_array_2 = Market_price

    # Plotting the price arrays
    plt.plot(price_array_1, label='AMM Prices ')
    plt.plot(price_array_2, label='Market Prices')

    # Adding labels, title, and legend
    plt.xlabel('Time')
    plt.ylabel('Prices')
    plt.title('Arbitrage Plot (With No Fee)')
    plt.legend()

    # Display the plot
    plt.show()


if __name__ == "__main__":
    main2()

    # for p in run_gbm(10, 0.1, 0.1, 10000):
    #     print(p)

