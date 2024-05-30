import numpy as np

class AMM:
    def __init__(self, initial_a, initial_b, fee=0.02):
        self.reserve_a = initial_a
        self.reserve_b = initial_b
        self.k = self.reserve_a * self.reserve_b
        self.fee = fee
        self.fee_a = 0
        self.fee_b = 0
        self.initial_a = initial_a
        self.initial_b = initial_b

    def get_price(self):
        return self.reserve_b / self.reserve_a

    def swap(self, amount):
        asset_delta = {'A': 0, 'B': 0}
        fees = {'A': 0, 'B': 0}
        if amount >= 0:
            # Deposit amount of B and get A
            if amount < 1:
                new_reserve_a = self.reserve_a * (1-amount)
            else:
                new_reserve_a = 1
                # print("error_info, resulting negative inventory. Reset inventory to minimum of 1")
            new_reserve_b = self.k / new_reserve_a
            deposit = new_reserve_b - self.reserve_b
            fee = deposit * self.fee
            asset_delta['A'] = new_reserve_a - self.reserve_a
            asset_delta['B'] = new_reserve_b - self.reserve_b
            self.reserve_b = new_reserve_b - fee
            self.reserve_a = new_reserve_a
            self.fee_a = 0
            self.fee_b = fee
            fees['A'] = 0
            fees['B'] = fee
        else:
            # Deposit -amount of A and get B
            if abs(amount) < 1:
                new_reserve_b = self.reserve_b * (1+amount)
            else:
                new_reserve_b = 1
                # print("error_info, resulting negative inventory. Reset inventory to minimum of 1")
            new_reserve_a = self.k / new_reserve_b
            deposit = new_reserve_a - self.reserve_a
            fee = deposit * self.fee
            asset_delta['A'] = new_reserve_a - self.reserve_a
            asset_delta['B'] = new_reserve_b - self.reserve_b
            self.reserve_a = new_reserve_a - fee
            self.reserve_b = new_reserve_b
            self.fee_a = fee
            self.fee_b = 0
            fees['A'] = fee
            fees['B'] = 0

        return {'asset_delta': asset_delta, 'fee': fees}

    def get_reserves(self):
        return self.reserve_a, self.reserve_b

    def reset(self):
        self.reserve_a = self.initial_a
        self.reserve_b = self.initial_b
        self.fee_a = 0
        self.fee_b = 0


    def __repr__(self):
        ret = '-' * 20 + '\n'
        ret += f"Reserve A: {self.reserve_a}\n"
        ret += f"Reserve B: {self.reserve_b}\n"
        ret += '-' * 20 + '\n'
        ret += f"FA: {self.fee_a}\n"
        ret += f"FB: {self.fee_b}\n"
        ret += '-' * 20 + '\n'
        return ret
    
if __name__ == '__main__':


    # Iterative adjustment process
    # Constants
    TARGET_RATIO = 0.5  # Desired ratio of Asset A to Asset B
    TOLERANCE = 0.00000001   # Tolerance for the ratio

    iteration_count = 0
    max_iterations = 1000000
    learning_rate = 0.001
    fee_a = 0
    delta_b = 0
    error = 1
    fee_rate = 0.0
    while iteration_count < max_iterations:

        initial_a = 11547.005306911517
        initial_b = 8660.254095505135
        amm = AMM(initial_a, initial_b, fee=fee_rate)  # 5% fee
        # Initial check
        initial_ratio = amm.reserve_b / amm.reserve_a
        print(f"Initial Ratio: {initial_ratio}")
        print(amm)


        if abs(error) < TOLERANCE:
            print("Desired ratio achieved within tolerance.")
            break

        # Adjust deposit of A based on the error
        # If the current ratio is less than the target, increase A
        # Scale adjustment by the magnitude of the error (this is a simple proportional control)

        # Perform swap
        trading_info = amm.swap(delta_b)
        current_ratio = amm.reserve_b / amm.reserve_a
        error = TARGET_RATIO - current_ratio
        delta_b += error * learning_rate

        asset_delta = trading_info['asset_delta']
        fee = trading_info['fee']
        fee_a += fee['A']
        print(f"Iteration {iteration_count + 1}: Deposit {asset_delta['A']} A, Withdraw {asset_delta['B']} B")
        print(amm)
        print(f"ratio: {amm.reserve_b/amm.reserve_a}")

        iteration_count += 1
    
    print("---------start testing---------")
    

    initial_a = 11547.005306911517
    initial_b = 8660.254095505135
    # initial_a = 10000
    # initial_b = 10000
    amm = AMM(initial_a, initial_b, fee=fee_rate)  
    # Initial check
    initial_ratio = amm.reserve_b / amm.reserve_a
    print(f"Initial Ratio: {initial_ratio}")
    print(amm)
    # Reserve A: 13906.415875594466
    # Reserve B: 6954.5976057473
    rate = (initial_b - 7071.067882497644) / initial_b
    trading_info = amm.swap(-rate)
    print(amm)
    print(f"ratio: {amm.reserve_b / amm.reserve_a}")
    asset_delta = trading_info['asset_delta']
    fee = trading_info['fee']
    amm_order_cost = asset_delta['B'] + fee['B']
    market_order_gain = (asset_delta['A'] + fee['A']) * (TARGET_RATIO)
    rew = - (market_order_gain + amm_order_cost)
    print(f"asset_delta: {asset_delta}")
    print(f"fee: {fee}")
    print(f"reward: {rew}")

