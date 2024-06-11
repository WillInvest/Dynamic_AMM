import numpy as np

class AMM:
    def __init__(self, initial_a=10000, initial_b=10000, fee=0.06):
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
            new_reserve_b = self.k / new_reserve_a
            deposit = new_reserve_b - self.reserve_b
            fee = (deposit / (1-self.fee)) * self.fee
            asset_delta['A'] = new_reserve_a - self.reserve_a
            asset_delta['B'] = new_reserve_b - self.reserve_b
            self.reserve_b = new_reserve_b
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
            new_reserve_a = self.k / new_reserve_b
            deposit = new_reserve_a - self.reserve_a
            fee = (deposit / (1-self.fee)) * self.fee
            asset_delta['A'] = new_reserve_a - self.reserve_a
            asset_delta['B'] = new_reserve_b - self.reserve_b
            self.reserve_a = new_reserve_a
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
        ret += f"product: {self.reserve_a * self.reserve_b}\n"
        ret += '-' * 20 + '\n'
        return ret
    
if __name__ == '__main__':
    
    amount = np.random.rand(10000) * 0.2 - 0.1
    print(amount)
    initial_a = 10000
    initial_b = 9000
    amm = AMM(initial_a, initial_b, fee=0.2)
    print(amm)
    amm.swap(-0.1)
    print(amm)
    amm.swap(0.8)
    print(amm)
    
    
    
    
    