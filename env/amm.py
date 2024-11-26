import numpy as np
import random

class AMM:
    def __init__(self, initial_ls=1000000, initial_lr=1000000, fee_rate=0.003, fee_distribute=True, fee_source=1):
        self.initial_ls = initial_ls
        self.initial_lr = initial_lr
        self.ls = initial_ls
        self.lr = initial_lr
        self.k = self.ls * self.lr
        self.f = fee_rate 
        self.fee_distribute = fee_distribute
        self.fee_source = fee_source

    def get_price(self):
        amm_ask = (self.ls / self.lr) / (1-self.f)
        amm_bid = (self.ls / self.lr) * (1-self.f)
        return amm_ask, amm_bid

    def swap(self, xr):
        assert self.f >= 0, "Fee rate must be non-negative"        
        pre_lr = self.lr
        pre_ls = self.ls
        step_fee = {'r' : 0, 's' : 0}
        # Calculate the amount of asset to be swapped
        if self.fee_source>0:
            if xr > 0:
                xs = (self.ls * self.lr) / (self.lr + (1-self.f)*xr) - self.ls # negative and swap out 
                fee = xr * self.f
                step_fee['r'] += fee
                self.fee['r'] += fee
                # add the fee to the fee pool if fee_pool is True
                if not self.fee_distribute:
                    self.lr += fee 
                    fee = 0
                xr *= (1-self.f)
                self.lr += xr
                self.ls += xs
            elif xr < 0:
                xs = (1/(1-self.f)) * ((self.ls * self.lr) / (self.lr + xr) - self.ls)
                fee = xs * self.f
                step_fee['s'] += fee
                self.fee['s'] += fee
                # add the fee to the fee pool if fee_pool is True
                if not self.fee_distribute:
                    self.ls += fee
                    fee = 0
                xs *= (1-self.f)
                self.ls += xs
                self.lr += xr
            else:
                xs = 0
                fee = 0
        else:
            if xr > 0:
                xs = (1-self.f) * ((self.ls * self.lr) / (self.lr + xr) - self.ls)
                fee = -xs * self.f
                step_fee['s'] += fee
                self.fee['s'] += fee
                # add the fee to the fee pool if fee_pool is True
                if not self.fee_distribute:
                    self.lr += fee 
                    fee = 0
                xs *= (1-self.f)
                self.lr += xr
                self.ls += xs
            elif xr < 0:
                xs = (self.ls * self.lr) / (self.lr + (1-self.f)*xr) - self.ls
                fee = -xr * self.f
                step_fee['r'] += fee
                self.fee['r'] += fee
                # add the fee to the fee pool if fee_pool is True
                if not self.fee_distribute:
                    self.lr += fee
                    fee = 0
                xr *= (1-self.f)
                self.ls += xs
                self.lr += xr
            else:
                xs = 0
                fee = 0   
        # else:
        #     if xr > 0:
        #         xs = (1-self.f) * ((self.ls * self.lr) / (self.lr + xr) - self.ls)
        #         fee = -xs * self.f
        #         step_fee['s'] = fee
        #         # add the fee to the fee pool if fee_pool is True
        #         if not self.fee_distribute:
        #             self.lr += fee 
        #             fee = 0
        #         xs *= (1-self.f)
        #         self.lr += xr
        #         self.ls += xs
        #     elif xr < 0:
        #         xs = (self.ls * self.lr) / (self.lr + (1-self.f)*xr) - self.ls
        #         fee = -xr * self.f
        #         step_fee['r'] = fee
        #         # add the fee to the fee pool if fee_pool is True
        #         if not self.fee_distribute:
        #             self.lr += fee
        #             fee = 0
        #         xr *= (1-self.f)
        #         self.ls += xs
        #         self.lr += xr
        #     else:
        #         xs = 0
        #         fee = 0  

        # return the information of the swap
        info = {'xs': xs, 'xr': xr, 'pre_ls': pre_ls, 'pre_lr': pre_lr,
                'ls':self.ls, 'lr':self.lr, 'token_fee': step_fee}
        return info

    def reset(self):
        self.lr = self.initial_lr
        self.ls = self.initial_ls
        self.fee = {'r' : 0, 's' :0}

    
if __name__ == '__main__':
    import os
    import sys
    import numpy as np
    import matplotlib.pyplot as plt

    # Add path for module import
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from env.amm import AMM
    from env.oracle import OracleSimulator
    from env.trader import Arbitrager
    import matplotlib.pyplot as plt
    import numpy as np

    amm = AMM(distribute=True, fee_source=1)
    oracle = OracleSimulator(spread=0, sigma=1)
    trader = Arbitrager(amm, oracle)

    income_fee_value = []
    output_fee_value = []
    initial_token_size = amm.initial_lr
    swap_sizes = np.arange(100, initial_token_size*100000, 100)
    for swap_size in swap_sizes:

        info = amm.swap(swap_size)

        pr = info['ls'] / info['lr']
        ps = info['lr'] / info['ls']

        total_fee_value = info['token_fee']['r'] * pr + info['token_fee']['s'] * ps
        income_fee_value.append(total_fee_value)
        amm.reset()
    
    amm = AMM(distribute=True, fee_source=-1)
    oracle = OracleSimulator(spread=0, sigma=1)
    trader = Arbitrager(amm, oracle)

    for swap_size in swap_sizes:

        info = amm.swap(swap_size)
        pr = info['ls'] / info['lr']
        ps = info['lr'] / info['ls']
        if info['token_fee']['r'] * pr != 0:
            print(f"info['token_fee']['r'] * pr: {info['token_fee']['r'] * pr}")
        total_fee_value = info['token_fee']['r'] * pr + info['token_fee']['s'] * ps
        output_fee_value.append(total_fee_value)
        amm.reset()

    # draw the plot for both income and output fee use left and right y-axis
    # use existed list of swap_sizes and income_fee_value, output_fee_value
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color = 'tab:red'
    ax1.set_xlabel('Swap Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Fee Value (input)', color=color, fontsize=12, fontweight='bold')
    ax1.plot(swap_sizes, income_fee_value, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Total Fee Value (output)', color=color, fontsize=12, fontweight='bold')
    ax2.plot(swap_sizes, output_fee_value, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.title('Total Fee Value vs Swap Size', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.show()
    plt.savefig('total_fee_value_vs_swap_size 10000 times.png')
    
    