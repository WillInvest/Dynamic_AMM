# importing src directory
import sys
sys.path.append('')
# experiment imports
# from src.amm import amm

from src.amm.amm import AMM
from src.amm.amm import SimpleFeeAMM
from src.amm.utils import parse_input, set_market_trade
from src.amm.fee import TriangleFee, PercentFee
from threading import Thread
# Function to execute in the setup thread


def set_trade_after_execution(amm_instance, MP, inv1, inv2):
    # Define the function to run after trade execution
    set_market_trade(amm_instance, MP, inv1, inv2)


# class MarketGBM:
#     def get_best_bid()
#
#     def get_best_ask()
#
#     def next()
#
# market = MarketGBM()
#
# class AmmArbitrageEnv:
#     def __init__(self, market, AMM) -> None:
#         self.obs_space:
#         self.act_space: (-1, 1), (0, 1)
#
#
#     def step(self, action: ):
#         reward = self.__get_reward()
#
#         self.AMM.trade_swap()
#         self.market.trade
#
#         obs.append(self.market.get_best_bid())
#
#         self.market.next()
#
#
#         return next_obs, reward, done, trucated, info
#
#     def __get_reward(self):
#         '''
#
#         '''
#         pass
#
#     def reset(self):
#         pass


def main():
    fee1 = PercentFee(0.)
    # fee2 = TriangleFee(0.2, -1)
    # amm = AMM()
    amm = SimpleFeeAMM(fee_structure=fee1)
    print("Initial AMM: ")
    print(SimpleFeeAMM)
    # while True:
    for i in range(100000):
        s2string = input("Input string (i.e. A B 1): ")
        if s2string == 'r':
            amm = SimpleFeeAMM(fee_structure=fee1)
            print("Reset amm")
            print(amm)
            continue  # reset
        order = parse_input(s2string)
        s1, s2, s2_in = order
        print(order)
        # amm.track_asset_ratio('A','B')
        success, trade_info = amm.trade_swap(s1, s2, s2_in)
        print(trade_info)
        # if success:
        #     t = Thread(target=set_trade_after_execution,
        #                args=(amm, 10, 'B', 'A'))
        #     t.start()
        # Function call for resetting ratio to market value
        # Parameter 1: Market Value
        # Parameter 2: Numerator for asset ratio eg: (B(parameter 2)/A(parameter 3))
        # amm.set_market_trade(10,'B','A')
        # function_to_solve = amm.helper_gen(s1, s2, s2_in)
        # s1_in, info = find_root_bisection(
        #     function_to_solve, left_bound=-amm.portfolio[s1] + 1)
        # print('solver info')
        # print(info)
        # print(f"s1_in: {s1_in}")
        # updates = {s1: s1_in, s2: s2_in}
        # amm.update_portfolio(delta_assets=updates, asset_in=s2, fee='triangle')
        print("Updated portfolio:")
        print(amm)


#     def update_portfolio(self, *, delta_assets: dict = {}, check=True, asset_in: str = None, fee=None):
if __name__ == "__main__":
    # amm = AMM()
    # print(amm)
    # print(amm.target_function())
    # amm.update_portfolio(delta_assets={'A': -1, 'B': 0.1})
    # print(amm)
    # print(amm.target_function())
    main()


    '''
    No fee
    in your AMM pool, asset X, and asset Y, pa > pb in numerireur X/Y, in outside market (no market impact), mpa > mpb (GBM x_t*(1+epsilon), x_t/(1+epsilon))
    arbitrage opportunity yields, when mpb > pa, or mpa < pb
    
    With fee:
    
    
    
    
    '''