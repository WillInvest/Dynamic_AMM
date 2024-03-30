from amm.amm import AMM, SimpleFeeAMM
from amm.utils import parse_input
from amm.fee import TriangleFee, PercentFee, NoFee
from random import random
import numpy as np


def experiment1():
    print("here")

# def main():
#     experiment1()






# def experiment1():
#     fee1 = PercentFee(0.01)
#     # fee2 = TriangleFee({1: 0.3, 100: 0.05, 500: 0.005, 1000: 0.0005, 10000: 0.00005})
#     fee2 = TriangleFee(0.2, -1)
#     fee3  =TriangleFee()
#     # amm = AMM()
#     amm = SimpleFeeAMM(fee_structure=fee1)

#     print("Initial AMM: ")
#     print(amm)
#     # while True:
#     for i in range(100000):
#         print()
#         s2string = "B A 10"
#         if s2string == 'r':
#             amm = AMM()
#             print("Reset amm")
#             print(amm)
#             continue  # reset
#         order = parse_input(s2string)
#         s1, s2, s2_in = order
#         # print(order)
#         # amm.track_asset_ratio('A','B')
#         succ, info = amm.trade_swap(s1, s2, s2_in)
#         if succ:
#             print(f"User pay {s1}: {info['pay_s1']}")
#         else:
#             print(f"unsuccessful trade: {info}")

#         # Function call for resetting ratio to market value
#         # Parameter 1: Market Value
#         # Parameter 2: Numerator for asset ratio eg: (B(parameter 2)/A(parameter 3))
#         # amm.set_market_trade(10,'B','A')

#         # function_to_solve = amm.helper_gen(s1, s2, s2_in)

#         # s1_in, info = find_root_bisection(
#         #     function_to_solve, left_bound=-amm.portfolio[s1] + 1)
#         # print('solver info')
#         # print(info)
#         # print(f"s1_in: {s1_in}")
#         # updates = {s1: s1_in, s2: s2_in}
#         # amm.update_portfolio(delta_assets=updates, asset_in=s2, fee='triangle')
#         print("Updated portfolio:")
#         print(amm)
#         print()
# #     def update_portfolio(self, *, delta_assets: dict = {}, check=True, asset_in: str = None, fee=None):

# def experiment2():
#     amm0 = SimpleFeeAMM(fee_structure=NoFee())
#     amm1 = SimpleFeeAMM(fee_structure=PercentFee(0.01))
#     amm2 = SimpleFeeAMM(fee_structure=TriangleFee(0.2, -1))

#     print("Initial AMM: ")
#     print(amm0)



#     # while True:
#     for i in range(3):
#         print()

#         print("DF")
#         print(type(amm0.data))
#         print(amm1.data)
        
#         amm1.data.append(list(amm1.portfolio + amm1.fees)) #, ignore_index=True)

#         # for tokens in amm1.portfolio:
#         #     amm1.data[i, tokens] = amm1.portfolio[tokens]
#         # for fees in amm1.fees:
#         #     amm1.data[i, fees] = amm1.fees[fees]

#         print(i)
        
#         # generate random number [0.0, 1,0)
#         # to find the direction of the trade (A -> B : 75% or B -> A : 25%)
#         direction_probability = random()
#         if direction_probability < 1/4: s1, s2 = "A", "B"
#         else: s1, s2 = "B", "A"

#         # generate poison number for the amount of asset to trade
#         trade_value = np.random.poisson(5)
#         s2_in = trade_value
        
#         print("Pre AMM: ")
#         print(amm1.portfolio, amm1.fees)
    
#         print(s1, s2, trade_value)
#         print('-' * 20 + '\n')

#         succ1, info1 = amm1.trade_swap(s1, s2, s2_in)
#         print("Post AMM: ")
#         print(amm1.portfolio)
#         print(info1)
#         print('-' * 20 + '\n')

#         if succ1:
#             print(f"User Receives {s1}: {abs(info1['asset_delta'][s1])}")
#         else:
#             print(f"unsuccessful trade: {info1}")

#         # succ1, info1 = amm1.trade_swap(s1, s2, s2_in)
#         # if succ1:
#         #     print(f"User Receives {s1}: {['asset_delta'][s1]}")
#         # else:
#         #     print(f"unsuccessful trade: {info1}")

#         # succ2, info2 = amm2.trade_swap(s1, s2, s2_in)
#         # if succ0:
#         #     print(f"User Receives {s1}: {['asset_delta'][s1]}")
#         # else:
#         #     print(f"unsuccessful trade: {info2}")


#         # Function call for resetting ratio to market value
#         # Parameter 1: Market Value
#         # Parameter 2: Numerator for asset ratio eg: (B(parameter 2)/A(parameter 3))
#         # amm.set_market_trade(10,'B','A')

#         # function_to_solve = amm.helper_gen(s1, s2, s2_in)

#         # s1_in, info = find_root_bisection(
#         #     function_to_solve, left_bound=-amm.portfolio[s1] + 1)
#         # print('solver info')
#         # print(info)
#         # print(f"s1_in: {s1_in}")
#         # updates = {s1: s1_in, s2: s2_in}
#         # amm.update_portfolio(delta_assets=updates, asset_in=s2, fee='triangle')
            

#     # print("Updated portfolios (none, %, tri):")
#     # print(amm0, amm1, amm2)
#     print()




# def main():
#     # experiment1()
#     experiment2()



# if __name__ == "__main__":
#     # amm = AMM()
#     # print(amm)
#     # print(amm.target_function())
#     # amm.update_portfolio(delta_assets={'A': -1, 'B': 0.1})
#     # print(amm)
#     # print(amm.target_function())
#     main()
