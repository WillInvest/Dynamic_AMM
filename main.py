from AMM import AMM
from utils import parse_input
from fee import TriangleFee, PercentFee

def main():
    
    fee1 = PercentFee(0.01)
    fee2 = TriangleFee({1: 0.3, 100: 0.05, 500: 0.005, 1000: 0.0005, 10000: 0.00005})
    
    # amm = AMM()
    amm = AMM(fee_structure=fee2)

    print("Initial AMM: ")
    print(amm)
    # while True:
    for i in range(100000):
        s2string = input("Input string (i.e. A B 1): ")
        if s2string == 'r':
            amm = AMM()
            print("Reset amm")
            print(amm)
            continue  # reset
        order = parse_input(s2string)
        s1, s2, s2_in = order
        print(order)
        ##amm.track_asset_ratio('A','B')
        amm.trade(s1, s2, s2_in)

        #Function call for resetting ratio to market value
        #Parameter 1: Market Value
        #Parameter 2: Numerator for asset ratio eg: (B(parameter 2)/A(parameter 3))
        amm.set_market_trade(10,'B','A')
        
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