import numpy as np
from solver import find_root_bisection

from fee import NoFee
from utils import add_dict


# Define class AMM
class AMM:

    # Initalize default values of A and B in portfolio
    default_init_portfolio = {'A': 1000.0, 'B': 10000.0, "L": None}
    fee_init_portfolio = {'A': 0.0, 'B': 0.0, "L": 0.0}

    # Create a new AMM object
    # set the initial portfolio when creating the object
    def __init__(self, *, 
                 initial_portfolio: dict = None, 
                 initial_fee_portfolio: dict = None, 
                 ratio_denomination: str = "None",
                 fee_structure = None,
                 solver = 'bisec') -> None:
        
        # Set solver
        if solver == 'bisec':
            self.solver = find_root_bisection
        else:
            raise RuntimeError(f"No solver found for {solver}.")
        
        # Set fee structure
        if fee_structure is not None:
            self.fee_structure = fee_structure
        else:
            self.fee_structure = NoFee()
        
        
        # Set the portfolio if initial portfolio is given
        if initial_portfolio:
            self.portfolio = initial_portfolio
        # Set the portfolio to default portfolio otherwise
        else:
            self.portfolio = self.default_init_portfolio.copy()

        # Set the fee portfolio if fee portfolio is given
        if initial_fee_portfolio:
            self.fees = initial_fee_portfolio
        # Set the fee portfolio to default fee portfolio otherwise
        else:
            self.fees = self.fee_init_portfolio.copy()

        # Create a list to store asset ratios
        self.AfB = []
        self.BfA = []

        # Set the ratio denomination if given
        if ratio_denomination:
            self.denomination = ratio_denomination
        else:
            # set to first value in portfolio
            self.denomination = [self.portfolio.keys()][0]

        # Number of assets excluding L, ie A, B -> 2
        self.num_assets = len(self.portfolio) - 1

        # This loops through each asset in the portfolio
        # Sets L = to the Nth root of the product
        p = 1.
        for key in self.portfolio:
            if key != 'L':
                p *= self.portfolio[key]
        self.portfolio['L'] = p ** (1 / self.num_assets)
        
        

    # Create list to print AMM assets
    def __repr__(self) -> str:
        ret = '-' * 20 + '\n'
        # add each asset to the list
        for key in self.portfolio:
            ret += f"{key}: {self.portfolio[key]}\n"
        ret += '-' * 20 + '\n'
        # add each fee to the list
        for fee in self.fees:
            ret += f"F{fee}: {self.fees[fee]}\n"
        ret += '-' * 20 + '\n'
        return ret

    @staticmethod
    # Checks that each value in the portfolio aside from L is not 0
    # Finds value based on log: Can Zhiyuan explain this?
    def _utility_log(portfolio):
        p = 0.
        for key in portfolio:
            if key != 'L':
                assert portfolio[key] > 0, f"invalid value: {key}: {portfolio[key]}"
                p += np.log(portfolio[key])
        return p

    @staticmethod
    def _v_log(l, n):
        # Check both l and n are greater than 0
        assert l > 0 and n > 0, f'invalid value: l = {l}, n = {n}'
        # Return for what??
        return n * np.log(l)

    @staticmethod
    def _utility(portfolio):
        # Return the value of all assets multiplied together
        # Not including L
        p = 1.
        for key in portfolio:
            if key != 'L':
                assert portfolio[key] > 0, f"invalid value: {key} - {portfolio[key]}"
                p *= portfolio[key]
        return p

    @staticmethod
    def _v(l, n):
        # Check both l and n are greater than 0
        assert l > 0 and n > 0, f'invalid value: l = {l}, n = {n}'
        return l ** n

    def utility(self):
        # Calculate portfolio using log for easier calc
        # Convert to normal for readability
        return np.exp(self._utility_log(self.portfolio))

    def track_asset_ratio(self, asset_to_track, reference_asset):
        self.AfB.append(
            self.portfolio[asset_to_track] / self.portfolio[reference_asset])
        self.BfA.append(
            self.portfolio[reference_asset] / self.portfolio[asset_to_track])

# new_amount = self.percent_fee(
#                         delta_assets, self.fees, asset_in, 0.01)
#                     self.portfolio[asset_in] += new_amount
#                 elif fee == 'triangle':
#                     new_amount = self.triangle_fee(delta_assets, self.fees, asset_in, {
#                         1: 0.3, 100: 0.05, 500: 0.005, 1000: 0.0005, 10000: 0.00005})

    # def percent_fee(self, delta_assets: dict, asset: str, fee: float):  # fee_assets: dict,
    #     if asset != "L":
    #         # Apply fixed percetn fee for buying asset
    #         fee_delta = delta_assets[asset] * fee
    #         # Charge fee based on size of order
    #         delta_assets[asset] -= fee_delta
    #         # Update fee assets
    #         self.fees[asset] += fee_delta
    #     return delta_assets[asset]

    # def triangle_fee(self, delta_assets: dict, asset: str, bracket_fees: dict):  # fee_assets: dict,
    #     if asset != "L":
    #         # These values can be changed, simply here to test if it works
    #         # Also implementation can be easily changed, again just testing
    #         # bracket_fees = {0.1: (100, 0.01), 0.2: (200, 0.02), 0.3: (300, 0.03)}
    #         # keep track of change coming in
    #         tracker = float(delta_assets[asset])
    #         try:
    #             while tracker > 0.0:
    #                 for amount, fee in bracket_fees.items():
    #                     fee_delta = min(amount, tracker) * fee
    #                     # Charge fee based on size/remaining size of order
    #                     delta_assets[asset] -= fee_delta
    #                     # Update fee assets
    #                     # print("HERE1:",
    #                     #       fee_delta, fee_assets[asset])
    #                     self.fees[asset] += fee_delta
    #                     # print("HERE2", fee_assets[asset])
    #                     tracker -= amount  # Reduce delta remaining by how much we have already assessed fees
    #                     if tracker < 1e-10:  # You can adjust the epsilon value as needed
    #                         break
    #         # catch any errors - added bcs not sure if theres edge case whole in the above logic
    #         except Exception as e:
    #             print(f"Assertion Error: {e}")
    #     # return updated changes dicitonary
    #     return delta_assets[asset]

    def target_function(self, *, delta_assets: dict = {}):
        tmp_portfolio = self.portfolio.copy()
        # Check for change in asset
        for asset in tmp_portfolio:
            tmp_portfolio[asset] += delta_assets.get(asset, 0.)
        # regular calculation
        # target = self._utility(tmp_portfolio)/self._v(tmp_portfolio['L'], self.num_assets);
        # print(target, self._utility(tmp_portfolio), self._v(tmp_portfolio['L'], self.num_assets))
        # return target-1.

        # log calculation
        target = self._utility_log(
            tmp_portfolio) - self._v_log(tmp_portfolio['L'], self.num_assets)
        # print('target', target, self._utility_log(tmp_portfolio), self._v_log(tmp_portfolio['L'], self.num_assets))
        return np.exp(target) - 1.

    def get_cummulative_fees(self):
        total_fees = 0
        for key, value in self.fees.items():
            if key == 'L':
                pass
            elif key == self.denomination:
                total_fees += value
            else:
                total_fees += (self.fees[key] /
                               self.fees[self.denomination])
        return total_fees

    def update_portfolio(self, *, delta_assets:dict={}, check = True):
        '''
        Manually update portfolio, this may lead to a unbalanced portfolio. 
        '''
        if check:
            target = self.target_function(delta_assets=delta_assets)
            assert abs(target) < 1e-8, f"Target: {target}"
        try:
            for k in delta_assets: 
                assert k in self.portfolio, k
                temp_result = self.portfolio[k] + delta_assets[k]
                if temp_result <= 0.: return False, {"error_info": f"Value: resulting non-positive amount of {k}."}
                self.portfolio[k] = temp_result

        except AssertionError:
            return False, {"error_info": f"AssertionError: {k} not in {delta_assets}."}
        return True, {}

    # def update_portfolio(self, *, delta_assets: dict = {}, check=True, asset_in: str = None, fee: str = None):
    #     '''
    #     Manually update portfolio, this may lead to a unbalanced portfolio.
    #     '''
    #     if check:
    #         target = self.target_function(delta_assets=delta_assets)
    #         # Ensure the target is greater than 0
    #         assert abs(target) < 1e-8, f"Target: {target}"
    #         # Update portfolio with changes in assets
    #         if fee:
    #             if fee == 'percent':
    #                 # provides negative value
    #                 new_amount = self.percent_fee(
    #                     delta_assets, asset_in, 0.01)
    #                 self.portfolio[asset_in] += new_amount
    #             elif fee == 'triangle':
    #                 new_amount = self.triangle_fee(delta_assets, asset_in, )
    #                 self.portfolio[asset_in] += new_amount

    #         # # update portfolio based on which fee structure we apply - needs updating to accomodate what fees is being applied
    #         # self.portfolio[k] += delta_assets[k]

    def helper_gen(self, s1, s2, s2_in):
        # Calculates target value of changed value
        assert s1 in self.portfolio and s2 in self.portfolio, f"{s1} or {s2} not in portfolio"
        assert self.portfolio[s2] + s2_in > 0., f"No enough assets/tokens"
        assert s1 != s2, f"same s1 and s2 input"

        def func(x: float):
            # if  s2 == 'L': # i.e. A L 1 -> L increases 1 and A increases x
            #     delta_assets = {s1: x}
            #     delta_L = s2_in
            # elif s1 == 'L':# i.e. L A 1 -> A increases 1 and L increases x
            #     delta_assets = {s2: s2_in}
            #     delta_L = x
            # else:# i.e. A B 1 -> B increases 1 and A *increases* x (x is negative)
            #     delta_assets = {s1: x, s2: s2_in}
            #     delta_L = 0.
            delta_assets = {s1: x, s2: s2_in}
            return self.target_function(delta_assets=delta_assets)

        return func
    
    def quote(self, s1, s2, s2_in):

        function_to_solve = self.helper_gen(s1, s2, s2_in)

        s1_in, _ = self.solver(
            function_to_solve, left_bound=-self.portfolio[s1] + 1)
        
        fee_dict = self.fee_structure.calculate_fee({s1: s1_in, s2: s2_in}, s2)
        info = {'asset_delta': {s1: s1_in, s2: s2_in}, 'fee': fee_dict}
        return s1_in, info
        
    def trade(self, s1, s2, s2_in):
        s1_in, info = self.quote(s1, s2, s2_in)
        fees = info['fee']
        updates = {s1: s1_in, s2: s2_in}
        success1, update_info1 = self.update_portfolio(delta_assets=updates, check=True)
        success2, update_info2 = self.update_portfolio(delta_assets=fees, check=False)
        
        info['update_info_before_fee'] = update_info1
        info['update_info_after_fee'] = update_info2
        
        for keys in fees:
            self.fees[keys] += fees[keys] # update fee portfolio
            
        if not (success1 or success2):
            print(f"update success before fee: {success1}, after fee: {success2}")
            print(info)
            raise ValueError("illegal trade")
        
        return success1 and success2, info
        
        


# def parse_input(string: str):
#     results = string.split(" ")
#     results[-1] = float(results[-1])
#     return tuple(results)


# def find_root_bisection(func, tolerance=1e-20, max_iterations=10000, left_bound=-np.inf, right_bound=np.inf):
#     # Find the initial range where the root lies
#     a = -1
#     b = 1.

#     while func(a) * func(b) > 0:
#         a = max(2 * a, left_bound)
#         b = min(2 * b, right_bound)
#     init_a, init_b = a, b
#     # Perform bisection method
#     iterations = 0
#     while (b - a) / 2 > tolerance and iterations < max_iterations:
#         c = (a + b) / 2
#         if func(c) == 0:
#             break
#         if func(c) * func(a) < 0:
#             b = c
#         else:
#             a = c
#         iterations += 1
#     if iterations >= max_iterations:
#         print("MAX Iteraion reached.")
#     root = (a + b) / 2
#     return root, {'final_interval': (a, b),
#                   'init': (init_a, init_b),
#                   "final_func_values": (func(a), func((a + b) / 2), func(b)),
#                   'init_func_values': (func(init_a), func(init_b))}



