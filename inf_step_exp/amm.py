import numpy as np

class AMM:
    def __init__(self, x=1e6, y=1e6, f=0.003, fee_source='incoming'):
        """
        Automated Market Maker (AMM) with constant product formula and configurable fee structure.
   
        Parameters:
            x (int): Initial risky token liquidity
            y (int): Initial stable token liquidity
            f (float): Trading fee percentage (e.g., 0.003 for 0.3%)
            fee_source (str): 'incoming' or 'outgoing'
   
        Methods:
            reset(): Resets AMM state to initial values
            get_price(): Returns AMM ask/bid prices 
            swap(xr): Executes token swap based on xr amount
                - xr > 0: Swap risky tokens for stable tokens
                - xr < 0: Swap stable tokens for risky tokens
                Returns dict with swap details including:
                - Amounts swapped (xs, xr)
                - Pre/post liquidity levels
                - Fees collected
           
        Private Methods:
            _handle_positive_xr(): Handles swaps with positive xr (r->s)
            _handle_negative_xr(): Handles swaps with negative xr (s->r)
        """
        self._x = x
        self._y = y
        self._initial_x = x
        self._initial_y = y
        self._prev_x = x
        self._prev_y = y
        self._k = self._x * self._y
        self._p = self._y / self._x
        self._s = self._p
        self.f = f
        self.fee_source = fee_source
        self._accumulative_fee = 0
        self._arbitrage_revenue = 0
        self.reset()
    
    # Add property decorators for rounded values
    @property
    def x(self):
        return round(self._x, 8)

    @property
    def y(self):
        return round(self._y, 8)
    
    @property
    def accumulative_fee(self):
        return round(self._accumulative_fee, 8)
    
    @property
    def arbitrage_revenue(self):
        return round(self._arbitrage_revenue, 8)

    def reset(self, fee_source=None):
        self._x = self._initial_x
        self._y = self._initial_y
        self._p = self._y / self._x
        self._s = self._p
        self._accumulative_fee = 0
        self._arbitrage_revenue = 0
        self.swap_count = 0
        self.swap_size = {'x': 0, 'y': 0}
        self.fee_source = fee_source if fee_source is not None else self.fee_source
        
    def pool_state(self):
        """
        Prints current pool state in a tabulated format showing:
        - Token balances (x, y)
        - Current and previous AMM prices (including ask/bid)
        - Market price
        - Fee rate and accumulated fees
        - Arbitrage revenue
        - Last swap details
        """
        # Calculate previous AMM price and ask/bid
        prev_p = self.prev_y / self.prev_x
        prev_ask = prev_p / (1 - self.f)
        prev_bid = prev_p * (1 - self.f)
        
        # Calculate current ask/bid
        current_ask = self._p / (1 - self.f)
        current_bid = self._p * (1 - self.f)
        
        print(f"\n=== AMM Pool State (Swap {self.swap_count}) ===")
        print(f"{'Metric':<20} {'Value':<15} {'Previous':<15}")
        print("-" * 50)
        print(f"{'Risky tokens (x)':<20} {self._x:<15.4f} {self._prev_x:<15.4f}")
        print(f"{'Stable tokens (y)':<20} {self._y:<15.4f} {self._prev_y:<15.4f}")
        
        print("\n=== Price Info ===")
        print(f"{'AMM price':<20} {self._p:<15.4f} {prev_p:<15.4f}")
        print(f"{'AMM ask':<20} {current_ask:<15.4f} {prev_ask:<15.4f}")
        print(f"{'AMM bid':<20} {current_bid:<15.4f} {prev_bid:<15.4f}")
        print(f"{'Market price':<20} {self._s:<15.4f}")
        print(f"{'Fee rate':<20} {self.f:<14.4f}")
        
        print("\n=== Last Swap ===")
        print(f"{'Δx (risky)':<20} {self.swap_size['x']:<15.4f}")
        print(f"{'Δy (stable)':<20} {self.swap_size['y']:<15.4f}")
        
        print("\n=== Trading Stats ===")
        print(f"{'Cumulative fees':<20} {self._accumulative_fee:<15.4f}")
        print(f"{'Arbitrage revenue':<20} {self._arbitrage_revenue:<15.4f}")
        print("-" * 50)

    def _handle_dn_movement(self, delta_x):
        """
        Handles downward price movement (when market price < AMM price).
        Trader provides risky tokens (x) to receive stable tokens (y).
        
        Args:
            delta_x: Amount of risky tokens being added to pool
        """
        self.swap_count += 1
        if self.fee_source == 'incoming':
            delta_y = self._y - self._k / (self._x + (1-self.f)*delta_x)
            self._x += (1-self.f) * delta_x
            self._y -= delta_y
            self._accumulative_fee += delta_y * self.f
            self._arbitrage_revenue += delta_y - delta_x * self._s
        else:
            delta_y = self._y - self._k / (self._x + delta_x) 
            self._x += delta_x
            self._y -= delta_y
            self._accumulative_fee += delta_y * self.f
            self._arbitrage_revenue += (1-self.f) * delta_y - delta_x * self._s
        self.swap_size['x'] = delta_x
        self.swap_size['y'] = -delta_y

    def _handle_up_movement(self, delta_x):
        """
        Handles upward price movement (when market price > AMM price).
        Trader provides stable tokens (y) to receive risky tokens (x).
        
        Args:
            delta_x: Amount of risky tokens being removed from pool
        """
        self.swap_count += 1
        if self.fee_source == 'incoming':
            delta_y = self._k/((1-self.f) * (self._x - delta_x)) - self._y/(1-self.f)
            self._x -= delta_x
            self._y += (1-self.f) * delta_y
            self._accumulative_fee += delta_y * self.f
            self._arbitrage_revenue += delta_x * self._s - delta_y
        else:
            delta_y = self._k/(self._x - delta_x) - self._y 
            self._x -= delta_x
            self._y += delta_y
            self._accumulative_fee += delta_x * self.f
            self._arbitrage_revenue += (1-self.f) * delta_x * self.s - delta_y
        self.swap_size['x'] = -delta_x
        self.swap_size['y'] = delta_y

    def swap(self, S_t):
        """
        Executes a swap based on the current market price S_t.
        
        Args:
            S_t: Current market price
            
        The swap direction is determined by comparing S_t to AMM price:
        - If S_t > AMM ask price: Traders want to buy stable tokens
        - If S_t < AMM bid price: Traders want to buy risky tokens
        - If price is between bid/ask: No arbitrage opportunity
        """
        assert S_t > 0, "Market price must be positive"
        self._s = S_t
        self._prev_x = self._x
        self._prev_y = self._y
        # Check if price is above ask - traders sell risky token to AMM
        if self._s > self._p/(1-self.f):
            delta_x = self._x - np.sqrt(self._k / (self._s * (1-self.f)))
            self._handle_up_movement(delta_x)
        # Check if price is below bid - traders buy risky token from AMM    
        elif self._s < self._p * (1-self.f):
            if self.fee_source == 'incoming':
                delta_x = (np.sqrt((1-self.f) * self._k / S_t) - self._x) / (1-self.f)
            else:
                delta_x = np.sqrt((1-self.f) * self._k / S_t) - self._x
            self._handle_dn_movement(delta_x)
        else:
            self.swap_size['x'] = 0
            self.swap_size['y'] = 0

        # Update pool state
        self._k = self._x * self._y
        self._p = self._y / self._x
        
        # check constant product invariant
        assert abs(self._k - self._initial_x * self._initial_y) < 1e-6, "Pool invariant violated"
