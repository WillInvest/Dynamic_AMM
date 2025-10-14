import numpy as np
from scipy.stats import lognorm
from scipy import integrate

class TwoStepIntegrate:
    def __init__(self, gamma: float, sigma: float, theta: float, dt: float = 12/(365*24*60*60), incoming_fee: bool = True):
        self.gamma = gamma
        self.sigma = sigma
        self.dt = dt
        self.incoming_fee = incoming_fee
        self.y0 = 1e6
        self.x0 = 1e6
        self.p0 = self.y0 / self.x0
        self.S0 = self.p0 * (1-self.gamma)**theta
        self.L = np.sqrt(self.y0 * self.x0)
        self.eps = 1e-2
        
        # For GBM with zero drift: log(v1) ~ N(-1/2 σ²dt, σ²dt)
        s = self.sigma * np.sqrt(self.dt)  # shape parameter
        scale = np.exp(-0.5 * self.sigma**2 * self.dt)  # scale parameter
        self.dist = lognorm(s=s, scale=scale)
        
    def fee_integrand_u(self, v1: float) -> float:
        """
        Integrand for the fee revenue calculation in the upper region
        including the probability density
        """
        S1 = self.S0 * v1
        delta_y = (self.L * np.sqrt((1-self.gamma) * S1) - self.y0) / (1-self.gamma)
        return self.gamma * delta_y * self.dist.pdf(v1)

    def fee_integrand_d(self, v1: float) -> float:
        """
        Integrand for the fee revenue calculation in the lower region
        including the probability density
        """
        S1 = self.S0 * v1
        delta_x = (self.L * np.sqrt((1-self.gamma) / S1) - self.x0) / (1-self.gamma)
        return self.gamma * delta_x * S1 * self.dist.pdf(v1)

        
    def fee_integrand_uu(self, v2: float, v1: float) -> float:
        """
        Integrand for the fee revenue calculation
        """
        S1 = self.S0 * v1
        S2 = self.S0 * v1 * v2
        x1 = self.L / np.sqrt((1-self.gamma) * S1)
        y1 = self.L * np.sqrt((1-self.gamma) * S1)
        assert abs(x1 * y1 - self.L**2) < self.eps, f"x1 * y1 != L^2: {x1 * y1} != {self.L**2}"
        delta_y = (self.L * np.sqrt((1-self.gamma) * S2) - y1) / (1-self.gamma)
        assert delta_y >= 0, f"delta_x is negative: {delta_y}"
        return self.gamma * delta_y * self.dist.pdf(v1) * self.dist.pdf(v2)
        
    def fee_integrand_ud(self, v2: float, v1: float) -> float:
        """
        Integrand for the fee revenue calculation
        """
        S1 = self.S0 * v1
        S2 = self.S0 * v1 * v2
        x1 = self.L / np.sqrt((1-self.gamma) * S1)
        y1 = self.L * np.sqrt((1-self.gamma) * S1)
        delta_x = (self.L * np.sqrt((1-self.gamma) / S2) - x1) / (1-self.gamma)
        assert delta_x >= 0, f"delta_x is negative: {delta_x}, v1: {v1}, v2: {v2}, gamma: {self.gamma}, x1: {x1}, y1: {y1}, S1: {S1}, S2: {S2}"
        assert abs(x1 * y1 - self.L**2) < self.eps, f"x1 * y1 != L^2: {x1 * y1} != {self.L**2}"
        return self.gamma * delta_x * S2 * self.dist.pdf(v1) * self.dist.pdf(v2)
    
    def fee_integrand_mu(self, v2: float, v1: float) -> float:
        """
        Integrand for the fee revenue calculation
        """
        S1 = self.S0 * v1
        S2 = self.S0 * v1 * v2
        x1 = self.x0
        y1 = self.y0
        delta_y = (self.L * np.sqrt((1-self.gamma) * S2) - y1) / (1-self.gamma)
        assert delta_y >= 0, f"delta_y is negative: {delta_y}"
        assert abs(x1 * y1 - self.L**2) < self.eps, f"x1 * y1 != L^2: {x1 * y1} != {self.L**2}"
        return self.gamma * delta_y * self.dist.pdf(v1) * self.dist.pdf(v2)
    
    def fee_integrand_md(self, v2: float, v1: float) -> float:
        """
        Integrand for the fee revenue calculation
        """
        S1 = self.S0 * v1
        S2 = self.S0 * v1 * v2
        x1 = self.x0
        y1 = self.y0
        delta_x = (self.L * np.sqrt((1-self.gamma) / S2) - x1) / (1-self.gamma)
        assert delta_x >= 0, f"delta_x is negative: {delta_x}"
        assert abs(x1 * y1 - self.L**2) < self.eps, f"x1 * y1 != L^2: {x1 * y1} != {self.L**2}"
        return self.gamma * delta_x * S2 * self.dist.pdf(v1) * self.dist.pdf(v2)
    
    def fee_integrand_du(self, v2: float, v1: float) -> float:
        """
        Integrand for the fee revenue calculation
        """
        S1 = self.S0 * v1
        S2 = self.S0 * v1 * v2
        x1 = self.L * np.sqrt((1-self.gamma) / S1)
        y1 = self.L * np.sqrt(S1 / (1-self.gamma))
        delta_y = (self.L * np.sqrt((1-self.gamma) * S2) - y1) / (1-self.gamma)
        assert delta_y >= 0, f"delta_x is negative: {delta_y}"
        assert abs(x1 * y1 - self.L**2) < self.eps, f"x1 * y1 != L^2: {x1 * y1} != {self.L**2}"
        return self.gamma * delta_y * self.dist.pdf(v1) * self.dist.pdf(v2)
    
    def fee_integrand_dd(self, v2: float, v1: float) -> float:
        """
        Integrand for the fee revenue calculation
        """
        S1 = self.S0 * v1
        S2 = self.S0 * v1 * v2
        x1 = self.L * np.sqrt((1-self.gamma) / S1)
        y1 = self.L * np.sqrt(S1 / (1-self.gamma))
        delta_x = (self.L * np.sqrt((1-self.gamma) / S2) - x1) / (1-self.gamma)
        assert delta_x >= 0, f"delta_x is negative: {delta_x}"
        assert abs(x1 * y1 - self.L**2) < self.eps, f"x1 * y1 != L^2: {x1 * y1} != {self.L**2}"
        return self.gamma * delta_x * S2 * self.dist.pdf(v1) * self.dist.pdf(v2)
    
        
    def pv_integrand_u(self, v1: float) -> float:
        """
        Integrand for the pool value calculation in the upper region
        including the probability density
        """
        S1 = self.S0 * v1
        x1 = self.L * np.sqrt((1-self.gamma) / S1)
        y1 = self.L * np.sqrt(S1 / (1-self.gamma))
        assert abs(x1 * y1 - self.L**2) < self.eps, f"x1 * y1 != L^2: {x1 * y1} != {self.L**2}"
        return (x1 * S1 + y1) * self.dist.pdf(v1)
    
    
    def pv_integrand_m(self, v1: float) -> float:
        """
        Integrand for the pool value calculation in the middle region
        including the probability density
        """
        S1 = self.S0 * v1
        x1 = self.x0
        y1 = self.y0
        assert abs(x1 * y1 - self.L**2) < self.eps, f"x1 * y1 != L^2: {x1 * y1} != {self.L**2}"
        return (x1 * S1 + y1) * self.dist.pdf(v1)
    
    def pv_integrand_d(self, v1: float) -> float:
        """
        Integrand for the pool value calculation in the lower region
        including the probability density
        """
        S1 = self.S0 * v1
        x1 = self.L * np.sqrt((1-self.gamma) / S1)
        y1 = self.L * np.sqrt(S1 / (1-self.gamma))
        assert abs(x1 * y1 - self.L**2) < self.eps, f"x1 * y1 != L^2: {x1 * y1} != {self.L**2}"
        return (x1 * S1 + y1) * self.dist.pdf(v1)
        
    def pv_integrand_uu(self, v2: float, v1: float) -> float:
        """
        Integrand for the pool value calculation
        """
        S1 = self.S0 * v1
        S2 = self.S0 * v1 * v2
        x1 = self.L / np.sqrt((1-self.gamma) * S1)
        y1 = self.L * np.sqrt((1-self.gamma) * S1)
        x2 = self.L / np.sqrt((1-self.gamma) * S2)
        y2 = self.L * np.sqrt((1-self.gamma) * S2)
        assert abs(x2 * y2 - x1 * y1) < self.eps, f"x2 * y2 != x1 * y1: {x2 * y2} != {x1 * y1}"
        assert abs(x2 * y2 - self.L**2) < self.eps, f"x2 * y2 != L^2: {x2 * y2} != {self.L**2}"
        return (x2 * S2 + y2) * self.dist.pdf(v1) * self.dist.pdf(v2)
    
    def pv_integrand_um(self, v2: float, v1: float) -> float:
        """
        Integrand for the pool value calculation
        """
        S1 = self.S0 * v1
        S2 = self.S0 * v1 * v2
        x1 = self.L / np.sqrt((1-self.gamma) * S1)
        y1 = self.L * np.sqrt((1-self.gamma) * S1)
        x2 = x1
        y2 = y1
        assert abs(x2 * y2 - x1 * y1) < self.eps, f"x2 * y2 != x1 * y1: {x2 * y2} != {x1 * y1}"
        assert abs(x2 * y2 - self.L**2) < self.eps, f"x2 * y2 != L^2: {x2 * y2} != {self.L**2}"
        return (x2 * S2 + y2) * self.dist.pdf(v1) * self.dist.pdf(v2)
    
    def pv_integrand_ud(self, v2: float, v1: float) -> float:
        """
        Integrand for the pool value calculation
        """
        S1 = self.S0 * v1
        S2 = self.S0 * v1 * v2
        x1 = self.L / np.sqrt((1-self.gamma) * S1)
        y1 = self.L * np.sqrt((1-self.gamma) * S1)
        x2 = self.L * np.sqrt((1-self.gamma) / S2)
        y2 = self.L * np.sqrt(S2 / (1-self.gamma))
        assert abs(x2 * y2 - x1 * y1) < self.eps, f"x2 * y2 != x1 * y1: {x2 * y2} != {x1 * y1}"
        assert abs(x2 * y2 - self.L**2) < self.eps, f"x2 * y2 != L^2: {x2 * y2} != {self.L**2}"
        return (x2 * S2 + y2) * self.dist.pdf(v1) * self.dist.pdf(v2)
    
    def pv_integrand_mu(self, v2: float, v1: float) -> float:
        """
        Integrand for the pool value calculation
        """
        S1 = self.S0 * v1
        S2 = self.S0 * v1 * v2
        x1 = self.x0
        y1 = self.y0
        x2 = self.L / np.sqrt((1-self.gamma) * S2)
        y2 = self.L * np.sqrt((1-self.gamma) * S2)
        assert abs(x2 * y2 - x1 * y1) < self.eps, f"x2 * y2 != x1 * y1: {x2 * y2} != {x1 * y1}"
        assert abs(x2 * y2 - self.L**2) < self.eps, f"x2 * y2 != L^2: {x2 * y2} != {self.L**2}"
        return (x2 * S2 + y2) * self.dist.pdf(v1) * self.dist.pdf(v2)
    
    def pv_integrand_mm(self, v2: float, v1: float) -> float:
        """
        Integrand for the pool value calculation
        """
        S1 = self.S0 * v1
        S2 = self.S0 * v1 * v2
        x1 = self.x0
        y1 = self.y0
        x2 = x1
        y2 = y1
        assert abs(x2 * y2 - x1 * y1) < self.eps, f"x2 * y2 != x1 * y1: {x2 * y2} != {x1 * y1}"
        assert abs(x2 * y2 - self.L**2) < self.eps, f"x2 * y2 != L^2: {x2 * y2} != {self.L**2}"
        return (x2 * S2 + y2) * self.dist.pdf(v1) * self.dist.pdf(v2)
    
    def pv_integrand_md(self, v2: float, v1: float) -> float:
        """
        Integrand for the pool value calculation
        """
        S1 = self.S0 * v1
        S2 = self.S0 * v1 * v2
        x1 = self.x0
        y1 = self.y0
        x2 = self.L * np.sqrt((1-self.gamma) / S2)
        y2 = self.L * np.sqrt(S2 / (1-self.gamma))
        assert abs(x2 * y2 - x1 * y1) < self.eps, f"x2 * y2 != x1 * y1: {x2 * y2} != {x1 * y1}"
        assert abs(x2 * y2 - self.L**2) < self.eps, f"x2 * y2 != L^2: {x2 * y2} != {self.L**2}"
        return (x2 * S2 + y2) * self.dist.pdf(v1) * self.dist.pdf(v2)
    
    def pv_integrand_du(self, v2: float, v1: float) -> float:
        """
        Integrand for the pool value calculation
        """
        S1 = self.S0 * v1
        S2 = self.S0 * v1 * v2
        x1 = self.L * np.sqrt((1-self.gamma) / S1)
        y1 = self.L * np.sqrt(S1 / (1-self.gamma))
        x2 = self.L / np.sqrt((1-self.gamma) * S2)
        y2 = self.L * np.sqrt((1-self.gamma) * S2)
        assert abs(x2 * y2 - x1 * y1) < self.eps, f"x2 * y2 != x1 * y1: {x2 * y2} != {x1 * y1}"
        assert abs(x2 * y2 - self.L**2) < self.eps, f"x2 * y2 != L^2: {x2 * y2} != {self.L**2}"
        return (x2 * S2 + y2) * self.dist.pdf(v1) * self.dist.pdf(v2)
    
    def pv_integrand_dm(self, v2: float, v1: float) -> float:
        """
        Integrand for the pool value calculation
        """
        S1 = self.S0 * v1
        S2 = self.S0 * v1 * v2
        x1 = self.L * np.sqrt((1-self.gamma) / S1)
        y1 = self.L * np.sqrt(S1 / (1-self.gamma))
        x2 = x1
        y2 = y1
        assert abs(x2 * y2 - x1 * y1) < self.eps, f"x2 * y2 != x1 * y1: {x2 * y2} != {x1 * y1}"
        assert abs(x2 * y2 - self.L**2) < self.eps, f"x2 * y2 != L^2: {x2 * y2} != {self.L**2}"
        return (x2 * S2 + y2) * self.dist.pdf(v1) * self.dist.pdf(v2)
    
    def pv_integrand_dd(self, v2: float, v1: float) -> float:
        """
        Integrand for the pool value calculation
        """
        S1 = self.S0 * v1
        S2 = self.S0 * v1 * v2
        x1 = self.L * np.sqrt((1-self.gamma) / S1)
        y1 = self.L * np.sqrt(S1 / (1-self.gamma))
        x2 = self.L * np.sqrt((1-self.gamma) / S2)
        y2 = self.L * np.sqrt(S2 / (1-self.gamma))
        assert abs(x2 * y2 - x1 * y1) < self.eps, f"x2 * y2 != x1 * y1: {x2 * y2} != {x1 * y1}"
        assert abs(x2 * y2 - self.L**2) < self.eps, f"x2 * y2 != L^2: {x2 * y2} != {self.L**2}"
        return (x2 * S2 + y2) * self.dist.pdf(v1) * self.dist.pdf(v2)
    
    def calculate_fee_revenue_single_step(self) -> float:
        """
        Calculate the expected fee revenue for a single step
        where log(v1) follows Normal(-1/2 σ²dt, σ²dt)
        """
        # Define the integration bounds
        v1_lower = 1 - self.gamma
        v1_upper = 1 / (1 - self.gamma)
        epsilon = 1e-12
        lb = self.dist.ppf(epsilon / 2)
        ub = self.dist.ppf(1 - epsilon / 2)
        
        # Perform the integration over all three regions
        result_u, _ = integrate.quad(self.fee_integrand_u, v1_upper, ub)
        result_d, _ = integrate.quad(self.fee_integrand_d, lb, v1_lower)
        
        return result_u + result_d
    
    def calculate_pool_value_single_step(self) -> float:
        """
        Calculate the pool value for a single step
        """
        # Define the integration bounds
        v1_lower = 1 - self.gamma
        v1_upper = 1 / (1 - self.gamma)
        epsilon = 1e-12
        lb = self.dist.ppf(epsilon / 2)
        ub = self.dist.ppf(1 - epsilon / 2)
        
        # Perform the integration over all three regions
        result_u, _ = integrate.quad(self.pv_integrand_u, v1_upper, ub)
        result_m, _ = integrate.quad(self.pv_integrand_m, v1_lower, v1_upper)
        result_d, _ = integrate.quad(self.pv_integrand_d, lb, v1_lower)
        
        return result_u + result_m + result_d
    
    def calculate_fee_revenue_second_step(self) -> float:
        """
        Calculate the expected fee revenue for a two step
        """
        epsilon = 1e-6
        to_ask = self.p0 / (1-self.gamma) / self.S0
        to_bid = self.p0 * (1-self.gamma) / self.S0
        lb = self.dist.ppf(epsilon / 2)
        ub = self.dist.ppf(1 - epsilon / 2)
        
        result_uu, _ = integrate.dblquad(self.fee_integrand_uu, to_ask, ub * to_ask, lambda v1: 1, lambda v1: ub, epsabs=1e-12, epsrel=1e-12)
        result_ud, _ = integrate.dblquad(self.fee_integrand_ud, to_ask, ub * to_ask, lambda v1: lb*(1-self.gamma)**2, lambda v1: (1-self.gamma)**2, epsabs=1e-12, epsrel=1e-12)
        result_mu, _ = integrate.dblquad(self.fee_integrand_mu, to_bid, to_ask, lambda v1: to_ask/v1, lambda v1: ub*to_ask/v1, epsabs=1e-12, epsrel=1e-12)
        result_md, _ = integrate.dblquad(self.fee_integrand_md, to_bid, to_ask, lambda v1: lb*to_bid/v1, lambda v1: to_bid/v1, epsabs=1e-12, epsrel=1e-12)
        result_du, _ = integrate.dblquad(self.fee_integrand_du, lb*to_bid, to_bid, lambda v1: 1/(1-self.gamma)**2, lambda v1: ub/(1-self.gamma)**2, epsabs=1e-12, epsrel=1e-12)
        result_dd, _ = integrate.dblquad(self.fee_integrand_dd, lb*to_bid, to_bid, lambda v1: lb, lambda v1: 1, epsabs=1e-12, epsrel=1e-12)
        
        # result_uu, _ = integrate.dblquad(self.fee_integrand_uu, 1/(1-self.gamma), ub, lambda v1: 1, lambda v1: ub, epsabs=1e-12, epsrel=1e-12)
        # result_ud, _ = integrate.dblquad(self.fee_integrand_ud, 1/(1-self.gamma), ub, lambda v1: lb, lambda v1: (1-self.gamma)**2, epsabs=1e-12, epsrel=1e-12)
        # result_mu, _ = integrate.dblquad(self.fee_integrand_mu, 1-self.gamma, 1/(1-self.gamma), lambda v1: 1/(v1*(1-self.gamma)), lambda v1: ub, epsabs=1e-12, epsrel=1e-12)
        # result_md, _ = integrate.dblquad(self.fee_integrand_md, 1-self.gamma, 1/(1-self.gamma), lambda v1: lb, lambda v1: (1-self.gamma)/v1, epsabs=1e-12, epsrel=1e-12)
        # result_du, _ = integrate.dblquad(self.fee_integrand_du, lb, 1-self.gamma, lambda v1: 1/(1-self.gamma)**2, lambda v1: ub, epsabs=1e-12, epsrel=1e-12)
        # result_dd, _ = integrate.dblquad(self.fee_integrand_dd, lb, 1-self.gamma, lambda v1: lb, lambda v1: 1, epsabs=1e-12, epsrel=1e-12)
        
        # assert all results are positive
        assert result_uu >= 0, f"fee_uu is negative: {result_uu}"
        assert result_ud >= 0, f"fee_ud is negative: {result_ud}"
        assert result_mu >= 0, f"fee_mu is negative: {result_mu}"
        assert result_md >= 0, f"fee_md is negative: {result_md}"
        assert result_du >= 0, f"fee_du is negative: {result_du}"
        assert result_dd >= 0, f"fee_dd is negative: {result_dd}"
        
        return result_uu + result_ud + result_mu + result_md + result_du + result_dd
    
    def calculate_pool_value_second_step(self) -> float:
        """
        Calculate the pool value for a two step
        """
        epsilon = 1e-6
        to_ask = self.p0 / (1-self.gamma) / self.S0
        to_bid = self.p0 * (1-self.gamma) / self.S0
        lb = self.dist.ppf(epsilon / 2) * to_bid
        ub = self.dist.ppf(1 - epsilon / 2) * to_ask
        
        result_uu, _ = integrate.dblquad(self.pv_integrand_uu, to_ask, ub*to_ask, lambda v1: 1, lambda v1: ub, epsabs=1e-12, epsrel=1e-12)
        result_um, _ = integrate.dblquad(self.pv_integrand_um, to_ask, ub*to_ask, lambda v1: (1-self.gamma)**2, lambda v1: 1, epsabs=1e-12, epsrel=1e-12)
        result_ud, _ = integrate.dblquad(self.pv_integrand_ud, to_ask, ub*to_ask, lambda v1: lb * (1-self.gamma)**2, lambda v1: (1-self.gamma)**2, epsabs=1e-12, epsrel=1e-12)
        result_mu, _ = integrate.dblquad(self.pv_integrand_mu, to_bid, to_ask, lambda v1: to_ask/v1, lambda v1: ub*to_ask/v1, epsabs=1e-12, epsrel=1e-12)
        result_mm, _ = integrate.dblquad(self.pv_integrand_mm, to_bid, to_ask, lambda v1: to_bid/v1, lambda v1: to_ask/v1, epsabs=1e-12, epsrel=1e-12)
        result_md, _ = integrate.dblquad(self.pv_integrand_md, to_bid, to_ask, lambda v1: lb*to_bid/v1, lambda v1: to_bid/v1, epsabs=1e-12, epsrel=1e-12)
        result_du, _ = integrate.dblquad(self.pv_integrand_du, lb*to_bid, to_bid, lambda v1: 1/(1-self.gamma)**2, lambda v1: ub/(1-self.gamma)**2, epsabs=1e-12, epsrel=1e-12)
        result_dm, _ = integrate.dblquad(self.pv_integrand_dm, lb*to_bid, to_bid, lambda v1: 1, lambda v1: 1/(1-self.gamma)**2, epsabs=1e-12, epsrel=1e-12)
        result_dd, _ = integrate.dblquad(self.pv_integrand_dd, lb*to_bid, to_bid, lambda v1: lb, lambda v1: 1, epsabs=1e-12, epsrel=1e-12)
        
        
        # result_uu, _ = integrate.dblquad(self.pv_integrand_uu, 1/(1-self.gamma), ub, lambda v1: 1, lambda v1: ub, epsabs=1e-12, epsrel=1e-12)
        # result_um, _ = integrate.dblquad(self.pv_integrand_um, 1/(1-self.gamma), ub, lambda v1: (1-self.gamma)**2, lambda v1: 1, epsabs=1e-12, epsrel=1e-12)
        # result_ud, _ = integrate.dblquad(self.pv_integrand_ud, 1/(1-self.gamma), ub, lambda v1: lb, lambda v1: (1-self.gamma)**2, epsabs=1e-12, epsrel=1e-12)
        # result_mu, _ = integrate.dblquad(self.pv_integrand_mu, 1-self.gamma, 1/(1-self.gamma), lambda v1: 1/(v1*(1-self.gamma)), lambda v1: ub, epsabs=1e-12, epsrel=1e-12)
        # result_mm, _ = integrate.dblquad(self.pv_integrand_mm, 1-self.gamma, 1/(1-self.gamma), lambda v1: (1-self.gamma)/v1, lambda v1: 1/(v1*(1-self.gamma)), epsabs=1e-12, epsrel=1e-12)
        # result_md, _ = integrate.dblquad(self.pv_integrand_md, 1-self.gamma, 1/(1-self.gamma), lambda v1: lb, lambda v1: (1-self.gamma)/v1, epsabs=1e-12, epsrel=1e-12)
        # result_du, _ = integrate.dblquad(self.pv_integrand_du, lb, 1-self.gamma, lambda v1: 1/(1-self.gamma)**2, lambda v1: ub, epsabs=1e-12, epsrel=1e-12)
        # result_dm, _ = integrate.dblquad(self.pv_integrand_dm, lb, 1-self.gamma, lambda v1: 1, lambda v1: 1/(1-self.gamma)**2, epsabs=1e-12, epsrel=1e-12)
        # result_dd, _ = integrate.dblquad(self.pv_integrand_dd, lb, 1-self.gamma, lambda v1: lb, lambda v1: 1, epsabs=1e-12, epsrel=1e-12)
        
        # assert all results are positive
        assert result_uu >= 0, f"pv_uu is negative: {result_uu}"
        assert result_um >= 0, f"pv_um is negative: {result_um}"
        assert result_ud >= 0, f"pv_ud is negative: {result_ud}"
        assert result_mu >= 0, f"pv_mu is negative: {result_mu}"
        assert result_mm >= 0, f"pv_mm is negative: {result_mm}"
        assert result_md >= 0, f"pv_md is negative: {result_md}"
        assert result_du >= 0, f"pv_du is negative: {result_du}"
        assert result_dm >= 0, f"pv_dm is negative: {result_dm}"
        assert result_dd >= 0, f"pv_dd is negative: {result_dd}"
        
        total_result = result_uu + result_um + result_ud + result_mu + result_mm + result_md + result_du + result_dm + result_dd
        
        return total_result
    
def process_gamma(args):
    sigma, gamma, theta = args
    two_step_integrate = TwoStepIntegrate(gamma, sigma, theta)
    first_step_fee_revenue = two_step_integrate.calculate_fee_revenue_single_step()
    first_step_pool_value = two_step_integrate.calculate_pool_value_single_step()
    second_step_fee_revenue = two_step_integrate.calculate_fee_revenue_second_step()
    second_step_pool_value = two_step_integrate.calculate_pool_value_second_step()
    return {
        "sigma": sigma,
        "gamma": gamma,
        "theta": theta,
        "first_step_fee_revenue": first_step_fee_revenue,
        "first_step_pool_value": first_step_pool_value,
        "second_step_fee_revenue": second_step_fee_revenue,
        "second_step_pool_value": second_step_pool_value
    }

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from multiprocessing import Pool, cpu_count
    import os
    
    path = '/Users/haofu/Desktop/AMM/Dynamic_AMM/CPMM/TwoStep/results'
    os.makedirs(path, exist_ok=True)
    
    sigma_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    gamma_list = np.arange(0.00001, 0.0005, 0.00001)
    theta_list = [-1]
    
    # Create all combinations of sigma and gamma
    args_list = [(sigma, gamma, theta) for sigma in sigma_list for gamma in gamma_list for theta in theta_list]
    
    # Use all available CPU cores
    num_cores = cpu_count()
    print(f"Using {num_cores} CPU cores")
    
    # Process in parallel with progress bar
    with Pool(num_cores) as pool:
        result_list = list(tqdm(
            pool.imap(process_gamma, args_list),
            total=len(args_list),
            desc="Processing combinations"
        ))
    
    result_df = pd.DataFrame(result_list)
    result_df.to_csv(f"{path}/two_step_result_thetan1.csv", index=False)
    
        
        
        
        
        
        
        
        
        
        
        