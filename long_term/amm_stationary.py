# amm_stationary.py
from typing import Union
import numpy as np
from scipy.stats import norm

class AMMStationaryDistributionFast:
    """
    Stationary theta distribution + GBM fee simulation for a constant-product AMM
    with fee-on-input arbitrage alignment. Includes:
      - Column-stochastic transition P (vectorized) and power-iteration stationary dist.
      - Analytic expectation E_theta[ fee(theta, S_t) ] integrated over S_t via Gauss–Hermite on Z.
      - Simulation that accumulates fees from an arbitrary step N0 forward (to match analytics).

    Conventions
    -----------
    - Time is in YEARS; sigma is ANNUALIZED.
    - GBM: S_t = S_0 * exp( (-0.5 sigma^2) t + sigma sqrt(t) Z ),  Z ~ N(0,1)  (mu=0 here)
    - AMM: constant-product x*y = L (Uniswap v2 style), price P_AMM = y/x, fee = gamma on input token.
    - Arbitrage rule: jump reserves to align AMM price to P_ext instantly at each step, with fee-on-input.
    """

    def __init__(self, gamma, mu, sigma, dt, bins, bin_centers, L=1_000_000.0, x=1_000_000.0):
        self.gamma = float(gamma)
        self.mu    = float(mu)         # (not used in GBM below; set to 0 or r-q if you want)
        self.sigma = float(sigma)
        self.dt    = float(dt)

        # pool scales (kept fixed)
        self.L = float(L)
        self.x = float(x)
        self.y = (self.L ** 2) / self.x

        # discretization for theta
        self.bins        = np.asarray(bins)
        self.bin_centers = np.asarray(bin_centers)
        self.N           = self.bins.size
        self.num_states  = self.N + 1  # K interior + 2 boundaries, but we store as K+2 rows/cols

        # fee scale -> theta drift/variance (you already used this convention)
        self.lambda_fee = -np.log(1.0 - self.gamma)
        self.m = (self.mu - 0.5 * self.sigma**2) / self.lambda_fee
        self.v = (self.sigma**2) / (self.lambda_fee**2)

        # build transition once (column-stochastic)
        self.P = self._build_transition_matrix_vectorized()

        # stationary distribution via power iteration
        self.pi_star = self._stationary_power(tol=1e-11, max_iter=10_000)

    # ---------- Transition matrix (vectorized) ----------
    def _build_transition_matrix_vectorized(self):
        """
        Build column-stochastic transition matrix P with ordering:
        [interior bins 0..K-1, boundary_minus (@-1), boundary_plus (@+1)] as rows,
        same ordering for columns (origin states).
        """
        N = self.N
        K = N - 1  # number of interior bins
        P = np.zeros((K + 2, K + 2), dtype=np.float64)

        dt  = self.dt
        std = np.sqrt(self.v * dt)

        # interior origins: normal with means (center - m dt), same std
        mean_interior = self.bin_centers - self.m * dt  # shape (K,)
        edges = self.bins[:, None]                      # (N, 1)
        cdfs  = norm.cdf(edges, loc=mean_interior[None, :], scale=std)  # (N, K)

        interior_block = (cdfs[1:, :] - cdfs[:-1, :])   # (K, K)
        prob_minus_interior = cdfs[0, :]                # (K,)
        prob_plus_interior  = 1.0 - cdfs[-1, :]         # (K,)

        # Fill interior-origin columns 0..K-1
        P[:K, :K]  = interior_block
        P[K,  :K]  = prob_minus_interior
        P[K+1,:K]  = prob_plus_interior

        # boundaries as origins: put state exactly at left/right edge then step
        x_minus = self.bins[0]
        x_plus  = self.bins[-1]
        means_b = np.array([x_minus, x_plus]) - self.m * dt    # shape (2,)
        cdfs_b  = norm.cdf(edges, loc=means_b, scale=std)      # (N, 2)

        # boundary -1 (column K)
        interior_from_minus = cdfs_b[1:, 0] - cdfs_b[:-1, 0]
        P[:K,  K]   = interior_from_minus
        P[K,   K]   = cdfs_b[0, 0]
        P[K+1, K]   = 1.0 - cdfs_b[-1, 0]

        # boundary +1 (column K+1)
        interior_from_plus = cdfs_b[1:, 1] - cdfs_b[:-1, 1]
        P[:K,  K+1] = interior_from_plus
        P[K,   K+1] = cdfs_b[0, 1]
        P[K+1, K+1] = 1.0 - cdfs_b[-1, 1]

        # Normalize columns to sum to 1 (guard against FP drift)
        colsum = P.sum(axis=0, keepdims=True)
        # Avoid divide-by-zero if a pathological column ever appears
        colsum[colsum == 0.0] = 1.0
        P /= colsum
        return P

    # ---------- Stationary distribution ----------
    def _stationary_power(self, tol=1e-12, max_iter=10_000):
        n = self.P.shape[0]
        pi = np.full(n, 1.0 / n, dtype=np.float64)
        for _ in range(max_iter):
            new = self.P @ pi
            s = new.sum()
            if s == 0.0:
                new = np.full_like(new, 1.0 / n)
            else:
                new /= s
            if np.max(np.abs(new - pi)) < tol:
                return new
            pi = new
        return pi

    # ---------- Fee formulas ----------
    def _incoming_fee_vec(self, theta, St=None):
        """
        Expected incoming-fee per (theta, S_t) over one step dt for constant-product AMM
        with fee-on-input (your closed-form). Vectorized over theta and St.
        """
        L, dt, sigma, g = self.L, self.dt, self.sigma, self.gamma
        if St is None:
            x = self.x
            y = (self.L ** 2) / x
            St = (y / x) * (1 - g) ** (-theta)

        exp_term = np.exp(-sigma**2 * dt / 8.0)
        sigma_sqrt_dt = sigma * np.sqrt(dt)
        lg = np.log(1 - g)

        d1 = (1 + theta) * lg / sigma_sqrt_dt
        d2 = (1 - theta) * lg / sigma_sqrt_dt

        first_term  = np.sqrt(1 - g) * exp_term * norm.cdf(d1)
        second_term = (1 - g) ** (-theta / 2.0) * norm.cdf(d1 - sigma_sqrt_dt/2)
        third_term  = np.sqrt(1 - g) * exp_term * norm.cdf(d2)
        fourth_term = (1 - g) ** (theta / 2.0) * norm.cdf(d2 - sigma_sqrt_dt/2)

        common_factor = (g / (1 - g)) * L * np.sqrt(St)
        return common_factor * (first_term - second_term + third_term - fourth_term)

    def _expected_incoming_fee_wrt_theta(self, St):
        """
        E_theta[ fee(theta, St) ] with theta distributed by pi_star over [bin_centers, -1, +1].
        Vectorized in St.
        """
        theta_list = np.concatenate([self.bin_centers, [-1.0, 1.0]])
        pi = self.pi_star
        pi = pi / pi.sum()
        inc = self._incoming_fee_vec(theta_list[:, None], St[None, :])  # (T, M)
        # sum over theta axis
        return (pi[:, None] * inc).sum(axis=0)  # shape (M,)

    # ---------- Analytic integral over S_t using Gauss–Hermite on Z ----------
    def _expected_theta_incoming_fee_wrt_st(self, S0, n, n_samples=1, seed=None):
        """
        Calculate E[ E_theta[fee(theta, S_n)] ] using Monte Carlo sampling.
    
        The calculation follows these steps:
        1. Generate standard normal samples Z ~ N(0,1)
        2. For each S0 and Z sample, calculate the future price S_n:
           S_n = S0 * exp(-0.5*sigma^2*t + sigma*sqrt(t)*Z)
        3. For each S_n, calculate E_theta[fee | S_n]
        4. Average over Z samples
    
        Parameters:
        -----------
        S0 : array-like, shape (K,)
            Vector of initial prices
        n : int
            Time step number (t = n*dt)
        n_samples : int
            Number of Monte Carlo samples
        seed : int, optional
            Random seed for reproducibility
    
        Returns:
        --------
        array of shape (K,) : Expected fees for each S0
        """
        rng = np.random.default_rng(seed)
        t = n * self.dt
        sigma = self.sigma
    
        # Step 1: Generate standard normal samples
        Z = rng.standard_normal((n_samples,))
    
        # Step 2: Calculate future prices S_n for each (S0, Z) pair
        # Shape: S0 (K,1) × exp(stuff + Z(1,n_samples)) -> (K, n_samples)
        drift = -0.5 * sigma**2 * t
        diffusion = sigma * np.sqrt(t)
        S_n = S0[:, None] * np.exp(drift + diffusion * Z[None, :])
    
        # Step 3: Calculate conditional expected fees E_theta[fee | S_n]
        # Shape: (K, n_samples)
        expected_fees_given_Sn = self._expected_incoming_fee_wrt_theta(S_n)
    
        # Step 4: Average over Z samples
        # Shape: (K, n_samples) -> mean over samples -> (K,)
        expected_fees = np.mean(expected_fees_given_Sn, axis=1)
    
        return expected_fees

    def _expected_fee_with_distributed_initial_S0(self, n):
        """
        Calculate expected fee with S0 distributed according to theta distribution.
        Each theta value has its own S0, and we compute expectations in parallel.
        """
        theta_list = np.concatenate([self.bin_centers, [-1.0, 1.0]])
        pi = self.pi_star / self.pi_star.sum()
    
        # Calculate S0 for each theta - shape (K,)
        # S0_values = (self.y / self.x) * (1.0 - self.gamma) ** (-theta_list)
        S0_values = np.repeat(self.y / self.x, theta_list.size)
    
        # Calculate expected fees for all S0 values at once - shape (K,)
        expected_fees = self._expected_theta_incoming_fee_wrt_st(S0_values, n)

        # Weight by probabilities and sum
        return float(np.sum(pi * expected_fees))

    def cumulative_fee_analytic(self, N0, window_len):
        """
        Sum_{n=N0}^{N0+window_len-1} E[ E_theta[fee(theta, S_n)] ].
        """
        total = 0.0
        for n in range(N0, N0 + window_len):
            total += self._expected_fee_with_distributed_initial_S0(n)
        return total

    # ---------- Simulation from N0 ----------
    def simulate_fees_from_N0(
        self,
        n_paths: int,
        n_steps: int,
        N0: int,
        mode: str = "Incoming",
        S0_mode: str = "flat",
        seed: Union[int, None] = None,
    ):
        """
        Simulate GBM paths and AMM fees. Accumulate fees from step index N0 onward (inclusive).
        S0_mode:
           - "flat": all paths start S0 = y/x
           - "theta_shifted": each path draws theta ~ pi_star and sets S0(theta) = (y/x)*(1-gamma)^(-theta)
                              (matches your earlier sim pattern)
        Returns a dict with per-path tail-fees and final reserves, plus the S matrix if you need it.
        """
        rng = np.random.default_rng(seed)
        sigma = self.sigma
        dt = self.dt
        gamma = self.gamma
        eps = 1e-12

        # theta support + stationary pmf
        theta_list = np.concatenate([self.bin_centers, [-1.0, 1.0]])
        pi = self.pi_star / self.pi_star.sum()

        # S0 per path
        if S0_mode == "flat":
            S0 = np.full(n_paths, self.y / self.x, dtype=float)
        elif S0_mode == "theta_shifted":
            theta_samples = rng.choice(theta_list, size=n_paths, p=pi)
            S0 = (self.y / self.x) * (1.0 - gamma) ** (-theta_samples)
        else:
            raise ValueError("S0_mode must be 'flat' or 'theta_shifted'.")

        # GBM path (exact)
        dW = rng.standard_normal((n_paths, n_steps+N0)) * np.sqrt(dt)
        W = np.cumsum(dW, axis=1)
        tgrid = np.arange(1, n_steps + N0 + 1) * dt
        S = S0[:, None] * np.exp((-0.5 * sigma**2) * tgrid + sigma * W)  # (n_paths, n_steps)

        # AMM init (constant-product)
        x0, y0 = self.x, self.y
        L = x0 * y0
        xt = np.full(n_paths, x0, dtype=float)
        yt = np.full(n_paths, y0, dtype=float)

        fee_y = np.zeros(n_paths)
        fee_y_equiv_from_x = np.zeros(n_paths)

        for i in range(N0 + n_steps):
            P_ext = S[:, i]
            P_AMM = yt / np.maximum(xt, eps)
            bid = P_AMM * (1.0 - gamma)
            ask = P_AMM / (1.0 - gamma)

            up_mask = P_ext > ask
            down_mask = P_ext < bid
            in_window = (i >= N0)

            if mode != "Incoming":
                raise ValueError("Only 'Incoming' (fee-on-input) supported here.")

            # Upward arbitrage: external > ask (trader buys X, pays Y)
            if np.any(up_mask):
                x_new = np.sqrt(L / np.maximum(P_ext[up_mask], eps))
                delta_x = xt[up_mask] - x_new
                xt[up_mask] -= np.maximum(delta_x, 0.0)

                y_new = L / np.maximum(x_new, eps)
                delta_y = (y_new - yt[up_mask]) / (1.0 - gamma)
                if in_window:
                    fee_y[up_mask] += gamma * np.maximum(delta_y, 0.0)
                yt[up_mask] += np.maximum(delta_y, 0.0)

            # Downward arbitrage: external < bid (trader sells X, receives Y)
            if np.any(down_mask):
                y_new = np.sqrt(L * P_ext[down_mask])
                delta_y = yt[down_mask] - y_new
                yt[down_mask] -= np.maximum(delta_y, 0.0)

                x_new = L / np.maximum(y_new, eps)
                delta_x = (x_new - xt[down_mask]) / (1.0 - gamma)
                fee_x = gamma * np.maximum(delta_x, 0.0)
                if in_window:
                    fee_y_equiv_from_x[down_mask] += fee_x * P_ext[down_mask]
                xt[down_mask] += np.maximum(delta_x, 0.0)

            # numerical floors
            xt = np.maximum(xt, eps)
            yt = np.maximum(yt, eps)

        fees_tail_y_equiv = fee_y + fee_y_equiv_from_x

        return {
            "fees_tail_y_equiv": fees_tail_y_equiv,    # per-path totals in Y units
            "fees_tail_y": fee_y,
            "fees_tail_from_x_in_y": fee_y_equiv_from_x,
            "final_xt": xt,
            "final_yt": yt,
            "S": S,
        }
