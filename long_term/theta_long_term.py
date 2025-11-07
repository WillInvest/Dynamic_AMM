
import numpy as np
from scipy.stats import norm
from numpy.linalg import solve
from joblib import Parallel, delayed

class AMMStationaryDistributionFast:
    """
    Vectorized transition construction + fast stationary distribution.
    """
    def __init__(self, gamma, mu, sigma, dt, bins, bin_centers):
        self.gamma = float(gamma)
        self.mu    = float(mu)
        self.sigma = float(sigma)
        self.dt    = float(dt)

        # fixed pool scales (only scale the fee, so leave as given)
        self.L = 1_000_000.0
        self.x = 1_000_000.0
        self.y = self.L ** 2 / self.x

        # shared discretization (passed in to avoid rebuilding)
        self.bins        = bins                   # shape: (N,)
        self.bin_centers = bin_centers            # shape: (N-1,)
        self.N           = bins.size              # N edges ⇒ N-1 interior bins
        self.num_states  = self.N + 1             # (N-1 interiors) + 2 boundaries

        # fee scale + induced drift/variance of theta
        self.lambda_fee = -np.log(1.0 - self.gamma)
        self.m = (mu - 0.5 * sigma**2) / self.lambda_fee
        self.v = sigma**2 / (self.lambda_fee**2)

        # build transition once
        self.P = self._build_transition_matrix_vectorized()

        # stationary distribution (right eigenvector for column-stochastic P)
        self.pi_star = self._stationary_power(tol=1e-11, max_iter=10000)

    # ---------- Transition matrix (vectorized) ----------
    def _build_transition_matrix_vectorized(self):
        """
        Column-stochastic P with ordering:
        [interior bins 0..N-2, boundary_minus (@-1), boundary_plus (@+1)]
        """
        N = self.N
        K = N - 1  # interior count
        P = np.zeros((K + 2, K + 2), dtype=np.float64)

        # Means / std for steps starting at each state (interior + two boundaries)
        dt  = self.dt
        mean_interior = self.bin_centers - self.m * dt           # shape: (K,)
        std           = np.sqrt(self.v * dt)

        # CDFs at all edges for all interior starting states: shape (N edges, K states)
        # edges along rows, states along cols
        edges = self.bins[:, None]                               # (N, 1)
        cdfs  = norm.cdf(edges, loc=mean_interior[None, :], scale=std)  # (N, K)

        # Interior→interior probabilities: difference across edges
        # rows: destination interior bin (0..K-1), cols: origin interior state (0..K-1)
        interior_block = (cdfs[1:, :] - cdfs[:-1, :])            # (K, K)

        # Interior→boundaries
        prob_minus_interior = cdfs[0, :]                         # (K,)
        prob_plus_interior  = 1.0 - cdfs[-1, :]                  # (K,)

        # Fill interior-origin columns (0..K-1)
        P[:K, :K]     = interior_block
        P[K,  :K]     = prob_minus_interior
        P[K+1, :K]    = prob_plus_interior

        # Boundary origins: treat x as exactly at edges -1 and +1
        x_minus = self.bins[0]
        x_plus  = self.bins[-1]
        means_b = np.array([x_minus, x_plus]) - self.m * dt      # shape (2,)
        cdfs_b  = norm.cdf(edges, loc=means_b, scale=std)        # (N, 2)

        # boundary -1 origin -> interior/boundaries goes to column K
        interior_from_minus = (cdfs_b[1:, 0] - cdfs_b[:-1, 0])
        P[:K,  K]   = interior_from_minus
        P[K,   K]   = cdfs_b[0, 0]
        P[K+1, K]   = 1.0 - cdfs_b[-1, 0]

        # boundary +1 origin -> interior/boundaries goes to column K+1
        interior_from_plus = (cdfs_b[1:, 1] - cdfs_b[:-1, 1])
        P[:K,  K+1] = interior_from_plus
        P[K,   K+1] = cdfs_b[0, 1]
        P[K+1, K+1] = 1.0 - cdfs_b[-1, 1]

        # Optional (cheap) normalization guard:
        # P /= P.sum(axis=0, keepdims=True)

        return P

    # ---------- Stationary distribution ----------
    def _stationary_power(self, tol=1e-12, max_iter=10000):
        """
        Power iteration on column-stochastic P for right eigenvector with λ=1.
        """
        n = self.P.shape[0]
        pi = np.full(n, 1.0 / n, dtype=np.float64)
        for _ in range(max_iter):
            new = self.P @ pi
            # normalize (L1)
            s = new.sum()
            if s == 0.0:
                # extremely pathological; fall back to uniform
                new = np.full_like(new, 1.0 / n)
            else:
                new /= s
            if np.max(np.abs(new - pi)) < tol:
                return new
            pi = new
        return pi  # best effort

    # ---------- Fee vectors (fully vectorized) ----------
    def _incoming_fee_vec(self, theta, St=None):
        L, dt, sigma, g = self.L, self.dt, self.sigma, self.gamma
        if St is None:
            x = self.x
            y = L ** 2 / x
            St = (y/x) * (1-g) ** (-theta)
        exp_term = np.exp(-sigma**2 * dt / 8)
        sigma_sqrt_dt = sigma * np.sqrt(dt)
        d1 = (1+theta) * np.log(1-g) / sigma_sqrt_dt
        d2 = (1-theta) * np.log(1-g) / sigma_sqrt_dt
        first_term = np.sqrt(1-g) * exp_term * norm.cdf(d1)
        second_term = (1-g) ** (-theta/2) * norm.cdf(d1 - sigma_sqrt_dt)
        third_term = np.sqrt(1-g) * exp_term * norm.cdf(d2)
        fourth_term = (1-g) ** (theta/2) * norm.cdf(d2 - sigma_sqrt_dt)
        common_factor = (g / (1-g)) * L * np.sqrt(St)
        return common_factor * (first_term - second_term + third_term - fourth_term)
    
    def _log_normal_density_vec(self, S0, St, sigma, n, dt):
        first_term = 1 / (St * sigma * np.sqrt(2 * np.pi * n * dt))
        numerator = np.log(St) - np.log(S0) + (n * (sigma**2) * dt)/2
        denominator = 2 * (sigma**2) * n * dt
        second_term = np.exp(-(numerator**2) / denominator)
        return first_term * second_term
    
    def _expected_incoming_fee_wrt_theta(self, St):
        theta_list = np.concatenate([self.bin_centers, [-1.0, 1.0]])
        pi         = self.pi_star
        pi = pi / pi.sum()
        inc = self._incoming_fee_vec(theta_list, St)
        return float(pi @ inc)
    
    def _expected_theta_incoming_fee_wrt_st(self, n, nodes=128):
        t = n * self.dt
        S0 = self.y / self.x # initialize the external price
        sigma = self.sigma
        tail_std = 10.0

        m = np.log(S0) - 0.5 * (sigma**2) * t
        v = (sigma**2) * t
        sdev = np.sqrt(v)

        # truncate the positive support of S_t to a very high-probability band in log-space
        a = np.exp(m - tail_std * sdev)
        b = np.exp(m + tail_std * sdev)
        
        z, w = np.polynomial.hermite.hermgauss(nodes)
        St = 0.5 * (b - a) * z + 0.5 * (a + b)
        W  = 0.5 * (b - a) * w
        
        fee_cond = self._expected_incoming_fee_wrt_theta(St)
        dens     = self._log_normal_density_vec(S0=S0, St=St, sigma=sigma, n=n, dt=dt)
        return float(W @ (fee_cond * dens))
    
    def _cumulative_fee(self, N0, nSteps):
        cumulative_fee = 0
        for i in range(nSteps):
            fee = self._expected_theta_incoming_fee_wrt_st(N0 + i)
            cumulative_fee += fee
        return cumulative_fee
    
    def simulate_fees_tail_only(
        self,
        n_paths,
        n_steps=100,
        tail_len=100,
        mode="Incoming"
    ):
        sigma = self.sigma
        dt = self.dt
        gamma = self.gamma

        # sample theta and compute path-specific S0
        theta_list = np.concatenate([self.bin_centers, [-1.0, 1.0]])
        pi = self.pi_star / self.pi_star.sum()
        theta_samples = np.random.choice(theta_list, size=n_paths, p=pi)
        S0 = (self.y / self.x) * (1 - gamma) ** (-theta_samples)

        # GBM path (exact discretization)
        dW = np.random.randn(n_paths, n_steps) * np.sqrt(dt)
        W = np.cumsum(dW, axis=1)
        t = np.arange(1, n_steps + 1) * dt
        S = S0[:, None] * np.exp((-0.5 * sigma**2) * t + sigma * W)

        # --- AMM init (constant-product CFMM) ---
        x0, y0 = self.x, self.y
        L = x0 * y0

        xt = np.full(n_paths, x0, dtype=float)
        yt = np.full(n_paths, y0, dtype=float)

        # fee tracking
        fee_y = np.zeros(n_paths)
        fee_y_equiv_from_x = np.zeros(n_paths)
        last_start = n_steps - tail_len

        eps = 1e-12

        for i in range(n_steps):
            P_ext = S[:, i]
            P_AMM = yt / np.maximum(xt, eps)
            bid = P_AMM * (1 - gamma)
            ask = P_AMM / (1 - gamma)

            up_mask = P_ext > ask
            down_mask = P_ext < bid
            in_tail = (i >= last_start)

            if mode == "Incoming":
                # --- Upward arbitrage: external > ask (trader buys X, pays Y)
                if np.any(up_mask):
                    # invariant: x*y = L
                    x_new = np.sqrt(L / P_ext[up_mask])
                    delta_x = xt[up_mask] - x_new
                    xt[up_mask] -= np.maximum(delta_x, 0)

                    y_new = L / np.maximum(x_new, eps)
                    delta_y = (y_new - yt[up_mask]) / (1 - gamma)
                    if in_tail:
                        fee_y[up_mask] += gamma * np.maximum(delta_y, 0)
                    yt[up_mask] += np.maximum(delta_y, 0)

                # --- Downward arbitrage: external < bid (trader sells X, receives Y)
                if np.any(down_mask):
                    y_new = np.sqrt(L * P_ext[down_mask])
                    delta_y = yt[down_mask] - y_new
                    yt[down_mask] -= np.maximum(delta_y, 0)

                    x_new = L / np.maximum(y_new, eps)
                    delta_x = (x_new - xt[down_mask]) / (1 - gamma)
                    fee_x = gamma * np.maximum(delta_x, 0)
                    if in_tail:
                        fee_y_equiv_from_x[down_mask] += fee_x * P_ext[down_mask]
                    xt[down_mask] += np.maximum(delta_x, 0)

            else:
                raise ValueError("Only 'Incoming' mode implemented in this snippet.")

            # numerical guard
            xt = np.maximum(xt, eps)
            yt = np.maximum(yt, eps)

        fees_tail_y_equiv = fee_y + fee_y_equiv_from_x

        return {
            "fees_tail_y_equiv": fees_tail_y_equiv,
            "fees_tail_y": fee_y,
            "fees_tail_from_x_in_y": fee_y_equiv_from_x,
            "final_xt": xt,
            "final_yt": yt,
            "S": S
        }

# ---------- Batch evaluation (parallel) ----------
def sweep_grid(gamma_list, sigma_list, mu, dt, N=500, n_jobs=-1):
    bins = np.linspace(-1.0, 1.0, N)           # N edges → N-1 interior
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # make Cartesian product once
    tasks = [(g, s) for g in gamma_list for s in sigma_list]

    def run_one(g, s):
        mdl = AMMStationaryDistributionFast(g, mu, s, dt, bins, bin_centers)
        inc, out, terminal_wealth, profit, inc_mid, out_mid, terminal_wealth_mid, profit_mid = mdl.collect_results()
        return g, s, inc, out, terminal_wealth, profit, inc_mid, out_mid, terminal_wealth_mid, profit_mid

    out = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(run_one)(g, s) for g, s in tasks
    )
    # to DataFrame without importing pandas here:
    return np.array(out, dtype=[("gamma", "f8"), ("sigma", "f8"),
                                ("expected_incoming_fee", "f8"),
                                ("expected_outgoing_fee", "f8"),
                                ("terminal_wealth", "f8"),
                                ("profit", "f8"),
                                ("incoming_fee_mid", "f8"),
                                ("outgoing_fee_mid", "f8"),
                                ("terminal_wealth_mid", "f8"),
                                ("profit_mid", "f8")])
    
    
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from datetime import datetime
    N   = 500
    mu  = 0.0
    dt  = 12.0 / (365*24*60*60)

    gamma_list = np.arange(0.0001, 0.0201, 0.0001)  
    sigma_list = np.arange(0.1, 2.1, 0.1)

    rec = sweep_grid_total(gamma_list, sigma_list, mu, dt, N=N, n_jobs=-1)
    df = pd.DataFrame(rec)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"vector_theta_analysis_results_total_large_gamma_{timestamp}.csv", index=False)
    
    