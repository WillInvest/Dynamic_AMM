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
    def _incoming_fee_vec(self, theta):
        L, x, mu, dt, sigma, g = self.L, self.x, self.mu, self.dt, self.sigma, self.gamma
        y   = L**2 / x
        p0  = (y/x) * np.power(1.0 - g, -theta)

        sqrt_dt     = np.sqrt(dt)
        sig_sqrt_dt = sigma * sqrt_dt

        d1 = (np.log((1.0 - g) * y / (p0 * x)) - mu * dt) / sig_sqrt_dt
        d2 = (np.log(y / ((1.0 - g) * x * p0)) - mu * dt) / sig_sqrt_dt

        alpha = L * np.sqrt((1.0 - g) * p0) * np.exp(mu/2 - (sigma**2) * dt / 8.0)

        return (g / (1.0 - g)) * (
            alpha * (norm.cdf(d1) + norm.cdf(-d2))
            - np.exp(mu * dt) * p0 * x * norm.cdf(d1 - sig_sqrt_dt / 2.0)
            - y * norm.cdf(-d2 - sig_sqrt_dt / 2.0)
        )

    def _outgoing_fee_vec(self, theta):
        L, x, mu, dt, sigma, g = self.L, self.x, self.mu, self.dt, self.sigma, self.gamma
        y   = L**2 / x
        p0  = (y/x) * np.power(1.0 - g, -theta)

        sqrt_dt     = np.sqrt(dt)
        sig_sqrt_dt = sigma * sqrt_dt

        d1 = (np.log((1.0 - g) * y / (p0 * x)) - mu * dt) / sig_sqrt_dt
        d2 = (np.log(y / ((1.0 - g) * x * p0)) - mu * dt) / sig_sqrt_dt

        alpha = L * np.sqrt(p0 / (1.0 - g)) * np.exp(mu/2 - (sigma**2) * dt / 8.0)

        return g * (
            - alpha * (norm.cdf(d1) + norm.cdf(-d2))
            + np.exp(mu * dt) * p0 * x * norm.cdf(-d2 + sig_sqrt_dt / 2.0)
            + y * norm.cdf(d1 + sig_sqrt_dt / 2.0)
        )
        
    def _analytical_pool_value(self, theta):
        L, x, mu, delta_t, sigma, gamma = self.L, self.x, self.mu, self.dt, self.sigma, self.gamma
        y = L**2 / x
        p0 = (y/x) * np.power(1.0 - gamma, -theta)
        initial_wealth = x * p0 + y # initial external wealth
        sigma_sqrt_dt = sigma * np.sqrt(delta_t)
        lambda_1 = y / ((1 - gamma) * p0 * x)
        lambda_2 = (1 - gamma) * y / (p0 * x)
    
        # Calculate D terms
        D1 = (1 / sigma_sqrt_dt) * np.log(lambda_1) 
        D2 = (1 / sigma_sqrt_dt) * np.log(lambda_2) 
        D1_plus = (1 / sigma_sqrt_dt) * (np.log(lambda_1) + 0.5 * sigma**2 * delta_t)
        D1_minus = (1 / sigma_sqrt_dt) * (np.log(lambda_1) - 0.5 * sigma**2 * delta_t)
        D2_plus = (1 / sigma_sqrt_dt) * (np.log(lambda_2) + 0.5 * sigma**2 * delta_t)
        D2_minus = (1 / sigma_sqrt_dt) * (np.log(lambda_2) - 0.5 * sigma**2 * delta_t)
    
        beta = L * (2-gamma) * np.sqrt(p0/(1-gamma)) * np.exp(-sigma**2/8 * delta_t)
    
        first_term = beta * (norm.cdf(-D1) + norm.cdf(D2))
        second_term = y * (norm.cdf(D1_plus) - norm.cdf(D2_plus)) + \
            x * p0 * (norm.cdf(D1_minus) - norm.cdf(D2_minus))
        terminal_wealth = first_term + second_term
        profit = terminal_wealth - initial_wealth
        return terminal_wealth, profit
    
    

    def collect_results(self):
        # theta list (interiors + boundaries)
        theta_list = np.concatenate([self.bin_centers, [-1.0, 1.0]])
        pi         = self.pi_star

        inc = self._incoming_fee_vec(theta_list)
        out = self._outgoing_fee_vec(theta_list)
        
        terminal_wealth, profit = self._analytical_pool_value(theta_list)

        # (Optional) small renorm for safety
        pi = pi / pi.sum()

        return float(pi @ inc), float(pi @ out), float(pi @ terminal_wealth), float(pi @ profit)

# ---------- Batch evaluation (parallel) ----------
def sweep_grid(gamma_list, sigma_list, mu, dt, N=500, n_jobs=-1):
    bins = np.linspace(-1.0, 1.0, N)           # N edges → N-1 interior
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # make Cartesian product once
    tasks = [(g, s) for g in gamma_list for s in sigma_list]

    def run_one(g, s):
        mdl = AMMStationaryDistributionFast(g, mu, s, dt, bins, bin_centers)
        inc, out, terminal_wealth, profit = mdl.collect_results()
        return g, s, inc, out, terminal_wealth, profit

    out = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(run_one)(g, s) for g, s in tasks
    )
    # to DataFrame without importing pandas here:
    return np.array(out, dtype=[("gamma", "f8"), ("sigma", "f8"),
                                ("expected_incoming_fee", "f8"),
                                ("expected_outgoing_fee", "f8"),
                                ("terminal_wealth", "f8"),
                                ("profit", "f8")])
    
    
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from datetime import datetime
    N   = 500
    mu  = 0.0
    dt  = 12.0 / (365*24*60*60)

    gamma_list = np.arange(0.001, 0.201, 0.001)  
    sigma_list = np.arange(0.1, 2.1, 0.1)

    rec = sweep_grid(gamma_list, sigma_list, mu, dt, N=N, n_jobs=-1)
    df = pd.DataFrame(rec)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"vector_theta_analysis_results_large_gamma_{timestamp}.csv", index=False)