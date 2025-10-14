import numpy as np
from scipy.stats import norm
from scipy.linalg import eig
import matplotlib.pyplot as plt
from tqdm import tqdm

class AMMStationaryDistribution:
    """
    Computes the stationary distribution of AMM mispricing Θ and expected fee revenue.
    
    Based on the paper: "Using Linear Algebra to Solve the Stationary Distribution 
    of Θ and Its Application to AMM Fee Revenue"
    """
    
    def __init__(self, gamma, mu, sigma, dt, N=50):
        """
        Initialize the AMM model.
        
        Parameters:
        -----------
        gamma : float
            Fee rate (e.g., 0.003 for 0.3%)
        mu : float
            Drift of external price (annualized)
        sigma : float
            Volatility of external price (annualized)
        dt : float
            Time step size
        N : int
            Number of interior bins + 1 (total states = N + 1)
        """
        self.gamma = gamma
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.N = N
        self.L = 1_000_000.0
        self.x = 1_000_000.0
        self.delta_t = dt
        
        # Fee scale
        self.lambda_fee = -np.log(1 - gamma)
        
        # Drift and variance parameters for Θ dynamics
        self.m = (mu - 0.5 * sigma**2) / self.lambda_fee
        self.v = sigma**2 / self.lambda_fee**2
        
        # Discretization: boundaries at -1 and +1, N-1 interior bins
        self.bins = np.linspace(-1, 1, N)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2
        
        # State ordering: [bin_1, ..., bin_{N-1}, boundary_minus, boundary_plus]
        self.num_states = N + 1
        
        # Build transition matrix
        self.P = self._build_transition_matrix()
        
        # Compute stationary distribution
        self.pi_star = self._compute_stationary_distribution()
        
    def _kernel_density(self, y, x):
        """Gaussian kernel k(y|x) for interior y."""
        mean = x - self.m * self.dt
        std = np.sqrt(self.v * self.dt)
        return norm.pdf(y, loc=mean, scale=std)
    
    def _kernel_boundary_minus(self, x):
        """Probability of hitting boundary at -1 given x."""
        mean = x - self.m * self.dt
        std = np.sqrt(self.v * self.dt)
        return norm.cdf(-1, loc=mean, scale=std)
    
    def _kernel_boundary_plus(self, x):
        """Probability of hitting boundary at +1 given x."""
        mean = x - self.m * self.dt
        std = np.sqrt(self.v * self.dt)
        return 1 - norm.cdf(1, loc=mean, scale=std)
    
    def _transition_from_state(self, x):
        """
        Compute transition probabilities from state x.
        
        Returns:
        --------
        interior_probs : array of length N-1
            Probabilities of transitioning to each interior bin
        prob_minus : float
            Probability of transitioning to boundary at -1
        prob_plus : float
            Probability of transitioning to boundary at +1
        """
        mean = x - self.m * self.dt
        std = np.sqrt(self.v * self.dt)
        
        # Interior bin probabilities
        interior_probs = np.zeros(self.N - 1)
        for j in range(self.N - 1):
            a_j_minus = self.bins[j]
            a_j = self.bins[j + 1]
            interior_probs[j] = norm.cdf(a_j, loc=mean, scale=std) - \
                               norm.cdf(a_j_minus, loc=mean, scale=std)
        
        # Boundary probabilities
        prob_minus = norm.cdf(self.bins[0], loc=mean, scale=std)
        prob_plus = 1 - norm.cdf(self.bins[-1], loc=mean, scale=std)
        
        return interior_probs, prob_minus, prob_plus
    
    def _build_transition_matrix(self):
        """Build the column-stochastic transition matrix P."""
        P = np.zeros((self.num_states, self.num_states))
        
        # Transitions from interior bins
        for i in range(self.N - 1):
            x_i = self.bin_centers[i]
            interior_probs, prob_minus, prob_plus = self._transition_from_state(x_i)
            
            # Fill column i
            P[:self.N-1, i] = interior_probs
            P[self.N-1, i] = prob_minus
            P[self.N, i] = prob_plus
        
        # Transitions from boundary at -1 and +1 (column N-1 and N)

        x_minus = self.bins[0]
        interior_probs, prob_minus, prob_plus = self._transition_from_state(x_minus)
        P[:self.N-1, self.N-1] = interior_probs
        P[self.N-1, self.N-1] = prob_minus
        P[self.N, self.N-1] = prob_plus
                
        x_plus = self.bins[-1]
        interior_probs, prob_minus, prob_plus = self._transition_from_state(x_plus)
        P[:self.N-1, self.N] = interior_probs
        P[self.N-1, self.N] = prob_minus
        P[self.N, self.N] = prob_plus
        
        # Verify column-stochastic property
        col_sums = P.sum(axis=0)
        assert np.allclose(col_sums, 1.0), f"Column sums not 1: {col_sums}"
        
        return P
    
    def _compute_stationary_distribution(self):
        """Compute stationary distribution as eigenvector with eigenvalue 1."""
        eigenvalues, eigenvectors = eig(self.P)
        
        # Find eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        
        # Normalize to probability distribution
        pi = np.abs(pi)
        pi = pi / pi.sum()
        
        return pi
    
    def _incoming_fee(self, theta, gamma, sigma):
        L, x, mu, dt = self.L, self.x, self.mu, self.delta_t
        y = L**2 / x
        p0 = (y/x) * np.power(1 - gamma, -theta)

        sqrt_dt = np.sqrt(dt)
        sig_sqrt_dt = sigma * sqrt_dt

        d1 = (np.log((1 - gamma) * y / (p0 * x)) - mu * dt) / sig_sqrt_dt
        d2 = (np.log(y / ((1 - gamma) * x * p0)) - mu * dt) / sig_sqrt_dt

        alpha = L * np.sqrt((1 - gamma) * p0) * np.exp(mu/2 - (sigma**2) * dt / 8)

        return (gamma / (1 - gamma)) * (
            alpha * (norm.cdf(d1) + norm.cdf(-d2))
            - np.exp(mu * dt) * p0 * x * norm.cdf(d1 - sig_sqrt_dt / 2)
            - y * norm.cdf(-d2 - sig_sqrt_dt / 2)
        )

    def _outgoing_fee(self, theta, gamma, sigma):
        L, x, mu, dt = self.L, self.x, self.mu, self.delta_t
        y = L**2 / x
        p0 = (y/x) * np.power(1 - gamma, -theta)

        sqrt_dt = np.sqrt(dt)
        sig_sqrt_dt = sigma * sqrt_dt

        d1 = (np.log((1 - gamma) * y / (p0 * x)) - mu * dt) / sig_sqrt_dt
        d2 = (np.log(y / ((1 - gamma) * x * p0)) - mu * dt) / sig_sqrt_dt

        alpha = L * np.sqrt(p0 / (1 - gamma)) * np.exp(mu/2 - (sigma**2) * dt / 8)

        return (gamma * (
            - alpha * (norm.cdf(d1) + norm.cdf(-d2))
            + np.exp(mu * dt) * p0 * x * norm.cdf(-d2 + sig_sqrt_dt / 2)
            + y * norm.cdf(d1 + sig_sqrt_dt / 2)
        ))
        
    def collect_results(self):
        theta_list = np.concatenate([self.bin_centers, [-1, 1]])
        expected_incoming_fee = 0
        expected_outgoing_fee = 0
        total_prob = 0
        eps = 1e-6
        for idx in range(len(theta_list)):
            theta = theta_list[idx]
            prob = self.pi_star[idx]
            total_prob += prob
            expected_incoming_fee += prob * self._incoming_fee(theta, self.gamma, self.sigma)
            expected_outgoing_fee += prob * self._outgoing_fee(theta, self.gamma, self.sigma)
        
        assert abs(total_prob - 1) < eps, f"Total probability is not 1: {total_prob}"
        return expected_incoming_fee, expected_outgoing_fee
        
    def plot_stationary_distribution(
        self,
        show_empirical: bool = True,
        steps: int = 500_000,
        runs: int = 5000,
        warmup: int = 50_000,
        X0: float = 1_000_000.0,
        Y0: float = 1_000_000.0,
        rng: np.random.Generator = None,
    ):
        """Plot theoretical and empirical stationary distributions side-by-side with a shared y-axis."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        ax_th, ax_emp = axes

        # --- Theoretical distribution (eigenvector) ---
        interior_probs = self.pi_star[: self.N - 1]
        width = self.bins[1] - self.bins[0]

        ax_th.bar(
            self.bin_centers,
            interior_probs,
            width=width,
            alpha=0.7,
            edgecolor="black",
            label="Interior bins",
        )
        ax_th.bar([-1.0], [self.pi_star[self.N - 1]], width=width, color="black", alpha=0.9, label="Boundary -1")
        ax_th.bar([ 1.0], [self.pi_star[self.N    ]], width=width, color="black", alpha=0.9, label="Boundary +1")

        ax_th.set_title("Theoretical", fontsize=13)
        ax_th.set_xlabel("Mispricing Θ", fontsize=12)
        ax_th.set_ylabel("Probability", fontsize=12)
        ax_th.grid(True, alpha=0.3)
        ax_th.legend()

        # --- Empirical distribution (simulation) ---
        if show_empirical:
            centers, pdf_interior, p_left, p_right = self.simulate_histogram(
                steps=steps, runs=runs, warmup=warmup, X0=X0, Y0=Y0, rng=rng, bins=self.bins
            )
            # interior bars
            ax_emp.bar(
                centers,
                pdf_interior,
                width=width,
                alpha=0.7,
                edgecolor="black",
                label="Empirical",
            )

            # empirical boundary masses as thin bars
            thin = min(width, 0.05)
            ax_emp.bar([-1.0], [p_left ], width=thin, color="black", alpha=0.9, label="Empirical -1")
            ax_emp.bar([ 1.0], [p_right], width=thin, color="black", alpha=0.9, label="Empirical +1")
        else:
            ax_emp.text(
                0.5, 0.5, "Empirical simulation skipped",
                ha="center", va="center", transform=ax_emp.transAxes, fontsize=12
            )

        ax_emp.set_title("Empirical (Simulation)", fontsize=13)
        ax_emp.set_xlabel("Mispricing Θ", fontsize=12)
        ax_emp.grid(True, alpha=0.3)
        ax_emp.legend()

        fig.suptitle("Stationary Distribution of AMM Mispricing", fontsize=15)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    
    def simulate_histogram(
        self,
        steps: int,
        runs: int = 500,
        warmup: int = 50_000,
        X0: float = 1_000_000.0,
        Y0: float = 1_000_000.0,
        rng: np.random.Generator = None,
        bins: np.ndarray = None,
        tol: float = 1e-4,  
    ):
        
        if rng is None:
            rng = np.random.default_rng()

        gamma = self.gamma
        dt    = self.dt
        mu    = self.mu
        sigma = self.sigma

        lam   = np.log(1.0 - gamma)  # negative
        diff  = sigma * np.sqrt(dt)
        drift = (mu - 0.5 * sigma * sigma) * dt

        # Use class edges; ensures perfect alignment with analytical bins
        edges = self.bins if bins is None else bins
        K = len(edges) - 1                         # # interior bins
        centers = 0.5 * (edges[:-1] + edges[1:])   # interior bin centers

        counts = np.zeros(K, dtype=np.int64)
        left_count = 0
        right_count = 0

        # Vector state
        S = np.full(runs, 1.0, dtype=float)
        X = np.full(runs, 1_000_000.0 if X0 is None else X0, dtype=float)
        Y = np.full(runs, 1_000_000.0 if Y0 is None else Y0, dtype=float)
        L0 = np.sqrt(X * Y)

        for t in tqdm(range(steps)):
            # GBM step
            z = rng.standard_normal(runs)
            S *= np.exp(drift + diff * z)

            P  = Y / X
            ub = P / (1.0 - gamma)
            lb = P * (1.0 - gamma)

            hi = S > ub
            lo = S < lb

            if hi.any():
                X_hi = L0[hi] / np.sqrt((1.0 - gamma) * S[hi])
                Y_hi = L0[hi] * np.sqrt((1.0 - gamma) * S[hi])
                X[hi], Y[hi] = X_hi, Y_hi

            if lo.any():
                X_lo = L0[lo] * np.sqrt((1.0 - gamma) / S[lo])
                Y_lo = L0[lo] * np.sqrt(S[lo] / (1.0 - gamma))
                X[lo], Y[lo] = X_lo, Y_lo

            # Update P after any rebalancing
            P = Y / X

            # Mispricing θ in [-1, 1]
            theta = np.clip(np.log(P / S) / lam, -1.0, 1.0)

            if t >= warmup:
                # Boundary masks (exact hits after clip)
                left_mask  = theta <= (-1.0 + tol)
                right_mask = theta >= ( 1.0 - tol)

                # Count boundaries
                left_count  += np.count_nonzero(left_mask)
                right_count += np.count_nonzero(right_mask)

                # Interior indices (strictly inside (-1,1))
                interior = ~(left_mask | right_mask)
                if interior.any():
                    idx = np.searchsorted(edges, theta[interior], side="right") - 1
                    # idx should lie in [0, K-1]; guard just in case of FP edge
                    np.clip(idx, 0, K - 1, out=idx)
                    # accumulate counts
                    for k in range(K):
                        counts[k] += np.count_nonzero(idx == k)

        total = counts.sum() + left_count + right_count
        if total > 0:
            pdf_interior = counts / total
            p_left  = left_count  / total
            p_right = right_count / total
        else:
            pdf_interior = counts.astype(float)
            p_left = p_right = 0.0

        return centers, pdf_interior, p_left, p_right


if __name__ == "__main__":
    
    import pandas as pd
    from tqdm import tqdm
    
    N = 500
    mu = 0.0
    dt    = 12.0 / (365*24*60*60)
    results = []
    gamma_list = np.arange(0.003, 0.004, 0.0005)
    sigma_list = np.arange(0.1, 0.5, 0.1)
    total_combinations = len(gamma_list) * len(sigma_list)
    pbar = tqdm(total=total_combinations, desc="Processing combinations")
    for gamma in gamma_list:
        for sigma in sigma_list:
            model = AMMStationaryDistribution(gamma, mu, sigma, dt, N=N)
            expected_incoming_fee, expected_outgoing_fee = model.collect_results()
            results.append({
                "gamma": gamma,
                "sigma": sigma,
                "expected_incoming_fee": expected_incoming_fee,
                "expected_outgoing_fee": expected_outgoing_fee
            })
            pbar.update(1)
    pbar.close()
    df = pd.DataFrame(results)
    df.to_csv("theta_analysis_results.csv", index=False)

   