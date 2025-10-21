import os
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


class ThetaAnalysis:
    def __init__(
        self,
        npz_path,
        L=1_000_000_000,
        x=1_000_000_000,
        mu=0.0,
        delta_t=1/(365*24*60*5),
        out_dir=None,
        show_plots=True,
        force_compute=True
    ):
        """
        Parameters
        ----------
        npz_path : str
            Path to the .npz file with arrays:
            - bin_edges, sigmas, gammas, counts (S,G,B), underflow (S,G), overflow (S,G)
        L, x, mu, delta_t : float
            Model parameters used in the analytical fee formulas.
        out_dir : str or None
            Where to write CSVs and PNGs. Defaults to directory of the npz file.
        force_compute : first check whether the results are already computed, if not, compute them.
                        if True, compute them even if they are already computed.
                        if False, use the already computed results.
        """
        self.npz_path = npz_path
        self.L = float(L)
        self.x = float(x)
        self.mu = float(mu)
        self.delta_t = float(delta_t)
        self.out_dir = out_dir or os.path.dirname(os.path.abspath(npz_path)) or "."
        self.show_plots = show_plots
        self.force_compute = force_compute
        self._loaded = False
        self._prepared = False
        self._results_df = None  # long-form rows across all S,G,B
        os.makedirs(self.out_dir, exist_ok=True)

    # ---------- Public entry point ----------
    def run(self,
            gamma_fixed_index=3,
            sigma_fixed_index=3,
            make_no_theta_grid=False,
            no_theta_SIGMAS=None,
            no_theta_GAMMAS=None,
            no_theta_thetas=(-1, 0, 1),
            show_plots=True):
        """
        Loads data, vectorizes calculations, saves intermediate CSV, and generates plots.

        Parameters
        ----------
        gamma_fixed_index : int
            Index into `gammas` for the 3x3 histograms (fix gamma).
        sigma_fixed_index : int
            Index into `sigmas` for the 3x3 histograms (fix sigma).
        make_no_theta_grid : bool
            If True, compute & save a grid of fees for given SIGMAS, GAMMAS and discrete thetas.
        no_theta_SIGMAS : array-like or None
            Custom sigma grid (defaults to a 9-point demo grid if None).
        no_theta_GAMMAS : array-like or None
            Custom gamma grid (defaults to np.arange(1e-4, 0.03, 1e-4) if None).
        no_theta_thetas : tuple/list
            Discrete theta values for the no-theta grid.
        """
        self._load()
        if self.force_compute:
            self._prepare_counts()
            self._compute_vectorized_results()
        else:
            self._load_results()
        inter_csv = self._save_intermediate_csv()
        # Plots
        p1 = self.plot_histograms_fix_gamma(gamma_fixed_index=gamma_fixed_index)
        p2 = self.plot_histograms_fix_sigma()
        p3 = self.plot_expected_fees_by_sigma_grid()
        p4 = self.plot_compare_expected_vs_fixed()

        return {
            "intermediate_csv": inter_csv,
            "hist_fixed_gamma_png": p1,
            "hist_fixed_sigma_png": p2,
            "expected_fees_grid_png": p3,
            "compare_expected_vs_fixed_png": p4
        }

    # ---------- Load & prepare ----------
    def _load(self):
        data = np.load(self.npz_path)
        self.edges   = data["bin_edges"]      # (B+1) if edges, or (B) if centers
        self.sigmas  = data["sigmas"]         # (S,)
        self.gammas  = data["gammas"]         # (G,)
        self.counts  = data["counts"]         # (S,G,B)
        self.underflow = data["underflow"]    # (S,G)
        self.overflow  = data["overflow"]     # (S,G)
        self._loaded = True

    def _prepare_counts(self):
        assert self._loaded, "Call _load() first"

        # Merge under/overflow into the first/last bins (avoid double-adding overflow)
        self.counts_mod = self.counts.copy()

        # Probabilities per (sigma, gamma)
        totals = self.counts_mod.sum(axis=2, keepdims=True)
        self.theta_probs = np.divide(
            self.counts_mod, totals,
            out=np.zeros_like(self.counts_mod, dtype=float),
            where=totals > 0
        )  # (S,G,B)

        # Theta values: centers if edges are true edges
        B = self.counts_mod.shape[2]
        if self.edges.size == B + 1:
            self.theta_vals = 0.5 * (self.edges[:-1] + self.edges[1:])
        elif self.edges.size == B:
            self.theta_vals = self.edges
        else:
            raise ValueError("bin_edges length doesn't match counts' bin dimension")

        self._prepared = True

    # ---------- Analytical fee formulas (vectorized) ----------
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

    # ---------- Vectorized results over (S,G,B) ----------
    def _compute_vectorized_results(self):
        assert self._prepared, "Call _prepare_counts() first"

        S, G, B = self.counts_mod.shape

        theta_grid = self.theta_vals[None, None, :]   # (1,1,B)
        sigma_grid = self.sigmas[:, None, None]       # (S,1,1)
        gamma_grid = self.gammas[None, :, None]       # (1,G,1)

        incoming = self._incoming_fee(theta_grid, gamma_grid, sigma_grid)  # (S,G,B)
        outgoing = self._outgoing_fee(theta_grid, gamma_grid, sigma_grid)  # (S,G,B)

        # Flatten to long-form DataFrame
        self._results_df = pd.DataFrame({
            "sigma":        np.repeat(self.sigmas, G*B),
            "gamma":        np.tile(np.repeat(self.gammas, B), S),
            "theta":        np.tile(self.theta_vals, S*G),
            "theta_probs":  self.theta_probs.reshape(-1),
            "incoming_fee": incoming.reshape(-1),
            "outgoing_fee": outgoing.reshape(-1),
        })

        # Add expected fees (per row)
        self._results_df["expected_incoming_fee"] = (
            self._results_df["theta_probs"] * self._results_df["incoming_fee"]
        )
        self._results_df["expected_outgoing_fee"] = (
            self._results_df["theta_probs"] * self._results_df["outgoing_fee"]
        )

        # Aggregate expected fees per (sigma, gamma)
        self._agg_df = (self._results_df
                        .groupby(["sigma", "gamma"], as_index=False)
                        .agg(expected_incoming_fee=("expected_incoming_fee", "sum"),
                             expected_outgoing_fee=("expected_outgoing_fee", "sum")))

    def _load_results(self):
        csv_path = os.path.join(self.out_dir, "theta_distribution_results.csv")
        self._results_df = pd.read_csv(csv_path)
        self._agg_df = self._results_df.groupby(["sigma", "gamma"], as_index=False).agg(
            expected_incoming_fee=("expected_incoming_fee", "sum"),
            expected_outgoing_fee=("expected_outgoing_fee", "sum"))
    
    def _save_intermediate_csv(self):
        assert self._results_df is not None, "Call _compute_vectorized_results() first"
        csv_path = os.path.join(self.out_dir, "theta_distribution_results.csv")
        self._results_df.to_csv(csv_path, index=False)
        return csv_path


    def _rebin_counts_and_edges(self, counts, edges_or_centers, target_bins):
        """
        counts: (S, G, B)
        edges_or_centers: (B+1) if edges, else (B) if centers
        target_bins: desired number of bins (e.g., 100)

        Returns:
            counts_reb: (S, G, target_bins)
            centers_new: (target_bins,)
            widths_new: (target_bins,)  # use for bar width
        """
        S, G, B = counts.shape
        q, r = divmod(B, target_bins)
        # sizes: first r groups have (q+1) bins, rest have q bins
        sizes = np.array([q+1]*r + [q]*(target_bins - r))
        idx = np.cumsum(np.r_[0, sizes[:-1]])  # start indices for each group

        # sum counts across groups along axis=2
        counts_reb = np.add.reduceat(counts, idx, axis=2)

        if edges_or_centers.size == B + 1:
            # we were given true edges → build new edges by grouping
            edges = edges_or_centers
            # group edges: start at edges[idx], end at edges[idx + sizes]
            starts = edges[idx]
            ends   = edges[idx + sizes]
            widths_new = ends - starts
            centers_new = (starts + ends) / 2.0
        else:
            # we were given centers → approximate new centers/widths by grouping
            ctr = edges_or_centers
            # centers: average of centers in the group; widths: span of group
            centers_new = np.array([ctr[i:i+sizes[k]].mean() for k, i in enumerate(idx)])
            # estimate widths using span of centers in each group
            widths_new = np.array([
                (ctr[i:i+sizes[k]].max() - ctr[i:i+sizes[k]].min()) if sizes[k] > 1 else
                (ctr[1] - ctr[0] if ctr.size > 1 else 1.0)
                for k, i in enumerate(idx)
            ])

        return counts_reb, centers_new, widths_new

    
    # ---------- Plots ----------
    def plot_histograms_fix_gamma(self, gamma_fixed_index=3, nrows=3, ncols=3, target_bins=100):
        """3x3 histograms for 9 sigmas, gamma fixed, y-axes independent."""
        S, G, B = self.counts_mod.shape
        if not (0 <= gamma_fixed_index < G):
            raise IndexError("gamma_fixed_index out of range")

        # Build centers if you have edges; otherwise pass centers directly
        edges_or_centers = (self.edges if self.edges.size == B+1
                            else (self.edges if self.edges.size == B else None))
        if edges_or_centers is None:
            raise ValueError("bin_edges length doesn't match counts bins")

        # Rebin from B to target_bins
        counts_reb, ctr_new, width_new = self._rebin_counts_and_edges(
            self.counts_mod, edges_or_centers, target_bins
        )

        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10), sharex=True)
        axes = axes.ravel()

        gi = gamma_fixed_index
        gamma_val = self.gammas[gi]
        sub_sigmas = self.sigmas[:nrows*ncols]

        for si, sigma_val in enumerate(sub_sigmas):
            ax = axes[si]
            ax.bar(ctr_new, counts_reb[si, gi], width=width_new)
            ax.set_title(f"sigma = {sigma_val:.3f}", fontsize=10)
            if si % ncols == 0:
                ax.set_ylabel("Count")
            if si // ncols == (nrows - 1):
                ax.set_xlabel("theta")

        for k in range(len(sub_sigmas), nrows*ncols):
            fig.delaxes(axes[k])

        fig.suptitle(f"Theta counts for gamma = {gamma_val:.4f}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        out_png = os.path.join(self.out_dir, f"hist_theta_fix_gamma_idx{gi}_bins{target_bins}.png")
        fig.savefig(out_png, dpi=150)
        if getattr(self, "show_plots", False):
            plt.show()
        plt.close(fig)
        return out_png


    def plot_histograms_fix_sigma(self, nrows=3, ncols=3, target_bins=100):
        """3x3 histograms for 9 gammas (skip-10 selection), sigma fixed, y-axes independent."""
        S, G, B = self.counts_mod.shape
        sigma_fixed_index = self.sigmas.size // 2 + 1
        if not (0 <= sigma_fixed_index < S):
            raise IndexError("sigma_fixed_index out of range")

        edges_or_centers = (self.edges if self.edges.size == B+1
                            else (self.edges if self.edges.size == B else None))
        if edges_or_centers is None:
            raise ValueError("bin_edges length doesn't match counts bins")

        # Rebin from B to target_bins
        counts_reb, ctr_new, width_new = self._rebin_counts_and_edges(
            self.counts_mod, edges_or_centers, target_bins
        )

        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10), sharex=True)
        axes = axes.ravel()

        si = sigma_fixed_index
        sigma_val = self.sigmas[si]

        # --- Select 9 gammas, every 10th ---
        indices = np.arange(0, 9)
        sub_gammas = self.gammas[indices]

        for idx, (gi, gamma_val) in enumerate(zip(indices, self.gammas)):
            ax = axes[idx]
            ax.bar(ctr_new, counts_reb[si, gi], width=width_new)
            ax.set_title(f"gamma = {gamma_val:.4f}", fontsize=10)
            if idx % ncols == 0:
                ax.set_ylabel("Count")
            if idx // ncols == (nrows - 1):
                ax.set_xlabel("theta")

        # Hide unused panels if fewer than 9 gammas
        for k in range(len(sub_gammas), nrows * ncols):
            fig.delaxes(axes[k])

        fig.suptitle(f"Theta counts for sigma = {sigma_val:.3f}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        out_png = os.path.join(self.out_dir,
                               f"hist_theta_fix_sigma_idx{si}_bins{target_bins}.png")
        fig.savefig(out_png, dpi=150)
        if self.show_plots:
            plt.show()
        plt.close(fig)
        return out_png


    def plot_expected_fees_by_sigma_grid(self, nrows=3, ncols=3, sigma_values=None):
        """
        Grid of subplots: x=gamma, y=expected_incoming_fee, one subplot per sigma.
        """
        assert self._agg_df is not None, "Call _compute_vectorized_results() first"

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*6, nrows*6))
        axes = axes.ravel()

        # Set global font size for readability
        plt.rcParams.update({
            "font.size": 14,        # default text
            "axes.titlesize": 18,   # subplot titles
            "axes.labelsize": 16,   # axis labels
            "xtick.labelsize": 14,  # tick labels
            "ytick.labelsize": 14,
            "legend.fontsize": 14
        })

        sigma_vals = np.sort(self._agg_df["sigma"].unique()) if sigma_values is None else sigma_values
        for i, sigma in enumerate(sigma_vals[:nrows*ncols]):
            ax = axes[i]
            df_s = self._agg_df[self._agg_df["sigma"] == sigma].sort_values("gamma")
            ax.plot(df_s["gamma"].to_numpy(),
                    df_s["expected_incoming_fee"].to_numpy(),
                    marker='o', linewidth=2, markersize=6)
            ax.set_xlabel("Gamma", labelpad=8)
            ax.set_ylabel("Expected Fee", labelpad=8)
            ax.set_title(f"σ={sigma:.3f}")
            ax.grid(True, alpha=0.3)

        # hide unused panels
        for j in range(len(sigma_vals), nrows*ncols):
            axes[j].set_visible(False)

        plt.tight_layout()
        out_png = os.path.join(self.out_dir, "expected_incoming_fee_by_sigma_grid.png")
        fig.savefig(out_png, dpi=200, bbox_inches="tight")  # higher dpi for projection
        if self.show_plots:
            plt.show()
        plt.close(fig)
        return out_png


    # ---------- Optional: discrete-theta grid over custom SIGMAS × GAMMAS ----------
    def _compute_no_theta_grid(self, thetas=(0,)):
        """
        Computes fees for a grid of SIGMAS×GAMMAS×thetas and writes CSV.
        Default is theta = 0 only.
        """
        # Ensure array of thetas
        thetas = np.atleast_1d(thetas).astype(float)

        # Broadcast to (S,G,T)
        S, G, T = self.sigmas.size, self.gammas.size, thetas.size
        sigma_grid = self.sigmas[:, None, None]
        gamma_grid = self.gammas[None, :, None]
        theta_grid = thetas[None, None, :]

        incoming = self._incoming_fee(theta_grid, gamma_grid, sigma_grid)  # (S,G,T)
        outgoing = self._outgoing_fee(theta_grid, gamma_grid, sigma_grid)  # (S,G,T)

        df = pd.DataFrame({
            "sigma":        np.repeat(self.sigmas, G*T),
            "gamma":        np.tile(np.repeat(self.gammas, T), S),
            "theta":        np.tile(thetas, S*G),
            "incoming_fee": incoming.reshape(-1),
            "outgoing_fee": outgoing.reshape(-1),
        })

        csv_path = os.path.join(self.out_dir, "no_theta_results.csv")
        df.to_csv(csv_path, index=False)
        return csv_path

    
    def plot_compare_expected_vs_fixed(self, nrows=3, ncols=3):
        """
        Plot expected incoming fees (distribution-based) vs fixed-theta fees
        for each sigma. One subplot per sigma.
        """
        # Load the two result files
        self._compute_no_theta_grid()
        dist_df = pd.read_csv(os.path.join(self.out_dir, "theta_distribution_results.csv"))
        agg_df = (dist_df.groupby(["sigma", "gamma"], as_index=False)
                         .agg(expected_incoming_fee=("expected_incoming_fee", "sum")))

        fixed_df = pd.read_csv(os.path.join(self.out_dir, "no_theta_results.csv"))

        # Setup grid of subplots
        sigma_vals = np.sort(agg_df["sigma"].unique())
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 18))
        axes = axes.ravel()

        for i, sigma in enumerate(sigma_vals[:nrows*ncols]):
            ax = axes[i]
            # Distribution-based (expected)
            df_s = agg_df[agg_df["sigma"] == sigma].sort_values("gamma")
            ax.plot(df_s["gamma"], df_s["expected_incoming_fee"],
                    label="Expected θ", color="black", lw=2)

            # Fixed-theta overlays
            for theta in sorted(fixed_df["theta"].unique()):
                df_f = fixed_df[(fixed_df["sigma"] == sigma) &
                                (fixed_df["theta"] == theta)].sort_values("gamma")
                ax.plot(df_f["gamma"], df_f["incoming_fee"],
                        label=f"θ={theta}", linestyle="--")

            ax.set_title(f"σ={sigma:.3f}")
            ax.set_xlabel("Gamma")
            ax.set_ylabel("Incoming Fee")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        # Hide unused panels
        for j in range(len(sigma_vals), nrows*ncols):
            axes[j].set_visible(False)

        plt.tight_layout()
        out_png = os.path.join(self.out_dir, "compare_expected_vs_fixed_theta.png")
        fig.savefig(out_png, dpi=150)
        if getattr(self, "show_plots", False):
            plt.show()
        plt.close(fig)
        return out_png




# ---------- Example usage ----------
if __name__ == "__main__":
    npz_path = "/home/shiftpub/Dynamic_AMM/theta_hist_out/hists_20250916_104317.npz"

    ta = ThetaAnalysis(npz_path)
    outputs = ta.run(
        gamma_fixed_index=3,     # gamma ~ 0.03 if that's the 4th element
        sigma_fixed_index=3,     # sigma ~ 0.10 if that's the 4th element
        make_no_theta_grid=True  # also produce no_theta_results.csv
    )
    print("Artifacts written:")
    for k, v in outputs.items():
        print(f" - {k}: {v}")
