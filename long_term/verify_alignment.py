# verify_alignment.py
import numpy as np
from amm_stationary import AMMStationaryDistributionFast

def build_bins(K=101):
    """
    Make symmetric bins for theta in [-1, 1] with K-1 interior bins.
    Returns (bins, bin_centers).
    """
    bins = np.linspace(-1.0, 1.0, K)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return bins, centers

def main():
    # ---------- Model params ----------
    gamma = 0.0003          # fee (e.g., 30 bps)
    mu = 0.0               # drift used in theta dynamics formula (not in GBM below)
    sigma = 1.0            # annual vol for GBM
    dt = 12.0 / (365 * 24 * 60 * 60)  # 1-minute in trading years

    # theta discretization
    bins, centers = build_bins(K=101)

    # Instance
    amm = AMMStationaryDistributionFast(
        gamma=gamma, mu=mu, sigma=sigma, dt=dt, bins=bins, bin_centers=centers,
        L=1_000_000.0, x=1_000_000.0
    )

    # ---------- Window and simulation settings ----------
    n_steps = 1      # total steps
    # window_len = 1000       # compare over last 100 steps
    # N0 = n_steps - window_len

    # For apples-to-apples: use the SAME S0 handling on both sides
    S0_mode = "flat"       # "flat" or "theta_shifted"
    nodes = 64           # GH nodes for analytic integration
    n_paths = 10000      # Monte Carlo paths
    seed = 42

    # ---------- Analytic cumulative expectation ----------
    # analytic_total = amm.cumulative_fee_analytic(N0=N0, window_len=window_len,
    #                                              nodes=nodes, S0_mode=S0_mode)
    for N0 in np.arange(0, 10000, 1000):
    # ---------- Simulation ----------
        analytic_total = amm.cumulative_fee_analytic(N0=N0, window_len=n_steps)
        print(f"N0={N0}, avg={analytic_total}")
        # sim = amm.simulate_fees_from_N0(
        #     n_paths=n_paths,
        #     n_steps=n_steps,
        #     N0=N0,
        #     mode="Incoming",
        #     S0_mode=S0_mode,
        #     seed=seed,
        # )
        # print(f"N0={N0}, avg={sim['fees_tail_y_equiv'].mean()}")

    # mc_samples = sim["fees_tail_y_equiv"]
    # mc_mean = float(mc_samples.mean())
    # mc_std  = float(mc_samples.std(ddof=1))
    # mc_se   = mc_std / np.sqrt(n_paths)

    # # 95% CI for the MC mean
    # ci_lo = mc_mean - 1.96 * mc_se
    # ci_hi = mc_mean + 1.96 * mc_se

    # # ---------- Report ----------
    # print("=== Alignment Check (Analytic vs Simulation) ===")
    # print(f"sigma={sigma:.4f}, dt={dt:.5e}, gamma={gamma:.4f}")
    # print(f"n_steps={n_steps}, window_len={window_len}, N0={N0}, n_paths={n_paths}, nodes={nodes}")
    # print(f"S0_mode={S0_mode}")
    # print()
    # print(f"Analytic sum over [N0, N0+K-1]: {analytic_total:,.6f}")
    # print(f"MC mean   over [N0, N0+K-1]:   {mc_mean:,.6f}")
    # print(f"MC 95% CI: [{ci_lo:,.6f}, {ci_hi:,.6f}]")
    # print(f"Abs diff: {abs(mc_mean - analytic_total):,.6f}")
    # if analytic_total != 0:
    #     print(f"Rel diff: {abs(mc_mean - analytic_total)/abs(analytic_total):.4%}")

    # # Optional: sanity prints of reserves
    # xt, yt = sim["final_xt"], sim["final_yt"]
    # print("\nFinal reserves (means):")
    # print(f"  E[X_T] = {xt.mean():,.2f}")
    # print(f"  E[Y_T] = {yt.mean():,.2f}")

if __name__ == "__main__":
    main()
